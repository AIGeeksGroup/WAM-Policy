"""Joint video + 10D state -> action sampling with a bounded observed-history KV cache.

Drop-in for :class:`wam_policy.inference.sampler.VideoActionSampler` inside
:class:`wam_policy.inference.session.OnlineSession`: the "latent" handed to ``reset``/``commit`` is
the pair ``(video [1,48,1,24,20], proprioception [1,10,1,n,1])`` produced by
:class:`wam_policy.ablations.proprioception.serving.ProprioceptionPipeline`.

Per group g the video latent and its states share logical window unit 2g and one timestep
(``forward_embodied``); actions use unit 2g+1. Group 0 holds the initial observation and one state
and, unlike the RGB-only teacher, no action placeholder. Predicted K/V are temporary until the
execution feedback of the group is committed.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from wan_va.utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id
from wam_policy.common.flow_config import FlowConfig
from wam_policy.inference.sampler import SamplingConfig

VIDEO_SHAPE = (1, 48, 1, 24, 20)
STATE_SHAPE = (1, 10, 1, 4, 1)
ACTION_SHAPE = (1, 9, 1, 16, 1)


def video_grid(model, group, device):
    return get_mesh_id(1, 24 // model.patch_size[1], 20 // model.patch_size[2], 0, f_shift=group).to(device)[None]


def state_grid(group, count, device):
    if (group == 0 and count != 1) or (group > 0 and count != 4):
        raise ValueError("Group 0 has one state, every later group four")
    temporal = (torch.zeros(count, device=device) if group == 0 else
                group - 1 + torch.arange(1, count + 1, device=device) / count)
    return torch.stack([temporal, torch.full_like(temporal, -2.), torch.full_like(temporal, -2.),
                        torch.full_like(temporal, 2.)], dim=0)[None]


def action_input(action, text, group, timestep):
    return {"noisy_latents": action,
            "timesteps": torch.full((action.shape[0], 1), float(timestep), device=action.device),
            "grid_id": get_mesh_id(1, 16, 1, 1, f_shift=group, action=True).to(action.device)[None],
            "text_emb": text}


class ProprioceptionSampler:
    def __init__(self, model, action_normalizer, config=SamplingConfig.preset("25/50")):
        self.model = model.eval().requires_grad_(False)
        self.normalizer = action_normalizer
        self.config = config.validate()
        if self.config.video_guidance != 1 or self.config.action_guidance != 1:
            raise ValueError("The proprioception teacher samples without CFG (guidance 1)")
        if getattr(model, "proprioception_dim", 0) != 10 or model.config.action_dim != 9:
            raise ValueError("Native9 model with proprioception_dim=10 required")
        if model.config.attn_mode == "flex":
            raise ValueError("Online inference uses cached torch/flashattn attention, not the flex training mask")
        weight = next(model.parameters())
        self.device, self.dtype = weight.device, weight.dtype
        self.cache_name = "wam-online-proprioception"
        self.clear()

    def clear(self):
        self.model.clear_cache(self.cache_name)
        self.group = 0
        self.pending = False
        self.ready = False
        self.generator = None
        self.text = None
        self.last_diagnostics = {}

    def _caches(self):
        return [b.attn1.attn_caches[self.cache_name] for b in self.model.blocks]

    def _query(self, logical):
        for cache in self._caches():
            cache["query_logical_id"] = logical

    def _embodied(self, video, state, group, timestep, update):
        self._query(2 * group)
        return self.model.forward_embodied(video, state, self.text, video_grid(self.model, group, self.device),
                                           state_grid(group, state.shape[3], self.device),
                                           torch.full((1, 1), float(timestep), device=self.device),
                                           update_cache=update, cache_name=self.cache_name)

    def _action(self, action, group, timestep, update):
        self._query(2 * group + 1)
        return self.model(action_input(action, self.text, group, timestep), action_mode=True,
                          update_cache=update, cache_name=self.cache_name)

    @staticmethod
    def _split(observation, initial):
        video, state = observation
        if tuple(video.shape) != VIDEO_SHAPE or tuple(state.shape) != (1, 10, 1, 1 if initial else 4, 1) \
                or not torch.isfinite(video).all() or not torch.isfinite(state).all():
            raise ValueError("Expected finite video [1,48,1,24,20] and states [1,10,1,1|4,1]")
        return video, state

    @torch.inference_mode()
    def reset(self, observation, text_emb, *, seed, empty_text_emb=None):
        self.clear()
        video, state = self._split(observation, initial=True)
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Explicit nonnegative per-episode policy seed required")
        if tuple(text_emb.shape) != (1, 512, 4096) or not torch.isfinite(text_emb).all():
            raise ValueError("Finite [1,512,4096] prompt required")
        self.text = text_emb.to(self.device, self.dtype)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        tokens = (24 // self.model.patch_size[1]) * (20 // self.model.patch_size[2]) + 4
        # Two extra group capacities keep predictions from evicting the observed window.
        self.model.create_empty_cache(self.cache_name, self.config.window_units + 4, tokens, 16,
                                      self.device, self.dtype, 1)
        for cache in self._caches():
            cache["logical_id"] = torch.full_like(cache["id"], -1)
            cache["window_units"] = self.config.window_units
            cache["query_logical_id"] = 0
        self._embodied(video.to(self.device, self.dtype), state.to(self.device, self.dtype), 0, 0, 2)
        self.group = 1
        self.ready = True

    def _evict(self):
        """Hide observed groups that left the training window (state+video and action units)."""
        floor = 2 * max(0, self.group - self.config.window_units // 2)
        for cache in self._caches():
            cache["mask"][cache["mask"] & (cache["logical_id"] < floor)] = False

    @torch.inference_mode()
    def predict(self):
        if not self.ready or self.pending:
            raise ValueError("One prediction requires initialized, confirmed history")
        self._evict()
        flow, cfg = FlowConfig(), self.config
        video = torch.randn(VIDEO_SHAPE, generator=self.generator, dtype=self.dtype, device=self.device)
        state = torch.randn(STATE_SHAPE, generator=self.generator, dtype=self.dtype, device=self.device)
        scheduler = FlowMatchScheduler(shift=flow.video_snr_shift, sigma_min=0.0, extra_one_step=True)
        scheduler.set_timesteps(cfg.video_steps)
        for index, t in enumerate([*scheduler.timesteps, torch.tensor(0.)]):
            final = index == cfg.video_steps
            vfield, sfield = self._embodied(video, state, self.group, t, 1 if final else 0)
            if not final:
                video = scheduler.step(data_seq_to_patch(self.model.patch_size, vfield, 1, 24, 20, batch_size=1), t, video)
                state = scheduler.step(sfield.transpose(1, 2).reshape_as(state), t, state)
        action = torch.randn(ACTION_SHAPE, generator=self.generator, dtype=self.dtype, device=self.device)
        scheduler = FlowMatchScheduler(shift=flow.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        scheduler.set_timesteps(cfg.action_steps)
        for index, t in enumerate([*scheduler.timesteps, torch.tensor(0.)]):
            final = index == cfg.action_steps
            field = self._action(action, self.group, t, 1 if final else 0)
            if not final:
                action = scheduler.step(field.transpose(1, 2).reshape_as(action), t, action)
        commands = self.normalizer.decode(action[0, :, 0, :, 0].float().cpu().numpy().T)
        commands[:, 6:] = 0  # explicit physical force boundary, after affine decode
        if commands.shape != (16, 9) or not np.isfinite(commands).all():
            raise ValueError("Finite native9 command group required")
        self.pending = True
        self.last_diagnostics = {"latent_group": self.group, "sampling": asdict(cfg),
                                 "predicted_state": state[0, :, 0, :, 0].T.float().cpu().numpy().tolist()}
        return commands

    @torch.inference_mode()
    def commit(self, observation, executed_commands):
        if not self.pending:
            raise ValueError("No pending action prediction")
        video, state = self._split(observation, initial=False)
        commands = np.asarray(executed_commands, np.float32)
        if commands.shape != (16, 9):
            raise ValueError("Only a complete execution-confirmed group may commit")
        action = torch.from_numpy(self.normalizer.encode(commands).T.copy())[None, :, None, :, None]
        self.model.clear_pred_cache(self.cache_name)
        self._embodied(video.to(self.device, self.dtype), state.to(self.device, self.dtype), self.group, 0, 2)
        self._action(action.to(self.device, self.dtype), self.group, 0, 2)
        self.group += 1
        self.pending = False

    def cache_summary(self):
        if not self.ready:
            return {"observed_units": [], "predicted_units": []}
        cache = self._caches()[0]
        return {name: sorted(set(cache["logical_id"][cache["mask"] & pred].cpu().tolist()))
                for name, pred in (("observed_units", ~cache["is_pred"]), ("predicted_units", cache["is_pred"]))}
