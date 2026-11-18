"""Native9 video→action flow sampling, with a bounded observed-history cache.

One group is one causal video latent (four new 12.5 Hz RGB frames) and
sixteen completed 50 Hz actions. Window units follow training: video=2g,
action=2g+1. Predicted K/V are temporary until actual execution feedback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from wan_va.utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id
from wam_policy.common.flow_config import FlowConfig


NFE_PRESETS = {"1/2": (1, 2), "25/50": (25, 50)}
ACTION_NOISE_SHAPE = (1, 9, 1, 16, 1)


@dataclass(frozen=True)
class SamplingConfig:
    video_steps: int = 5
    action_steps: int = 10
    video_guidance: float = 1.0
    action_guidance: float = 1.0
    window_units: int = 8

    @classmethod
    def preset(cls, name, *, window_units=8):
        """Named NFE settings : '1/2' (OPD student / undistilled low-NFE) or '25/50' (teacher)."""
        if name not in NFE_PRESETS:
            raise ValueError(f"Unknown NFE preset {name!r}; use one of {sorted(NFE_PRESETS)}")
        video, action = NFE_PRESETS[name]
        return cls(video, action, window_units=window_units).validate()

    @property
    def nfe(self):
        return f"{self.video_steps}/{self.action_steps}"

    def validate(self):
        for key in ("video_steps", "action_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("Positive full-denoising step counts required")
        if type(self.window_units) is not int or not 4 <= self.window_units <= 64 or self.window_units % 2:
            raise ValueError("Even training logical window units in [4,64] required")
        if any(not math.isfinite(v) or v < 1 for v in (self.video_guidance, self.action_guidance)):
            raise ValueError("Finite CFG scales >=1 required")
        return self


class VideoActionSampler:
    """No dataset, simulator, pretrained conversion or global RNG dependency."""

    def __init__(self, model, normalizer, config=SamplingConfig()):
        self.model = model.eval().requires_grad_(False)
        self.normalizer = normalizer
        self.config = config.validate()
        weight = next(model.parameters())
        self.device, self.dtype = weight.device, weight.dtype
        if model.config.action_dim != 9 or tuple(model.config.patch_size) != (1, 2, 2):
            raise ValueError("Native9 / original video patch geometry required")
        # Training's flex/reference masks are global, not a serving backend.
        if model.config.attn_mode != "torch":
            raise ValueError("Online inference uses explicit cached torch SDPA attention")
        self.cache_name = "wam-online-v2"
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

    def _forward(self, value, timestep, *, group, action=False, update=0):
        """Respect training's modality-level window without evicting hidden history.

        Ordinary diffusion uses scratch K/V in WanAttention. Predicted persistent
        writes have reserved space; occupied observed slots retain their physical
        allocation while attention filters their logical positions independently.
        """
        logical = 2 * group + int(action)
        batch = self.text.shape[0]
        grid = get_mesh_id(1, 16 if action else 12, 1 if action else 10,
                           int(action), 1, group, action=action).to(self.device)
        payload = {"noisy_latents": value.repeat(batch, 1, 1, 1, 1),
                   "text_emb": self.text,
                   "timesteps": torch.full((batch, 1), float(timestep), device=self.device),
                   "grid_id": grid[None].repeat(batch, 1, 1)}
        for cache in self._caches():
            cache["query_logical_id"] = logical
            cache["window_units"] = self.config.window_units
        return self.model(payload, update_cache=update, cache_name=self.cache_name, action_mode=action)

    @torch.inference_mode()
    def reset(self, latent, text_emb, *, seed, empty_text_emb=None):
        self.clear()
        self._validate_latent(latent)
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Explicit nonnegative per-episode policy seed required")
        if text_emb.shape != (1, 512, 4096) or not torch.isfinite(text_emb).all():
            raise ValueError("Finite [1,512,4096] prompt required")
        cfg = max(self.config.video_guidance, self.config.action_guidance) > 1
        if cfg and (empty_text_emb is None or empty_text_emb.shape != text_emb.shape or not torch.isfinite(empty_text_emb).all()):
            raise ValueError("CFG requires verified empty-prompt embedding")
        self.text = (torch.cat((text_emb, empty_text_emb)) if cfg else text_emb).to(self.device, self.dtype)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        # The extra two group capacities prevent prediction from evicting observation.
        self.model.create_empty_cache(self.cache_name, self.config.window_units + 4, 120, 16,
                                      self.device, self.dtype, self.text.shape[0])
        for cache in self._caches():
            cache["logical_id"] = torch.full_like(cache["id"], -1)
        self._forward(latent.to(self.device, self.dtype), 0, group=0, update=2)
        # The initial group has no executed actions: its zero, masked action placeholders still
        # provide context (as in training), but never decode or send these slots as physical EEF targets.
        self._forward(torch.zeros(1, 9, 1, 16, 1, device=self.device, dtype=self.dtype),
                      0, group=0, action=True, update=2)
        self.group = 1
        self.ready = True

    @staticmethod
    def _validate_latent(latent):
        if latent.shape != (1, 48, 1, 24, 20) or not torch.isfinite(latent).all():
            raise ValueError("One finite dual-view causal latent group required")

    def _sample(self, shape, *, steps, shift, action, guidance, noise_offset=None):
        value = torch.randn(shape, generator=self.generator, device=self.device, dtype=self.dtype)
        if noise_offset is not None:
            # The offset shifts the initial action noise; the generator draw is unchanged.
            if tuple(noise_offset.shape) != tuple(shape) or not torch.isfinite(noise_offset).all():
                raise ValueError(f"Finite initial-noise offset of shape {tuple(shape)} required")
            value = value + noise_offset.to(self.device, self.dtype)
        scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
        scheduler.set_timesteps(steps)
        for t in scheduler.timesteps:
            pred = self._forward(value, t, group=self.group, action=action)
            if action:
                pred = pred.transpose(1, 2).reshape(self.text.shape[0], 9, 1, 16, 1)
            else:
                pred = data_seq_to_patch((1, 2, 2), pred, 1, 24, 20, batch_size=self.text.shape[0])
            pred = pred[:1] if self.text.shape[0] == 1 else pred[1:] + guidance * (pred[:1] - pred[1:])
            value = scheduler.step(pred, t, value).to(self.dtype)
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite flow sampling result")
        self._forward(value, 0, group=self.group, action=action, update=1)
        return value

    @torch.inference_mode()
    def predict(self, *, action_noise_offset=None):
        """One command group. ``action_noise_offset`` (optional, shaped like the action noise) is added
        to the initial action noise draw (used by Task RL's window-shared noise). The caller commits the
        commands it actually executed."""
        if not self.ready or self.pending:
            raise ValueError("One prediction requires initialized, confirmed history")
        cfg, flow = self.config, FlowConfig()
        self._sample((1, 48, 1, 24, 20), steps=cfg.video_steps, shift=flow.video_snr_shift,
                     action=False, guidance=cfg.video_guidance)
        actions = self._sample(ACTION_NOISE_SHAPE, steps=cfg.action_steps, shift=flow.action_snr_shift,
                               action=True, guidance=cfg.action_guidance, noise_offset=action_noise_offset)
        commands = self.normalizer.decode(actions[0, :, 0, :, 0].float().cpu().numpy().T)
        commands[:, 6:] = 0  # explicit physical force boundary, after affine decode
        if commands.shape != (16, 9) or not np.isfinite(commands).all():
            raise ValueError("Finite native9 command group required")
        self.pending = True
        self.last_diagnostics = {"latent_group": self.group, "sampling": asdict(cfg),
                                 "history_source": "confirmed_physical_feedback"}
        if action_noise_offset is not None:
            self.last_diagnostics["noise_offset"] = True
        return commands

    @torch.inference_mode()
    def commit(self, latent, executed_commands):
        if not self.pending:
            raise ValueError("No pending action prediction")
        self._validate_latent(latent)
        commands = np.asarray(executed_commands, np.float32)
        if commands.shape != (16, 9):
            raise ValueError("Only a complete execution-confirmed group may commit")
        normalized = self.normalizer.encode(commands)
        value = torch.from_numpy(normalized.T.copy()).reshape(1, 9, 1, 16, 1)
        self._commit_normalized(latent, value)

    def _commit_normalized(self, latent, value):
        """Commit an already encoded, execution-confirmed history group."""
        self.model.clear_pred_cache(self.cache_name)
        self._forward(latent.to(self.device, self.dtype), 0, group=self.group, update=2)
        self._forward(value.to(self.device, self.dtype), 0, group=self.group, action=True, update=2)
        self.group += 1
        # Evict complete observed video/action groups, never part of a tensor row.
        minimum = 2 * max(0, self.group - self.config.window_units // 2)
        for cache in self._caches():
            expired = cache["mask"] & (cache["logical_id"] < minimum)
            cache["mask"][expired] = False
            cache["id"][expired] = -1
        self.pending = False

    def cache_summary(self):
        if not self.ready:
            return {"observed_units": [], "predicted_units": []}
        cache = self._caches()[0]
        return {name: sorted(set(cache["logical_id"][cache["mask"] & pred].cpu().tolist()))
                for name, pred in (("observed_units", ~cache["is_pred"]), ("predicted_units", cache["is_pred"]))}

