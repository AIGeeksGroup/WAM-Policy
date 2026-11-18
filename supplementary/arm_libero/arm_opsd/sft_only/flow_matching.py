"""SFT only: the upstream LingBot-VA joint flow-matching loss (clean/noisy packing with explicit, sample-isolated
SDPA masks), applied to the LoRA student on whole demonstration episodes."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class TrainingSchedule:
    """Exactly the native 1000-step, extra-one-step training schedule."""
    sigmas: torch.Tensor
    weights: torch.Tensor

    @classmethod
    def native(cls, shift: float) -> "TrainingSchedule":
        if shift <= 0:
            raise ValueError("Noise shift must be positive")
        base = torch.linspace(1.0, 0.0, 1001, dtype=torch.float32)[:-1]
        sigma = shift * base / (1.0 + (shift - 1.0) * base)
        time = sigma * 1000
        weights = torch.exp(-2.0 * ((time - 500.0) / 1000.0).square())
        weights = weights - weights.min()
        weights = weights * (1000 / weights.sum())
        return cls(sigma, weights)


def mesh_grid(frames: int, height: int, width: int, *, action: bool, device) -> torch.Tensor:
    ff, hh, ww = torch.meshgrid(
        torch.arange(frames, device=device), torch.arange(height, device=device),
        torch.arange(width, device=device), indexing="ij",
    )
    ff = ff.float()
    if action:
        ff = ff + torch.arange(1, height + 1, device=device).reshape(1, -1, 1) / (height + 1)
        hh, ww = torch.full_like(hh, -1), torch.full_like(ww, -1)
    grid = torch.stack([ff, hh.float(), ww.float()]).flatten(1)
    return torch.cat([grid, torch.full_like(grid[:1], int(action))], dim=0)


def joint_attention_masks(
    latent_shape, action_shape, *, patch_size=(1, 2, 2), chunk_size: int,
    window_size: int, text_length: int = 512, padding: int = 0, device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense boolean equivalent of native FlexAttnFunc._get_*_mask_mod.

    Flattened packing is noisy video, clean video, noisy action, clean action.
    Clean queries see clean past/current blocks; noisy queries see clean strict
    past plus noisy current block. Video/action have even/odd block IDs.
    """
    b, _, frames, height, width = latent_shape
    ab, _, af, ah, aw = action_shape
    pt, ph, pw = patch_size
    if b != ab or frames != af or pt != 1:
        raise ValueError("Joint native packing requires equal batches/frames and temporal patch size 1")
    if chunk_size < 1 or window_size < 0 or padding < 0 or text_length < 1:
        raise ValueError("Invalid joint mask dimensions")
    if height % ph or width % pw:
        raise ValueError("Latent spatial dimensions must be patch-aligned")
    ls = torch.arange(b, device=device)[:, None, None, None].expand(b, frames, height // ph, width // pw).reshape(-1)
    acs = torch.arange(b, device=device)[:, None, None, None].expand(b, af, ah, aw).reshape(-1)
    lf = torch.arange(frames, device=device)[None, :, None, None].expand(b, frames, height // ph, width // pw).reshape(-1)
    acf = torch.arange(af, device=device)[None, :, None, None].expand(b, af, ah, aw).reshape(-1)
    seq = torch.cat([ls, ls, acs, acs])
    block = torch.cat([lf // chunk_size * 2, lf // chunk_size * 2,
                       acf // chunk_size * 2 + 1, acf // chunk_size * 2 + 1])
    clean = torch.cat([torch.zeros_like(ls), torch.ones_like(ls),
                       torch.zeros_like(acs), torch.ones_like(acs)])
    if padding:
        seq, block, clean = (F.pad(x, (0, padding), value=-1) for x in (seq, block, clean))
    qseq, kseq = seq[:, None], seq[None, :]
    qb, kb = block[:, None], block[None, :]
    qc, kc = clean[:, None], clean[None, :]
    visible = (((qc == 1) & (kc == 1) & (kb <= qb)) |
               ((qc == 0) & (kc == 1) & (kb < qb)) |
               ((qc == 0) & (kc == 0) & (kb == qb)))
    visible &= (qseq == kseq) & (qseq >= 0) & ((qb - kb).abs() <= window_size)
    text_seq = torch.arange(b, device=device).repeat_interleave(text_length)
    cross = (seq[:, None] == text_seq[None, :]) & (seq[:, None] >= 0)
    # A padding token is still a query, and a query with no visible key is backend-defined in masked SDPA.
    # It sees only itself and the first text token; no real token sees it and its output is discarded.
    padded = seq < 0
    visible |= torch.diag(padded)
    cross[padded, 0] = True
    return visible[None, None], cross[None, None]


class JointBatchPreparer:
    """Native per-frame schedule, independent modality noise and optional CFG dropout."""
    def __init__(self, patch_size=(1, 2, 2), *, video_shift=5.0, action_shift=0.05,
                 video_condition_noise_probability=0.5, cfg_probability=0.1):
        self.patch_size = tuple(patch_size)
        self.video_schedule = TrainingSchedule.native(video_shift)
        self.action_schedule = TrainingSchedule.native(action_shift)
        self.video_condition_noise_probability = video_condition_noise_probability
        self.cfg_probability = cfg_probability
        if not 0 <= video_condition_noise_probability <= 1 or not 0 <= cfg_probability <= 1:
            raise ValueError("Noise/CFG probabilities must be in [0, 1]")

    def _modality(self, clean: torch.Tensor, schedule: TrainingSchedule, *, action: bool,
                  mask: torch.Tensor | None, generator: torch.Generator) -> dict:
        b, _, frames, height, width = clean.shape
        # Upstream samples one F-vector, shared across B, separately for each modality.
        ids = (torch.rand(frames, generator=generator) * 1000).long().clamp_max(999)
        sigma = schedule.sigmas[ids].to(clean.device).reshape(1, 1, frames, 1, 1)
        noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(clean)
        noised = (1 - sigma) * clean + sigma * noise
        target = noise - clean
        cond_time = torch.zeros(frames, device=clean.device)
        condition = clean.clone()
        if not action and torch.rand((), generator=generator).item() < self.video_condition_noise_probability:
            cond_ids = (500 + torch.rand(frames, generator=generator) * 500).long().clamp_max(999)
            cond_sigma = schedule.sigmas[cond_ids].to(clean.device).reshape(1, 1, frames, 1, 1)
            cond_noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(clean)
            condition = (1 - cond_sigma) * clean + cond_sigma * cond_noise
            cond_time = schedule.sigmas[cond_ids].to(clean.device) * 1000
        if mask is not None:
            noised, target, condition = (x * mask for x in (noised, target, condition))
        pt, ph, pw = (1, 1, 1) if action else self.patch_size
        result = {
            "noisy_latents": noised, "latent": condition, "targets": target,
            "timesteps": (schedule.sigmas[ids].to(clean.device) * 1000)[None].expand(b, -1),
            "cond_timesteps": cond_time[None].expand(b, -1),
            "weights": schedule.weights[ids].to(clean.device)[None].expand(b, -1),
            "grid_id": mesh_grid(frames // pt, height // ph, width // pw, action=action,
                                 device=clean.device)[None].expand(b, -1, -1),
        }
        if mask is not None:
            result["actions_mask"] = mask
        return result

    @torch.no_grad()
    def __call__(self, batch: dict, *, generator: torch.Generator,
                 empty_text: torch.Tensor | None = None) -> dict:
        video = self._modality(batch["latents"], self.video_schedule, action=False,
                               mask=None, generator=generator)
        action = self._modality(batch["actions"], self.action_schedule, action=True,
                                mask=batch["actions_mask"], generator=generator)
        text = batch["text_emb"]
        if self.cfg_probability:
            if empty_text is None or empty_text.shape != text.shape[1:]:
                raise ValueError("Native CFG dropout requires a matching explicit empty embedding")
            drop = (torch.rand(text.shape[0], generator=generator) < self.cfg_probability).to(text.device)
            text = torch.where(drop[:, None, None], empty_text.to(text)[None], text)
        video["text_emb"] = action["text_emb"] = text
        return {"latent_dict": video, "action_dict": action,
                "chunk_size": int(torch.randint(1, 5, (), generator=generator)),
                "window_size": int(torch.randint(4, 65, (), generator=generator))}


class JointModel(nn.Module):
    """Native joint forward using explicit SDPA masks, without mutable/global KV.

    Uses the existing model parameters and blocks. The native block extension
    accepts self_attention_mask and cross_attention_mask, so no global Flex mask
    or attention monkeypatch survives this call. Dense masks are intentionally
    recomputed per batch; full-scale memory is measured on the target hardware.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.model
        latent, action = inputs["latent_dict"], inputs["action_dict"]
        dtype = model.patch_embedding_mlp.weight.dtype
        batch = latent["noisy_latents"].shape[0]
        def embedded(tensor, kind):
            return model._input_embed(tensor.to(dtype), input_type=kind).flatten(0, 1)[None]
        vn = embedded(latent["noisy_latents"], "latent")
        vc = embedded(latent["latent"], "latent")
        an = embedded(action["noisy_latents"], "action")
        ac = embedded(action["latent"], "action")
        text = embedded(latent["text_emb"], "text")
        hidden = torch.cat([vn, vc, an, ac], dim=1)
        vg = latent["grid_id"].permute(1, 0, 2).flatten(1)[None]
        ag = action["grid_id"].permute(1, 0, 2).flatten(1)[None]
        rope = model.rope(torch.cat([vg, vg, ag, ag], dim=2))[:, :, None]
        vt = torch.cat([latent["timesteps"].flatten(), latent["cond_timesteps"].flatten()])[None]
        at = torch.cat([action["timesteps"].flatten(), action["cond_timesteps"].flatten()])[None]
        ve, vp = model._time_embed(vt, *latent["noisy_latents"].shape[-2:], dtype=dtype)
        ae, ap = model._time_embed(at, *action["noisy_latents"].shape[-2:], dtype=dtype, action_mode=True)
        temb, proj = torch.cat([ve, ae], 1), torch.cat([vp, ap], 1)
        padding = (-hidden.shape[1]) % 128
        hidden = F.pad(hidden, (0, 0, 0, padding))
        rope = F.pad(rope, (0, 0, 0, 0, 0, padding))
        temb = F.pad(temb, (0, 0, 0, padding))
        proj = F.pad(proj, (0, 0, 0, 0, 0, padding))
        self_mask, cross_mask = joint_attention_masks(
            latent["noisy_latents"].shape, action["noisy_latents"].shape,
            patch_size=model.patch_size, chunk_size=inputs["chunk_size"],
            window_size=inputs["window_size"], padding=padding,
            text_length=latent["text_emb"].shape[1], device=hidden.device,
        )
        # No mutable KV is used here (update_cache=0, no attn_caches), so activations can be recomputed.
        call = partial(checkpoint, use_reentrant=False) if model.gradient_checkpointing and torch.is_grad_enabled() else None
        for block in model.blocks:
            arguments = (hidden, text, proj, rope)
            options = {"update_cache": 0, "self_attention_mask": self_mask, "cross_attention_mask": cross_mask}
            hidden = call(block, *arguments, **options) if call else block(*arguments, **options)
        shift, scale = (model.scale_shift_table[None] + temb[:, :, None]).unbind(2)
        hidden = (model.norm_out(hidden.float()) * (1 + scale) + shift).to(hidden.dtype)
        vn, _, an, _, _ = torch.split(hidden, [vn.shape[1], vc.shape[1], an.shape[1], ac.shape[1], padding], dim=1)
        video_out = model.proj_out(vn).reshape(batch, -1, math.prod(model.patch_size),
                                               latent["noisy_latents"].shape[1]).flatten(1, 2)
        action_out = model.action_proj_out(an).reshape(batch, -1, action["noisy_latents"].shape[1])
        return video_out, action_out


def unpatch_video(sequence: torch.Tensor, shape, patch_size) -> torch.Tensor:
    b, c, frames, height, width = shape
    pt, ph, pw = patch_size
    return (sequence.reshape(b, frames // pt, height // ph, width // pw, pt, ph, pw, c)
            .permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(b, c, frames, height, width))


def joint_fm_loss(inputs: dict, output: tuple, patch_size) -> tuple[torch.Tensor, torch.Tensor]:
    """Native framewise scheduler-weighted means of the video and action flow-matching losses."""
    video, action = inputs["latent_dict"], inputs["action_dict"]
    vp = unpatch_video(output[0], video["targets"].shape, patch_size)
    b, c, frames, count, width = action["targets"].shape
    ap = output[1].reshape(b, frames, count, width, c).permute(0, 4, 1, 2, 3)
    vl = (vp.float() - video["targets"].float().detach()).square()
    vl = vl * video["weights"][:, None, :, None, None]
    vl = vl.permute(0, 2, 1, 3, 4).flatten(2)
    vl = (vl.sum(-1) / (vl.shape[-1] + 1e-6)).mean()
    mask = action["actions_mask"].float()
    al = (ap.float() - action["targets"].float().detach()).square() * mask
    al = al * action["weights"][:, None, :, None, None]
    al = al.permute(0, 2, 1, 3, 4).flatten(2).sum(-1)
    count = mask.permute(0, 2, 1, 3, 4).flatten(2).sum(-1)
    al = (al / (count + 1e-6)).mean()
    return vl, al
