"""The upstream LIBERO offline action codec and sample checks of the SFT-only demonstrations.

Reproduces the sample preparation of the upstream LatentLeRobotDataset for LIBERO; latents are the
frozen VAE mean latents shipped with robbyant/libero-long-lerobot.
"""
from __future__ import annotations


import torch


def offline_action_postprocess(
    actions: torch.Tensor,
    latent_frame_ids: torch.Tensor,
    local_start_frame: int,
    q01: torch.Tensor,
    q99: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Faithfully reproduce LatentLeRobotDataset._action_post_process for LIBERO.

    Unlike the online codec, four frame strides of *physical zero* precede the
    sequence, are quantile-normalized, and remain supervised. Padding channels
    are invalid; native offline clipping to [-1.5, 1.5] is intentional here only.
    """
    actions = torch.as_tensor(actions, dtype=torch.float64)
    frame_ids = torch.as_tensor(latent_frame_ids, dtype=torch.int64)
    lo = torch.as_tensor(q01, dtype=torch.float64).reshape(-1)
    hi = torch.as_tensor(q99, dtype=torch.float64).reshape(-1)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError("LIBERO offline actions must have shape [N, 7]")
    if frame_ids.ndim != 1 or frame_ids.numel() < 2:
        raise ValueError("At least two frame IDs are required")
    if lo.shape != (30,) or hi.shape != (30,):
        raise ValueError("Native quantile vectors must each have 30 channels")
    if not torch.isfinite(actions).all() or not torch.isfinite(lo).all() or not torch.isfinite(hi).all():
        raise ValueError("Nonfinite offline actions/quantiles")
    if not torch.all(hi[:7] > lo[:7]):
        raise ValueError("Active action quantiles must have a positive range")
    stride = int(frame_ids[1] - frame_ids[0])
    shift = int(frame_ids[0]) - int(local_start_frame)
    if stride <= 0 or shift < 0 or not torch.all(frame_ids[1:] - frame_ids[:-1] == stride):
        raise ValueError("Frame IDs must have a positive uniform stride and nonnegative shift")
    groups = (frame_ids.numel() - 1) // 4 + 1
    if (frame_ids.numel() - 1) % 4:
        raise ValueError("Native VAE frame selection requires 1 + 4*k frame IDs")
    per_group = stride * 4
    padded = torch.cat([actions.new_zeros(per_group, 7), actions[shift:]], dim=0)
    required = groups * per_group
    if padded.shape[0] < required:
        raise ValueError("Actions do not cover the requested native latent sequence")
    padded = padded[:required]
    aligned = actions.new_zeros(required, 30)
    aligned[:, :7] = padded
    mask = torch.zeros_like(aligned, dtype=torch.bool)
    mask[:, :7] = True
    aligned = ((aligned - lo) / (hi - lo + 1e-6) * 2.0 - 1.0).clamp(-1.5, 1.5)
    aligned = aligned * mask
    return (
        aligned.reshape(groups, per_group, 30).permute(2, 0, 1).unsqueeze(-1).float(),
        mask.reshape(groups, per_group, 30).permute(2, 0, 1).unsqueeze(-1),
    )


def validate_sample(sample: dict[str, torch.Tensor], *, native_geometry: bool = True) -> None:
    required = {"latents", "actions", "actions_mask", "text_emb"}
    if set(sample) != required:
        raise ValueError(f"Sample tensor keys must be {sorted(required)}")
    z, a, mask, text = (sample[k] for k in ("latents", "actions", "actions_mask", "text_emb"))
    if z.ndim != 4 or a.ndim != 4 or mask.shape != a.shape or text.ndim != 2:
        raise ValueError("Expected latent [C,F,H,W], action/mask [C,F,N,1], text [L,D]")
    if z.shape[1] != a.shape[1] or a.shape[-1] != 1 or mask.dtype != torch.bool:
        raise ValueError("Video/action frame groups and boolean mask must agree")
    if min(z.shape) <= 0 or min(a.shape) <= 0 or min(text.shape) <= 0:
        raise ValueError("Sample tensors must be nonempty")
    for key in ("latents", "actions", "text_emb"):
        if not sample[key].is_floating_point() or not torch.isfinite(sample[key]).all():
            raise ValueError(f"{key} must contain finite floating-point values")
    if native_geometry:
        if z.shape[0] != 48 or z.shape[2:] != (8, 16) or a.shape[0] != 30 or a.shape[2] != 4:
            raise ValueError("Dataset does not have native LIBERO latent/action geometry")
        if text.shape != (512, 4096):
            raise ValueError("Native LIBERO text embedding must be [512, 4096]")
        if not mask[:7].all() or mask[7:].any():
            raise ValueError("Native offline samples supervise channels 0..6, including the first group")
        if a[7:].count_nonzero() or a.abs().max() > 1.5:
            raise ValueError("Offline padded channels must be zero; active values clipped to [-1.5, 1.5]")
    if not mask.any() or torch.count_nonzero(a.masked_select(~mask)):
        raise ValueError("Action mask is empty or invalid coordinates are nonzero")


def native_camera_latents(agentview: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
    """Pack two already encoded [C,F,H,W] camera means along latent width."""
    if agentview.shape != wrist.shape or agentview.ndim != 4:
        raise ValueError("Camera mean latents must have the same [C,F,H,W] shape")
    return torch.cat([agentview, wrist], dim=-1)
