from __future__ import annotations

import torch


def video_ffm_average_velocity_loss(
    pred_u: torch.Tensor,
    target_u: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    _validate_tensor("pred_u", pred_u)
    _validate_tensor("target_u", target_u)
    if pred_u.shape != target_u.shape:
        raise ValueError(f"pred_u shape must match target_u shape: {pred_u.shape} != {target_u.shape}")
    if pred_u.device != target_u.device:
        raise ValueError(
            f"pred_u and target_u must be on the same device: {pred_u.device} != {target_u.device}"
        )
    _validate_finite("pred_u", pred_u)
    _validate_finite("target_u", target_u)

    diff = pred_u - target_u
    if mask is not None:
        _validate_tensor("mask", mask)
        if mask.device != diff.device:
            raise ValueError(f"mask must be on the same device as pred_u: {mask.device} != {diff.device}")
        mask_like = _expand_mask(mask, diff)
        _validate_finite("mask", mask_like)
        diff = diff * mask_like.to(dtype=diff.dtype)

    loss = diff.square().mean()
    _validate_finite("loss", loss)

    metrics = {
        "loss/video_ffm": float(loss.detach().cpu()),
        "stat/pred_u_abs_mean": float(pred_u.detach().abs().mean().cpu()),
        "stat/target_u_abs_mean": float(target_u.detach().abs().mean().cpu()),
    }
    return loss, metrics


def _expand_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mask.shape == target.shape:
        return mask
    if mask.ndim == 1 and mask.shape[0] == target.shape[0]:
        return mask.view(mask.shape[0], *([1] * (target.ndim - 1)))
    if not _is_broadcastable(mask.shape, target.shape):
        raise ValueError(f"mask shape must be broadcastable to pred_u: {mask.shape}")
    return mask


def _validate_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty.")


def _validate_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values.")


def _is_broadcastable(source_shape: torch.Size, target_shape: torch.Size) -> bool:
    for source_dim, target_dim in zip(reversed(source_shape), reversed(target_shape), strict=False):
        if source_dim not in (1, target_dim):
            return False
    return len(source_shape) <= len(target_shape)
