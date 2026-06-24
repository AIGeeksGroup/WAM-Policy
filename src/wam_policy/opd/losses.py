from __future__ import annotations

import torch


def video_ffm_average_velocity_loss(
    pred_u: torch.Tensor,
    target_u: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    return average_velocity_mse_loss(
        pred_u=pred_u,
        target_u=target_u,
        mask=mask,
        metric_prefix="video_ffm",
    )


def action_ffm_average_velocity_loss(
    pred_u: torch.Tensor,
    target_u: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    return average_velocity_mse_loss(
        pred_u=pred_u,
        target_u=target_u,
        mask=mask,
        metric_prefix="action_ffm",
    )


def average_velocity_mse_loss(
    pred_u: torch.Tensor,
    target_u: torch.Tensor,
    mask: torch.Tensor | None = None,
    metric_prefix: str = "ffm",
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
        f"loss/{metric_prefix}": float(loss.detach().cpu()),
        "stat/pred_u_abs_mean": float(pred_u.detach().abs().mean().cpu()),
        "stat/target_u_abs_mean": float(target_u.detach().abs().mean().cpu()),
    }
    return loss, metrics


def dmd_score_surrogate_loss(
    student_state: torch.Tensor,
    teacher_score: torch.Tensor,
    student_score: torch.Tensor,
    weight: torch.Tensor | float | None = None,
    metric_prefix: str = "video_dmd",
) -> tuple[torch.Tensor, dict[str, float]]:
    """DMD-style surrogate whose gradient flows only through student_state."""
    _validate_tensor("student_state", student_state)
    _validate_tensor("teacher_score", teacher_score)
    _validate_tensor("student_score", student_score)
    if student_state.shape != teacher_score.shape or student_state.shape != student_score.shape:
        raise ValueError("student_state, teacher_score, and student_score shapes must match.")
    if student_state.device != teacher_score.device or student_state.device != student_score.device:
        raise ValueError("student_state, teacher_score, and student_score must be on the same device.")
    _validate_finite("student_state", student_state)
    _validate_finite("teacher_score", teacher_score)
    _validate_finite("student_score", student_score)

    correction = student_score.detach() - teacher_score.detach()
    per_sample = (student_state * correction).reshape(student_state.shape[0], -1).sum(dim=1)
    if weight is not None:
        per_sample = per_sample * _normalize_weight(weight, per_sample)
    loss = per_sample.mean()
    _validate_finite("loss", loss)

    metrics = {
        f"loss/{metric_prefix}": float(loss.detach().cpu()),
        f"stat/{metric_prefix}/correction_abs_mean": float(correction.detach().abs().mean().cpu()),
        f"stat/{metric_prefix}/student_state_abs_mean": float(
            student_state.detach().abs().mean().cpu()
        ),
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


def _normalize_weight(weight: torch.Tensor | float, per_sample: torch.Tensor) -> torch.Tensor:
    if isinstance(weight, torch.Tensor):
        if weight.device != per_sample.device:
            raise ValueError("weight must be on the same device as student_state.")
        _validate_finite("weight", weight)
        if weight.ndim == 0:
            return weight.to(dtype=per_sample.dtype)
        if weight.shape != per_sample.shape:
            raise ValueError(f"weight shape must be scalar or [B], got {weight.shape}.")
        return weight.to(dtype=per_sample.dtype)
    return torch.tensor(float(weight), device=per_sample.device, dtype=per_sample.dtype)


def _is_broadcastable(source_shape: torch.Size, target_shape: torch.Size) -> bool:
    for source_dim, target_dim in zip(reversed(source_shape), reversed(target_shape), strict=False):
        if source_dim not in (1, target_dim):
            return False
    return len(source_shape) <= len(target_shape)
