from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch


@dataclass(frozen=True)
class VideoFlowMapPair:
    z_clean: torch.Tensor
    eps: torch.Tensor
    s_src: torch.Tensor
    s_tgt: torch.Tensor
    z_src: torch.Tensor
    z_tgt: torch.Tensor
    target_u: torch.Tensor


@dataclass(frozen=True)
class ActionFlowMapPair:
    a_clean: torch.Tensor
    eps: torch.Tensor
    s_src: torch.Tensor
    s_tgt: torch.Tensor
    a_src: torch.Tensor
    a_tgt: torch.Tensor
    target_u: torch.Tensor


def sample_s_pairs(
    batch_size: int,
    s_min: float,
    s_max: float,
    min_delta_s: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample LingBot-direction flow-map pairs with s_src < s_tgt."""
    _validate_s_pair_config(batch_size, s_min, s_max, min_delta_s)

    dtype = torch.float32
    s_src_high = s_max - min_delta_s
    s_src_range = s_src_high - s_min
    s_src = torch.rand(batch_size, device=device, dtype=dtype) * s_src_range + s_min

    s_tgt_low = s_src + min_delta_s
    s_tgt_range = s_max - s_tgt_low
    s_tgt = torch.rand(batch_size, device=device, dtype=dtype) * s_tgt_range + s_tgt_low
    return s_src, s_tgt


def expand_time_like(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Broadcast a [B] time tensor to match x as [B, 1, ...]."""
    _validate_tensor("x", x)
    _validate_time_tensor("s", s, batch_size=x.shape[0])
    if s.device != x.device:
        raise ValueError(f"s must be on the same device as x: {s.device} != {x.device}")

    return s.to(dtype=x.dtype).view(s.shape[0], *([1] * (x.ndim - 1)))


def build_video_flowmap_pair(
    z_clean: torch.Tensor,
    s_src: torch.Tensor,
    s_tgt: torch.Tensor,
    eps: torch.Tensor | None = None,
) -> VideoFlowMapPair:
    """Build Z_src, Z_tgt and the average-velocity target in LingBot s direction."""
    eps = _resolve_eps("eps", eps, "z_clean", z_clean, s_src, s_tgt)
    z_src, z_tgt = _interpolate_clean_noise(z_clean, eps, s_src, s_tgt)
    target_u = average_velocity_target_s(z_src, z_tgt, s_src, s_tgt)

    return VideoFlowMapPair(
        z_clean=z_clean,
        eps=eps,
        s_src=s_src,
        s_tgt=s_tgt,
        z_src=z_src,
        z_tgt=z_tgt,
        target_u=target_u,
    )


def build_action_flowmap_pair(
    a_clean: torch.Tensor,
    s_src: torch.Tensor,
    s_tgt: torch.Tensor,
    eps: torch.Tensor | None = None,
) -> ActionFlowMapPair:
    """Build A_src, A_tgt and the average-velocity target in LingBot s direction."""
    eps = _resolve_eps("eps", eps, "a_clean", a_clean, s_src, s_tgt)
    a_src, a_tgt = _interpolate_clean_noise(a_clean, eps, s_src, s_tgt)
    target_u = average_velocity_target_s(a_src, a_tgt, s_src, s_tgt)

    return ActionFlowMapPair(
        a_clean=a_clean,
        eps=eps,
        s_src=s_src,
        s_tgt=s_tgt,
        a_src=a_src,
        a_tgt=a_tgt,
        target_u=target_u,
    )


def average_velocity_target_s(
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    s_src: torch.Tensor,
    s_tgt: torch.Tensor,
) -> torch.Tensor:
    """Compute (z_tgt - z_src) / (s_tgt - s_src)."""
    _validate_tensor("z_src", z_src)
    _validate_tensor("z_tgt", z_tgt)
    if z_src.shape != z_tgt.shape:
        raise ValueError(f"z_src shape must match z_tgt shape: {z_src.shape} != {z_tgt.shape}")
    _validate_same_device("z_tgt", z_tgt, "z_src", z_src)
    _validate_time_pair(s_src, s_tgt, batch_size=z_src.shape[0])
    _validate_same_device("s_src", s_src, "z_src", z_src)
    _validate_same_device("s_tgt", s_tgt, "z_src", z_src)

    delta_s = expand_time_like(z_src, s_tgt - s_src)
    return (z_tgt - z_src) / delta_s


def _resolve_eps(
    eps_name: str,
    eps: torch.Tensor | None,
    clean_name: str,
    clean: torch.Tensor,
    s_src: torch.Tensor,
    s_tgt: torch.Tensor,
) -> torch.Tensor:
    _validate_tensor(clean_name, clean)
    _validate_time_pair(s_src, s_tgt, batch_size=clean.shape[0])
    _validate_same_device("s_src", s_src, clean_name, clean)
    _validate_same_device("s_tgt", s_tgt, clean_name, clean)

    if eps is None:
        return torch.randn_like(clean)

    _validate_tensor(eps_name, eps)
    if eps.shape != clean.shape:
        raise ValueError(f"{eps_name} shape must match {clean_name} shape: {eps.shape} != {clean.shape}")
    _validate_same_device(eps_name, eps, clean_name, clean)
    return eps


def _interpolate_clean_noise(
    clean: torch.Tensor,
    eps: torch.Tensor,
    s_src: torch.Tensor,
    s_tgt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    s_src_like = expand_time_like(clean, s_src)
    s_tgt_like = expand_time_like(clean, s_tgt)
    src = (1.0 - s_src_like) * eps + s_src_like * clean
    tgt = (1.0 - s_tgt_like) * eps + s_tgt_like * clean
    return src, tgt


def _validate_s_pair_config(
    batch_size: int,
    s_min: float,
    s_max: float,
    min_delta_s: float,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    values = {"s_min": s_min, "s_max": s_max, "min_delta_s": min_delta_s}
    for name, value in values.items():
        if not isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")
    if s_min < 0.0 or s_max > 1.0:
        raise ValueError("s_min and s_max must stay within [0, 1].")
    if s_min >= s_max:
        raise ValueError("s_min must be smaller than s_max.")
    if min_delta_s <= 0.0:
        raise ValueError("min_delta_s must be positive.")
    if min_delta_s > s_max - s_min:
        raise ValueError("min_delta_s must be no larger than s_max - s_min.")


def _validate_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension.")


def _validate_time_tensor(name: str, value: torch.Tensor, batch_size: int) -> None:
    _validate_tensor(name, value)
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [B].")
    if value.shape[0] != batch_size:
        raise ValueError(f"{name} batch size must be {batch_size}, got {value.shape[0]}.")


def _validate_time_pair(s_src: torch.Tensor, s_tgt: torch.Tensor, batch_size: int) -> None:
    _validate_time_tensor("s_src", s_src, batch_size=batch_size)
    _validate_time_tensor("s_tgt", s_tgt, batch_size=batch_size)
    if s_src.shape != s_tgt.shape:
        raise ValueError(f"s_src shape must match s_tgt shape: {s_src.shape} != {s_tgt.shape}")
    if not torch.all(s_src < s_tgt).item():
        raise ValueError("s_src must be strictly smaller than s_tgt for every sample.")


def _validate_same_device(
    left_name: str,
    left: torch.Tensor,
    right_name: str,
    right: torch.Tensor,
) -> None:
    if left.device != right.device:
        raise ValueError(
            f"{left_name} must be on the same device as {right_name}: "
            f"{left.device} != {right.device}"
        )
