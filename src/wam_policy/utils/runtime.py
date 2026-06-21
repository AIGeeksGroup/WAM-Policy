from __future__ import annotations

from typing import Any, Mapping

import torch


def resolve_device(cfg: Mapping[str, Any]) -> torch.device:
    return torch.device(resolve_device_name(cfg))


def resolve_device_name(cfg: Mapping[str, Any]) -> str:
    runtime_cfg = cfg.get("runtime")
    device_name = "auto"
    if isinstance(runtime_cfg, Mapping):
        raw_device = runtime_cfg.get("device", "auto")
        if isinstance(raw_device, str) and raw_device:
            device_name = raw_device.lower()

    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name in {"rocm", "hip"}:
        if not _torch_has_rocm():
            raise ValueError(
                f"runtime.device={device_name!r} requires a ROCm PyTorch build "
                "where torch.version.hip is set."
            )
        return "cuda"
    if device_name in {"cuda", "cpu"}:
        return device_name
    raise ValueError("runtime.device must be one of auto, cuda, rocm, hip, or cpu.")


def resolve_lingbot_attn_mode(cfg: Mapping[str, Any]) -> str:
    runtime_cfg = cfg.get("runtime")
    attn_mode = "torch"
    if isinstance(runtime_cfg, Mapping):
        raw_attn_mode = runtime_cfg.get("lingbot_attn_mode", attn_mode)
        if isinstance(raw_attn_mode, str) and raw_attn_mode:
            attn_mode = raw_attn_mode.lower()

    if attn_mode not in {"torch", "flashattn", "flex"}:
        raise ValueError("runtime.lingbot_attn_mode must be one of torch, flashattn, or flex.")
    return attn_mode


def resolve_torch_dtype(cfg: Mapping[str, Any]) -> torch.dtype:
    runtime_cfg = cfg.get("runtime")
    precision = "bf16"
    if isinstance(runtime_cfg, Mapping):
        raw_precision = runtime_cfg.get("precision", "bf16")
        if isinstance(raw_precision, str) and raw_precision:
            precision = raw_precision

    precision_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if precision not in precision_map:
        raise ValueError(f"Unsupported runtime.precision: {precision}")
    return precision_map[precision]


def _torch_has_rocm() -> bool:
    return bool(getattr(torch.version, "hip", None))
