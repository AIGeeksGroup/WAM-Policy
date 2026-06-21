from __future__ import annotations

from typing import Any, Mapping

import torch


def resolve_device(cfg: Mapping[str, Any]) -> torch.device:
    runtime_cfg = cfg.get("runtime")
    device_name = "auto"
    if isinstance(runtime_cfg, Mapping):
        raw_device = runtime_cfg.get("device", "auto")
        if isinstance(raw_device, str) and raw_device:
            device_name = raw_device

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


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
