from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


def save_training_checkpoint(
    student: nn.Module,
    optimizer: Any,
    cfg: Mapping[str, Any],
    step: int,
    metrics: Mapping[str, float],
    checkpoint_dir: str | Path,
) -> dict[str, Path]:
    if step < 0:
        raise ValueError("step must be non-negative.")

    output_dir = Path(checkpoint_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": dict(cfg),
        "metrics": dict(metrics),
    }
    latest_path = output_dir / "latest.pt"
    step_path = output_dir / f"step_{step:06d}.pt"
    torch.save(payload, latest_path)
    torch.save(payload, step_path)
    return {"latest": latest_path, "step": step_path}


def load_student_checkpoint(
    student: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a WAM student checkpoint saved by save_training_checkpoint."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Student checkpoint does not exist: {path}")

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Student checkpoint must contain a mapping: {path}")
    state_dict = payload.get("model")
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Student checkpoint missing model state_dict: {path}")

    student.load_state_dict(state_dict, strict=strict)
    return dict(payload)
