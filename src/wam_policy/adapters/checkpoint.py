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
