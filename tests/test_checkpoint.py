from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from wam_policy.adapters.checkpoint import save_training_checkpoint
from wam_policy.utils.optim import build_adamw_optimizer


def test_save_training_checkpoint_writes_latest_and_step_file(tmp_path: Path) -> None:
    student = nn.Linear(2, 1)
    optimizer = build_adamw_optimizer(student.parameters(), lr=0.01, weight_decay=0.0)
    cfg = {"experiment": {"stage": "video_ffm"}}
    metrics = {"loss/video_ffm": 1.25}

    paths = save_training_checkpoint(
        student=student,
        optimizer=optimizer,
        cfg=cfg,
        step=500,
        metrics=metrics,
        checkpoint_dir=tmp_path,
    )

    assert paths["latest"] == tmp_path.resolve() / "latest.pt"
    assert paths["step"] == tmp_path.resolve() / "step_000500.pt"
    assert paths["latest"].is_file()
    assert paths["step"].is_file()

    loaded = torch.load(paths["latest"], map_location="cpu", weights_only=False)
    assert loaded["step"] == 500
    assert loaded["cfg"] == cfg
    assert loaded["metrics"] == metrics
    assert set(loaded) == {"step", "model", "optimizer", "cfg", "metrics"}
