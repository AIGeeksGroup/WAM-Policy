from __future__ import annotations

from typing import Any, Mapping

from wam_policy.opd.trainer import VideoFFMTrainer


def train_video_forward_flowmap(cfg: Mapping[str, Any]) -> dict[str, Any]:
    trainer = VideoFFMTrainer(cfg)
    return trainer.train()
