from __future__ import annotations

from typing import Any, Mapping

from wam_policy.opd.trainer import ActionFFMTrainer, VideoDMDOPDTrainer, VideoFFMTrainer


def train_video_forward_flowmap(cfg: Mapping[str, Any]) -> dict[str, Any]:
    trainer = VideoFFMTrainer(cfg)
    return trainer.train()


def train_action_forward_flowmap(cfg: Mapping[str, Any]) -> dict[str, Any]:
    trainer = ActionFFMTrainer(cfg)
    return trainer.train()


def train_video_dmd_opd(cfg: Mapping[str, Any]) -> dict[str, Any]:
    trainer = VideoDMDOPDTrainer(cfg)
    return trainer.train()
