from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from wam_policy.opd.trainer import VideoFFMTrainer


class TinyVideoDataset(Dataset[dict[str, Any]]):
    def __init__(self, length: int = 2) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "z_clean": torch.full((1, 2), float(index + 1)),
            "context": {"index": index},
            "prompt": f"sample {index}",
            "meta": {"sample_id": f"sample_{index}"},
        }


class FakeAdapter:
    def build_video_context(self, batch: Mapping[str, Any]) -> Any:
        return batch["context"]


class FakeStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.built = False
        self.forward_calls: list[dict[str, torch.Tensor]] = []

    def build(self) -> None:
        self.built = True

    def forward_video_flowmap(
        self,
        z_src: torch.Tensor,
        s_src: torch.Tensor,
        s_tgt: torch.Tensor,
        context: Any,
    ) -> torch.Tensor:
        assert isinstance(context, list)
        self.forward_calls.append(
            {
                "z_src": z_src.detach().clone(),
                "s_src": s_src.detach().clone(),
                "s_tgt": s_tgt.detach().clone(),
            }
        )
        return self.bias * torch.ones_like(z_src)


def _cfg(tmp_path: Path, max_steps: int = 2) -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": str(tmp_path),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
        "runtime": {
            "device": "cpu",
            "precision": "fp32",
        },
        "data": {
            "batch_size": 2,
            "num_workers": 0,
        },
        "flowmap": {
            "s_min": 0.0,
            "s_max": 1.0,
            "min_delta_s": 0.1,
        },
        "train": {
            "seed": 123,
            "lr": 0.1,
            "weight_decay": 0.0,
            "max_steps": max_steps,
            "grad_clip_norm": 1.0,
            "log_every": 999,
            "save_every": 1,
        },
    }


def test_video_ffm_trainer_runs_cpu_smoke_and_saves_checkpoints(tmp_path: Path) -> None:
    student = FakeStudent()
    initial_bias = student.bias.detach().clone()
    trainer = VideoFFMTrainer(
        _cfg(tmp_path),
        adapter=FakeAdapter(),
        student=student,
        dataset=TinyVideoDataset(),
    )

    result = trainer.train()

    assert result["step"] == 2
    assert "loss/video_ffm" in result["metrics"]
    assert student.built is True
    assert len(student.forward_calls) == 2
    assert not torch.equal(student.bias.detach(), initial_bias)
    for call in student.forward_calls:
        assert call["z_src"].shape == (2, 1, 2)
        assert torch.all(call["s_src"] < call["s_tgt"])

    checkpoint_dir = tmp_path / "checkpoints"
    assert (checkpoint_dir / "latest.pt").is_file()
    assert (checkpoint_dir / "step_000001.pt").is_file()
    assert (checkpoint_dir / "step_000002.pt").is_file()
    latest = torch.load(checkpoint_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert latest["step"] == 2


def test_video_ffm_trainer_restarts_dataloader_until_max_steps(tmp_path: Path) -> None:
    student = FakeStudent()
    trainer = VideoFFMTrainer(
        _cfg(tmp_path, max_steps=3),
        adapter=FakeAdapter(),
        student=student,
        dataset=TinyVideoDataset(length=1),
    )

    result = trainer.train()

    assert result["step"] == 3
    assert len(student.forward_calls) == 3


def test_video_ffm_trainer_rejects_no_trainable_parameters(tmp_path: Path) -> None:
    class FrozenStudent(FakeStudent):
        def __init__(self) -> None:
            super().__init__()
            self.bias.requires_grad = False

    trainer = VideoFFMTrainer(
        _cfg(tmp_path),
        adapter=FakeAdapter(),
        student=FrozenStudent(),
        dataset=TinyVideoDataset(),
    )

    with pytest.raises(RuntimeError, match="No trainable parameters"):
        trainer.train()


def test_video_ffm_trainer_rejects_empty_dataset(tmp_path: Path) -> None:
    trainer = VideoFFMTrainer(
        _cfg(tmp_path),
        adapter=FakeAdapter(),
        student=FakeStudent(),
        dataset=TinyVideoDataset(length=0),
    )

    with pytest.raises(ValueError, match="dataset must not be empty"):
        trainer.train()
