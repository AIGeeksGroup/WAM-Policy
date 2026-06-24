from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from wam_policy.opd.trainer import ActionFFMTrainer, normalize_action_chunk_batch


class TinyActionDataset(Dataset[dict[str, Any]]):
    def __init__(self, length: int = 2) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "a_clean": torch.full((4, 2), float(index + 1)),
            "context": {"index": index},
            "z_condition": torch.full((1, 2), float(index)),
            "prompt": f"sample {index}",
            "meta": {"sample_id": f"sample_{index}"},
        }


class FakeAdapter:
    def build_action_context(self, batch: Mapping[str, Any]) -> Any:
        return batch["context"]


class FakeActionStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.built = False
        self.forward_calls: list[dict[str, torch.Tensor]] = []

    def build(self) -> None:
        self.built = True

    def forward_action_flowmap(
        self,
        a_src: torch.Tensor,
        s_src: torch.Tensor,
        s_tgt: torch.Tensor,
        context: Any,
        z_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert isinstance(context, list)
        assert isinstance(z_condition, torch.Tensor)
        self.forward_calls.append(
            {
                "a_src": a_src.detach().clone(),
                "s_src": s_src.detach().clone(),
                "s_tgt": s_tgt.detach().clone(),
                "z_condition": z_condition.detach().clone(),
            }
        )
        return self.bias * torch.ones_like(a_src)


def _cfg(tmp_path: Path, max_steps: int = 2) -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": str(tmp_path),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "lingbot_repo": "external/lingbot-va",
        },
        "teacher": {
            "checkpoint_path": "experiments/pretrained_models/lingbot-va/test",
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


def test_action_ffm_trainer_runs_cpu_smoke_and_saves_checkpoints(tmp_path: Path) -> None:
    student = FakeActionStudent()
    initial_bias = student.bias.detach().clone()
    trainer = ActionFFMTrainer(
        _cfg(tmp_path),
        adapter=FakeAdapter(),
        student=student,
        dataset=TinyActionDataset(),
    )

    result = trainer.train()

    assert result["step"] == 2
    assert "loss/action_ffm" in result["metrics"]
    assert student.built is True
    assert len(student.forward_calls) == 2
    assert not torch.equal(student.bias.detach(), initial_bias)
    for call in student.forward_calls:
        assert call["a_src"].shape == (2, 4, 2)
        assert call["z_condition"].shape == (2, 1, 2)
        assert torch.all(call["s_src"] < call["s_tgt"])

    checkpoint_dir = tmp_path / "checkpoints"
    assert (checkpoint_dir / "latest.pt").is_file()
    assert (checkpoint_dir / "step_000001.pt").is_file()
    assert (checkpoint_dir / "step_000002.pt").is_file()


def test_action_ffm_trainer_rejects_empty_dataset(tmp_path: Path) -> None:
    trainer = ActionFFMTrainer(
        _cfg(tmp_path),
        adapter=FakeAdapter(),
        student=FakeActionStudent(),
        dataset=TinyActionDataset(length=0),
    )

    with pytest.raises(ValueError, match="dataset must not be empty"):
        trainer.train()


def test_normalize_action_chunk_batch_accepts_action_shapes() -> None:
    a_clean = torch.zeros(2, 4, 14)
    a_clean_flat = torch.zeros(2, 14)
    a_clean_with_singleton = torch.zeros(2, 1, 4, 14)

    assert normalize_action_chunk_batch(a_clean) is a_clean
    assert normalize_action_chunk_batch(a_clean_flat) is a_clean_flat
    assert normalize_action_chunk_batch(a_clean_with_singleton).shape == (2, 4, 14)


def test_normalize_action_chunk_batch_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="a_clean must have shape"):
        normalize_action_chunk_batch(torch.zeros(2, 1, 1, 4, 14))
