from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from wam_policy.opd.trainer import ActionDMDOPDTrainer


class TinyActionRolloutDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        include_z_condition: bool = True,
        include_teacher_label: bool = False,
    ) -> None:
        self.include_z_condition = include_z_condition
        self.include_teacher_label = include_teacher_label

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        teacher_label = torch.zeros(4, 2) if self.include_teacher_label else None
        return {
            "context": {"index": index},
            "z_student": None,
            "a_student": torch.full((4, 2), float(index + 1)),
            "z_condition": torch.full((1, 2), float(index)) if self.include_z_condition else None,
            "teacher_video_label": None,
            "teacher_action_label": teacher_label,
            "record": None,
            "meta": {"sample_id": f"rollout_{index}"},
        }


class FakeAdapter:
    def __init__(self) -> None:
        self.teacher_z_condition: torch.Tensor | None = None
        self.teacher_calls = 0

    def build_action_context(self, batch: Mapping[str, Any]) -> Any:
        return batch["context"]

    def teacher_action_score_or_velocity(self, **kwargs: Any) -> torch.Tensor:
        self.teacher_calls += 1
        self.teacher_z_condition = kwargs["z_condition"]
        return torch.zeros_like(kwargs["a_state"])

    def student_action_score_or_velocity_from_flowmap(
        self,
        pred_u: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        return pred_u


class FakeActionStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.25))
        self.built = False
        self.forward_z_condition: torch.Tensor | None = None
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
        assert torch.all(s_src < s_tgt)
        assert z_condition is not None
        self.forward_z_condition = z_condition.detach().clone()
        self.forward_calls.append(
            {
                "s_src": s_src.detach().clone(),
                "s_tgt": s_tgt.detach().clone(),
                "z_condition": z_condition.detach().clone(),
            }
        )
        return self.bias * torch.ones_like(a_src)


def _cfg(
    tmp_path: Path,
    max_steps: int = 2,
    *,
    teacher_score_source: str = "lingbot_velocity_forward",
    surrogate_state: str = "pred_u",
    student_score_source: str = "flowmap_small_interval",
) -> dict[str, Any]:
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
        "opd": {
            "dmd": {
                "weight": 0.1,
                "teacher_score_source": teacher_score_source,
                "student_score_source": student_score_source,
                "surrogate_state": surrogate_state,
                "small_delta_s": 0.03,
            },
            "anchor_ffm": {"enabled": True, "weight": 0.5},
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


def test_action_dmd_trainer_uses_student_rollout_z_condition(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    student = FakeActionStudent()
    trainer = ActionDMDOPDTrainer(
        _cfg(tmp_path),
        adapter=adapter,
        student=student,
        rollout_dataset=TinyActionRolloutDataset(),
    )

    result = trainer.train()

    assert result["step"] == 2
    assert "loss/action_dmd" in result["metrics"]
    assert "loss/action_dmd_total" in result["metrics"]
    assert student.built is True
    assert student.forward_z_condition is not None
    assert student.forward_z_condition.shape == (2, 1, 2)
    assert len(student.forward_calls) == 4
    small_interval_calls = student.forward_calls[1::2]
    for call in small_interval_calls:
        assert torch.allclose(
            call["s_tgt"] - call["s_src"],
            torch.full_like(call["s_src"], 0.03),
        )
    assert adapter.teacher_z_condition is not None
    assert adapter.teacher_z_condition.shape == (2, 1, 2)
    assert (tmp_path / "checkpoints" / "latest.pt").is_file()


def test_action_dmd_trainer_rejects_missing_student_z_condition(tmp_path: Path) -> None:
    trainer = ActionDMDOPDTrainer(
        _cfg(tmp_path, max_steps=1),
        adapter=FakeAdapter(),
        student=FakeActionStudent(),
        rollout_dataset=TinyActionRolloutDataset(include_z_condition=False),
    )

    with pytest.raises(ValueError, match="z_condition"):
        trainer.train()


def test_action_dmd_trainer_supports_clean_sample_surrogate(tmp_path: Path) -> None:
    trainer = ActionDMDOPDTrainer(
        _cfg(tmp_path, max_steps=1, surrogate_state="clean_sample"),
        adapter=FakeAdapter(),
        student=FakeActionStudent(),
        rollout_dataset=TinyActionRolloutDataset(),
    )

    result = trainer.train()

    assert result["step"] == 1
    assert "loss/action_dmd" in result["metrics"]


def test_action_dmd_trainer_uses_cached_teacher_label(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    trainer = ActionDMDOPDTrainer(
        _cfg(tmp_path, max_steps=1, teacher_score_source="cached_label"),
        adapter=adapter,
        student=FakeActionStudent(),
        rollout_dataset=TinyActionRolloutDataset(include_teacher_label=True),
    )

    trainer.train()

    assert adapter.teacher_calls == 0


def test_action_dmd_trainer_rejects_not_implemented_teacher_source(tmp_path: Path) -> None:
    trainer = ActionDMDOPDTrainer(
        _cfg(tmp_path, max_steps=1, teacher_score_source="not_implemented"),
        adapter=FakeAdapter(),
        student=FakeActionStudent(),
        rollout_dataset=TinyActionRolloutDataset(),
    )

    with pytest.raises(RuntimeError, match="teacher_score_source=cached_label"):
        trainer.train()
