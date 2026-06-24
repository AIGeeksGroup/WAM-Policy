from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import Dataset

from wam_policy.opd.trainer import VideoDMDOPDTrainer


class TinyVideoRolloutDataset(Dataset[dict[str, Any]]):
    def __init__(self, include_teacher_label: bool = False) -> None:
        self.include_teacher_label = include_teacher_label

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        teacher_label = torch.zeros(1, 2, 2, 2) if self.include_teacher_label else None
        return {
            "context": {"index": index},
            "z_student": torch.full((1, 2, 2, 2), float(index + 1)),
            "a_student": None,
            "z_condition": None,
            "teacher_video_label": teacher_label,
            "teacher_action_label": None,
            "record": None,
            "meta": {"sample_id": f"rollout_{index}"},
        }


class FakeAdapter:
    def __init__(self) -> None:
        self.teacher_calls = 0

    def build_video_context(self, batch: Mapping[str, Any]) -> Any:
        return batch["context"]

    def teacher_video_score_or_velocity(self, **kwargs: Any) -> torch.Tensor:
        self.teacher_calls += 1
        return torch.zeros_like(kwargs["z_state"])

    def student_video_score_or_velocity_from_flowmap(
        self,
        pred_u: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        return pred_u


class FakeVideoStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.25))
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
        assert torch.all(s_src < s_tgt)
        self.forward_calls.append(
            {
                "s_src": s_src.detach().clone(),
                "s_tgt": s_tgt.detach().clone(),
            }
        )
        return self.bias * torch.ones_like(z_src)


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


def test_video_dmd_trainer_runs_fake_cpu_step_and_saves_checkpoint(tmp_path: Path) -> None:
    student = FakeVideoStudent()
    trainer = VideoDMDOPDTrainer(
        _cfg(tmp_path),
        adapter=FakeAdapter(),
        student=student,
        rollout_dataset=TinyVideoRolloutDataset(),
    )

    result = trainer.train()

    assert result["step"] == 2
    assert "loss/video_dmd" in result["metrics"]
    assert "loss/video_dmd_total" in result["metrics"]
    assert student.built is True
    assert len(student.forward_calls) == 4
    small_interval_calls = student.forward_calls[1::2]
    for call in small_interval_calls:
        assert torch.allclose(
            call["s_tgt"] - call["s_src"],
            torch.full_like(call["s_src"], 0.03),
        )
    assert (tmp_path / "checkpoints" / "latest.pt").is_file()


def test_video_dmd_trainer_supports_clean_sample_surrogate(tmp_path: Path) -> None:
    trainer = VideoDMDOPDTrainer(
        _cfg(tmp_path, max_steps=1, surrogate_state="clean_sample"),
        adapter=FakeAdapter(),
        student=FakeVideoStudent(),
        rollout_dataset=TinyVideoRolloutDataset(),
    )

    result = trainer.train()

    assert result["step"] == 1
    assert "loss/video_dmd" in result["metrics"]


def test_video_dmd_trainer_uses_cached_teacher_label(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    trainer = VideoDMDOPDTrainer(
        _cfg(tmp_path, max_steps=1, teacher_score_source="cached_label"),
        adapter=adapter,
        student=FakeVideoStudent(),
        rollout_dataset=TinyVideoRolloutDataset(include_teacher_label=True),
    )

    trainer.train()

    assert adapter.teacher_calls == 0


def test_video_dmd_trainer_rejects_not_implemented_teacher_source(tmp_path: Path) -> None:
    trainer = VideoDMDOPDTrainer(
        _cfg(tmp_path, max_steps=1, teacher_score_source="not_implemented"),
        adapter=FakeAdapter(),
        student=FakeVideoStudent(),
        rollout_dataset=TinyVideoRolloutDataset(),
    )

    try:
        trainer.train()
    except RuntimeError as exc:
        assert "teacher_score_source=lingbot_velocity_forward" in str(exc)
        assert "teacher_score_source=cached_label" in str(exc)
    else:
        raise AssertionError("trainer.train() should fail for teacher_score_source=not_implemented")
