from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from wam_policy.adapters.lingbot_va import LingBotVAAdapter
from wam_policy.opd.trainer import VideoFFMTrainer, normalize_video_latent_batch


class TinyVideoDataset(Dataset[dict[str, Any]]):
    def __init__(self, length: int = 2) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "z_clean": torch.full((1, 2, 2, 2), float(index + 1)),
            "context": {"index": index},
            "prompt": f"sample {index}",
            "meta": {"sample_id": f"sample_{index}"},
        }


class TinyTextEmbVideoDataset(Dataset[dict[str, Any]]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "z_clean": torch.full((1, 1, 2, 2), float(index + 1)),
            "context": {"text_emb": torch.ones(3, 4)},
            "prompt": f"sample {index}",
            "meta": {"sample_id": f"text_emb_sample_{index}"},
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


class TrainerFakeConditionEmbedder(nn.Module):
    def __init__(self, inner_dim: int = 4) -> None:
        super().__init__()
        self.time_proj = nn.Linear(inner_dim, inner_dim * 6)


class HookPathFakeLingBotModel(nn.Module):
    patch_size = (1, 1, 1)

    def __init__(self, inner_dim: int = 4) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.condition_embedder = TrainerFakeConditionEmbedder(inner_dim=inner_dim)
        self.forward_calls = 0

    def _time_embed(
        self,
        timesteps: torch.Tensor,
        height: int,
        width: int,
        dtype: torch.dtype,
        action_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del action_mode
        batch_size, token_frames = timesteps.shape
        token_count = token_frames * height * width
        temb = torch.ones(
            batch_size,
            token_count,
            self.inner_dim,
            device=timesteps.device,
            dtype=dtype,
        )
        timestep_proj = torch.ones(
            batch_size,
            token_count,
            6,
            self.inner_dim,
            device=timesteps.device,
            dtype=dtype,
        )
        return temb, timestep_proj

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        action_mode: bool = False,
        train_mode: bool = False,
    ) -> torch.Tensor:
        del train_mode
        self.forward_calls += 1
        z_src = input_dict["noisy_latents"]
        temb, _ = self._time_embed(
            input_dict["timesteps"],
            z_src.shape[3],
            z_src.shape[4],
            z_src.dtype,
            action_mode=action_mode,
        )
        scale = temb.mean(dim=(1, 2)).view(z_src.shape[0], 1, 1, 1, 1)
        return z_src + scale


class HookPathAdapter(LingBotVAAdapter):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg)
        self.model: HookPathFakeLingBotModel | None = None

    def load_pretrained_video_action_model(self) -> nn.Module:
        self.model = HookPathFakeLingBotModel()
        return self.model


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
            "lingbot_num_train_timesteps": 1000,
        },
        "model": {
            "freeze_backbone": True,
            "time_pair_hidden_dim": 4,
            "time_pair_mlp_hidden_dim": 4,
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
        assert call["z_src"].shape == (2, 1, 2, 2, 2)
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


def test_video_ffm_trainer_default_student_uses_adapter_hook_path(
    tmp_path: Path,
) -> None:
    adapter = HookPathAdapter(_cfg(tmp_path, max_steps=1))
    trainer = VideoFFMTrainer(
        _cfg(tmp_path, max_steps=1),
        adapter=adapter,
        dataset=TinyTextEmbVideoDataset(),
    )

    result = trainer.train()

    assert result["step"] == 1
    assert adapter.model is not None
    assert adapter.model.forward_calls == 1
    assert hasattr(adapter.model, "wam_forward_video_flowmap")


def test_normalize_video_latent_batch_accepts_video_latent_shapes() -> None:
    z_clean = torch.zeros(2, 3, 4, 5, 6)
    z_clean_with_singleton = torch.zeros(2, 1, 3, 4, 5, 6)

    assert normalize_video_latent_batch(z_clean) is z_clean
    assert normalize_video_latent_batch(z_clean_with_singleton).shape == (2, 3, 4, 5, 6)


def test_normalize_video_latent_batch_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="z_clean must have shape"):
        normalize_video_latent_batch(torch.zeros(2, 3, 4))
