from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from wam_policy.adapters.lingbot_va import LingBotVAAdapter
from wam_policy.models.flow_map_heads import FlowTimePairEmbedding


def _cfg(tmp_path: Path) -> dict:
    return {
        "paths": {
            "repo_root": str(tmp_path),
            "lingbot_repo": "external/lingbot-va",
        },
        "teacher": {
            "checkpoint_path": "experiments/pretrained_models/lingbot-va/test",
        },
        "runtime": {
            "device": "cpu",
            "precision": "fp32",
        },
        "flowmap": {
            "lingbot_num_train_timesteps": 1000,
        },
    }


class FakeConditionEmbedder(nn.Module):
    def __init__(self, inner_dim: int = 4) -> None:
        super().__init__()
        self.time_proj = nn.Linear(inner_dim, inner_dim * 6)


class FakeLingBotVideoModel(nn.Module):
    patch_size = (1, 2, 2)

    def __init__(self, return_sequence: bool = False, inner_dim: int = 4) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.return_sequence = return_sequence
        self.condition_embedder = FakeConditionEmbedder(inner_dim=inner_dim)
        self.captured_input: dict[str, torch.Tensor] | None = None
        self.last_time_embed: tuple[torch.Tensor, torch.Tensor] | None = None

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
        self.captured_input = input_dict
        z_src = input_dict["noisy_latents"]
        _, channels, frames, height, width = z_src.shape
        patch_f, patch_h, patch_w = self.patch_size
        self.last_time_embed = self._time_embed(
            input_dict["timesteps"],
            height // patch_h,
            width // patch_w,
            z_src.dtype,
            action_mode=action_mode,
        )
        if self.return_sequence:
            token_count = (
                frames
                // patch_f
                * (height // patch_h)
                * (width // patch_w)
                * patch_f
                * patch_h
                * patch_w
            )
            return torch.zeros(
                z_src.shape[0],
                token_count,
                channels,
                device=z_src.device,
                dtype=z_src.dtype,
            )
        return z_src + 1.0


class FakeLingBotActionModel(nn.Module):
    def __init__(self, inner_dim: int = 4) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.condition_embedder = FakeConditionEmbedder(inner_dim=inner_dim)
        self.captured_input: dict[str, torch.Tensor] | None = None
        self.captured_action_mode: bool | None = None
        self.last_time_embed: tuple[torch.Tensor, torch.Tensor] | None = None

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
        self.captured_input = input_dict
        self.captured_action_mode = action_mode
        a_src = input_dict["noisy_actions"]
        self.last_time_embed = self._time_embed(
            input_dict["timesteps"][:, None],
            1,
            1,
            a_src.dtype,
            action_mode=action_mode,
        )
        return a_src + 1.0


def test_lingbot_adapter_parses_paths_device_and_dtype(tmp_path: Path) -> None:
    adapter = LingBotVAAdapter(_cfg(tmp_path))

    assert adapter.repo_root == tmp_path.resolve()
    assert adapter.repo_path == (tmp_path / "external/lingbot-va").resolve()
    assert adapter.checkpoint_path == (
        tmp_path / "experiments/pretrained_models/lingbot-va/test"
    ).resolve()
    assert adapter.device == torch.device("cpu")
    assert adapter.torch_dtype is torch.float32


def test_build_video_context_returns_context_or_none(tmp_path: Path) -> None:
    adapter = LingBotVAAdapter(_cfg(tmp_path))
    tensor_context = torch.tensor([1.0])
    list_context = [{"tokens": [1, 2, 3]}]

    assert adapter.build_video_context({"context": tensor_context}) is tensor_context
    assert adapter.build_video_context({"context": list_context}) is list_context
    assert adapter.build_video_context({}) is None


def test_build_action_context_returns_context_or_none(tmp_path: Path) -> None:
    adapter = LingBotVAAdapter(_cfg(tmp_path))
    tensor_context = torch.tensor([1.0])

    assert adapter.build_action_context({"context": tensor_context}) is tensor_context
    assert adapter.build_action_context({}) is None


def test_attach_video_time_pair_conditioner_sets_non_intrusive_attribute(tmp_path: Path) -> None:
    adapter = LingBotVAAdapter(_cfg(tmp_path))
    model = nn.Linear(2, 2)
    conditioner = nn.Identity()

    returned = adapter.attach_video_time_pair_conditioner(model, conditioner)

    assert returned is model
    assert model.wam_time_pair_conditioner is conditioner


def test_attach_action_time_pair_conditioner_sets_mode_specific_attribute(tmp_path: Path) -> None:
    adapter = LingBotVAAdapter(_cfg(tmp_path))
    model = nn.Linear(2, 2)
    video_conditioner = nn.Identity()
    action_conditioner = nn.Identity()

    adapter.attach_video_time_pair_conditioner(model, video_conditioner)
    returned = adapter.attach_action_time_pair_conditioner(model, action_conditioner)

    assert returned is model
    assert model.wam_time_pair_conditioner is video_conditioner
    assert model.wam_video_time_pair_conditioner is video_conditioner
    assert model.wam_action_time_pair_conditioner is action_conditioner


def test_attach_installs_video_flowmap_hook_and_builds_lingbot_input(
    tmp_path: Path,
) -> None:
    adapter = LingBotVAAdapter(_cfg(tmp_path))
    model = FakeLingBotVideoModel()
    conditioner = FlowTimePairEmbedding(hidden_dim=8)
    z_src = torch.zeros(2, 3, 2, 4, 4)
    s_src = torch.tensor([0.25, 0.5])
    s_tgt = torch.tensor([0.75, 1.0])
    context = [
        {"text_emb": torch.ones(5, 6)},
        {"text_emb": torch.full((5, 6), 2.0)},
    ]

    adapter.attach_video_time_pair_conditioner(model, conditioner)
    pred_u = adapter.forward_video_flowmap(model, z_src, s_src, s_tgt, context=context)

    assert hasattr(model, "wam_forward_video_flowmap")
    assert torch.equal(pred_u, z_src + 1.0)
    assert model.captured_input is not None
    assert torch.equal(model.captured_input["noisy_latents"], z_src)
    assert torch.allclose(
        model.captured_input["timesteps"],
        torch.tensor([[750.0, 750.0], [500.0, 500.0]]),
    )
    assert model.captured_input["grid_id"].shape == (2, 4, 8)
    assert model.captured_input["text_emb"].shape == (2, 5, 6)
    assert model.last_time_embed is not None
    temb, timestep_proj = model.last_time_embed
    assert temb.shape == (2, 8, 4)
    assert timestep_proj.shape == (2, 8, 6, 4)


def test_hook_converts_lingbot_sequence_output_to_latent_shape(tmp_path: Path) -> None:
    adapter = LingBotVAAdapter(_cfg(tmp_path))
    model = FakeLingBotVideoModel(return_sequence=True)
    conditioner = FlowTimePairEmbedding(hidden_dim=8)
    z_src = torch.zeros(2, 3, 2, 4, 4)
    s_src = torch.tensor([0.1, 0.2])
    s_tgt = torch.tensor([0.6, 0.8])
    context = torch.ones(2, 5, 6)

    adapter.attach_video_time_pair_conditioner(model, conditioner)
    pred_u = adapter.forward_video_flowmap(model, z_src, s_src, s_tgt, context=context)

    assert pred_u.shape == z_src.shape
    assert torch.equal(pred_u, torch.zeros_like(z_src))


def test_attach_installs_action_flowmap_hook_and_builds_lingbot_input(
    tmp_path: Path,
) -> None:
    adapter = LingBotVAAdapter(_cfg(tmp_path))
    model = FakeLingBotActionModel()
    conditioner = FlowTimePairEmbedding(hidden_dim=8)
    a_src = torch.zeros(2, 4, 14)
    s_src = torch.tensor([0.25, 0.5])
    s_tgt = torch.tensor([0.75, 1.0])
    context = [{"text_emb": torch.ones(5, 6)}, {"text_emb": torch.full((5, 6), 2.0)}]
    z_condition = torch.zeros(2, 1, 2, 2, 2)

    adapter.attach_action_time_pair_conditioner(model, conditioner)
    pred_u = adapter.forward_action_flowmap(
        model,
        a_src,
        s_src,
        s_tgt,
        context=context,
        z_condition=z_condition,
    )

    assert hasattr(model, "wam_forward_action_flowmap")
    assert torch.equal(pred_u, a_src + 1.0)
    assert model.captured_action_mode is True
    assert model.captured_input is not None
    assert torch.equal(model.captured_input["noisy_actions"], a_src)
    assert torch.allclose(model.captured_input["timesteps"], torch.tensor([750.0, 500.0]))
    assert model.captured_input["context"] == context
    assert torch.equal(model.captured_input["z_condition"], z_condition)
    assert model.last_time_embed is not None
    temb, timestep_proj = model.last_time_embed
    assert temb.shape == (2, 1, 4)
    assert timestep_proj.shape == (2, 1, 6, 4)


def test_time_embed_patch_preserves_values_at_zero_init_and_skips_action_mode(
    tmp_path: Path,
) -> None:
    adapter = LingBotVAAdapter(_cfg(tmp_path))
    model = FakeLingBotVideoModel()
    conditioner = FlowTimePairEmbedding(hidden_dim=8)
    adapter.attach_video_time_pair_conditioner(model, conditioner)
    timesteps = torch.zeros(2, 2)

    model._wam_current_s_src = torch.tensor([0.1, 0.2])
    model._wam_current_s_tgt = torch.tensor([0.6, 0.9])
    temb, timestep_proj = model._time_embed(
        timesteps,
        2,
        2,
        torch.float32,
        action_mode=False,
    )
    action_temb, action_timestep_proj = model._time_embed(
        timesteps,
        2,
        2,
        torch.float32,
        action_mode=True,
    )

    assert torch.equal(temb, torch.ones_like(temb))
    assert torch.equal(timestep_proj, torch.ones_like(timestep_proj))
    assert torch.equal(action_temb, torch.ones_like(action_temb))
    assert torch.equal(action_timestep_proj, torch.ones_like(action_timestep_proj))


def test_normalize_video_context_rejects_missing_or_mismatched_text_emb(
    tmp_path: Path,
) -> None:
    adapter = LingBotVAAdapter(_cfg(tmp_path))

    with pytest.raises(ValueError, match="text_emb"):
        adapter.normalize_video_context(
            {"tokens": torch.ones(2, 3)},
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    with pytest.raises(ValueError, match="batch size"):
        adapter.normalize_video_context(
            torch.ones(2, 5, 6),
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    with pytest.raises(ValueError, match="single"):
        adapter.normalize_video_context(
            torch.ones(5, 6),
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_build_lingbot_video_input_rejects_non_divisible_patch_shape(
    tmp_path: Path,
) -> None:
    adapter = LingBotVAAdapter(_cfg(tmp_path))
    model = FakeLingBotVideoModel()
    z_src = torch.zeros(1, 3, 2, 3, 4)

    with pytest.raises(ValueError, match="not divisible"):
        adapter.build_lingbot_video_input(
            model,
            z_src,
            torch.tensor([0.1]),
            torch.tensor([0.9]),
            context=torch.ones(1, 5, 6),
        )


def test_load_pretrained_video_action_model_wraps_missing_runtime_dependency(
    tmp_path: Path,
) -> None:
    adapter = LingBotVAAdapter(_cfg(tmp_path))

    with pytest.raises(RuntimeError, match="Unable to import LingBot-VA runtime modules"):
        adapter.load_pretrained_video_action_model()


def test_forward_video_flowmap_without_hook_raises_not_implemented(tmp_path: Path) -> None:
    adapter = LingBotVAAdapter(_cfg(tmp_path))
    model = nn.Linear(2, 2)
    z_src = torch.zeros(2, 1, 1, 1, 1)
    s_src = torch.tensor([0.1, 0.2])
    s_tgt = torch.tensor([0.4, 0.8])

    with pytest.raises(NotImplementedError, match="forward binding is not implemented"):
        adapter.forward_video_flowmap(model, z_src, s_src, s_tgt, context=None)


def test_forward_video_flowmap_uses_fake_hook(tmp_path: Path) -> None:
    class FakeModel(nn.Module):
        def wam_forward_video_flowmap(
            self,
            *,
            z_src: torch.Tensor,
            s_src: torch.Tensor,
            s_tgt: torch.Tensor,
            context: object,
        ) -> torch.Tensor:
            assert torch.all(s_src < s_tgt)
            assert context == {"ok": True}
            return z_src + 1.0

    adapter = LingBotVAAdapter(_cfg(tmp_path))
    z_src = torch.zeros(2, 1, 1, 1, 1)
    s_src = torch.tensor([0.1, 0.2])
    s_tgt = torch.tensor([0.4, 0.8])

    pred_u = adapter.forward_video_flowmap(
        FakeModel(),
        z_src,
        s_src,
        s_tgt,
        context={"ok": True},
    )

    assert torch.equal(pred_u, torch.ones_like(z_src))


def test_forward_video_flowmap_rejects_invalid_time_order(tmp_path: Path) -> None:
    adapter = LingBotVAAdapter(_cfg(tmp_path))
    model = nn.Linear(2, 2)
    z_src = torch.zeros(2, 1, 1, 1, 1)
    s_src = torch.tensor([0.1, 0.9])
    s_tgt = torch.tensor([0.4, 0.8])

    with pytest.raises(ValueError, match="s_src must be strictly smaller"):
        adapter.forward_video_flowmap(model, z_src, s_src, s_tgt, context=None)
