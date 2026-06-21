from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from wam_policy.adapters.lingbot_va import LingBotVAAdapter


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
    }


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


def test_attach_video_time_pair_conditioner_sets_non_intrusive_attribute(tmp_path: Path) -> None:
    adapter = LingBotVAAdapter(_cfg(tmp_path))
    model = nn.Linear(2, 2)
    conditioner = nn.Identity()

    returned = adapter.attach_video_time_pair_conditioner(model, conditioner)

    assert returned is model
    assert model.wam_time_pair_conditioner is conditioner


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
