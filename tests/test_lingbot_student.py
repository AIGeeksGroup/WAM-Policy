from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from wam_policy.models.flow_map_heads import FlowTimePairEmbedding
from wam_policy.models.lingbot_student import (
    LingBotActionFlowMapStudent,
    LingBotVideoFlowMapStudent,
)


def test_flow_time_pair_embedding_outputs_expected_shape_and_zero_init() -> None:
    embedding = FlowTimePairEmbedding(hidden_dim=8, mlp_hidden_dim=4)
    s_src = torch.tensor([0.1, 0.2, 0.3])
    s_tgt = torch.tensor([0.4, 0.8, 1.0])

    output = embedding(s_src, s_tgt)

    assert output.shape == (3, 8)
    assert torch.allclose(output, torch.zeros_like(output))


def test_flow_time_pair_embedding_rejects_invalid_time_pair() -> None:
    embedding = FlowTimePairEmbedding(hidden_dim=8)

    with pytest.raises(ValueError, match="shape must match"):
        embedding(torch.tensor([0.1]), torch.tensor([0.2, 0.3]))

    with pytest.raises(ValueError, match="strictly smaller"):
        embedding(torch.tensor([0.4]), torch.tensor([0.2]))


def test_flow_time_pair_embedding_forward_conditioning_supports_batch_times() -> None:
    embedding = FlowTimePairEmbedding(hidden_dim=4, timestep_proj_dim=24)
    s_src = torch.tensor([0.1, 0.2])
    s_tgt = torch.tensor([0.6, 0.8])

    temb_delta, timestep_proj_delta = embedding.forward_conditioning(
        s_src,
        s_tgt,
        timestep_proj_shape=(2, 5, 6, 4),
    )

    assert temb_delta.shape == (2, 5, 4)
    assert timestep_proj_delta.shape == (2, 5, 6, 4)
    assert torch.equal(temb_delta, torch.zeros_like(temb_delta))
    assert torch.equal(timestep_proj_delta, torch.zeros_like(timestep_proj_delta))


def test_flow_time_pair_embedding_forward_conditioning_supports_token_times() -> None:
    embedding = FlowTimePairEmbedding(hidden_dim=4, timestep_proj_dim=24)
    s_src = torch.full((2, 5), 0.1)
    s_tgt = torch.full((2, 5), 0.8)

    temb_delta, timestep_proj_delta = embedding.forward_conditioning(
        s_src,
        s_tgt,
        timestep_proj_shape=(2, 5, 6, 4),
    )

    assert temb_delta.shape == (2, 5, 4)
    assert timestep_proj_delta.shape == (2, 5, 6, 4)
    assert torch.equal(temb_delta, torch.zeros_like(temb_delta))
    assert torch.equal(timestep_proj_delta, torch.zeros_like(timestep_proj_delta))


class FakeBaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.video_head_adapter = nn.Linear(2, 2)
        self.action_head_adapter = nn.Linear(2, 2)


class FakeAdapter:
    def __init__(self, pred_offset: float = 1.0) -> None:
        self.pred_offset = pred_offset
        self.loaded = False
        self.attached_conditioner: nn.Module | None = None

    def load_pretrained_video_action_model(self) -> nn.Module:
        self.loaded = True
        return FakeBaseModel()

    def attach_video_time_pair_conditioner(
        self,
        model: nn.Module,
        conditioner: nn.Module,
    ) -> nn.Module:
        self.attached_conditioner = conditioner
        model.wam_time_pair_conditioner = conditioner
        return model

    def attach_action_time_pair_conditioner(
        self,
        model: nn.Module,
        conditioner: nn.Module,
    ) -> nn.Module:
        self.attached_conditioner = conditioner
        model.wam_action_time_pair_conditioner = conditioner
        return model

    def forward_video_flowmap(
        self,
        model: nn.Module,
        z_src: torch.Tensor,
        s_src: torch.Tensor,
        s_tgt: torch.Tensor,
        context: Any,
    ) -> torch.Tensor:
        assert isinstance(model, FakeBaseModel)
        assert torch.all(s_src < s_tgt)
        assert context == {"context": True}
        return z_src + self.pred_offset

    def forward_action_flowmap(
        self,
        model: nn.Module,
        a_src: torch.Tensor,
        s_src: torch.Tensor,
        s_tgt: torch.Tensor,
        context: Any,
        z_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert isinstance(model, FakeBaseModel)
        assert torch.all(s_src < s_tgt)
        assert context == {"context": True}
        assert z_condition is None or isinstance(z_condition, torch.Tensor)
        return a_src + self.pred_offset


def _student_cfg() -> dict[str, Any]:
    return {
        "model": {
            "freeze_backbone": True,
            "trainable_modules": ["video_head_adapter"],
            "time_pair_hidden_dim": 16,
            "time_pair_mlp_hidden_dim": 8,
        }
    }


def _action_student_cfg() -> dict[str, Any]:
    return {
        "model": {
            "freeze_backbone": True,
            "trainable_modules": ["action_head_adapter"],
            "time_pair_hidden_dim": 16,
            "time_pair_mlp_hidden_dim": 8,
        }
    }


def test_lingbot_student_build_attaches_conditioner_and_freezes_by_config() -> None:
    adapter = FakeAdapter()
    student = LingBotVideoFlowMapStudent(_student_cfg(), adapter)  # type: ignore[arg-type]

    student.build()

    assert adapter.loaded is True
    assert adapter.attached_conditioner is student.time_pair_conditioner
    assert isinstance(student.base_model, FakeBaseModel)
    assert student.base_model.wam_time_pair_conditioner is student.time_pair_conditioner
    assert all(not param.requires_grad for param in student.base_model.backbone.parameters())
    assert all(param.requires_grad for param in student.base_model.video_head_adapter.parameters())
    assert all(param.requires_grad for param in student.time_pair_conditioner.parameters())


def test_lingbot_student_forward_delegates_to_adapter() -> None:
    student = LingBotVideoFlowMapStudent(_student_cfg(), FakeAdapter())  # type: ignore[arg-type]
    student.build()
    z_src = torch.zeros(2, 1, 1, 1, 1)
    s_src = torch.tensor([0.1, 0.2])
    s_tgt = torch.tensor([0.5, 0.9])

    pred_u = student.forward_video_flowmap(z_src, s_src, s_tgt, context={"context": True})

    assert torch.equal(pred_u, torch.ones_like(z_src))


def test_lingbot_student_forward_requires_build() -> None:
    student = LingBotVideoFlowMapStudent(_student_cfg(), FakeAdapter())  # type: ignore[arg-type]
    z_src = torch.zeros(1, 1, 1, 1, 1)

    with pytest.raises(RuntimeError, match="build"):
        student.forward_video_flowmap(
            z_src,
            torch.tensor([0.1]),
            torch.tensor([0.2]),
            context=None,
        )


def test_lingbot_action_student_build_attaches_conditioner_and_freezes_by_config() -> None:
    adapter = FakeAdapter()
    student = LingBotActionFlowMapStudent(_action_student_cfg(), adapter)  # type: ignore[arg-type]

    student.build()

    assert adapter.loaded is True
    assert adapter.attached_conditioner is student.time_pair_conditioner
    assert isinstance(student.base_model, FakeBaseModel)
    assert student.base_model.wam_action_time_pair_conditioner is student.time_pair_conditioner
    assert all(not param.requires_grad for param in student.base_model.backbone.parameters())
    assert all(param.requires_grad for param in student.base_model.action_head_adapter.parameters())
    assert all(param.requires_grad for param in student.time_pair_conditioner.parameters())


def test_lingbot_action_student_forward_delegates_to_adapter() -> None:
    student = LingBotActionFlowMapStudent(_action_student_cfg(), FakeAdapter())  # type: ignore[arg-type]
    student.build()
    a_src = torch.zeros(2, 4, 14)
    s_src = torch.tensor([0.1, 0.2])
    s_tgt = torch.tensor([0.5, 0.9])

    pred_u = student.forward_action_flowmap(
        a_src,
        s_src,
        s_tgt,
        context={"context": True},
        z_condition=torch.zeros(2, 1),
    )

    assert torch.equal(pred_u, torch.ones_like(a_src))


def test_lingbot_action_student_forward_requires_build() -> None:
    student = LingBotActionFlowMapStudent(_action_student_cfg(), FakeAdapter())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="build"):
        student.forward_action_flowmap(
            torch.zeros(1, 4, 14),
            torch.tensor([0.1]),
            torch.tensor([0.2]),
            context=None,
        )


def test_lingbot_action_student_forward_rejects_shape_mismatch() -> None:
    class BadAdapter(FakeAdapter):
        def forward_action_flowmap(
            self,
            model: nn.Module,
            a_src: torch.Tensor,
            s_src: torch.Tensor,
            s_tgt: torch.Tensor,
            context: Any,
            z_condition: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return torch.zeros(1)

    student = LingBotActionFlowMapStudent(_action_student_cfg(), BadAdapter())  # type: ignore[arg-type]
    student.build()

    with pytest.raises(ValueError, match="pred_u shape must match"):
        student.forward_action_flowmap(
            torch.zeros(2, 4, 14),
            torch.tensor([0.1, 0.2]),
            torch.tensor([0.5, 0.9]),
            context=None,
        )


def test_lingbot_student_forward_rejects_shape_mismatch() -> None:
    class BadAdapter(FakeAdapter):
        def forward_video_flowmap(
            self,
            model: nn.Module,
            z_src: torch.Tensor,
            s_src: torch.Tensor,
            s_tgt: torch.Tensor,
            context: Any,
        ) -> torch.Tensor:
            return torch.zeros(1)

    student = LingBotVideoFlowMapStudent(_student_cfg(), BadAdapter())  # type: ignore[arg-type]
    student.build()

    with pytest.raises(ValueError, match="pred_u shape must match"):
        student.forward_video_flowmap(
            torch.zeros(2, 1, 1, 1, 1),
            torch.tensor([0.1, 0.2]),
            torch.tensor([0.5, 0.9]),
            context=None,
        )
