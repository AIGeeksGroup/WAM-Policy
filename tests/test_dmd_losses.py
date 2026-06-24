from __future__ import annotations

import pytest
import torch

from wam_policy.adapters.anyflow import (
    estimate_instant_velocity_from_flowmap,
    flowmap_dmd_weight,
    sample_flowmap_backward_segment,
)
from wam_policy.opd.losses import dmd_score_surrogate_loss


def test_dmd_score_surrogate_loss_flows_gradient_only_through_student_state() -> None:
    student_state = torch.ones(2, 3, requires_grad=True)
    teacher_score = torch.full((2, 3), 0.25, requires_grad=True)
    student_score = torch.full((2, 3), 0.75, requires_grad=True)

    loss, metrics = dmd_score_surrogate_loss(
        student_state,
        teacher_score,
        student_score,
        metric_prefix="video_dmd",
    )
    loss.backward()

    assert metrics["loss/video_dmd"] == pytest.approx(1.5)
    assert student_state.grad is not None
    assert torch.equal(student_state.grad, torch.full((2, 3), 0.25))
    assert teacher_score.grad is None
    assert student_score.grad is None


def test_dmd_score_surrogate_loss_supports_batch_weight() -> None:
    student_state = torch.ones(2, 2, requires_grad=True)
    teacher_score = torch.zeros(2, 2)
    student_score = torch.ones(2, 2)
    weight = torch.tensor([1.0, 0.5])

    loss, _ = dmd_score_surrogate_loss(
        student_state,
        teacher_score,
        student_score,
        weight=weight,
        metric_prefix="action_dmd",
    )

    assert torch.allclose(loss, torch.tensor(1.5))


def test_dmd_score_surrogate_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        dmd_score_surrogate_loss(torch.zeros(2, 3), torch.zeros(2, 2), torch.zeros(2, 3))


def test_sample_flowmap_backward_segment_returns_valid_s_pair() -> None:
    segment = sample_flowmap_backward_segment(
        batch_size=8,
        s_min=0.0,
        s_max=1.0,
        min_delta_s=0.1,
        device=torch.device("cpu"),
        step_index=1,
        num_steps=4,
    )

    assert segment.s_src.shape == (8,)
    assert segment.s_tgt.shape == (8,)
    assert torch.all(segment.s_src < segment.s_tgt)
    assert segment.s_mid is None
    assert segment.step_index == 1
    assert segment.num_steps == 4


def test_estimate_instant_velocity_from_flowmap_is_identity_for_average_velocity() -> None:
    pred_u = torch.randn(2, 3)

    estimated = estimate_instant_velocity_from_flowmap(pred_u, delta_s=0.1)

    assert estimated is pred_u


def test_flowmap_dmd_weight_returns_segment_length() -> None:
    s_src = torch.tensor([0.1, 0.2])
    s_tgt = torch.tensor([0.4, 1.0])

    weight = flowmap_dmd_weight(s_src, s_tgt)

    assert torch.allclose(weight, torch.tensor([0.3, 0.8]))
