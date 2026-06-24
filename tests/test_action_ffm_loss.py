from __future__ import annotations

import pytest
import torch

from wam_policy.opd.losses import action_ffm_average_velocity_loss


def test_action_ffm_average_velocity_loss_computes_mse_and_metrics() -> None:
    pred_u = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target_u = torch.tensor([[0.0, 2.0], [1.0, 6.0]])

    loss, metrics = action_ffm_average_velocity_loss(pred_u, target_u)

    assert torch.allclose(loss, torch.tensor(2.25))
    assert metrics["loss/action_ffm"] == pytest.approx(2.25)
    assert metrics["stat/pred_u_abs_mean"] == pytest.approx(2.5)
    assert metrics["stat/target_u_abs_mean"] == pytest.approx(2.25)


def test_action_ffm_average_velocity_loss_supports_mask() -> None:
    pred_u = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    target_u = torch.ones_like(pred_u)
    mask = torch.tensor([1.0, 0.0])

    loss, _ = action_ffm_average_velocity_loss(pred_u, target_u, mask=mask)

    expected = torch.tensor([[0.0, 2.0], [0.0, 0.0]]).square().mean()
    assert torch.allclose(loss, expected)


def test_action_ffm_average_velocity_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape must match"):
        action_ffm_average_velocity_loss(torch.zeros(2, 3), torch.zeros(2, 2))


def test_action_ffm_average_velocity_loss_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        action_ffm_average_velocity_loss(torch.tensor([float("nan")]), torch.tensor([0.0]))
