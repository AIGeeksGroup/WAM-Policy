from __future__ import annotations

import pytest
import torch

from wam_policy.opd.losses import video_ffm_average_velocity_loss


def test_video_ffm_average_velocity_loss_computes_mse_and_metrics() -> None:
    pred_u = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target_u = torch.tensor([[0.0, 2.0], [1.0, 6.0]])

    loss, metrics = video_ffm_average_velocity_loss(pred_u, target_u)

    assert torch.allclose(loss, torch.tensor(2.25))
    assert metrics["loss/video_ffm"] == pytest.approx(2.25)
    assert metrics["stat/pred_u_abs_mean"] == pytest.approx(2.5)
    assert metrics["stat/target_u_abs_mean"] == pytest.approx(2.25)


def test_video_ffm_average_velocity_loss_supports_full_shape_mask() -> None:
    pred_u = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    target_u = torch.ones_like(pred_u)
    mask = torch.tensor([[1.0, 0.0], [0.5, 1.0]])

    loss, _ = video_ffm_average_velocity_loss(pred_u, target_u, mask=mask)

    expected = ((pred_u - target_u) * mask).square().mean()
    assert torch.allclose(loss, expected)


def test_video_ffm_average_velocity_loss_supports_batch_mask() -> None:
    pred_u = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    target_u = torch.ones_like(pred_u)
    mask = torch.tensor([1.0, 0.0])

    loss, _ = video_ffm_average_velocity_loss(pred_u, target_u, mask=mask)

    expected = torch.tensor([[0.0, 2.0], [0.0, 0.0]]).square().mean()
    assert torch.allclose(loss, expected)


def test_video_ffm_average_velocity_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape must match"):
        video_ffm_average_velocity_loss(torch.zeros(2, 3), torch.zeros(2, 2))


def test_video_ffm_average_velocity_loss_rejects_device_mismatch() -> None:
    pred_u = torch.empty((2, 2), device="meta")
    target_u = torch.zeros(2, 2)

    with pytest.raises(ValueError, match="same device"):
        video_ffm_average_velocity_loss(pred_u, target_u)


@pytest.mark.parametrize(
    ("pred_u", "target_u"),
    [
        (torch.tensor([float("nan")]), torch.tensor([0.0])),
        (torch.tensor([float("inf")]), torch.tensor([0.0])),
        (torch.tensor([0.0]), torch.tensor([float("nan")])),
    ],
)
def test_video_ffm_average_velocity_loss_rejects_non_finite_values(
    pred_u: torch.Tensor,
    target_u: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        video_ffm_average_velocity_loss(pred_u, target_u)


def test_video_ffm_average_velocity_loss_rejects_unbroadcastable_mask() -> None:
    pred_u = torch.zeros(2, 3)
    target_u = torch.zeros(2, 3)
    mask = torch.zeros(4)

    with pytest.raises(ValueError, match="broadcastable"):
        video_ffm_average_velocity_loss(pred_u, target_u, mask=mask)
