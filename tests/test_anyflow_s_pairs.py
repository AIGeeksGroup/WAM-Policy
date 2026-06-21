from __future__ import annotations

import pytest
import torch

from wam_policy.adapters.anyflow import (
    average_velocity_target_s,
    build_video_flowmap_pair,
    expand_time_like,
    sample_s_pairs,
)


def test_sample_s_pairs_respects_range_and_min_delta() -> None:
    torch.manual_seed(7)

    s_src, s_tgt = sample_s_pairs(
        batch_size=128,
        s_min=0.1,
        s_max=0.9,
        min_delta_s=0.2,
        device=torch.device("cpu"),
    )

    assert s_src.shape == (128,)
    assert s_tgt.shape == (128,)
    assert torch.all(s_src >= 0.1)
    assert torch.all(s_tgt <= 0.9)
    assert torch.all(s_src < s_tgt)
    assert torch.all((s_tgt - s_src) >= 0.2)


@pytest.mark.parametrize(
    ("batch_size", "s_min", "s_max", "min_delta_s"),
    [
        (0, 0.0, 1.0, 0.1),
        (1, -0.1, 1.0, 0.1),
        (1, 0.0, 1.1, 0.1),
        (1, 0.5, 0.5, 0.1),
        (1, 0.0, 1.0, 0.0),
        (1, 0.0, 0.1, 0.2),
    ],
)
def test_sample_s_pairs_rejects_invalid_config(
    batch_size: int,
    s_min: float,
    s_max: float,
    min_delta_s: float,
) -> None:
    with pytest.raises(ValueError):
        sample_s_pairs(
            batch_size=batch_size,
            s_min=s_min,
            s_max=s_max,
            min_delta_s=min_delta_s,
            device=torch.device("cpu"),
        )


def test_expand_time_like_broadcasts_to_latent_rank() -> None:
    x = torch.zeros(2, 3, 4, 5, 6, dtype=torch.float64)
    s = torch.tensor([0.25, 0.75], dtype=torch.float32)

    expanded = expand_time_like(x, s)

    assert expanded.shape == (2, 1, 1, 1, 1)
    assert expanded.dtype == x.dtype
    assert expanded.device == x.device
    assert torch.allclose(expanded.flatten(), s.to(dtype=x.dtype))


def test_build_video_flowmap_pair_matches_lingbot_s_formula() -> None:
    z_clean = torch.arange(2 * 1 * 2 * 2 * 2, dtype=torch.float32).reshape(2, 1, 2, 2, 2)
    eps = torch.full_like(z_clean, -2.0)
    s_src = torch.tensor([0.0, 0.25])
    s_tgt = torch.tensor([0.5, 1.0])

    pair = build_video_flowmap_pair(z_clean, s_src, s_tgt, eps=eps)

    expected_z_src = (1.0 - expand_time_like(z_clean, s_src)) * eps + expand_time_like(
        z_clean, s_src
    ) * z_clean
    expected_z_tgt = (1.0 - expand_time_like(z_clean, s_tgt)) * eps + expand_time_like(
        z_clean, s_tgt
    ) * z_clean

    assert pair.z_clean is z_clean
    assert pair.eps is eps
    assert torch.allclose(pair.z_src, expected_z_src)
    assert torch.allclose(pair.z_tgt, expected_z_tgt)
    assert torch.allclose(pair.target_u, z_clean - eps)


def test_average_velocity_target_reconstructs_target_state() -> None:
    z_src = torch.randn(3, 2, 2, 2, 2)
    target_u = torch.randn_like(z_src)
    s_src = torch.tensor([0.1, 0.3, 0.5])
    s_tgt = torch.tensor([0.2, 0.7, 1.0])
    delta_s = expand_time_like(z_src, s_tgt - s_src)
    z_tgt = z_src + target_u * delta_s

    computed = average_velocity_target_s(z_src, z_tgt, s_src, s_tgt)

    assert torch.allclose(computed, target_u, atol=1e-6)
    assert torch.allclose(z_src + computed * delta_s, z_tgt, atol=1e-6)


def test_build_video_flowmap_pair_rejects_invalid_time_order() -> None:
    z_clean = torch.zeros(2, 1, 1, 1, 1)
    s_src = torch.tensor([0.2, 0.8])
    s_tgt = torch.tensor([0.3, 0.7])

    with pytest.raises(ValueError, match="s_src must be strictly smaller"):
        build_video_flowmap_pair(z_clean, s_src, s_tgt)


def test_build_video_flowmap_pair_rejects_shape_mismatch() -> None:
    z_clean = torch.zeros(2, 1, 1, 1, 1)
    eps = torch.zeros(2, 1, 1, 1)
    s_src = torch.tensor([0.2, 0.4])
    s_tgt = torch.tensor([0.3, 0.7])

    with pytest.raises(ValueError, match="eps shape must match"):
        build_video_flowmap_pair(z_clean, s_src, s_tgt, eps=eps)


def test_build_video_flowmap_pair_rejects_device_mismatch() -> None:
    z_clean = torch.zeros(2, 1, 1, 1, 1)
    eps = torch.empty(z_clean.shape, device="meta")
    s_src = torch.tensor([0.2, 0.4])
    s_tgt = torch.tensor([0.3, 0.7])

    with pytest.raises(ValueError, match="eps must be on the same device"):
        build_video_flowmap_pair(z_clean, s_src, s_tgt, eps=eps)
