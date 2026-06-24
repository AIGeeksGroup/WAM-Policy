from __future__ import annotations

import pytest
import torch

from wam_policy.adapters.anyflow import (
    build_action_flowmap_pair,
    expand_time_like,
)


@pytest.mark.parametrize("shape", [(2, 4, 14), (2, 14), (2, 3, 4)])
def test_build_action_flowmap_pair_matches_lingbot_s_formula(shape: tuple[int, ...]) -> None:
    a_clean = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    eps = torch.full_like(a_clean, -2.0)
    s_src = torch.tensor([0.0, 0.25])
    s_tgt = torch.tensor([0.5, 1.0])

    pair = build_action_flowmap_pair(a_clean, s_src, s_tgt, eps=eps)

    expected_a_src = (1.0 - expand_time_like(a_clean, s_src)) * eps + expand_time_like(
        a_clean, s_src
    ) * a_clean
    expected_a_tgt = (1.0 - expand_time_like(a_clean, s_tgt)) * eps + expand_time_like(
        a_clean, s_tgt
    ) * a_clean

    assert pair.a_clean is a_clean
    assert pair.eps is eps
    assert torch.allclose(pair.a_src, expected_a_src)
    assert torch.allclose(pair.a_tgt, expected_a_tgt)
    assert torch.allclose(pair.target_u, a_clean - eps)


def test_build_action_flowmap_pair_rejects_invalid_time_order() -> None:
    a_clean = torch.zeros(2, 4, 14)
    s_src = torch.tensor([0.2, 0.8])
    s_tgt = torch.tensor([0.3, 0.7])

    with pytest.raises(ValueError, match="s_src must be strictly smaller"):
        build_action_flowmap_pair(a_clean, s_src, s_tgt)


def test_build_action_flowmap_pair_rejects_eps_shape_mismatch() -> None:
    a_clean = torch.zeros(2, 4, 14)
    eps = torch.zeros(2, 14)
    s_src = torch.tensor([0.2, 0.4])
    s_tgt = torch.tensor([0.3, 0.7])

    with pytest.raises(ValueError, match="eps shape must match"):
        build_action_flowmap_pair(a_clean, s_src, s_tgt, eps=eps)
