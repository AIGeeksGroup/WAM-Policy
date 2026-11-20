"""adapt_command / policy_gains: per-joint gains must land on the right Go2 motors."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wam_go2.adapter.command_adapter import (  # noqa: E402
    CommandError,
    adapt_command,
    policy_gains,
)
from wam_go2.constants import (  # noqa: E402
    GO2_JOINT_NAMES,
    POLICY_KD,
    POLICY_KP,
    RAMBO_JOINT_NAMES,
)
from wam_go2.tasks import QUADRUPED  # noqa: E402

Z = np.zeros(12)
GO2_FL = [3, 4, 5]          # FL hip/thigh/calf in Go2 motor order
GO2_REST = [i for i in range(12) if i not in GO2_FL]


def test_scalar_gains_are_broadcast_unchanged():
    _, _, kp, kd, _ = adapt_command(Z, Z, Z)
    assert kp.shape == (12,) and kd.shape == (12,)
    assert np.all(kp == POLICY_KP) and np.all(kd == POLICY_KD)


def test_vector_gains_follow_the_joint_mapping():
    kp = np.arange(12, dtype=float)      # RAMBO order: value == RAMBO index
    _, _, kp_go2, kd_go2, _ = adapt_command(Z, Z, Z, kp=kp, kd=2.0)
    for go2_idx, name in enumerate(GO2_JOINT_NAMES):
        assert kp_go2[go2_idx] == RAMBO_JOINT_NAMES.index(name)
    assert np.all(kd_go2 == 2.0)


def test_policy_gains_off_is_nominal():
    kp, kd = policy_gains(QUADRUPED.ee_joint_indices, False, 10.0, 0.5)
    assert np.all(kp == POLICY_KP) and np.all(kd == POLICY_KD)


def test_quadruped_compliance_lowers_only_fl_motors():
    kp, kd = policy_gains(QUADRUPED.ee_joint_indices, True, 10.0, 0.5)
    _, _, kp_go2, kd_go2, _ = adapt_command(Z, Z, Z, kp=kp, kd=kd)
    assert np.allclose(kp_go2[GO2_FL], 10.0) and np.allclose(kd_go2[GO2_FL], 0.5)
    assert np.allclose(kp_go2[GO2_REST], POLICY_KP)
    assert np.allclose(kd_go2[GO2_REST], POLICY_KD)


@pytest.mark.parametrize("bad", [np.full(11, 40.0), np.full((3, 4), 40.0)])
def test_wrong_gain_shape_is_rejected(bad):
    with pytest.raises(CommandError):
        adapt_command(Z, Z, Z, kp=bad, kd=1.0)


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_negative_or_nonfinite_gain_is_rejected(bad):
    with pytest.raises(CommandError):
        adapt_command(Z, Z, Z, kp=bad, kd=1.0)
    with pytest.raises(CommandError):
        adapt_command(Z, Z, Z, kp=40.0, kd=np.full(12, bad))


def test_zero_gain_is_allowed():
    _, _, kp, _, _ = adapt_command(Z, Z, Z, kp=0.0, kd=2.0)
    assert np.all(kp == 0.0)


def test_nonfinite_command_still_rejected():
    q = Z.copy()
    q[3] = np.nan
    with pytest.raises(CommandError):
        adapt_command(q, Z, Z)
