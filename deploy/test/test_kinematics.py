import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wam_go2.constants import DEFAULT_JOINT_POS  # noqa: E402
from wam_go2.control import kinematics  # noqa: E402
from wam_go2.control.math_utils import quat_normalize  # noqa: E402


def _random_q(rng):
    hip = rng.uniform(-0.8, 0.8, 4)
    thigh = rng.uniform(0.2, 1.4, 4)
    calf = rng.uniform(-2.4, -1.0, 4)
    return np.concatenate([hip, thigh, calf])


def test_fk_default_pose():
    feet = kinematics.foot_positions_b(DEFAULT_JOINT_POS)
    # symmetric pose: FL/FR mirrored in y, FL/RL mirrored around hip x
    assert np.allclose(feet[0, 0], feet[1, 0])
    assert np.allclose(feet[0, 1], -feet[1, 1])
    assert np.allclose(feet[0, 2], feet[1, 2])
    # all feet below the base at a plausible standing depth
    assert np.all(feet[:, 2] < -0.25)
    assert np.all(feet[:, 2] > -0.40)


def test_ik_fk_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = _random_q(rng)
        feet = kinematics.foot_positions_b(q)
        q_ik = kinematics.ik_all(feet)
        assert np.allclose(q_ik, q, atol=1e-6), (q, q_ik)


def test_ik_fl_matches_ik_all():
    rng = np.random.default_rng(2)
    for _ in range(20):
        q = _random_q(rng)
        feet = kinematics.foot_positions_b(q)
        q_fl = kinematics.ik_fl(feet[0])
        q_all = kinematics.ik_all(feet)
        assert np.allclose(q_fl, [q_all[0], q_all[4], q_all[8]], atol=1e-6)


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(3)
    for _ in range(20):
        q = _random_q(rng)
        J = kinematics.foot_jacobian(q)
        eps = 1e-6
        for j in range(12):
            dq = np.zeros(12)
            dq[j] = eps
            dp = (kinematics.foot_positions_b(q + dq)
                  - kinematics.foot_positions_b(q - dq)) / (2 * eps)
            assert np.allclose(J[:, j], dp.reshape(-1), atol=1e-5)


def test_gravity_compensation_sane():
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    tau = kinematics.gravity_compensation_torques(DEFAULT_JOINT_POS, quat)
    assert tau.shape == (12,)
    assert np.all(np.isfinite(tau))
    # magnitudes: small, dominated by thigh links (< 3 Nm each)
    assert np.all(np.abs(tau) < 3.0)
    # left/right symmetry at the symmetric default pose:
    # thigh and calf torques equal, hip torques mirrored
    assert np.allclose(tau[4], tau[5], atol=1e-9)   # FL/FR thigh
    assert np.allclose(tau[8], tau[9], atol=1e-9)   # FL/FR calf
    assert np.allclose(tau[0], -tau[1], atol=1e-9)  # FL/FR hip


def test_gravity_compensation_matches_potential_gradient():
    """tau must equal d/dq of sum(m * g_b . p(q)) - independent check via
    the total potential computed directly from the point-mass positions."""
    rng = np.random.default_rng(4)
    q = _random_q(rng)
    quat = quat_normalize(np.array([0.9, 0.05, -0.1, 0.02]))
    from wam_go2.control.kinematics import (
        _leg_point_positions_b,
        gravity_compensation_torques,
    )
    from wam_go2.control.math_utils import quat_rotate_inverse

    g_b = quat_rotate_inverse(quat, np.array([0.0, 0.0, 9.81]))

    def potential(qv):
        total = 0.0
        for leg in range(4):
            q_leg = np.array([qv[leg], qv[4 + leg], qv[8 + leg]])
            for mass, p in _leg_point_positions_b(leg, q_leg):
                total += mass * float(np.dot(g_b, p))
        return total

    tau = gravity_compensation_torques(q, quat)
    eps = 1e-6
    for j in range(12):
        dq = np.zeros(12)
        dq[j] = eps
        num = (potential(q + dq) - potential(q - dq)) / (2 * eps)
        assert abs(tau[j] - num) < 1e-5
