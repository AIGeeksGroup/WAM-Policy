import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wam_go2.constants import DEFAULT_JOINT_POS, FOOT_RADIUS  # noqa: E402
from wam_go2.control import kinematics  # noqa: E402
from wam_go2.estimator.state_estimator import StateEstimator  # noqa: E402

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
GRAV_ACCEL = np.array([0.0, 0.0, 9.81])


def test_static_convergence():
    est = StateEstimator()
    q = DEFAULT_JOINT_POS.copy()
    dq = np.zeros(12)
    contacts = np.ones(4, dtype=bool)
    for _ in range(500):  # 1 s at 500 Hz
        est.update(0.002, IDENTITY, np.zeros(3), GRAV_ACCEL, q, dq, contacts)
    v = est.base_lin_vel_b(IDENTITY)
    assert np.linalg.norm(v) < 0.02
    # height from FK: feet at z=-0.3113 -> base at 0.3113 + foot radius
    feet = kinematics.foot_positions_b(q)
    expect_h = -feet[0, 2] + FOOT_RADIUS
    assert abs(est.base_height - expect_h) < 0.01


def test_constant_velocity_tracking():
    """Treadmill construction: joints move so the stance feet slide backward
    at exactly -v in the body frame; the KF must recover v."""
    est = StateEstimator()
    v_true = np.array([0.3, 0.0, 0.0])
    q = DEFAULT_JOINT_POS.copy()
    dt = 0.002
    contacts = np.ones(4, dtype=bool)
    for _ in range(1500):
        jac = kinematics.foot_jacobian(q)
        dq = np.zeros(12)
        for leg in range(4):
            J_leg = jac[3 * leg:3 * leg + 3, [leg, 4 + leg, 8 + leg]]
            dq[[leg, 4 + leg, 8 + leg]] = np.linalg.solve(J_leg, -v_true)
        est.update(dt, IDENTITY, np.zeros(3), GRAV_ACCEL, q, dq, contacts)
        # keep joints near default (do not integrate far from workspace)
    v = est.base_lin_vel_b(IDENTITY)
    assert np.linalg.norm(v - v_true) < 0.05, v


def test_no_contact_free_integration_bounded():
    est = StateEstimator()
    q = DEFAULT_JOINT_POS.copy()
    for _ in range(100):
        est.update(0.002, IDENTITY, np.zeros(3), GRAV_ACCEL, q, np.zeros(12),
                   np.zeros(4, dtype=bool))
    # perfect IMU (a_w = 0): velocity stays finite and small
    assert np.linalg.norm(est.x[0:3]) < 1e-6


def test_nan_sample_does_not_poison_filter():
    """Regression: one non-finite IMU sample must not permanently corrupt the
    KF; subsequent good samples keep working."""
    est = StateEstimator()
    q = DEFAULT_JOINT_POS.copy()
    contacts = np.ones(4, dtype=bool)
    for _ in range(200):
        est.update(0.002, IDENTITY, np.zeros(3), GRAV_ACCEL, q, np.zeros(12),
                   contacts)
    bad_accel = np.array([np.nan, 0.0, 9.81])
    est.update(0.002, IDENTITY, np.zeros(3), bad_accel, q, np.zeros(12), contacts)
    assert np.all(np.isfinite(est.x))
    for _ in range(200):
        est.update(0.002, IDENTITY, np.zeros(3), GRAV_ACCEL, q, np.zeros(12),
                   contacts)
    assert np.all(np.isfinite(est.x))
    assert np.linalg.norm(est.base_lin_vel_b(IDENTITY)) < 0.02


def test_divergence_guard_resets():
    est = StateEstimator()
    est.x[0] = np.inf  # simulate an already-corrupted state
    q = DEFAULT_JOINT_POS.copy()
    est.update(0.002, IDENTITY, np.zeros(3), GRAV_ACCEL, q, np.zeros(12),
               np.ones(4, dtype=bool))
    assert np.all(np.isfinite(est.x))
