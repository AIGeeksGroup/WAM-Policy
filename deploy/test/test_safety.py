import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wam_go2.safety.safety_manager import SafetyManager  # noqa: E402

LEVEL_G = np.array([0.0, 0.0, -1.0])
Q = np.zeros(12)


def _mgr():
    return SafetyManager()


def test_nominal_passes():
    m = _mgr()
    assert m.check(0.002, LEVEL_G, Q, Q,
                   base_lin_vel_b=np.array([0.3, 0.0, 0.0]), base_height=0.3)


def test_estimated_speed_trips():
    """Regression: KF free-integration stays finite (measured 10 m/s in 20 s
    on real data) so the isfinite guard never fired; the bound must."""
    m = _mgr()
    assert not m.check(0.002, LEVEL_G, Q, Q,
                       base_lin_vel_b=np.array([2.0, 0.0, 0.0]),
                       base_height=0.3)
    assert m.tripped and "speed" in m.reason


def test_estimated_height_trips():
    m = _mgr()
    assert not m.check(0.002, LEVEL_G, Q, Q,
                       base_lin_vel_b=np.zeros(3), base_height=-96.4)
    assert m.tripped and "height" in m.reason
    m = _mgr()
    assert not m.check(0.002, LEVEL_G, Q, Q,
                       base_lin_vel_b=np.zeros(3), base_height=0.7)
    assert m.tripped


def test_prone_state_passes():
    """The real robot lying prone (h=0.071, v drift 0.02 m/s) must NOT trip."""
    m = _mgr()
    assert m.check(0.002, LEVEL_G, Q, Q,
                   base_lin_vel_b=np.array([0.02, 0.0, 0.0]),
                   base_height=0.071)


def test_legacy_call_without_estimates():
    m = _mgr()
    assert m.check(0.002, LEVEL_G, Q, Q)


def test_estimator_bounds_gated_off_when_not_passed():
    """DAMP/STAND pass no estimates (None) - prone's low FK height must not
    trip. Regression: guard false-tripped STAND at 'base height 0.020 m'."""
    m = _mgr()
    # simulate DAMP/STAND call: estimates omitted
    assert m.check(0.002, LEVEL_G, Q, Q)
    assert not m.tripped


def test_non_finite_gyro_trips():
    m = _mgr()
    assert not m.check(0.002, LEVEL_G, Q, Q, base_ang_vel_b=np.array([0.0, np.nan, 0.0]))
    assert "non-finite" in m.reason
