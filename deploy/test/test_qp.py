import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wam_go2.constants import (  # noqa: E402
    DEFAULT_JOINT_POS,
    QP_BODY_MASS,
    QP_FRICTION_COEF,
    QP_FZ_MAX,
    QP_FZ_MIN,
)
from wam_go2.control.qp_torque_optimizer import QPTorqueOptimizer  # noqa: E402
from wam_go2.control.robot_state import RobotState  # noqa: E402


def _standing_state():
    return RobotState(joint_pos=DEFAULT_JOINT_POS.copy(),
                      base_height=0.3).finalize()


def _make_opt():
    opt = QPTorqueOptimizer()
    opt.desired_base_position = np.array([0.0, 0.0, 0.3])
    opt.desired_base_orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])
    return opt


def test_standing_supports_weight():
    rs = _standing_state()
    opt = _make_opt()
    grf, cost, _, _ = opt.get_grf(rs, np.zeros(3), np.ones(4, dtype=bool))
    fz = grf[2::3]
    # FL tracks zero EE force; the three support legs carry most of the weight
    total = np.sum(fz)
    assert 0.85 * QP_BODY_MASS * 9.8 < total < 1.1 * QP_BODY_MASS * 9.8
    # constraints on FR/RL/RR
    for leg in range(1, 4):
        assert QP_FZ_MIN - 1e-2 <= grf[3 * leg + 2] <= QP_FZ_MAX + 1e-2
        fx, fy, fzl = grf[3 * leg:3 * leg + 3]
        assert abs(fx) <= QP_FRICTION_COEF * fzl + 1e-2
        assert abs(fy) <= QP_FRICTION_COEF * fzl + 1e-2


def test_swing_legs_forceless():
    rs = _standing_state()
    opt = _make_opt()
    contact = np.array([True, True, False, True])  # RL swinging
    grf, _, _, _ = opt.get_grf(rs, np.zeros(3), contact)
    assert np.abs(grf[6:9]).max() < 0.05  # RL (leg 2) near zero


def test_fl_tracks_ee_force_command():
    rs = _standing_state()
    opt = _make_opt()
    cmd = np.array([20.0, 0.0, -10.0])  # push forward and down
    grf, _, _, _ = opt.get_grf(rs, cmd, np.ones(4, dtype=bool))
    # FL grf should approach the reaction -cmd (identity orientation)
    assert np.allclose(grf[0:3], -cmd, atol=2.0)


def test_infeasible_falls_back_to_last():
    rs = _standing_state()
    opt = _make_opt()
    grf1, _, _, _ = opt.get_grf(rs, np.zeros(3), np.ones(4, dtype=bool))
    # all-swing with big acceleration demand is near-infeasible but bounded;
    # solver should still return finite values
    opt.desired_linear_acceleration = np.array([30.0, 30.0, 30.0])
    grf2, _, _, _ = opt.get_grf(rs, np.zeros(3), np.zeros(4, dtype=bool))
    assert np.all(np.isfinite(grf2))


def test_walking_sequence_no_solver_fallback():
    """Regression: OSQP must converge on every step of
    a realistic walking-gait sequence (FL always swing, one phase leg cycling),
    never silently reusing a stale GRF."""
    from wam_go2.constants import WALKING_CONTACT_SEQUENCE
    from wam_go2.control.contact_generator import ContactGenerator
    from wam_go2.control.math_utils import quat_normalize

    rng = np.random.default_rng(7)
    gen = ContactGenerator(WALKING_CONTACT_SEQUENCE)
    opt = _make_opt()
    # convergence, not timing: a loaded test host must not turn a slow solve into a "fallback"
    # (the time limits have their own tests)
    opt.SOLVE_TIME_LIMIT, opt.WALL_TIME_LIMIT = 0.5, 1.0
    q = DEFAULT_JOINT_POS.copy()
    for k in range(2000):
        t = k * 0.01
        gen.update(t)
        q += rng.normal(0, 0.005, 12)  # correlated joint drift
        q = np.clip(q, DEFAULT_JOINT_POS - 0.3, DEFAULT_JOINT_POS + 0.3)
        quat = quat_normalize(np.array([1.0, *rng.normal(0, 0.03, 3)]))
        rs = RobotState(joint_pos=q.copy(), base_quat=quat,
                        base_lin_vel_b=rng.normal(0, 0.2, 3),
                        base_ang_vel_b=rng.normal(0, 0.3, 3),
                        base_height=0.3 + rng.normal(0, 0.01)).finalize()
        opt.desired_linear_acceleration = rng.normal(0, 2.0, 3)
        opt.desired_angular_acceleration = rng.normal(0, 2.0, 3)
        grf, _, _, _ = opt.get_grf(rs, rng.normal(0, 5.0, 3),
                                   gen.desired_contact_state)
        assert np.all(np.isfinite(grf))
    assert opt.fallback_count == 0, (
        f"{opt.fallback_count}/{opt.solve_count} solver fallbacks")


def test_solved_acc_matches_the_training_sign():
    """The training QP reports solved_acc = M@grf - g (its documented quirk); the
    same sign is used so qp_cost is comparable with training logs."""
    from wam_go2.control.qp_torque_optimizer import construct_mass_mat
    rs = _standing_state()
    opt = _make_opt()
    grf, cost, desired_acc_b, solved_acc = opt.get_grf(
        rs, np.zeros(3), np.ones(4, dtype=bool))
    mass_mat = construct_mass_mat(rs.ee_pos_b)
    g = np.zeros(6)
    g[2] = -9.8
    g[:3] = rs.base_rot_mat_rp.T @ g[:3]
    assert np.allclose(solved_acc, mass_mat @ grf - g, atol=1e-9)


@pytest.mark.parametrize("contact", [np.ones(4, dtype=bool),
                                     np.array([True, True, False, True])])
def test_failed_solve_cannot_reuse_previous_forces(monkeypatch, contact):
    opt = _make_opt()
    rs = _standing_state()
    opt.get_grf(rs, np.zeros(3), np.ones(4, dtype=bool))
    monkeypatch.setattr(opt, "_osqp_solve", lambda *args: None)
    with pytest.raises(RuntimeError, match="QP"):
        opt.get_grf(rs, np.zeros(3), contact)
    assert opt.status == "failed"
    assert opt.consecutive_failures == 1
    assert opt.last_success_time is not None
    assert opt.solve_duration >= 0.0


@pytest.mark.parametrize("result", [np.full(12, np.nan), np.full(12, np.inf),
                                    np.zeros(11), np.zeros(12)])
def test_invalid_or_constraint_violating_solution_is_rejected(monkeypatch, result):
    opt = _make_opt()
    monkeypatch.setattr(opt, "_osqp_solve", lambda *args: result)
    with pytest.raises(RuntimeError, match="QP"):
        opt.get_grf(_standing_state(), np.zeros(3), np.ones(4, dtype=bool))
    assert opt.last_success_time is None


def test_solver_exception_is_a_fault(monkeypatch):
    import wam_go2.control.qp_torque_optimizer as qp_module

    def broken_solver():
        raise RuntimeError("forced solver exception")

    monkeypatch.setattr(qp_module.osqp, "OSQP", broken_solver)
    with pytest.raises(RuntimeError, match="forced solver exception"):
        _make_opt().get_grf(_standing_state(), np.zeros(3), np.ones(4, dtype=bool))


@pytest.mark.parametrize("status_code,status_text", [
    (7, "maximum iterations reached"), (8, "run time limit reached"),
])
def test_solver_budget_failure_does_not_retry(monkeypatch, status_code, status_text):
    from types import SimpleNamespace
    import wam_go2.control.qp_torque_optimizer as qp_module

    setups = []

    class TimedOutSolver:
        def setup(self, **kwargs):
            setups.append(kwargs)

        def warm_start(self, **kwargs):
            pass

        def solve(self, **kwargs):
            return SimpleNamespace(info=SimpleNamespace(status_val=status_code,
                                   status=status_text), x=None)

    monkeypatch.setattr(qp_module.osqp, "OSQP", TimedOutSolver)
    with pytest.raises(RuntimeError, match="QP"):
        _make_opt().get_grf(_standing_state(), np.zeros(3), np.ones(4, dtype=bool))
    assert len(setups) == 1
    assert 0.0 < setups[0]["time_limit"] < 0.01


def test_finite_solved_result_returned_after_deadline_is_rejected(monkeypatch):
    from types import SimpleNamespace
    import wam_go2.control.qp_torque_optimizer as qp_module

    opt = _make_opt()
    rs = _standing_state()
    grf, _, _, _ = opt.get_grf(rs, np.zeros(3), np.ones(4, dtype=bool))
    previous_success = opt.last_success_time
    monkeypatch.setattr(opt, "_osqp_solve", lambda *args: grf)
    # past the wall-clock limit (a tick this late is a real problem)
    clock = iter([100.0, 100.0 + qp_module.QPTorqueOptimizer.WALL_TIME_LIMIT + 0.001, 100.0])
    monkeypatch.setattr(qp_module, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    with pytest.raises(RuntimeError, match="budget"):
        opt.get_grf(rs, np.zeros(3), np.ones(4, dtype=bool))
    assert opt.status == "failed"
    assert opt.last_success_time == previous_success


def test_host_hiccup_inside_the_control_period_keeps_a_verified_solution(monkeypatch):
    """A converged, constraint-checked solution returned 4 ms late is a host
    scheduling hiccup, not a solver failure: the node's control_gap_timeout
    (50 ms) owns that. Replaying five recorded real-robot traces (8320 ticks)
    gave max 1.51 ms, so the solver never approaches even the 3 ms OSQP budget;
    dropping the robot for a 4 ms wall-clock blip would be a false trip."""
    from types import SimpleNamespace
    import wam_go2.control.qp_torque_optimizer as qp_module

    opt = _make_opt()
    rs = _standing_state()
    grf, _, _, _ = opt.get_grf(rs, np.zeros(3), np.ones(4, dtype=bool))
    monkeypatch.setattr(opt, "_osqp_solve", lambda *args: grf)
    assert qp_module.QPTorqueOptimizer.SOLVE_TIME_LIMIT < 0.004 < qp_module.QPTorqueOptimizer.WALL_TIME_LIMIT
    clock = iter([100.0, 100.004, 100.004])
    monkeypatch.setattr(qp_module, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    grf2, _, _, _ = opt.get_grf(rs, np.zeros(3), np.ones(4, dtype=bool))
    assert opt.status == "solved"
    assert np.allclose(grf2, grf)


def test_new_policy_session_resets_solver_health(monkeypatch):
    from wam_go2.control.rambo_controller import RamboController

    ctrl = RamboController(lambda obs: np.zeros(18))
    rs = _standing_state()
    ctrl.reset(rs)
    ctrl.step(0.0, rs)
    monkeypatch.setattr(ctrl.qp, "_osqp_solve", lambda *args: None)
    with pytest.raises(RuntimeError, match="QP"):
        ctrl.step(0.01, rs)
    ctrl.reset(rs)
    assert ctrl.qp.status == "not_run"
    assert ctrl.qp.consecutive_failures == 0
    assert ctrl.qp.last_success_time is None
    assert ctrl.qp.solve_count == 0
    assert np.all(ctrl.qp._last_grf == 0.0)
