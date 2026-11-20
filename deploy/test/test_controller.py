import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wam_go2.constants import (  # noqa: E402
    DEFAULT_EE_POS_CMD,
    DEFAULT_JOINT_POS,
    JOINT_TORQUE_LIMIT,
    NUM_OBS_PER_STEP,
)
from wam_go2.control import kinematics  # noqa: E402
from wam_go2.control.math_utils import quat_normalize  # noqa: E402
from wam_go2.control.rambo_controller import RamboController  # noqa: E402
from wam_go2.control.robot_state import RobotState  # noqa: E402


class ZeroPolicy:
    def __init__(self):
        self.last_obs = None

    def __call__(self, obs):
        self.last_obs = np.array(obs)
        return np.zeros(18)


def _standing_state():
    return RobotState(joint_pos=DEFAULT_JOINT_POS.copy(), base_height=0.31,
                      foot_contacts=np.ones(4, dtype=bool)).finalize()


def _make():
    policy = ZeroPolicy()
    ctrl = RamboController(policy)
    rs = _standing_state()
    ctrl.reset(rs)
    return ctrl, policy, rs


def test_step_shapes_and_finiteness():
    ctrl, policy, rs = _make()
    q_des, dq_des, tau = ctrl.step(0.0, rs)
    assert q_des.shape == (12,) and dq_des.shape == (12,) and tau.shape == (12,)
    assert np.all(np.isfinite(q_des)) and np.all(np.isfinite(tau))
    assert np.all(dq_des == 0.0)
    assert np.all(np.abs(tau) <= JOINT_TORQUE_LIMIT + 1e-9)


def test_zero_action_standing_tracks_default_pose():
    ctrl, policy, rs = _make()
    q_des, _, _ = ctrl.step(0.0, rs)
    # all-stance at t=0: raibert reference = default stance pose, zero action
    assert np.allclose(q_des, DEFAULT_JOINT_POS, atol=1e-6)


def test_observation_content():
    ctrl, policy, rs = _make()
    ctrl.set_commands(vel_cmd=[0.3, 0.0, 0.1])
    ctrl.step(0.0, rs)
    obs = policy.last_obs
    assert obs.shape == (405,)
    # first 4 history frames are zero right after reset
    assert np.allclose(obs[:4 * NUM_OBS_PER_STEP], 0.0)
    latest = obs[4 * NUM_OBS_PER_STEP:]
    assert abs(latest[0] - rs.base_height) < 1e-9            # base height
    assert np.allclose(latest[1:4], rs.projected_gravity_b)  # gravity
    assert np.allclose(latest[4:7], rs.base_lin_vel_b)
    assert np.allclose(latest[7:10], rs.base_ang_vel_b)
    assert np.allclose(latest[10:22], rs.joint_pos - DEFAULT_JOINT_POS)
    assert np.allclose(latest[22:34], rs.joint_vel)
    # t=0: all-stance -> phase all 1, mode all 1
    assert np.allclose(latest[34:38], 1.0)
    assert np.allclose(latest[38:42], 1.0)
    # raibert reference - default = 0 in stance
    assert np.allclose(latest[42:54], 0.0, atol=1e-6)
    # velocity commands are ZEROED during the all-stance segment
    assert np.allclose(latest[54:57], 0.0)
    assert np.allclose(latest[57:60], DEFAULT_EE_POS_CMD)
    assert np.allclose(latest[60:63], 0.0)  # ee force zeroed in all-stance
    assert np.allclose(latest[63:81], 0.0)  # last_action starts at zero


def test_commands_active_after_stance_phase():
    ctrl, policy, rs = _make()
    ctrl.set_commands(vel_cmd=[0.3, 0.0, 0.1])
    ctrl.step(1.5, rs)  # walking segment
    latest = policy.last_obs[4 * NUM_OBS_PER_STEP:]
    assert np.allclose(latest[54:57], [0.3, 0.0, 0.1])


def test_command_clipping():
    ctrl, _, _ = _make()
    ctrl.set_commands(vel_cmd=[2.0, -2.0, 5.0], ee_force_cmd=[100, 0, 0])
    assert np.allclose(ctrl.velocity_commands, [0.5, -0.5, 0.5])
    assert ctrl.ee_force_commands[0, 0] == 30.0


def test_history_rolls():
    ctrl, policy, rs = _make()
    for k in range(6):
        ctrl.step(k * 0.01, rs)
    obs = policy.last_obs
    # after 6 steps the full history is populated (no zero frames)
    frames = obs.reshape(5, NUM_OBS_PER_STEP)
    assert not np.allclose(frames[0], 0.0)
    # base height identical across frames (static robot)
    assert np.allclose(frames[:, 0], rs.base_height)


def test_ema_smoothing():
    ctrl, policy, rs = _make()
    q1, _, t1 = ctrl.step(0.0, rs)

    class StepPolicy:
        def __call__(self, obs):
            a = np.zeros(18)
            a[6:] = 1.0  # +0.15 rad on every joint after scaling
            return a

    ctrl.policy = StepPolicy()
    q2, _, _ = ctrl.step(0.01, rs)
    # EMA alpha=0.5: first step moves half of the 0.15 offset
    delta = q2 - q1
    moved = np.abs(delta) > 1e-4
    assert np.all(np.abs(delta[moved] - 0.075) < 1e-3)


def test_tilted_base_still_finite():
    ctrl, policy, _ = _make()
    rs = RobotState(joint_pos=DEFAULT_JOINT_POS + 0.1,
                    base_quat=quat_normalize(np.array([0.95, 0.2, -0.1, 0.05])),
                    base_ang_vel_b=np.array([0.3, -0.2, 0.1]),
                    base_lin_vel_b=np.array([0.2, 0.1, -0.05]),
                    base_height=0.27,
                    foot_contacts=np.array([True, False, True, True])).finalize()
    q_des, _, tau = ctrl.step(2.34, rs)
    assert np.all(np.isfinite(q_des)) and np.all(np.isfinite(tau))


def test_gravity_compensation_present_in_torque():
    ctrl, policy, rs = _make()
    _, _, tau_with = ctrl.step(0.0, rs)
    ctrl2 = RamboController(ZeroPolicy(), gravity_compensation=False)
    ctrl2.reset(rs)
    _, _, tau_without = ctrl2.step(0.0, rs)
    grav = kinematics.gravity_compensation_torques(rs.joint_pos, rs.base_quat)
    # difference between the two runs equals the gravity term (same QP inputs)
    assert np.allclose(tau_with - tau_without, grav, atol=0.2)


class ConstPolicy:
    """Every action saturated positive: visible residual on every joint."""

    def __call__(self, obs):
        return np.ones(18)


def _free_leg_step(ee_soft):
    ctrl = RamboController(ConstPolicy())
    ctrl.ee_soft = ee_soft
    rs = _standing_state()
    ctrl.reset(rs)
    q_des, _, tau = ctrl.step(1.5, rs)      # t = 1.5 s: FL is the free leg
    return ctrl, q_des, tau


def test_ee_soft_removes_fl_residual_and_qp_torque_but_keeps_gravity_comp():
    ctrl_hard, q_hard, tau_hard = _free_leg_step(False)
    ctrl_soft, q_soft, tau_soft = _free_leg_step(True)
    fl = ctrl_soft.task.ee_joint_indices              # [0, 4, 8]
    rest = [j for j in range(12) if j not in fl]
    # same JPC target on the stance legs (the soft FL target follows the
    # measured foot inside the fence, see the fence test below) ...
    assert np.allclose(ctrl_hard._desired_joint_pos[rest], ctrl_soft._desired_joint_pos[rest])
    # ... soft FL tracks its target exactly, hard FL carries the policy residual
    assert np.allclose(q_soft[fl], ctrl_soft._desired_joint_pos[fl], atol=1e-9)
    assert np.all(np.abs(q_hard[fl] - ctrl_hard._desired_joint_pos[fl]) > 0.05)
    # the stance legs are untouched by the flag
    assert np.allclose(q_soft[rest], q_hard[rest])
    assert np.allclose(tau_soft[rest], tau_hard[rest])
    # soft FL torque == gravity compensation only
    rs = _standing_state()
    grav = kinematics.gravity_compensation_torques(rs.joint_pos, rs.base_quat)
    assert np.allclose(tau_soft[fl], grav[fl], atol=1e-9)
    # default is off (training behaviour)
    assert RamboController(ZeroPolicy()).ee_soft is False


def _state_with_fl(q_fl):
    q = DEFAULT_JOINT_POS.copy()
    q[[0, 4, 8]] = q_fl
    return RobotState(joint_pos=q, base_height=0.31,
                      foot_contacts=np.array([False, True, True, True])).finalize()


def test_ee_soft_target_follows_the_measured_foot_inside_the_fence():
    ctrl = RamboController(ZeroPolicy())
    ctrl.ee_soft = True
    cmd = np.array([0.32, 0.142, 0.20])
    ctrl.set_commands(ee_pos_cmd=cmd, leg_slot=0)
    rs0 = _standing_state()
    ctrl.reset(rs0)
    # a foot 4 cm ahead / 3 cm above the command (inside the box): the JPC
    # target must be the measured position, i.e. the FL joints do not move
    d = np.array([0.04, 0.0, 0.03])
    local = rs0.base_rot_mat_rp_t @ (cmd + d - np.array([0, 0, rs0.base_height]))
    q_fl = kinematics.ik_leg(0, local)
    rs = _state_with_fl(q_fl)
    q_des, _, _ = ctrl.step(1.5, rs)
    assert np.allclose(q_des[[0, 4, 8]], q_fl, atol=2e-3)
    # the nominal command is what the policy observes, not the fenced target
    assert np.allclose(ctrl.ee_pos_commands[0], cmd)
    def foot_and_fenced(offset):
        q = kinematics.ik_leg(0, rs0.base_rot_mat_rp_t @ (cmd + offset
                                                            - np.array([0, 0, rs0.base_height])))
        rs_ = _state_with_fl(q)
        foot = rs_.base_rot_mat_rp @ kinematics.foot_positions_b(rs_.joint_pos)[0]
        foot[2] += rs_.base_height                       # where the foot really is
        return foot, ctrl._fenced_ee_targets(rs_)[0]
    # a foot far ahead (outside the box; the IK clips, so use the FK'd foot):
    # the target sits on the x fence, y/z follow the foot within their fences
    foot, fenced = foot_and_fenced(np.array([0.30, 0.0, 0.0]))
    assert foot[0] > cmd[0] + 0.10
    assert np.isclose(fenced[0], cmd[0] + 0.10)
    assert np.allclose(fenced, np.clip(foot, cmd - ctrl.ee_soft_fence_lo, cmd + ctrl.ee_soft_fence_hi))
    # sag floor: a foot 8 cm below the command -> target clamped 2 cm below
    foot, fenced = foot_and_fenced(np.array([0.0, 0.0, -0.08]))
    assert foot[2] < cmd[2] - 0.05
    assert np.isclose(fenced[2], cmd[2] - 0.02)
    # hard mode: the target is the command regardless of where the foot is
    ctrl.ee_soft = False
    ctrl.reset(rs0)
    q_des_hard, _, _ = ctrl.step(1.5, rs)
    local_cmd = rs0.base_rot_mat_rp_t @ (cmd - np.array([0, 0, rs0.base_height]))
    assert np.allclose(q_des_hard[[0, 4, 8]], kinematics.ik_leg(0, local_cmd), atol=2e-3)


@pytest.mark.parametrize("raw", [
    np.full(18, np.nan), np.full(18, np.inf), np.full(18, -np.inf),
    np.zeros(17), np.zeros(19), np.zeros((1, 18)), np.zeros((18, 1)),
])
def test_invalid_policy_output_fails_before_creating_command(raw):
    ctrl, _, rs = _make()
    ctrl.policy = lambda obs: raw
    with pytest.raises(ValueError, match="policy output") as caught:
        ctrl.step(0.0, rs)
    np.testing.assert_equal(ctrl.last_raw_action, raw)
    np.testing.assert_equal(caught.value.raw_action, raw)
    assert ctrl.qp.solve_count == 0
    assert ctrl._first_step
