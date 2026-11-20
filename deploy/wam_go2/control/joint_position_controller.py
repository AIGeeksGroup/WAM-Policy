"""Raibert-heuristic joint position controller, single robot (numpy).

Behaves as the joint position controller of the RAMBO training environment
(rambo_quadruped/modules/joint_position_controller.py). Produces the reference joint
positions that the policy's 12 joint-action offsets are added to
(qp_env.py:734). Per contact mode:
- stance mode      -> task stance pose (quadruped [0, 0.8, -1.5])
- swing mode       -> task swing pose, then the manipulation leg
                      (task.ee_legs: FL) tracks its end-effector position
                      command through IK
- phase mode, swing segment -> Raibert foothold + cubic-Bezier swing
  trajectory + analytic IK
- phase mode, stance segment -> hold the current measured joint positions

All "projected com frame" quantities follow qp_env.py conventions:
p_proj = R_rp @ p_body, with z offset by base height where noted.
"""

import numpy as np

from ..constants import (
    FOOT_CLEARANCE,
    FOOT_HEIGHT,
    HIP_POSITIONS_IN_BODY_FRAME,
)
from ..tasks import QUADRUPED
from . import kinematics


def _cubic_bezier_z(x0, x1, t):
    progress = t ** 3 + 3 * t ** 2 * (1 - t)
    return x0 + progress * (x1 - x0)


def _cubic_bezier_xy(x0, x1, t):
    progress = 2 * ((t / 2) ** 3 + 3 * (t / 2) ** 2 * (1 - (t / 2)))
    return x0 + progress * (x1 - x0)


def _cubic_bezier_xy_inv(x0, x1, t):
    progress = 2 * (((t / 2 + 0.5) ** 3 + 3 * (t / 2 + 0.5) ** 2 * (1 - (t / 2 + 0.5))) - 0.5)
    return x0 + progress * (x1 - x0)


def _gen_swing_foot_trajectory(input_phase, start_pos, mid_pos, end_pos):
    """(4,) phase, (4,3) positions -> (4,3) foot targets. All in projected frame."""
    cutoff = 0.5
    out = np.zeros((4, 3))
    for i in range(4):
        t = input_phase[i]
        if t < cutoff:
            out[i, :2] = _cubic_bezier_xy(start_pos[i, :2], mid_pos[i, :2], t / cutoff)
            out[i, 2] = _cubic_bezier_z(start_pos[i, 2], mid_pos[i, 2], t / cutoff)
        else:
            out[i, :2] = _cubic_bezier_xy_inv(mid_pos[i, :2], end_pos[i, :2],
                                              (t - cutoff) / (1 - cutoff))
            out[i, 2] = _cubic_bezier_z(mid_pos[i, 2], end_pos[i, 2],
                                        (t - cutoff) / (1 - cutoff))
    return out


def compute_desired_foot_positions(base_rot_mat_rp, base_lin_vel_b, base_ang_vel_b,
                                   stance_duration, normalized_phase,
                                   phase_switch_foot_positions,
                                   hip_positions_in_body_frame=HIP_POSITIONS_IN_BODY_FRAME):
    """Raibert foothold + Bezier swing target (joint_position_controller.py:386-423)."""
    hip_position = (base_rot_mat_rp @ hip_positions_in_body_frame.T).T

    mid_position = hip_position.copy()
    mid_position[:, 2] = FOOT_HEIGHT

    hip_velocity_b = base_lin_vel_b[None, :] + np.cross(
        np.tile(base_ang_vel_b, (4, 1)), hip_positions_in_body_frame)
    hip_velocity = (base_rot_mat_rp @ hip_velocity_b.T).T

    land_position = hip_velocity * stance_duration[:, None] / 2
    land_position[:, 0] = np.clip(land_position[:, 0], -0.2, 0.2)
    land_position[:, 1] = np.clip(land_position[:, 1], -0.2, 0.2)
    land_position += hip_position
    land_position[:, 2] = FOOT_CLEARANCE

    switch_foot_position = phase_switch_foot_positions.copy()
    switch_foot_position[:, 2] += FOOT_CLEARANCE

    return _gen_swing_foot_trajectory(normalized_phase, switch_foot_position,
                                      mid_position, land_position)


class JointPositionController:
    def __init__(self, contact_generator, task=None):
        self._task = task or QUADRUPED
        self._contact_generator = contact_generator
        self._current_contact_state = np.ones(4, dtype=bool)
        self._phase_switch_foot_positions = np.zeros((4, 3))
        self._desired_joint_positions = self._task.desired_joint_pos_stance.copy()

    def reset(self, robot_state, ee_pos_commands):
        """Mirror of reset_idx (joint_position_controller.py:52-129).

        ee_pos_commands: (3,) for a single manipulation leg or (n_ee, 3) in
        task.ee_legs order, projected com frame.
        """
        rs = robot_state
        self._current_contact_state = rs.foot_contacts.copy()
        self._phase_switch_foot_positions = (rs.base_rot_mat_rp @ rs.ee_pos_b.T).T
        self._phase_switch_foot_positions[:, 2] += rs.base_height
        self._compute(rs, ee_pos_commands)
        return self._desired_joint_positions.copy()

    def update(self, robot_state, ee_pos_commands):
        """Mirror of update (joint_position_controller.py:131-238)."""
        rs = robot_state
        new_leg_state = rs.foot_contacts
        new_foot_positions = (rs.base_rot_mat_rp @ rs.ee_pos_b.T).T
        new_foot_positions[:, 2] += rs.base_height
        changed = self._current_contact_state != new_leg_state
        self._phase_switch_foot_positions[changed] = new_foot_positions[changed]
        self._current_contact_state = new_leg_state.copy()
        self._compute(rs, ee_pos_commands)
        return self._desired_joint_positions.copy()

    def _compute(self, rs, ee_pos_commands):
        task = self._task
        ee_cmds = np.atleast_2d(np.asarray(ee_pos_commands, dtype=float))
        if ee_cmds.shape != (task.num_ee, 3):
            raise ValueError(
                f"ee_pos_commands must be ({task.num_ee}, 3) for task "
                f"'{task.name}', got {ee_cmds.shape}")

        mode = self._contact_generator.desired_contact_mode
        state = self._contact_generator.desired_contact_state
        # expand (4,) foot flags to (12,) joint flags: [hip x4, thigh x4, calf x4]
        mode_expand = np.tile(mode, 3)
        state_expand = np.tile(state, 3)

        # stance mode -> fixed standing pose
        idx = mode_expand >= 0.5
        self._desired_joint_positions[idx] = task.desired_joint_pos_stance[idx]

        # swing mode -> swing pose; the manipulation leg tracks its
        # end-effector position command through IK (FL: jpc:76-95)
        idx = mode_expand <= -0.5
        self._desired_joint_positions[idx] = task.desired_joint_pos_swing[idx]

        for slot, leg in enumerate(task.ee_legs):
            d = ee_cmds[slot].copy()
            d[2] -= rs.base_height
            local = rs.base_rot_mat_rp_t @ d
            q_leg = kinematics.ik_leg(leg, local)
            for k, joint in enumerate((leg, 4 + leg, 8 + leg)):
                if idx[joint]:
                    self._desired_joint_positions[joint] = q_leg[k]

        # phase mode
        idx = (mode_expand < 0.5) & (mode_expand > -0.5)
        desired_foot_positions = compute_desired_foot_positions(
            rs.base_rot_mat_rp,
            rs.base_lin_vel_b,
            rs.base_ang_vel_b,
            self._contact_generator.stance_duration,
            self._contact_generator.normalized_phase,
            self._phase_switch_foot_positions,
            task.hip_positions_in_body_frame,
        )
        d_foot = desired_foot_positions.copy()
        d_foot[:, 2] -= rs.base_height
        foot_local = (rs.base_rot_mat_rp_t @ d_foot.T).T
        q_phase_swing = kinematics.ik_all(foot_local)

        idx_c = idx & state_expand          # phase, stance segment: hold current q
        self._desired_joint_positions[idx_c] = rs.joint_pos[idx_c]
        idx_s = idx & ~state_expand         # phase, swing segment: Bezier + IK
        self._desired_joint_positions[idx_s] = q_phase_swing[idx_s]
