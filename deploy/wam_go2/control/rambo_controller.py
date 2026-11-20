"""RAMBO deployment controller: one 100 Hz control step.

Computes on the real robot the same observation and control as the training
environment's QPEnv.step + _get_observations (rambo_quadruped/qp_env.py:704-991)
for the task the checkpoint was trained on (tasks.QUADRUPED):

per tick (dt = 0.01 s):
  1. contact_generator.update(t); desired_joint_pos = jpc.update(state)
     [training does this at the END of the previous step - same thing]
  2. build the 81-dim per-step observation, push into the 5-step history,
     normalize, run the policy -> raw 18-dim action
  3. scale/clip the action; set QP setpoints; q_des = raibert + joint action,
     clamped to joint limits with margin
  4. solve the GRF QP -> tau_stance = -J^T f; add gravity compensation;
     mask by desired contact (manipulation leg always on); clip; EMA-filter
     pos and torque
  5. return (q_des, dq_des=0, tau_ff) with kp=40, kd=1

The simulation-only parts of the training step (external ee forces, resets,
rewards, domain randomization) are intentionally absent.
"""

import numpy as np

from ..constants import (
    ACTION_SCALE,
    HISTORY_LENGTH,
    JOINT_POS_LOWER,
    JOINT_POS_UPPER,
    JOINT_TORQUE_LIMIT,
    NUM_ACTIONS,
    POS_ALPHA,
    POS_LIMIT_MARGIN,
    TOR_ALPHA,
)
from ..tasks import QUADRUPED
from . import kinematics
from .contact_generator import ContactGenerator
from .joint_position_controller import JointPositionController
from .policy import PolicyOutputError, validate_policy_output
from .qp_torque_optimizer import QPTorqueOptimizer


# How the commands are taken (command_handling). Whatever the simulation can reach, the robot must reach too.
#   rambo: clipped to RAMBO's Isaac sampling ranges (tasks.py / constants.py) and velocity / force zeroed
#          while all four feet are in stance (qp_env.py:905-940). Every non-WAM demo (config/wam_go2.yaml default).
#   sim:   exactly as the QLM-Bench MuJoCo simulation that produced the WAM training data - RAMBOCore.submit
#          (qlm_bench/controllers/rambo/core.py) stores the base velocity and the FL target unclipped and nothing is
#          zeroed in stance. WAM runs (run_wam.sh / scripts/demo/wam_task.sh WAM_COMMAND_HANDLING).
COMMAND_HANDLING = {
    "rambo": {"clip_commands": True, "zero_commands_in_all_stance": True},
    "sim": {"clip_commands": False, "zero_commands_in_all_stance": False},
}


def command_handling_options(name, task):
    """RamboController keyword arguments for a command_handling name (rambo | sim)."""
    if name not in COMMAND_HANDLING:
        raise ValueError(f"command_handling must be one of {sorted(COMMAND_HANDLING)} (rambo = clipped as RAMBO "
                         f"was trained, sim = unclipped as the QLM-Bench MuJoCo simulation), not {name!r}")
    if name == "sim" and task.name != "quadruped":
        raise ValueError("command_handling sim is the QLM-Bench MuJoCo native9 path of the quadruped task "
                         f"(WAM); task {task.name!r} keeps rambo")
    return dict(COMMAND_HANDLING[name])


class RamboController:
    def __init__(self, policy, task=None, contact_sequence=None,
                 gravity_compensation=True, joint_limit_soft_factor=1.0,
                 clip_commands=True, zero_commands_in_all_stance=True):
        self.task = task or QUADRUPED
        # True/True = the RAMBO training ranges (the default); False/False = the QLM-Bench MuJoCo
        # simulation (command_handling "sim", WAM runs). See COMMAND_HANDLING above.
        self.clip_commands = bool(clip_commands)
        self.zero_commands_in_all_stance = bool(zero_commands_in_all_stance)
        self.policy = policy
        self.contact_generator = ContactGenerator(
            contact_sequence or self.task.contact_sequence)
        self.jpc = JointPositionController(self.contact_generator, self.task)
        self.qp = QPTorqueOptimizer(self.task)
        self._gravity_compensation = gravity_compensation

        # joint position clamp bounds (qp_env.py:748-752). soft factor covers
        # the (unverified) possibility that training used 0.9-scaled limits.
        center = 0.5 * (JOINT_POS_LOWER + JOINT_POS_UPPER)
        half = 0.5 * (JOINT_POS_UPPER - JOINT_POS_LOWER) * joint_limit_soft_factor
        self._pos_lower = center - half + POS_LIMIT_MARGIN
        self._pos_upper = center + half - POS_LIMIT_MARGIN

        # commands (projected com frame for ee pos/force; base frame for vel).
        # ee arrays are (n_ee, 3) in task.ee_legs order.
        self.velocity_commands = np.zeros(3)
        self.ee_pos_commands = self.task.ee_pos_default.copy()
        self.ee_force_commands = np.zeros((self.task.num_ee, 3))

        # buffers
        self._obs_history = np.zeros((HISTORY_LENGTH, self.task.num_obs_per_step))
        self._last_action = np.zeros(NUM_ACTIONS)
        self._desired_joint_pos = self.task.default_joint_pos.copy()
        self._desired_pos = self.task.default_joint_pos.copy()
        self._desired_tor = np.zeros(12)
        self._first_step = True
        self.last_raw_action = np.zeros(NUM_ACTIONS)
        self.last_qp_cost = 0.0
        self.last_grf = np.zeros(12)

        # entry blend (deployment only, see reset()): the controller can be
        # started from a pose that is NOT the task nominal and blend into it
        # under its own closed-loop QP/policy control.
        self._entry_ramp = 0.0
        self._blend_gait_time = None
        self._entry_action_ramp = True
        self._entry_q = self.task.default_joint_pos.copy()
        self._entry_height = self.task.nominal_base_height
        self.entry_fraction = 1.0
        self._gait_time0 = 0.0

    # ------------------------------------------------------------------
    def reset(self, robot_state, entry_ramp=0.0, action_ramp=True, gait_time=0.0,
              blend_gait_time=None):
        """Mirror of _reset_idx scheduler/obs handling.

        gait_time [s]: where the gait clock starts. 0 (default) = upstream
        reset at t = 0.

        entry_ramp > 0 [s]: start from the robot's CURRENT joint pose and
        base height and blend (raised cosine) into the task defaults over
        entry_ramp seconds - the JPC target, the QP height setpoint and (with
        action_ramp) the policy action all fade in with the same profile,
        and the gait clock is held until the blend is complete. 0 (default)
        = no blend.

        blend_gait_time [s]: where the gait clock is HELD during the blend.
        None (default) holds it at gait_time; the clock then runs from
        gait_time when the blend completes.
        """
        self._gait_time0 = max(0.0, float(gait_time))
        self._blend_gait_time = (None if blend_gait_time is None
                                 else max(0.0, float(blend_gait_time)))
        self.qp.reset()
        self.contact_generator.reset(self._gait_time0)
        self._desired_joint_pos = self.jpc.reset(robot_state, self.ee_pos_commands)
        self._obs_history[:] = 0.0
        self._last_action[:] = 0.0
        self._first_step = True
        self._entry_ramp = max(0.0, float(entry_ramp))
        self._entry_action_ramp = bool(action_ramp)
        self._entry_q = np.asarray(robot_state.joint_pos, dtype=float).copy()
        self._entry_height = float(robot_state.base_height)
        self.entry_fraction = 1.0 if self._entry_ramp <= 0.0 else 0.0
        if self._entry_ramp > 0.0:
            self._desired_joint_pos = self._entry_q.copy()

    def _entry_fraction(self, t):
        """Raised-cosine blend fraction in [0, 1] (1 = defaults reached)."""
        if self._entry_ramp <= 0.0:
            return 1.0
        u = float(np.clip(t / self._entry_ramp, 0.0, 1.0))
        return 0.5 - 0.5 * np.cos(np.pi * u)

    def set_commands(self, vel_cmd=None, ee_pos_cmd=None, ee_force_cmd=None,
                     leg_slot=0):
        """Set user commands: clipped to the RAMBO training ranges (clip_commands, the default), or stored as
        given like QLM-Bench's RAMBOCore.submit (command_handling "sim"). The ee force keeps its +-30 N clip in
        both (the simulation zeroes it; WAM's bridge refuses a non-zero force).

        leg_slot indexes task.ee_legs (0 = FL).
        """
        if vel_cmd is not None:
            self.velocity_commands = (self.task.clip_vel(vel_cmd) if self.clip_commands
                                      else np.array(vel_cmd, dtype=float))
        if ee_pos_cmd is not None:
            self.ee_pos_commands[leg_slot] = (self.task.clip_ee_pos(leg_slot, ee_pos_cmd) if self.clip_commands
                                              else np.array(ee_pos_cmd, dtype=float))
        if ee_force_cmd is not None:
            self.ee_force_commands[leg_slot] = self.task.clip_ee_force(
                leg_slot, ee_force_cmd)

    def set_ee_commands(self, leg_slot, pos=None, force=None):
        self.set_commands(ee_pos_cmd=pos, ee_force_cmd=force, leg_slot=leg_slot)

    # ------------------------------------------------------------------
    def step(self, time_since_start, robot_state):
        """Run one control step. Returns (q_des, dq_des, tau_ff), RAMBO order."""
        rs = robot_state
        task = self.task

        # 0. entry blend (deployment only, off by default): the gait clock is
        # HELD at blend_gait_time (default gait_time) while the pose blends,
        # and starts running at gait_time when the blend is complete - see
        # reset().
        s = self._entry_fraction(time_since_start)
        self.entry_fraction = s
        if self._entry_ramp <= 0.0:
            t_gait = self._gait_time0 + time_since_start
        elif time_since_start < self._entry_ramp:
            t_gait = (self._gait_time0 if self._blend_gait_time is None
                      else self._blend_gait_time)
        else:
            t_gait = self._gait_time0 + (time_since_start - self._entry_ramp)

        # 1. advance the contact schedule and the Raibert reference
        self.contact_generator.update(t_gait)
        jpc_target = self.jpc.update(rs, self.ee_pos_commands)
        if s >= 1.0:
            self._desired_joint_pos = jpc_target
        else:
            self._desired_joint_pos = self._entry_q + s * (jpc_target - self._entry_q)

        # training zeroes all commands for episodes sampled in an all-stance
        # segment (qp_env.py:905-940); do the same while all feet are in stance
        # (not in command_handling "sim": QLM-Bench MuJoCo never zeroes them)
        if (self.zero_commands_in_all_stance
                and np.sum(self.contact_generator.desired_contact_mode) > 3.9):
            vel_cmd = np.zeros(3)
            ee_force_cmd = np.zeros((task.num_ee, 3))
        else:
            vel_cmd = self.velocity_commands
            ee_force_cmd = self.ee_force_commands

        # 2. observation -> policy
        obs = self._build_obs(rs, vel_cmd, ee_force_cmd)
        try:
            action = self.policy(obs)
        except PolicyOutputError as exc:
            self.last_raw_action = exc.raw_action.copy()
            raise
        self.last_raw_action = np.array(action, copy=True)  # untouched fault evidence
        action = validate_policy_output(self.last_raw_action, (NUM_ACTIONS,))
        if self._entry_action_ramp and s < 1.0:
            action = action * s                      # fade the policy in

        # 3. action processing (qp_env.py:712-752). Base setpoints are the
        # task's default root state (position [0,0,h_nominal], identity
        # rotation).
        scaled = np.clip(action * ACTION_SCALE, -2.0 * ACTION_SCALE, 2.0 * ACTION_SCALE)

        h_des = task.nominal_base_height
        if s < 1.0:
            h_des = self._entry_height + s * (task.nominal_base_height - self._entry_height)
        self.qp.desired_base_position = np.array([0.0, 0.0, h_des])
        self.qp.desired_base_orientation_quat = task.desired_base_quat.copy()
        self.qp.desired_linear_velocity = np.array([vel_cmd[0], vel_cmd[1], 0.0])
        self.qp.desired_angular_velocity = np.array([0.0, 0.0, vel_cmd[2]])
        self.qp.desired_linear_acceleration = scaled[0:3]
        self.qp.desired_angular_acceleration = scaled[3:6]

        joint_offsets = scaled[6:18].copy()
        desired_motor_position = self._desired_joint_pos + joint_offsets
        desired_motor_position = np.clip(desired_motor_position,
                                         self._pos_lower, self._pos_upper)

        # 4. feedforward torque (qp_env.py:754-787)
        desired_contact = self.contact_generator.desired_contact_state
        grf, qp_cost, _, _ = self.qp.get_grf(rs, ee_force_cmd, desired_contact)
        self.last_grf = grf
        self.last_qp_cost = qp_cost
        jac = kinematics.foot_jacobian(rs.joint_pos)
        stance_torques = -(jac.T @ grf)

        if self._gravity_compensation:
            grav_torques = kinematics.gravity_compensation_torques(
                rs.joint_pos, rs.base_quat)
        else:
            grav_torques = np.zeros(12)

        contact_mask = np.tile(desired_contact, 3).astype(bool)
        # the manipulation leg (FL) always receives the QP torque
        # (qp_env.py:776-779)
        contact_mask[task.ee_joint_indices] = True
        desired_torque = grav_torques + np.where(contact_mask, stance_torques, 0.0)
        desired_torque = np.clip(desired_torque, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)

        # 5. EMA filters (qp_env.py:785-793)
        if self._first_step:
            self._desired_pos = desired_motor_position.copy()
            self._desired_tor = desired_torque.copy()
            self._first_step = False
        else:
            self._desired_pos = (POS_ALPHA * desired_motor_position
                                 + (1.0 - POS_ALPHA) * self._desired_pos)
            self._desired_tor = (TOR_ALPHA * desired_torque
                                 + (1.0 - TOR_ALPHA) * self._desired_tor)

        self._last_action = action.copy()

        return self._desired_pos.copy(), np.zeros(12), self._desired_tor.copy()

    # ------------------------------------------------------------------
    def _build_obs(self, rs, vel_cmd, ee_force_cmd):
        """Per-step observation in the exact training order
        (rambo_quadruped qp_env.py:969-983).
        """
        default = self.task.default_joint_pos
        latest = np.concatenate([
            [rs.base_height],
            rs.projected_gravity_b,
            rs.base_lin_vel_b,
            rs.base_ang_vel_b,
            rs.joint_pos - default,
            rs.joint_vel,
            self.contact_generator.desired_contact_phase,
            self.contact_generator.desired_contact_mode,
            self._desired_joint_pos - default,
            vel_cmd,
            np.asarray(self.ee_pos_commands).reshape(-1),
            np.asarray(ee_force_cmd).reshape(-1),
            self._last_action,
        ])
        assert latest.shape[0] == self.task.num_obs_per_step, latest.shape
        self._obs_history = np.vstack([self._obs_history[1:], latest[None, :]])
        return self._obs_history.reshape(-1)
