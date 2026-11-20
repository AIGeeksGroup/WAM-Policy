"""RAMBO task configuration of the trained checkpoint
(Isaac-RAMBO-Quadruped-Go2-v0).

The controller modules take a TaskConfig. Every value matches the task's
training environment, rambo/.../direct/rambo_quadruped/qp_env.py (see
constants.py for the line refs).

Foot / leg indices are RAMBO order [FL, FR, RL, RR]; joint vectors are
[hip x4, thigh x4, calf x4].
"""

from dataclasses import dataclass

import numpy as np

from . import constants as C


@dataclass(frozen=True)
class TaskConfig:
    name: str
    gym_id: str
    checkpoint_relpath: str          # relative to <share>/policies
    num_obs_per_step: int
    nominal_base_height: float
    default_joint_pos: np.ndarray
    desired_joint_pos_stance: np.ndarray
    desired_joint_pos_swing: np.ndarray
    hip_positions_in_body_frame: np.ndarray
    contact_sequence: dict
    ee_legs: tuple                   # manipulation legs (unconstrained in the QP,
                                     # IK-tracked in swing, always fed QP torque)
    ee_pos_default: np.ndarray       # (n_ee, 3) projected com frame
    ee_pos_range: np.ndarray         # (n_ee, 3, 2) [lo, hi]
    ee_force_range: np.ndarray       # (n_ee, 3, 2) [lo, hi]
    vel_range: np.ndarray            # (3, 2) vx, vy, wz
    desired_base_quat: np.ndarray    # QP orientation setpoint (default root rot)
    gravity_target_b: np.ndarray     # nominal projected gravity, base frame
    orientation_error_limit: float   # training termination ||g_b - target||
    min_base_height_terminate: float

    # ------------------------------------------------------------------
    @property
    def num_obs(self):
        return self.num_obs_per_step * C.HISTORY_LENGTH

    @property
    def num_ee(self):
        return len(self.ee_legs)

    @property
    def constrained_legs(self):
        return tuple(i for i in range(4) if i not in self.ee_legs)

    @property
    def ee_joint_indices(self):
        """Joint indices (RAMBO order) of all manipulation legs."""
        return [j for leg in self.ee_legs for j in (leg, 4 + leg, 8 + leg)]

    def clip_ee_pos(self, leg_slot, cmd):
        r = self.ee_pos_range[leg_slot]
        return np.clip(np.asarray(cmd, dtype=float), r[:, 0], r[:, 1])

    def clip_ee_force(self, leg_slot, cmd):
        r = self.ee_force_range[leg_slot]
        return np.clip(np.asarray(cmd, dtype=float), r[:, 0], r[:, 1])

    def clip_vel(self, cmd):
        return np.clip(np.asarray(cmd, dtype=float),
                       self.vel_range[:, 0], self.vel_range[:, 1])

    def gravity_error(self, projected_gravity_b):
        return float(np.linalg.norm(
            np.asarray(projected_gravity_b) - self.gravity_target_b))


def _rng(*pairs):
    return np.array(pairs, dtype=float)


QUADRUPED = TaskConfig(
    name="quadruped",
    gym_id="Isaac-RAMBO-Quadruped-Go2-v0",
    checkpoint_relpath="quadruped/model_2000.pt",
    num_obs_per_step=C.NUM_OBS_PER_STEP,
    nominal_base_height=C.NOMINAL_BASE_HEIGHT,
    default_joint_pos=C.DEFAULT_JOINT_POS,
    desired_joint_pos_stance=C.DESIRED_JOINT_POS_STANCE,
    desired_joint_pos_swing=C.DESIRED_JOINT_POS_SWING,
    hip_positions_in_body_frame=C.HIP_POSITIONS_IN_BODY_FRAME,
    contact_sequence=C.WALKING_CONTACT_SEQUENCE,
    ee_legs=(0,),
    ee_pos_default=C.DEFAULT_EE_POS_CMD[None, :],
    ee_pos_range=np.array([_rng(C.FL_POS_X, C.FL_POS_Y, C.FL_POS_Z)]),
    ee_force_range=np.array([_rng(*[(-C.FL_FORCE_LIM, C.FL_FORCE_LIM)] * 3)]),
    vel_range=_rng(C.VEL_RANGE_X, C.VEL_RANGE_Y, C.AVEL_RANGE_Z),
    desired_base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
    gravity_target_b=np.array([0.0, 0.0, -1.0]),
    orientation_error_limit=0.75,    # rambo_quadruped/qp_env.py:1105
    min_base_height_terminate=0.1,   # rambo_quadruped/qp_env.py:1096
)

TASKS = {QUADRUPED.name: QUADRUPED}


def get_task(name):
    try:
        return TASKS[str(name).strip().lower()]
    except KeyError:
        raise ValueError(f"unknown RAMBO task '{name}'; expected one of "
                         f"{sorted(TASKS)}")
