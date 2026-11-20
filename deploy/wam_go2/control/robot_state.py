"""Estimated robot state consumed by the RAMBO controller modules.

Field conventions mirror the QPEnv properties (qp_env.py:1173-1268):
- base_quat: IMU quaternion (w, x, y, z), body -> world
- projected_gravity_b: unit gravity direction in base frame
- base_rot_mat_rp:   R with p_projected = R @ p_body   (qp_env.py:1186-1189)
- base_rot_mat_rp_t: R with p_body = R @ p_projected   (qp_env.py:1198-1201)
- base_lin_vel_b / base_ang_vel_b: base frame
- base_height: world z of the base origin (estimated)
- joint_pos / joint_vel: RAMBO order [hip x4, thigh x4, calf x4]
- ee_pos_b: (4,3) foot positions in base frame, legs [FL, FR, RL, RR]
- foot_contacts: (4,) bool, measured foot contact, RAMBO foot order
"""

from dataclasses import dataclass, field

import numpy as np

from . import kinematics
from .math_utils import (
    quat_from_matrix,
    quat_rotate_inverse,
    rp_rotation_from_gravity_b,
)


@dataclass
class RobotState:
    joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(12))
    joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(12))
    base_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    base_ang_vel_b: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_lin_vel_b: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_height: float = 0.3
    foot_contacts: np.ndarray = field(default_factory=lambda: np.ones(4, dtype=bool))

    # derived, filled by finalize()
    projected_gravity_b: np.ndarray = field(default_factory=lambda: np.array([0.0, 0, -1]))
    base_rot_mat_rp: np.ndarray = field(default_factory=lambda: np.eye(3))
    base_rot_mat_rp_t: np.ndarray = field(default_factory=lambda: np.eye(3))
    base_quat_rp: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    ee_pos_b: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))

    def finalize(self):
        """Compute derived quantities after the raw fields are set."""
        self.projected_gravity_b = quat_rotate_inverse(
            self.base_quat, np.array([0.0, 0.0, -1.0]))
        self.base_rot_mat_rp_t = rp_rotation_from_gravity_b(self.projected_gravity_b)
        self.base_rot_mat_rp = self.base_rot_mat_rp_t.T
        self.base_quat_rp = quat_from_matrix(self.base_rot_mat_rp)
        self.ee_pos_b = kinematics.foot_positions_b(self.joint_pos)
        return self
