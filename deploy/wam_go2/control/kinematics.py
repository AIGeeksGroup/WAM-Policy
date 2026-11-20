"""Analytic Go2 leg kinematics, single robot (numpy).

Written for the real robot to give the same results as the quantities the RAMBO
training environment computes (rambo/source/isaaclab_tasks/isaaclab_tasks/direct/
rambo_quadruped/qp_env.py), so the trained policy sees the same inputs:
- foot_positions_b   = QPEnv.ee_pos_b            (qp_env.py:1240-1268)
- foot_jacobian      = QPEnv.all_foot_jacobian   (qp_env.py:1271-1304)
- ik_leg / ik_all    = motor_angles_from_foot_positions[_fl] (qp_env.py:1614-1674)
- gravity_compensation_torques ~ QPEnv.get_gravity_compensation_torques
  (qp_env.py:1306-1329), derived from the go2_description.urdf point-mass
  model because the PhysX whole-body jacobian is not available on hardware.

Joint vectors are in RAMBO order: [hip x4, thigh x4, calf x4], legs [FL,FR,RL,RR].
"""

import numpy as np

from ..constants import (
    GRAV_COMP,
    HIP_OFFSET,
    L_HIP_SIGN,
    LINK_LENGTHS,
)
from .math_utils import quat_rotate_inverse, wrap_to_pi


def foot_positions_b(joint_pos):
    """Foot positions in base frame, shape (4, 3), legs [FL, FR, RL, RR]."""
    theta_hip = joint_pos[0:4]
    theta_thigh = joint_pos[4:8]
    theta_knee = joint_pos[8:12]
    l_hip = L_HIP_SIGN * LINK_LENGTHS[0]
    l_up = LINK_LENGTHS[1]
    l_low = LINK_LENGTHS[2]

    leg_distance = np.sqrt(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(theta_knee))
    eff_swing = theta_thigh + theta_knee / 2

    off_x_hip = -leg_distance * np.sin(eff_swing)
    off_z_hip = -leg_distance * np.cos(eff_swing)
    off_y_hip = l_hip

    off_x = off_x_hip
    off_y = np.cos(theta_hip) * off_y_hip - np.sin(theta_hip) * off_z_hip
    off_z = np.sin(theta_hip) * off_y_hip + np.cos(theta_hip) * off_z_hip

    return np.stack([off_x, off_y, off_z], axis=1) + HIP_OFFSET


def foot_jacobian(joint_pos):
    """Flattened foot jacobian, shape (12, 12).

    Rows 3i..3i+2: linear velocity of foot i (base frame), i in [FL,FR,RL,RR].
    Columns are in RAMBO joint order (leg i uses columns [i, 4+i, 8+i]).
    """
    theta_hip = joint_pos[0:4]
    theta_thigh = joint_pos[4:8]
    theta_knee = joint_pos[8:12]
    l_hip = L_HIP_SIGN * LINK_LENGTHS[0]
    l_up = LINK_LENGTHS[1]
    l_low = LINK_LENGTHS[2]
    l_eff = np.sqrt(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(theta_knee))
    t_eff = theta_thigh + theta_knee / 2

    J = np.zeros((4, 3, 3))
    J[:, 0, 0] = 0.0
    J[:, 0, 1] = -l_eff * np.cos(t_eff)
    J[:, 0, 2] = (l_low * l_up * np.sin(theta_knee) * np.sin(t_eff) / l_eff
                  - l_eff * np.cos(t_eff) / 2)
    J[:, 1, 0] = -l_hip * np.sin(theta_hip) + l_eff * np.cos(theta_hip) * np.cos(t_eff)
    J[:, 1, 1] = -l_eff * np.sin(theta_hip) * np.sin(t_eff)
    J[:, 1, 2] = (-l_low * l_up * np.sin(theta_hip) * np.sin(theta_knee) * np.cos(t_eff) / l_eff
                  - l_eff * np.sin(theta_hip) * np.sin(t_eff) / 2)
    J[:, 2, 0] = l_hip * np.cos(theta_hip) + l_eff * np.sin(theta_hip) * np.cos(t_eff)
    J[:, 2, 1] = l_eff * np.sin(t_eff) * np.cos(theta_hip)
    J[:, 2, 2] = (l_low * l_up * np.sin(theta_knee) * np.cos(theta_hip) * np.cos(t_eff) / l_eff
                  + l_eff * np.sin(t_eff) * np.cos(theta_hip) / 2)

    flat = np.zeros((12, 12))
    for leg in range(4):
        flat[3 * leg:3 * leg + 3, [leg, 4 + leg, 8 + leg]] = J[leg]
    return flat


def ik_all(foot_local_positions):
    """IK for all four feet: (4,3) base-frame foot positions -> 12 joint angles.

    Same result as motor_angles_from_foot_positions (qp_env.py:1645-1674).
    """
    p = foot_local_positions - HIP_OFFSET
    l_hip = LINK_LENGTHS[0] * L_HIP_SIGN
    l_up = LINK_LENGTHS[1]
    l_low = LINK_LENGTHS[2]

    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    theta_calf = -np.arccos(np.clip(
        (x ** 2 + y ** 2 + z ** 2 - l_hip ** 2 - l_low ** 2 - l_up ** 2)
        / (2 * l_low * l_up), -1.0, 1.0))
    l = np.sqrt(np.clip(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(theta_calf),
                        1e-7, 1.0))
    theta_thigh = np.arcsin(np.clip(-x / l, -1.0, 1.0)) - theta_calf / 2
    c1 = l_hip * y - l * np.cos(theta_thigh + theta_calf / 2) * z
    s1 = l * np.cos(theta_thigh + theta_calf / 2) * y + l_hip * z
    theta_hip = np.arctan2(s1, c1)

    return wrap_to_pi(np.concatenate([theta_hip, theta_thigh, theta_calf]))


def ik_leg(leg, foot_local_position):
    """IK for one leg: (3,) base-frame foot position -> (hip, thigh, calf).

    Same result as motor_angles_from_foot_positions_fl / _fr (rambo_quadruped
    qp_env.py:1614-1641): the same closed-form
    leg IK for every leg, only the hip offset row and the abduction-link sign differ
    (l_hip = +link_lengths[0] for left legs, -link_lengths[0] for right).
    """
    p = np.asarray(foot_local_position, dtype=float) - HIP_OFFSET[leg]
    l_hip = LINK_LENGTHS[0] * L_HIP_SIGN[leg]
    l_up = LINK_LENGTHS[1]
    l_low = LINK_LENGTHS[2]

    x, y, z = p
    theta_calf = -np.arccos(np.clip(
        (x ** 2 + y ** 2 + z ** 2 - l_hip ** 2 - l_low ** 2 - l_up ** 2)
        / (2 * l_low * l_up), -1.0, 1.0))
    l = np.sqrt(np.clip(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(theta_calf),
                        1e-7, 1.0))
    theta_thigh = np.arcsin(np.clip(-x / l, -1.0, 1.0)) - theta_calf / 2
    c1 = l_hip * y - l * np.cos(theta_thigh + theta_calf / 2) * z
    s1 = l * np.cos(theta_thigh + theta_calf / 2) * y + l_hip * z
    theta_hip = np.arctan2(s1, c1)

    return wrap_to_pi(np.array([theta_hip, theta_thigh, theta_calf]))


def ik_fl(foot_local_position):
    """IK for the FL leg only: (3,) base-frame position -> (hip, thigh, calf).

    Same result as motor_angles_from_foot_positions_fl (qp_env.py:1614-1641).
    """
    p = foot_local_position - HIP_OFFSET[0]
    l_hip = LINK_LENGTHS[0]
    l_up = LINK_LENGTHS[1]
    l_low = LINK_LENGTHS[2]

    x, y, z = p
    theta_calf = -np.arccos(np.clip(
        (x ** 2 + y ** 2 + z ** 2 - l_hip ** 2 - l_low ** 2 - l_up ** 2)
        / (2 * l_low * l_up), -1.0, 1.0))
    l = np.sqrt(np.clip(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(theta_calf),
                        1e-7, 1.0))
    theta_thigh = np.arcsin(np.clip(-x / l, -1.0, 1.0)) - theta_calf / 2
    c1 = l_hip * y - l * np.cos(theta_thigh + theta_calf / 2) * z
    s1 = l * np.cos(theta_thigh + theta_calf / 2) * y + l_hip * z
    theta_hip = np.arctan2(s1, c1)

    return wrap_to_pi(np.array([theta_hip, theta_thigh, theta_calf]))


# ---------------------------------------------------------------------------
# Gravity compensation
# ---------------------------------------------------------------------------
_FRONT_SIGN = np.array([1.0, 1.0, -1.0, -1.0])   # FL, FR, RL, RR
_SIDE_SIGN = np.array([1.0, -1.0, 1.0, -1.0])


def _leg_point_masses(leg):
    """Point masses of one leg chain: list of (mass, frame, offset).

    frame: 0 = hip link frame (rotates with hip joint about x),
           1 = thigh link frame, 2 = calf link frame.
    Offsets mirror across y for right legs and across x for rear hips.
    """
    f = _FRONT_SIGN[leg]
    s = _SIDE_SIGN[leg]
    gc = GRAV_COMP
    hip_com = gc["hip_com"] * np.array([f, s, 1.0])
    thigh_rotor = gc["thigh_rotor_pos"] * np.array([1.0, s, 1.0])
    thigh_com = gc["thigh_com"] * np.array([1.0, s, 1.0])
    calf_rotor = gc["calf_rotor_pos"] * np.array([1.0, s, 1.0])
    calf_com = gc["calf_com"] * np.array([1.0, s, 1.0])
    foot_pos = gc["foot_pos"]
    return [
        (gc["hip_mass"], 0, hip_com),
        (gc["thigh_rotor_mass"], 0, thigh_rotor),
        (gc["thigh_mass"], 1, thigh_com),
        (gc["calf_rotor_mass"], 1, calf_rotor),
        (gc["calf_mass"], 2, calf_com),
        (gc["foot_mass"], 2, foot_pos),
    ]


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _leg_point_positions_b(leg, q_leg):
    """Base-frame positions of the point masses of one leg. q_leg=(hip,thigh,calf)."""
    s = _SIDE_SIGN[leg]
    hip_origin = HIP_OFFSET[leg]
    R_hip = _rot_x(q_leg[0])
    thigh_origin_in_hip = np.array([0.0, s * LINK_LENGTHS[0], 0.0])
    R_thigh = _rot_y(q_leg[1])
    calf_origin_in_thigh = np.array([0.0, 0.0, -LINK_LENGTHS[1]])
    R_calf = _rot_y(q_leg[2])

    positions = []
    for mass, frame, offset in _leg_point_masses(leg):
        if frame == 0:
            p = hip_origin + R_hip @ offset
        elif frame == 1:
            p = hip_origin + R_hip @ (thigh_origin_in_hip + R_thigh @ offset)
        else:
            p = hip_origin + R_hip @ (
                thigh_origin_in_hip + R_thigh @ (calf_origin_in_thigh + R_calf @ offset))
        positions.append((mass, p))
    return positions


def gravity_compensation_torques(joint_pos, base_quat):
    """Gravity compensation torques (12,), RAMBO joint order.

    Equivalent of QPEnv.get_gravity_compensation_torques (qp_env.py:1306-1329):
    tau = sum_links J_lin(world)^T * m * (0,0,+9.81). Computed here in the base
    frame: tau = J_b^T * m * R_bw @ (0,0,+9.81), with J_b from central
    differences of the analytic chain FK (exact to O(eps^2)).
    """
    g_b = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, GRAV_COMP["gravity"]]))
    tau = np.zeros(12)
    eps = 1e-6
    for leg in range(4):
        q_leg = np.array([joint_pos[leg], joint_pos[4 + leg], joint_pos[8 + leg]])
        base_points = _leg_point_positions_b(leg, q_leg)
        # jacobian of each point mass wrt the 3 leg joints (central differences)
        for j in range(3):
            dq = np.zeros(3)
            dq[j] = eps
            pts_p = _leg_point_positions_b(leg, q_leg + dq)
            pts_m = _leg_point_positions_b(leg, q_leg - dq)
            joint_idx = [leg, 4 + leg, 8 + leg][j]
            for (mass, pp), (_, pm) in zip(pts_p, pts_m):
                dp_dq = (pp - pm) / (2 * eps)
                tau[joint_idx] += mass * np.dot(dp_dq, g_b)
        del base_points
    return tau
