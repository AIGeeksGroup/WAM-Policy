"""Quaternion / rotation helpers (numpy, single robot).

NumPy equivalents of the isaaclab.utils.math functions the RAMBO training environment uses.
Quaternions are (w, x, y, z), matching both IsaacLab and the Go2 IMU.
"""

import numpy as np


def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_rotate(q, v):
    """Rotate vector v by quaternion q (body -> world if q is base orientation)."""
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a + b + c


def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q (world -> body)."""
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def matrix_from_quat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_matrix(m):
    """Rotation matrix -> quaternion (w, x, y, z). Shepperd's method."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        return quat_normalize(np.array([
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        ]))
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return quat_normalize(np.array([
            (m[2, 1] - m[1, 2]) / s, 0.25 * s,
            (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s,
        ]))
    if m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return quat_normalize(np.array([
            (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
            0.25 * s, (m[1, 2] + m[2, 1]) / s,
        ]))
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return quat_normalize(np.array([
        (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
        (m[1, 2] + m[2, 1]) / s, 0.25 * s,
    ]))


def axis_angle_from_quat(q):
    """Quaternion -> axis-angle vector (isaaclab math.py axis_angle_from_quat)."""
    q = q * (1.0 if q[0] >= 0.0 else -1.0)  # ensure w >= 0
    mag = np.linalg.norm(q[1:])
    half_angle = np.arctan2(mag, q[0])
    angle = 2.0 * half_angle
    # Taylor expansion of sin(x/2)/x near zero (matches isaaclab eps handling)
    if np.abs(angle) > 1e-6:
        sin_half_angles_over_angles = np.sin(half_angle) / angle
    else:
        sin_half_angles_over_angles = 0.5 - angle ** 2 / 48.0
    return q[1:] / sin_half_angles_over_angles


def quat_error(q1, q2):
    """Rotation difference q1 * q2^-1 as axis-angle (isaaclab math.py:702-713)."""
    return axis_angle_from_quat(quat_mul(q1, quat_conjugate(q2)))


def rp_rotation_from_gravity_b(grav_b):
    """Roll-pitch (gravity-aligned) rotation from projected gravity in base frame.

    Same as isaaclab math.py:295-300. Returns R with p_base = R @ p_projected,
    i.e. this is base_rot_mat_rp_t in qp_env.py terminology.
    """
    col1 = np.array([-grav_b[2], 0.0, grav_b[0]])
    col1 = col1 / np.linalg.norm(col1)
    col2 = np.cross(-grav_b, col1)
    col2 = col2 / np.linalg.norm(col2)
    col3 = -grav_b
    return np.stack((col1, col2, col3), axis=1)


def wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi
