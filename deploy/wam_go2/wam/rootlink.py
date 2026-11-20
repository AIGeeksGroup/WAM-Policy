"""The controller's state as QLM's five root-link fields.

QLM's ``make_observation`` needs the measured root state of every sample: it
computes each camera frame's world pose from the root link and the fixed camera
mount. WAM-Policy takes no body state as policy input, so the live packet carries
none (``root_state`` empty, no state semantics); these fields only place the
frames. Fields (``qlm_bench.policy_interface.physical.ROOT_FIELDS``): root-link
position, orientation as a unit xyzw quaternion, linear and angular velocity in
the WORLD frame, body-frame projected gravity.

The controller provides the Kalman base height and body-frame linear velocity,
the IMU quaternion (Go2 order w, x, y, z) and the gyro, so the bridge converts
the quaternion order and rotates the velocities into the world frame. The world
x/y position is unknown on the robot; it is sent as zero.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

ROOT_FIELDS = (
    "observation.state.base.position",
    "observation.state.base.orientation",
    "observation.state.base.linear_velocity",
    "observation.state.base.angular_velocity",
    "observation.state.projected_gravity",
)
GROUND_Z = 0.0          # the floor is the world z = 0 plane (QLM push-box-v2-plane-world-z0)


def _vector(value: Sequence[float], n: int, name: str) -> np.ndarray:
    v = np.asarray(value, dtype=np.float64)
    if v.shape != (n,) or not np.isfinite(v).all():
        raise ValueError(f"{name} must be {n} finite numbers")
    return v


def quat_xyzw_from_wxyz(quat_wxyz: Sequence[float]) -> list:
    """Go2/IMU (w, x, y, z) -> normalised protocol (x, y, z, w) as plain floats."""
    q = _vector(quat_wxyz, 4, "quaternion")
    norm = float(np.linalg.norm(q))
    if norm < 1e-6:
        raise ValueError("quaternion must not be zero")
    w, x, y, z = (q / norm).tolist()
    return [float(x), float(y), float(z), float(w)]


def rotate_inverse_xyzw(quaternion: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    """World -> body rotation by a unit xyzw quaternion."""
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    xyz, w = q[:3], q[3]
    twice = 2.0 * np.cross(xyz, v)
    return v - w * twice + np.cross(xyz, twice)


def rotate_xyzw(quaternion: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    """Body -> world rotation by a unit xyzw quaternion (inverse of the above)."""
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    xyz, w = q[:3], q[3]
    twice = 2.0 * np.cross(xyz, v)
    return v + w * twice + np.cross(xyz, twice)


def root_state(*, height: float, quat_wxyz: Sequence[float], lin_vel_b: Sequence[float],
               ang_vel_b: Sequence[float], ground_z: float = 0.0) -> dict:
    """The five ROOT_FIELDS (JSON-ready lists of floats) for one sample."""
    if not isinstance(height, (int, float)) or not math.isfinite(height):
        raise ValueError("height must be a finite number")
    if not isinstance(ground_z, (int, float)) or not math.isfinite(ground_z):
        raise ValueError("ground_z must be a finite number")
    q = quat_xyzw_from_wxyz(quat_wxyz)
    v_b = _vector(lin_vel_b, 3, "linear velocity")
    w_b = _vector(ang_vel_b, 3, "angular velocity")
    gravity = rotate_inverse_xyzw(q, [0.0, 0.0, -1.0])
    fields = {
        ROOT_FIELDS[0]: [0.0, 0.0, float(height) + float(ground_z)],
        ROOT_FIELDS[1]: q,
        ROOT_FIELDS[2]: [float(x) for x in rotate_xyzw(q, v_b)],
        ROOT_FIELDS[3]: [float(x) for x in rotate_xyzw(q, w_b)],
        ROOT_FIELDS[4]: [float(x) for x in gravity],
    }
    return fields
