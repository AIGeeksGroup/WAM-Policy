"""Deployment position limits, separate from RAMBO's training constants."""

import numpy as np

from ..constants import JOINT_POS_LOWER, JOINT_POS_UPPER

# Hip output fence: +-57 degrees. Unitree's URDF - used by RAMBO and by the QLM-Bench MuJoCo simulation of
# the WAM training data - says +-60 degrees, and RAMBO's own clamp stops at 60 deg - POS_LIMIT_MARGIN
# (0.1 rad) = 54.27 deg, which the training simulation reaches (FL hip > 48 deg in 4.9 % of the training
# rows: oven 26 %, door 14 %, valve 11 %). At 57 degrees this fence does not cut any hip target RAMBO
# produces; it is the output guard only. Other joints keep the URDF bounds.
HIP_FENCE_DEG = 57.0
HARDWARE_JOINT_POS_LOWER = JOINT_POS_LOWER.copy()
HARDWARE_JOINT_POS_UPPER = JOINT_POS_UPPER.copy()
HARDWARE_JOINT_POS_LOWER[:4] = -np.deg2rad(HIP_FENCE_DEG)
HARDWARE_JOINT_POS_UPPER[:4] = np.deg2rad(HIP_FENCE_DEG)


def clip_joint_positions(q_des):
    """Return hardware-bounded positions in RAMBO order without mutating input."""
    return np.clip(q_des, HARDWARE_JOINT_POS_LOWER, HARDWARE_JOINT_POS_UPPER)
