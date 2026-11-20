"""Joint order mapping between RAMBO (IsaacLab BFS) and Go2 hardware order.

The authoritative index arrays live in constants.py and are derived from the
joint NAME lists (never hand-typed indices), so a mistake in either list
fails loudly in the round-trip test.

RAMBO order: [FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, ..., FL_calf, ...]
Go2 order:   [FR_hip, FR_thigh, FR_calf, FL_..., RR_..., RL_...]
"""

import numpy as np

from ..constants import GO2_TO_RAMBO, RAMBO_TO_GO2


def rambo_to_go2(arr):
    """(12,) RAMBO order -> Go2 motor order."""
    arr = np.asarray(arr)
    return arr[GO2_TO_RAMBO]


def go2_to_rambo(arr):
    """(12,) Go2 motor order -> RAMBO order."""
    arr = np.asarray(arr)
    return arr[RAMBO_TO_GO2]
