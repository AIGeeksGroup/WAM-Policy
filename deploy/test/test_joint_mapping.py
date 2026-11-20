import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wam_go2.adapter.joint_mapping import go2_to_rambo, rambo_to_go2  # noqa: E402
from wam_go2.constants import (  # noqa: E402
    GO2_JOINT_NAMES,
    GO2_TO_RAMBO,
    RAMBO_JOINT_NAMES,
    RAMBO_TO_GO2,
)


def test_expected_index_arrays():
    # hand-derived from the two orderings; guards against edits to the lists
    assert list(RAMBO_TO_GO2) == [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
    assert list(GO2_TO_RAMBO) == [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]


def test_round_trip():
    x = np.arange(12.0)
    assert np.allclose(go2_to_rambo(rambo_to_go2(x)), x)
    assert np.allclose(rambo_to_go2(go2_to_rambo(x)), x)


def test_name_correspondence():
    # value at RAMBO slot i lands at the Go2 slot with the same joint name
    x_rambo = np.arange(12.0)
    x_go2 = rambo_to_go2(x_rambo)
    for i, name in enumerate(RAMBO_JOINT_NAMES):
        j = GO2_JOINT_NAMES.index(name)
        assert x_go2[j] == x_rambo[i]
