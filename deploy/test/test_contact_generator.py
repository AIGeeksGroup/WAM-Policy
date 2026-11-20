import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wam_go2.constants import WALKING_CONTACT_SEQUENCE  # noqa: E402
from wam_go2.control.contact_generator import ContactGenerator  # noqa: E402


def _gen(t):
    g = ContactGenerator(WALKING_CONTACT_SEQUENCE)
    g.update(t)
    return g


def test_initial_stance():
    g = _gen(0.0)
    assert np.allclose(g.desired_contact_mode, [1.0, 1.0, 1.0, 1.0])
    assert np.all(g.desired_contact_state)
    assert np.allclose(g.desired_contact_phase, 1.0)


def test_after_one_second_fl_swings():
    g = _gen(1.5)
    # FL permanent swing
    assert g.desired_contact_mode[0] == -1.0
    assert not g.desired_contact_state[0]
    assert g.desired_contact_phase[0] == 0.0
    # FR/RL/RR in phase mode
    assert np.allclose(g.desired_contact_mode[1:], 0.0)
    # FR: after=0.5, phase=(0.5+0)/0.7 mod 1 = 0.714 < 0.8 -> contact
    assert abs(g.desired_contact_phase[1] - 0.5 / 0.7) < 1e-6
    assert g.desired_contact_state[1]
    # RL: (0.5 + 0.33*0.7) mod 0.7 = 0.031 -> phase 0.0443
    assert abs(g.desired_contact_phase[2] - (0.731 % 0.7) / 0.7) < 1e-6
    assert g.desired_contact_state[2]
    # RR: (0.5 + 0.67*0.7) mod 0.7 = 0.269 -> phase 0.3841
    assert abs(g.desired_contact_phase[3] - (0.969 % 0.7) / 0.7) < 1e-6
    assert g.desired_contact_state[3]


def test_fr_swing_window():
    # FR swings when phase >= 0.8: after in [0.56, 0.7) -> t in [1.56, 1.7)
    assert _gen(1.55).desired_contact_state[1]
    assert not _gen(1.60).desired_contact_state[1]
    assert not _gen(1.69).desired_contact_state[1]
    assert _gen(1.71).desired_contact_state[1]


def test_only_one_phase_leg_swings_at_a_time():
    # walking gait with 0.33/0.67 offsets and 0.2 swing window
    for t in np.arange(1.01, 12.0, 0.013):
        g = _gen(t)
        swinging = np.sum(~g.desired_contact_state[1:])
        assert swinging <= 1, f"t={t}: more than one phase leg in swing"


def test_extends_beyond_plan_horizon():
    # the plan totals 10 s; beyond it the last segments hold
    g = _gen(25.3)
    assert g.desired_contact_mode[0] == -1.0
    assert np.allclose(g.desired_contact_mode[1:], 0.0)
    # FR phase continuous: after = 25.3 - 1.0 = 24.3, (24.3 mod 0.7)/0.7
    expect = (24.3 % 0.7) / 0.7
    assert abs(g.desired_contact_phase[1] - expect) < 1e-6


def test_normalized_phase_and_stance_duration():
    g = _gen(1.5)
    # FR phase 0.714 in stance segment -> normalized 0.714/0.8
    assert abs(g.normalized_phase[1] - (0.5 / 0.7) / 0.8) < 1e-6
    # stance duration for phase legs = 0.7 * 0.8
    assert np.allclose(g.stance_duration[1:], 0.56)
    # FL in swing -> 0
    assert g.stance_duration[0] == 0.0
    g2 = _gen(1.6)  # FR in swing segment: phase 0.857 -> (0.857-0.8)/0.2
    expect = ((0.6 / 0.7) - 0.8) / 0.2
    assert abs(g2.normalized_phase[1] - expect) < 1e-6
