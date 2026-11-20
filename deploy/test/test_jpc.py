import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wam_go2.constants import (  # noqa: E402
    DEFAULT_EE_POS_CMD,
    DEFAULT_JOINT_POS,
    WALKING_CONTACT_SEQUENCE,
)
from wam_go2.control import kinematics  # noqa: E402
from wam_go2.control.contact_generator import ContactGenerator  # noqa: E402
from wam_go2.control.joint_position_controller import (  # noqa: E402
    JointPositionController,
)
from wam_go2.control.robot_state import RobotState  # noqa: E402


def _standing_state():
    return RobotState(joint_pos=DEFAULT_JOINT_POS.copy(), base_height=0.31,
                      foot_contacts=np.ones(4, dtype=bool)).finalize()


def _setup(t):
    gen = ContactGenerator(WALKING_CONTACT_SEQUENCE)
    gen.reset(0.0)
    jpc = JointPositionController(gen)
    rs = _standing_state()
    jpc.reset(rs, DEFAULT_EE_POS_CMD.copy())
    gen.update(t)
    return gen, jpc, rs


def test_stance_gives_default_pose():
    _, jpc, rs = _setup(0.0)
    q = jpc.update(rs, DEFAULT_EE_POS_CMD.copy())
    assert np.allclose(q, DEFAULT_JOINT_POS, atol=1e-9)


def test_fl_swing_tracks_ee_command():
    gen, jpc, rs = _setup(1.5)  # FL in permanent swing
    ee_cmd = np.array([0.30, 0.15, 0.10])
    q = jpc.update(rs, ee_cmd)
    q_fl = np.array([q[0], q[4], q[8]])
    # verify by FK: resulting FL foot position (projected frame + height offset)
    q_full = DEFAULT_JOINT_POS.copy()
    q_full[[0, 4, 8]] = q_fl
    feet = kinematics.foot_positions_b(q_full)
    fl_proj = rs.base_rot_mat_rp @ feet[0]
    fl_proj[2] += rs.base_height
    assert np.allclose(fl_proj, ee_cmd, atol=1e-5), (fl_proj, ee_cmd)


def test_phase_stance_holds_current_q():
    gen, jpc, _ = _setup(1.5)
    rs = _standing_state()
    rs.joint_pos = DEFAULT_JOINT_POS + 0.05
    rs.finalize()
    q = jpc.update(rs, DEFAULT_EE_POS_CMD.copy())
    # FR/RL/RR are phase-mode in stance segment at t=1.5 -> hold current q
    for leg in (1, 2, 3):
        for j in (leg, 4 + leg, 8 + leg):
            assert np.isclose(q[j], rs.joint_pos[j])


def test_phase_swing_leg_uses_bezier_ik():
    gen, jpc, rs = _setup(1.6)  # FR in swing segment (see contact gen test)
    q = jpc.update(rs, DEFAULT_EE_POS_CMD.copy())
    assert not gen.desired_contact_state[1]
    # FR joints deviate from both the hold value and the stance pose
    assert not np.allclose([q[1], q[5], q[9]],
                           [rs.joint_pos[1], rs.joint_pos[5], rs.joint_pos[9]],
                           atol=1e-6)
    assert np.all(np.isfinite(q))
