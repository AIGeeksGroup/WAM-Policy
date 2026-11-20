"""Final motor commands respect the hip output fence: +-57 deg.

Unitree's URDF (used by RAMBO and the QLM-Bench MuJoCo training simulation) allows +-60 deg and RAMBO's own clamp stops
at 60 deg - 0.1 rad = 54.27 deg, which the training simulation reaches (~55 deg measured). At 57 deg the fence does
not cut any hip target RAMBO produces.
"""

import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wam_go2.adapter.command_adapter import adapt_command  # noqa: E402
from wam_go2.constants import DEFAULT_JOINT_POS  # noqa: E402


@pytest.mark.parametrize("rambo_hip,go2_motor", [(0, 3), (1, 0), (2, 9), (3, 6)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_each_hip_is_clamped_to_the_57_degree_fence(rambo_hip, go2_motor, sign):
    q = DEFAULT_JOINT_POS.copy()
    q[rambo_hip] = sign * np.deg2rad(59.5)
    output, *_ = adapt_command(q, np.zeros(12), np.zeros(12))
    assert output[go2_motor] == pytest.approx(sign * np.deg2rad(57.0))
    other_hips = [i for i in (0, 3, 6, 9) if i != go2_motor]
    assert np.array_equal(output[other_hips], np.zeros(3))


def test_exact_hip_boundaries_and_just_inside_are_unchanged():
    limit = np.deg2rad(57.0)
    q = DEFAULT_JOINT_POS.copy()
    q[:4] = [-limit, limit, -limit + 1e-9, limit - 1e-9]
    output, *_ = adapt_command(q, np.zeros(12), np.zeros(12))
    assert np.array_equal(output[[3, 0, 9, 6]], q[:4])


def test_existing_legal_command_preserves_all_values_and_inputs():
    q = np.array([.12, -.23, .34, -.45, .61, .72, .83, .94, -1.2, -1.3, -1.4, -1.5])
    dq = np.arange(12, dtype=float) / 100
    tau = np.arange(12, dtype=float) / 10
    kp = np.arange(12, dtype=float) + 30
    kd = np.arange(12, dtype=float) / 10
    inputs = [a.copy() for a in (q, dq, kp, kd, tau)]
    outputs = adapt_command(q, dq, tau, kp, kd)
    for source, before, output in zip((q, dq, kp, kd, tau), inputs, outputs):
        assert np.array_equal(source, before)
        assert np.array_equal(output, before[[1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]])


def test_other_joint_limits_remain_the_existing_urdf_limits():
    q = np.array([0.] * 4 + [-100., 100., -100., 100.] + [-100., 100., -100., 100.])
    output, *_ = adapt_command(q, np.zeros(12), np.zeros(12))
    assert np.array_equal(output[[4, 1, 10, 7]], [-1.5708, 3.4907, -.5236, 4.5379])
    assert np.array_equal(output[[5, 2, 11, 8]], [-2.7227, -.83776, -2.7227, -.83776])


def test_rambos_own_hip_clamp_lies_inside_the_fence():
    # what the training simulation could command (URDF +-60 deg minus RAMBO's 0.1 rad margin) passes unaltered
    from wam_go2.constants import JOINT_POS_LOWER, JOINT_POS_UPPER, POS_LIMIT_MARGIN
    assert np.degrees(JOINT_POS_UPPER[:4] - POS_LIMIT_MARGIN) == pytest.approx([54.27] * 4, abs=0.01)
    assert np.all(np.degrees(JOINT_POS_UPPER[:4] - POS_LIMIT_MARGIN) < 57.0)
    assert np.all(np.degrees(JOINT_POS_LOWER[:4] + POS_LIMIT_MARGIN) > -57.0)


CHECKPOINT = Path(__file__).resolve().parents[1] / "policies/quadruped/model_2000.pt"


@pytest.mark.skipif(not CHECKPOINT.is_file(),
                    reason="RAMBO checkpoint policies/quadruped/model_2000.pt not present (download it separately)")
def test_bundled_policy_legal_ee_target_reaches_the_motor_unaltered():
    pytest.importorskip("torch")
    from wam_go2.control.policy import RamboPolicy
    from wam_go2.control.rambo_controller import RamboController
    from wam_go2.control.robot_state import RobotState

    policy = RamboPolicy(str(CHECKPOINT))
    state = RobotState(joint_pos=DEFAULT_JOINT_POS.copy(), base_height=.30,
                       foot_contacts=np.ones(4, dtype=bool)).finalize()
    controller = RamboController(policy)
    controller.reset(state)
    # The exact setter used by the ROS ee_pos_cmd callback, with a legal target.
    controller.set_commands(ee_pos_cmd=[.30, .20, .35])
    command = controller.step(1.1, state)
    assert np.array_equal(controller.ee_pos_commands[0], [.30, .20, .35])
    from wam_go2.constants import JOINT_POS_UPPER, POS_LIMIT_MARGIN
    assert 48.0 < command[0][0] * 180 / np.pi and command[0][0] <= JOINT_POS_UPPER[0] - POS_LIMIT_MARGIN + 1e-12  # beyond Unitree's documented 48 deg, at most RAMBO's clamp
    output, *_ = adapt_command(*command)
    assert output[3] == command[0][0]                         # the 57 deg fence leaves it alone
