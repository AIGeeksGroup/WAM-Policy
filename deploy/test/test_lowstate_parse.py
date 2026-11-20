import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wam_go2.constants import GO2_JOINT_NAMES, RAMBO_JOINT_NAMES  # noqa: E402
from wam_go2.go2.lowstate import Go2State, parse_lowstate  # noqa: E402


def _mock_lowstate():
    m = types.SimpleNamespace()
    m.tick = 12345
    m.motor_state = []
    for i in range(20):
        ms = types.SimpleNamespace()
        ms.q = float(i)          # motor index encoded in q
        ms.dq = float(i) * 0.1
        ms.tau_est = float(i) * 0.01
        ms.reserve = [0, 500]
        m.motor_state.append(ms)
    imu = types.SimpleNamespace()
    imu.quaternion = [0.9, 0.1, 0.2, 0.3]
    imu.gyroscope = [0.01, 0.02, 0.03]
    imu.accelerometer = [0.1, 0.2, 9.8]
    imu.rpy = [0.0, 0.1, 0.2]
    m.imu_state = imu
    m.bms_state = types.SimpleNamespace()
    m.foot_force = [100, 200, 300, 400]  # FR, FL, RR, RL
    m.foot_force_est = [0, 0, 0, 0]
    m.power_v = 28.5
    m.bit_flag = 0
    m.wireless_remote = [0] * 40
    return m


def test_parse_reorders_joints():
    s = parse_lowstate(_mock_lowstate())
    # q value encodes the original Go2 motor index; verify by joint name
    for i, name in enumerate(RAMBO_JOINT_NAMES):
        go2_index = GO2_JOINT_NAMES.index(name)
        assert s.q[i] == float(go2_index), name


def test_parse_reads_no_foot_force():
    # this release reads no foot contact force (LowState foot_force is not used)
    assert not hasattr(parse_lowstate(_mock_lowstate()), "foot_force")


def test_parse_imu():
    s = parse_lowstate(_mock_lowstate())
    assert np.allclose(s.quat, [0.9, 0.1, 0.2, 0.3])
    assert np.allclose(s.gyro, [0.01, 0.02, 0.03])


def test_motor_errors_default_is_per_state_uint32_array():
    first, second = Go2State(), Go2State()
    assert first.motor_errors.shape == (12,)
    assert first.motor_errors.dtype == np.uint32
    assert not first.motor_errors.any()
    first.motor_errors[0] = 0x100
    assert not second.motor_errors.any()


def test_parse_motor_errors_reorders_all_twelve_full_uint32_values():
    msg = _mock_lowstate()
    # Include communication-interrupted bit 8 and upper uint32 bits, which
    # must survive parsing even when the safety checker ignores unknown bits.
    for go2_index, motor in enumerate(msg.motor_state):
        motor.reserve = [0xA0000100 | go2_index, 500 + go2_index]
    state = parse_lowstate(msg)
    assert state.motor_errors.shape == (12,)
    assert state.motor_errors.dtype == np.uint32
    for rambo_index, name in enumerate(RAMBO_JOINT_NAMES):
        go2_index = GO2_JOINT_NAMES.index(name)
        assert int(state.motor_errors[rambo_index]) == 0xA0000100 | go2_index


def test_parse_motor_errors_ignores_unused_motor_slots():
    msg = _mock_lowstate()
    for motor in msg.motor_state[12:]:
        motor.reserve[0] = 0xFFFFFFFF
    assert not parse_lowstate(msg).motor_errors.any()
