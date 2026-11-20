"""Parsing of unitree_go/LowState into numpy arrays.

Motor order conversion happens HERE: everything this module returns is in
RAMBO joint / foot order (see constants.py) unless suffixed with _go2.
"""

from dataclasses import dataclass, field

import numpy as np

from ..constants import RAMBO_TO_GO2

@dataclass
class Go2State:
    """Snapshot of one LowState message, RAMBO ordering."""
    tick_ms: int = 0
    q: np.ndarray = field(default_factory=lambda: np.zeros(12))
    dq: np.ndarray = field(default_factory=lambda: np.zeros(12))
    tau_est: np.ndarray = field(default_factory=lambda: np.zeros(12))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    power_v: float = 0.0
    bit_flag: int = 0
    # MotorState.reserve[0], including bit 8 (communication interrupted).
    # Keep all 32 wire bits; deciding which bits are faults belongs to safety.
    motor_errors: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.uint32))


def parse_lowstate(msg):
    """unitree_go/LowState -> Go2State (RAMBO joint & foot order)."""
    s = Go2State()
    s.tick_ms = int(msg.tick)
    q_go2 = np.array([msg.motor_state[i].q for i in range(12)])
    dq_go2 = np.array([msg.motor_state[i].dq for i in range(12)])
    tau_go2 = np.array([msg.motor_state[i].tau_est for i in range(12)])
    s.q = _reorder(q_go2)
    s.dq = _reorder(dq_go2)
    s.tau_est = _reorder(tau_go2)
    motor_errors_go2 = np.array(
        [msg.motor_state[i].reserve[0] for i in range(12)], dtype=np.uint32)
    s.motor_errors = _reorder(motor_errors_go2)
    imu = msg.imu_state
    s.quat = np.array([imu.quaternion[0], imu.quaternion[1],
                       imu.quaternion[2], imu.quaternion[3]])  # (w, x, y, z)
    s.gyro = np.array(imu.gyroscope[:3])
    s.accel = np.array(imu.accelerometer[:3])
    s.rpy = np.array(imu.rpy[:3])
    s.power_v = float(msg.power_v)
    s.bit_flag = int(msg.bit_flag)
    return s


def _reorder(arr_go2):
    """(12,) Go2 motor order -> RAMBO order (RAMBO slot i <- Go2 motor
    RAMBO_TO_GO2[i])."""
    return arr_go2[RAMBO_TO_GO2]
