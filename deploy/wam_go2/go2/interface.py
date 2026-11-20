"""ROS2 interface to the Go2 low-level controller.

Publishes unitree_go/LowCmd on /lowcmd and subscribes unitree_go/LowState on
/lowstate (the rmw maps these to the DDS topics rt/lowcmd / rt/lowstate).
LowCmd initialization follows the official examples (unitree_sdk2
go2_stand_example InitLowCmd): head 0xFE,0xEF; level_flag 0xFF; all 20 motor
slots mode 0x01, q=PosStopF, dq=VelStopF, gains zero.

This class works with arrays in GO2 MOTOR ORDER; the command adapter converts
from RAMBO order before calling send().
"""

import time

import numpy as np
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from unitree_go.msg import LowCmd, LowState

from ..constants import (
    DAMPING_KD,
    GO2_NUM_MOTORS,
    LOWLEVEL_FLAG,
    POS_STOP_F,
    VEL_STOP_F,
)
from .crc import compute_lowcmd_crc
from .lowstate import parse_lowstate

# /lowstate is a 500 Hz sensor stream: BEST_EFFORT (sensor-data semantics) so
# that a lost packet is simply a lost frame and never a retransmission cycle.
# QoS alone does not prevent gaps: RELIABLE and BEST_EFFORT readers side by
# side see identical 20-250 ms gaps when the default FastDDS participant also
# announces the host's WiFi address and the robot streams /lowstate over
# WiFi. Binding DDS to the wired NIC (config/fastdds_wired.xml;
# scripts/demo/common.sh does it) gives 500.0 Hz, 2 ms spacing, zero gaps.
# KEEP_LAST 10 = the robot's last 20 ms.
LOWSTATE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE)


class Go2Interface:
    def __init__(self, node, lowcmd_topic="/lowcmd", lowstate_topic="/lowstate",
                 state_callback=None):
        self._node = node
        self._state_callback = state_callback
        self._pub = node.create_publisher(LowCmd, lowcmd_topic, 10)
        self._sub = node.create_subscription(
            LowState, lowstate_topic, self._on_lowstate, LOWSTATE_QOS)

        self.last_state = None            # parsed Go2State (RAMBO ordering)
        self.last_state_stamp = None      # accepted source state, monotonic [s]
        self.last_receive_stamp = None    # any callback, monotonic [s]
        self.state_count = 0              # accepted forward source ticks only
        self.receive_count = 0
        self.accepted_dt = 0.002
        self.source_fault = None         # discontinuity latch; restart to reset
        self._last_tick = None
        self._regressed_tick = None

        self._cmd = self._init_lowcmd()

    # ------------------------------------------------------------------
    @staticmethod
    def _init_lowcmd():
        cmd = LowCmd()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = LOWLEVEL_FLAG
        cmd.gpio = 0
        for i in range(GO2_NUM_MOTORS):
            cmd.motor_cmd[i].mode = 0x01  # FOC servo mode
            cmd.motor_cmd[i].q = float(POS_STOP_F)
            cmd.motor_cmd[i].dq = float(VEL_STOP_F)
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = 0.0
        return cmd

    def _on_lowstate(self, msg):
        now = time.monotonic()
        self.receive_count += 1
        self.last_receive_stamp = now
        tick = int(msg.tick)
        if not self._accept_tick(tick, now):
            return
        self.last_state = parse_lowstate(msg)
        self._last_tick = tick
        self.last_state_stamp = now
        self.state_count += 1
        if self._state_callback is not None:
            self._state_callback(self.last_state)

    def _accept_tick(self, tick, now):
        """Separate uint32 source progress from transport receive activity.

        A single nearby reordered frame is dropped. A >1 s regression, or a
        progressing stream behind the accepted tick, may be a robot reboot:
        latch closed because its estimator epoch cannot safely be reused.
        Reordered batches and reboot ticks cannot be distinguished from the
        tick alone, so the ambiguous case also requires a controller restart.
        """
        if self.source_fault is not None:
            return False
        if not 0 <= tick <= 0xFFFFFFFF:
            self.source_fault = "LowState tick outside uint32 range; restart controller"
            return False
        if self._last_tick is None:
            return True
        delta = (tick - self._last_tick) & 0xFFFFFFFF
        if delta == 0:
            return False
        if delta >= 0x80000000:
            backward = (self._last_tick - tick) & 0xFFFFFFFF
            progressing_regression = (self._regressed_tick is not None and
                0 < ((tick - self._regressed_tick) & 0xFFFFFFFF) < 0x80000000)
            if backward > 1000 or progressing_regression:
                self.source_fault = "LowState source tick regressed/restarted; restart controller"
            self._regressed_tick = tick
            return False
        # A reset from the upper uint32 half can resemble a wrap; accept only
        # progress plausible for elapsed host time (1 s scheduling tolerance).
        max_delta = max(1000.0, (now - self.last_state_stamp) * 1000.0 + 100.0)
        if delta > max_delta:
            self.source_fault = "LowState source tick jumped/discontinuous; restart controller"
            return False
        self._regressed_tick = None
        self.accepted_dt = delta * 1e-3
        return True

    def seconds_since_last_state(self):
        """Monotonic age of the last accepted source state, not last callback."""
        if self.last_state_stamp is None:
            return float("inf")
        return time.monotonic() - self.last_state_stamp

    def seconds_since_last_receive(self):
        if self.last_receive_stamp is None:
            return float("inf")
        return time.monotonic() - self.last_receive_stamp

    # ------------------------------------------------------------------
    def send(self, q_go2, dq_go2, kp_go2, kd_go2, tau_go2):
        """Publish a LowCmd. All arrays are length 12 in GO2 motor order."""
        cmd = self._cmd
        for i in range(12):
            m = cmd.motor_cmd[i]
            m.mode = 0x01
            m.q = float(q_go2[i])
            m.dq = float(dq_go2[i])
            m.kp = float(kp_go2[i])
            m.kd = float(kd_go2[i])
            m.tau = float(tau_go2[i])
        cmd.crc = compute_lowcmd_crc(cmd)
        self._pub.publish(cmd)

    def send_damping(self, kd=DAMPING_KD):
        """Soft e-stop: kp=0, kd>0, zero targets (official damping pattern)."""
        z = np.zeros(12)
        self.send(z, z, z, np.full(12, float(kd)), z)

    def send_passive(self):
        """All motors limp (kp=kd=tau=0)."""
        z = np.zeros(12)
        self.send(z, z, z, z, z)

    def resend_last(self):
        """Re-publish the last command (CRC already computed)."""
        self._pub.publish(self._cmd)
