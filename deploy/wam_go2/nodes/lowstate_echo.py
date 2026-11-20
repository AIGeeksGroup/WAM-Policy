"""Bring-up tool: print a live summary of /lowstate.

Verifies communication, joint mapping, IMU and motor states before any command
is ever sent. Usage:
    ros2 run wam_go2_deploy lowstate_echo
"""

import time

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from unitree_go.msg import LowState

from ..constants import RAMBO_JOINT_NAMES
from ..go2.lowstate import parse_lowstate


class LowStateEcho(Node):
    PRINT_PERIOD = 0.5
    # Diagnostic only: mark silence lasting one display interval as stale.
    # This does not set or replace any controller safety timeout.
    STALE_AFTER = PRINT_PERIOD

    def __init__(self):
        super().__init__("lowstate_echo")
        self._count = 0
        self._last_tick = None
        self._rate_est = None
        self._last_receive = None
        self._report_time = time.monotonic()
        self._report_count = 0
        self.create_subscription(LowState, "/lowstate", self._on_state, 10)
        # The silence diagnostic must still print when /clock is paused or
        # absent; use the same steady time base as the receive ages below.
        self._diagnostic_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(self.PRINT_PERIOD, self._print, clock=self._diagnostic_clock)
        self._state = None
        self.get_logger().info("waiting for /lowstate ...")

    def _on_state(self, msg):
        now = time.monotonic()
        self._count += 1
        tick = int(msg.tick)
        if self._last_receive is not None and now - self._last_receive < self.STALE_AFTER:
            delta = (tick - self._last_tick) & 0xFFFFFFFF
            # Same tick-progress sanity bound as Go2Interface: normal uint32
            # wrap is forward, but duplicate/reset/jump intervals give no rate.
            max_delta = max(1000.0, (now - self._last_receive) * 1000.0 + 100.0)
            if 0 < delta < 0x80000000 and delta <= max_delta:
                rate = 1000.0 / delta
                self._rate_est = (rate if self._rate_est is None
                                  else 0.95 * self._rate_est + 0.05 * rate)
            else:
                self._rate_est = None
        else:
            self._rate_est = None
        self._last_tick = tick
        self._last_receive = now
        self._state = parse_lowstate(msg)

    def _print(self):
        if self._state is None:
            return
        now = time.monotonic()
        elapsed = now - self._report_time
        rx_rate = (self._count - self._report_count) / elapsed if elapsed > 0.0 else 0.0
        self._report_time, self._report_count = now, self._count
        age = now - self._last_receive
        stale = age >= self.STALE_AFTER
        tick_rate = ("unknown" if stale or self._rate_est is None
                     else f"{self._rate_est:6.1f} Hz")
        s = self._state
        np.set_printoptions(precision=3, suppress=True)
        lines = [
            f"msgs={self._count}  rx_rate={rx_rate:6.1f} Hz  tick_rate={tick_rate}  "
            f"age={age:.3f}s {'STALE' if stale else 'LIVE'}  "
            f"tick={s.tick_ms}  V={s.power_v:.1f}  bit_flag=0x{s.bit_flag:02x}",
            f"quat(wxyz)={s.quat}  rpy={s.rpy}",
            f"gyro={s.gyro}  accel={s.accel}",
            "joints (RAMBO order):",
        ]
        for i, name in enumerate(RAMBO_JOINT_NAMES):
            lines.append(f"  {name:16s} q={s.q[i]:+7.3f}  dq={s.dq[i]:+7.3f}  "
                         f"tau={s.tau_est[i]:+7.3f}")
        self.get_logger().info("\n" + "\n".join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = LowStateEcho()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
