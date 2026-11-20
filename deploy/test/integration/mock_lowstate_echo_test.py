"""Isolated diagnostic regression; only synthetic LowState, no actuator topics."""
from pathlib import Path
import os
import re
import sys
import time

# Isolated synthetic topic only; never join the robot's default domain.
_domain = int(os.environ.get("ROS_DOMAIN_ID", "42"))
if not 1 <= _domain <= 232:
    raise SystemExit("mock_lowstate_echo_test requires ROS_DOMAIN_ID in 1..232")
os.environ["ROS_DOMAIN_ID"] = str(_domain)
os.environ["ROS_LOCALHOST_ONLY"] = "1"
for key in ("FASTRTPS_DEFAULT_PROFILES_FILE", "FASTDDS_DEFAULT_PROFILES_FILE", "CYCLONEDDS_URI"):
    os.environ.pop(key, None)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from unitree_go.msg import LowState

from wam_go2.nodes.lowstate_echo import LowStateEcho

def main():
    rclpy.init()
    executor = SingleThreadedExecutor()
    echo = LowStateEcho()
    source = Node('synthetic_lowstate_echo_source')
    reports = []
    real_logger = echo.get_logger()
    class LoggerTap:
        def info(self, text, *args, **kwargs):
            reports.append(text)
            return real_logger.info(text, *args, **kwargs)
        def __getattr__(self, key):
            return getattr(real_logger, key)
    echo.get_logger = lambda: LoggerTap()
    publisher = source.create_publisher(LowState, '/lowstate', 10)
    count = [0]
    def publish():
        msg = LowState()
        msg.tick = count[0] * 2
        count[0] += 1
        msg.imu_state.quaternion[0] = 1.
        publisher.publish(msg)
    timer = source.create_timer(.002, publish)
    executor.add_node(echo)
    executor.add_node(source)
    def spin(duration):
        end = time.monotonic() + duration
        while time.monotonic() < end:
            executor.spin_once(timeout_sec=.005)
    try:
        spin(.8)
        assert echo._count > 20 and any('LIVE' in r for r in reports), 'positive live-data control failed'
        reports.clear()
        result = echo.set_parameters([Parameter('use_sim_time', value=True)])[0]
        assert result.successful
        timer.cancel()
        spin(1.3)
        assert echo.get_clock().now().nanoseconds == 0, 'ROS clock was not paused'
        assert any('STALE' in r and re.search(r'rx_rate=\s*0\.0 Hz', r) for r in reports), (
            'diagnostic stopped reporting STALE when ROS clock paused and LowState stopped')
        reports.clear()
        timer.reset()
        spin(.8)
        assert any('LIVE' in r for r in reports), 'diagnostic did not recover on fresh reception'
        print('ALL LOWSTATE ECHO CHECKS PASSED: LIVE -> zero receive rate/STALE with paused ROS clock -> LIVE on recovery')
    finally:
        timer.cancel()
        executor.remove_node(echo)
        executor.remove_node(source)
        echo.destroy_node()
        source.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
