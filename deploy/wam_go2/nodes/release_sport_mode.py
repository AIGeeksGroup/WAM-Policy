"""Release the Go2 motion service (sport mode) for low-level control.

Runs the official CheckMode/ReleaseMode loop (go2_stand_example.cpp:128-142)
over the unitree_api ROS2 topics and exits when the robot reports no active
motion mode. Run this once before starting the controller:

    ros2 run wam_go2_deploy release_sport_mode
"""

import sys

import rclpy
from rclpy.node import Node

from ..go2.motion_switcher import MotionSwitcherClient


def main(args=None):
    rclpy.init(args=args)
    node = Node("release_sport_mode")
    client = MotionSwitcherClient(node)

    def spin():
        rclpy.spin_once(node, timeout_sec=0.05)

    node.get_logger().info("checking motion mode ...")
    ok = client.release_until_clear(spin)
    if ok:
        node.get_logger().info("motion service released; low-level control is free")
    else:
        name = client.check_mode(spin)
        node.get_logger().error(
            f"could not release motion service (current mode: {name!r}); "
            "check the connection and try again, or use the Unitree app")
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
