#!/usr/bin/env python3
"""Hand the Go2 back to a high-level motion service (the opposite of
release_sport_mode), e.g. to record an official sport-mode motion:

    python3 scripts/select_motion_mode.py normal     # Go2 sport controller
    python3 scripts/select_motion_mode.py            # just print the current mode

Uses MotionSwitcher SelectMode (api 1002) + CheckMode (api 1001). Stop the
low-level controller first - two publishers on /lowcmd fight each other.
"""

import os
import sys

import rclpy
from rclpy.node import Node

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wam_go2.go2.motion_switcher import MotionSwitcherClient  # noqa: E402


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else None
    rclpy.init()
    node = Node("select_motion_mode")
    ms = MotionSwitcherClient(node)

    def spin():
        rclpy.spin_once(node, timeout_sec=0.02)

    current = ms.check_mode(spin)
    print(f"current motion mode: {current!r}")
    if name:
        ok, code = ms.select_mode(name, spin)
        print(f"SelectMode({name!r}) -> ok={ok} status={code}")
        current = ms.check_mode(spin)
        print(f"motion mode now: {current!r}")
        if current != name and not ok:
            sys.exit(1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
