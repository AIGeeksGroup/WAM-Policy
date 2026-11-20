#!/usr/bin/env python3
"""Synthetic camera streams for the WAM bridge mock test, in their own process.

Publishes what ego_stream and d435i_sim_camera.py would: /camera/go2/ego/image_raw
(rgb8 640x360 at ~14.3 fps, header.frame_id tf:<counter>) and
/camera/d435i/sim/color/image_raw (rgb8, --d435i-size, default 640x360 = what d435i_sim_camera.py
gives for the policy's profile, at 30 fps), stamped with the
wall clock at publication. A separate process because serialising 2.8 MB frames
inside the mock robot's node starves its 500 Hz LowState timer (140 ms gaps).

    python3 synthetic_cameras.py --control-dir DIR    # DIR/ego_off present -> no ego frames
"""
import argparse
import array
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def image(img, frame_id):
    m = Image()
    t = time.time()
    m.header.stamp.sec = int(t)
    m.header.stamp.nanosec = int((t - int(t)) * 1e9)
    m.header.frame_id = frame_id
    m.height, m.width = img.shape[0], img.shape[1]
    m.encoding = "rgb8"
    m.step = img.shape[1] * 3
    # an array.array takes rosidl's fast path; a bytes payload is checked element by
    # element (107 ms per 720p frame: the stream then runs at ~9 fps instead of 30)
    m.data = array.array("B", img.tobytes())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-dir", required=True)
    ap.add_argument("--only", choices=("ego", "d435i"), default=None, help="publish one stream (one process each)")
    ap.add_argument("--ego-period", type=float, default=0.070)
    ap.add_argument("--d435i-period", type=float, default=1.0 / 30.0)
    ap.add_argument("--d435i-size", default="640x360", help="WxH of the D435i sim frames")
    args = ap.parse_args()
    d_w, d_h = (int(v) for v in args.d435i_size.split("x"))
    rclpy.init()
    node = Node(f"synthetic_cameras_{args.only or 'both'}", enable_rosout=False, start_parameter_services=False)
    ego_pub = node.create_publisher(Image, "/camera/go2/ego/image_raw", qos_profile_sensor_data)
    d_pub = node.create_publisher(Image, "/camera/d435i/sim/color/image_raw", qos_profile_sensor_data)
    ego_img = np.full((360, 640, 3), 90, np.uint8)
    d_img = np.full((d_h, d_w, 3), 150, np.uint8)
    ego_off = os.path.join(args.control_dir, "ego_off")
    state = {"ego": 0}

    def ego():
        if os.path.exists(ego_off):
            return
        state["ego"] += 1
        ego_pub.publish(image(ego_img, f"tf:{6667 * state['ego']}"))      # time_frame steps ~6667 per frame

    def d435i():
        d_pub.publish(image(d_img, "d435i_sim"))
    if args.only in (None, "ego"):
        node.create_timer(args.ego_period, ego)
    if args.only in (None, "d435i"):
        node.create_timer(args.d435i_period, d435i)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
