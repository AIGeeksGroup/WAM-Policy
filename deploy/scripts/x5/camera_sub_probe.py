#!/usr/bin/env python3
"""Read-only camera receive probe (laptop side): subscribes to the D435i colour
and depth images with SENSOR_DATA-compatible QoS, counts frames and bytes for
--duration seconds and prints per-topic rate, inter-arrival max and payload.
Used by camera_check.sh concurrently with lowstate_probe.py."""
import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Imu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--ns", default="/camera/d435i")
    ap.add_argument("--label", default="camera")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    topics = {"color": f"{args.ns}/color/image_raw", "depth": f"{args.ns}/depth/image_rect_raw"}
    imu_topics = {"gyro": f"{args.ns}/gyro/sample", "accel": f"{args.ns}/accel/sample", "imu": f"{args.ns}/imu"}

    rclpy.init()
    node = Node("camera_receive_probe")
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5,
                     reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
    stats = {k: {"n": 0, "bytes": 0, "t": [], "enc": None, "wh": None} for k in topics}
    imu = {k: {"n": 0, "t": [], "last": None} for k in imu_topics}

    def cb(key):
        def _cb(msg):
            st = stats[key]
            st["n"] += 1; st["bytes"] += len(msg.data); st["t"].append(time.monotonic())
            st["enc"] = msg.encoding; st["wh"] = (msg.width, msg.height)
        return _cb

    def imu_cb(key):
        def _cb(msg):
            st = imu[key]; st["n"] += 1; st["t"].append(time.monotonic())
            st["last"] = (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                          msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z)
        return _cb

    for k, t in topics.items():
        node.create_subscription(Image, t, cb(k), qos)
    for k, t in imu_topics.items():
        node.create_subscription(Imu, t, imu_cb(k), qos)
    t0 = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic() - t0 < args.duration:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    dur = time.monotonic() - t0
    node.destroy_node(); rclpy.shutdown()

    lines = [f"label= {args.label}", f"duration_s= {dur:.1f}"]
    total_mb = 0.0
    for k, st in stats.items():
        if st["n"] == 0:
            lines.append(f"{k}: NO FRAMES"); continue
        ts = st["t"]; gaps = [b - a for a, b in zip(ts, ts[1:])]
        mb = st["bytes"] / 1e6; total_mb += mb
        lines.append(f"{k}: frames= {st['n']} rate_hz= {st['n'] / dur:.2f} max_gap_ms= {max(gaps) * 1e3 if gaps else 0:.0f} "
                     f"enc= {st['enc']} size= {st['wh'][0]}x{st['wh'][1]} payload_MB= {mb:.1f}")
    lines.append(f"total_payload_MB_per_s= {total_mb / dur:.2f}  (~{total_mb * 8 / dur:.0f} Mbit/s image payload)")
    for k, st in imu.items():
        if st["n"] == 0:
            lines.append(f"{k}: no messages (IMU disabled or topic absent)"); continue
        ts = st["t"]; gaps = [b - a for a, b in zip(ts, ts[1:])]
        g = st["last"]
        lines.append(f"{k}: msgs= {st['n']} rate_hz= {st['n'] / dur:.1f} max_gap_ms= {max(gaps) * 1e3 if gaps else 0:.0f} "
                     f"last gyro=({g[0]:+.3f},{g[1]:+.3f},{g[2]:+.3f}) accel=({g[3]:+.2f},{g[4]:+.2f},{g[5]:+.2f})")
    text = "\n".join(lines); print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(time.strftime("# %Y-%m-%d %H:%M:%S\n") + text + "\n")


if __name__ == "__main__":
    main()
