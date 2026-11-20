#!/usr/bin/env python3
"""Read-only /lowstate receive-timing probe (RDK X5 / switch bring-up).
Subscribes BEST_EFFORT, timestamps every callback on the
monotonic clock, and prints/saves arrival-gap and tick statistics. Creates no
publisher, sends nothing to the robot.

    python3 scripts/x5/lowstate_probe.py --duration 120 --label lowstate_only \
        --out data/logs/camera_checks_<date>/lowstate_only.txt

Also samples rx_bytes of the wired NIC (--nic, the laptop's NIC on the robot
cable) and of --other-nic (WiFi, if present) over the run so the report shows
where the bytes actually arrived.
"""
import argparse
import os
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from unitree_go.msg import LowState


def rx_bytes(nic):
    try:
        with open(f"/sys/class/net/{nic}/statistics/rx_bytes") as f:
            return int(f.read())
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--label", default="probe")
    ap.add_argument("--out", default=None)
    ap.add_argument("--nic", default="eth0", help="the laptop's NIC on the robot cable")
    ap.add_argument("--other-nic", default="wlo1")
    args = ap.parse_args()

    rclpy.init()
    node = Node("lowstate_timing_probe")
    received_ns, ticks = [], []
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                     reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE)

    def on_state(msg):
        received_ns.append(time.monotonic_ns())
        ticks.append(int(msg.tick))

    node.create_subscription(LowState, "/lowstate", on_state, qos)
    rx0, ox0 = rx_bytes(args.nic), rx_bytes(args.other_nic)
    started = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic() - started < args.duration:
            rclpy.spin_once(node, timeout_sec=0.05)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        observed_s = time.monotonic() - started
        rx1, ox1 = rx_bytes(args.nic), rx_bytes(args.other_nic)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    lines = [f"label= {args.label}", f"observation_s= {observed_s:.3f}", f"samples= {len(received_ns)}"]
    if len(received_ns) < 2:
        lines.append("FAIL: fewer than two frames - check device, DDS binding, message type, QoS")
    else:
        arrival = np.asarray(received_ns, dtype=np.int64)
        gap_ms = np.diff(arrival) / 1e6
        tick = np.asarray(ticks, dtype=np.int64)
        tick_step = (np.diff(tick) + (1 << 32)) % (1 << 32)
        active_s = (arrival[-1] - arrival[0]) / 1e9
        p50, p99, p999 = np.percentile(gap_ms, [50, 99, 99.9])
        lines += [
            f"active_interval_hz= {(len(arrival) - 1) / active_s:.2f}",
            f"whole_window_hz= {len(arrival) / observed_s:.2f}",
            f"arrival_gap_ms median/p99/p99.9/max= {p50:.3f} {p99:.3f} {p999:.3f} {gap_ms.max():.3f}",
            f"arrival_gaps_gt_10ms= {int(np.count_nonzero(gap_ms > 10))}",
            f"arrival_gaps_gt_20ms= {int(np.count_nonzero(gap_ms > 20))}",
            f"arrival_gaps_gt_50ms= {int(np.count_nonzero(gap_ms > 50))}",
            f"tick_step_ms median/max= {float(np.median(tick_step)):.1f} {int(tick_step.max())}",
            f"tick_duplicates= {int(np.count_nonzero(tick_step == 0))}",
            f"tick_steps_gt_2ms= {int(np.count_nonzero(tick_step > 2))}",
        ]
    if rx0 is not None and rx1 is not None:
        lines.append(f"rx_{args.nic}_MB= {(rx1 - rx0) / 1e6:.2f}  ({(rx1 - rx0) / 1e6 / observed_s:.3f} MB/s)")
    if ox0 is not None and ox1 is not None:
        lines.append(f"rx_{args.other_nic}_MB= {(ox1 - ox0) / 1e6:.2f}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(time.strftime("# %Y-%m-%d %H:%M:%S\n") + text + "\n")


if __name__ == "__main__":
    main()
