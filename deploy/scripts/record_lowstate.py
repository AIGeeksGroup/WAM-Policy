#!/usr/bin/env python3
"""Record raw /lowstate samples to an .npz for offline analysis.

Usage (sourced ROS2 env):
    python3 scripts/record_lowstate.py [duration_s] [out.npz]

Default output: deploy/data/captures/lowstate_<n>.npz (next to scripts/).
"""
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from unitree_go.msg import LowState

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
if len(sys.argv) > 2:
    OUT = sys.argv[2]
else:
    base = os.path.join(os.path.dirname(__file__), "..", "data", "captures")
    os.makedirs(base, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(base, f"lowstate_{n}.npz")):
        n += 1
    OUT = os.path.join(base, f"lowstate_{n}.npz")


class Cap(Node):
    def __init__(self):
        super().__init__("lowstate_capture")
        self.rows = []
        self.create_subscription(LowState, "/lowstate", self.cb, 50)

    def cb(self, m):
        self.rows.append((
            time.time(), m.tick,
            [m.motor_state[i].q for i in range(12)],
            [m.motor_state[i].dq for i in range(12)],
            [m.motor_state[i].tau_est for i in range(12)],
            [m.motor_state[i].temperature for i in range(12)],
            [m.motor_state[i].mode for i in range(12)],
            list(m.imu_state.quaternion),
            list(m.imu_state.gyroscope),
            list(m.imu_state.accelerometer),
            list(m.imu_state.rpy),
            float(m.power_v), float(m.power_a),
            int(m.bit_flag),
        ))


def main():
    rclpy.init()
    n = Cap()
    end = time.time() + DURATION
    while time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.05)
    r = n.rows
    np.savez_compressed(
        OUT,
        t=np.array([x[0] for x in r]),
        tick=np.array([x[1] for x in r]),
        q=np.array([x[2] for x in r]),
        dq=np.array([x[3] for x in r]),
        tau_est=np.array([x[4] for x in r]),
        temperature=np.array([x[5] for x in r]),
        motor_mode=np.array([x[6] for x in r]),
        quat=np.array([x[7] for x in r]),
        gyro=np.array([x[8] for x in r]),
        accel=np.array([x[9] for x in r]),
        rpy=np.array([x[10] for x in r]),
        power_v=np.array([x[11] for x in r]),
        power_a=np.array([x[12] for x in r]),
        bit_flag=np.array([x[13] for x in r]),
    )
    print(f"saved {len(r)} samples over {DURATION}s -> {os.path.abspath(OUT)}")
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
