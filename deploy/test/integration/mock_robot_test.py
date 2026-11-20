#!/usr/bin/env python3
"""End-to-end closed-loop test against a mock Go2.

Publishes synthetic unitree_go/LowState at 500 Hz, launches the real
controller node in a separate process (`ros2 run wam_go2_deploy
controller`), drives the FSM via the rambo/fsm_cmd topic, and checks the
resulting /lowcmd stream:

  DAMP    kp=0, kd=DAMPING_KD, valid CRC
  STAND   kp=stand_kp, converges to DEFAULT_JOINT_POS (in Go2 motor order)
  POLICY  kp=40, kd=1, ~100 Hz, finite q, non-zero feedforward torque,
          valid CRC on every message

The mock robot tracks commanded joint positions whenever kp > 0 (perfect
motors), so the stand-up completes and tracking watchdogs stay quiet.

This is NOT a pytest unit test: it needs a sourced ROS2 environment and the
built package. Run it manually:

    source /opt/ros/<distro>/setup.bash
    source <unitree_msgs workspace>/install/setup.bash   # scripts/install_unitree_msgs.sh
    source <workspace with wam_go2_deploy built>/install/setup.bash
    python3 test/integration/mock_robot_test.py [--checkpoint PATH]

Exits non-zero if any check fails.
"""

import argparse
import os
import signal
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from unitree_go.msg import LowCmd, LowState
from unitree_api.msg import Request, Response

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from wam_go2.constants import (  # noqa: E402
    DAMPING_KD,
    DEFAULT_JOINT_POS,
    RAMBO_TO_GO2,
)
from wam_go2.go2.crc import compute_lowcmd_crc  # noqa: E402
from wam_go2.go2.motion_switcher import (  # noqa: E402
    API_CHECK_MODE, MOTION_SWITCHER_REQUEST_TOPIC, MOTION_SWITCHER_RESPONSE_TOPIC,
)


def to_go2(q_rambo):
    """RAMBO order -> Go2 motor order (the wire order)."""
    out = np.zeros(12)
    out[RAMBO_TO_GO2] = q_rambo
    return out


DEFAULT_Q_GO2 = to_go2(DEFAULT_JOINT_POS)
LEVEL_QUAT = np.array([1.0, 0.0, 0.0, 0.0])



class MockRobot(Node):
    """Fake Go2: 500 Hz LowState publisher + /lowcmd recorder."""

    def __init__(self):
        super().__init__("mock_go2")
        self.pub = self.create_publisher(LowState, "/lowstate", 10)
        self.create_subscription(LowCmd, "/lowcmd", self._on_cmd, 50)
        # Even auto_release_sport_mode=False verifies that no high-level
        # controller owns the robot. This isolated mock has no sport service.
        self.motion_pub = self.create_publisher(Response, MOTION_SWITCHER_RESPONSE_TOPIC, 10)
        self.create_subscription(Request, MOTION_SWITCHER_REQUEST_TOPIC,
                                 self._on_motion_request, 10)
        self.fsm_pub = self.create_publisher(String, "rambo/fsm_cmd", 10)
        # rambo/state: the controller's state stream for the WAM bridge
        self.states = []
        self.create_subscription(Float64MultiArray, "rambo/state", self._on_state_stream, 50)
        self.cmds = []
        self.tick = 0
        self.q = DEFAULT_Q_GO2.copy()   # Go2 order, tracks commands
        self.quat = LEVEL_QUAT.copy()
        self.gyro = np.zeros(3)
        self.create_timer(0.002, self._publish_state)

    def _on_motion_request(self, request):
        response = Response()
        response.header.identity.id = request.header.identity.id
        response.header.identity.api_id = request.header.identity.api_id
        response.header.status.code = 0 if request.header.identity.api_id == API_CHECK_MODE else 1
        response.data = '{"name": ""}' if response.header.status.code == 0 else '{}'
        self.motion_pub.publish(response)

    def _publish_state(self):
        msg = LowState()
        msg.tick = self.tick
        self.tick += 2  # ms
        for i in range(12):
            msg.motor_state[i].q = float(self.q[i])
            msg.motor_state[i].dq = 0.0
            msg.motor_state[i].tau_est = 0.0
        msg.imu_state.quaternion = [float(v) for v in self.quat]
        msg.imu_state.gyroscope = [float(v) for v in self.gyro]
        # specific force = -gravity in the body frame
        w, x, y, z = self.quat
        g_b = np.array([2 * (x * z - w * y), 2 * (y * z + w * x),
                        1 - 2 * (x * x + y * y)]) * -1.0   # R^T (0,0,-1)
        msg.imu_state.accelerometer = [float(v) for v in (-9.81 * g_b)]
        msg.imu_state.rpy = [0.0, 0.0, 0.0]
        msg.foot_force = [200, 200, 200, 200]      # all feet in contact
        msg.power_v = 30.0
        msg.wireless_remote = [0] * 40             # no buttons pressed
        self.pub.publish(msg)

    def _on_state_stream(self, msg):
        self.states.append((time.time(), list(msg.data)))
        if len(self.states) > 20000:
            del self.states[:10000]

    def _on_cmd(self, msg):
        kp = np.array([msg.motor_cmd[i].kp for i in range(12)])
        q = np.array([msg.motor_cmd[i].q for i in range(12)])
        if kp.max() > 0.0 and np.all(np.isfinite(q)):
            self.q = q.copy()    # perfect tracking
        self.cmds.append({
            "t": time.time(),
            "q": q,
            "kp": kp,
            "kd": np.array([msg.motor_cmd[i].kd for i in range(12)]),
            "tau": np.array([msg.motor_cmd[i].tau for i in range(12)]),
            "crc_ok": compute_lowcmd_crc(msg) == msg.crc,
        })

    def request(self, mode):
        m = String()
        m.data = mode
        self.fsm_pub.publish(m)


def check_state_stream(robot, check):
    """rambo/state: ~100 Hz, decodable, and consistent with the controller's mode."""
    from wam_go2 import state_stream
    states = robot.states
    check(len(states) > 100, f"rambo/state: only {len(states)} messages")
    if len(states) < 2:
        return {}
    last_t = states[-1][0]
    recent = [s for s in states if s[0] >= last_t - 1.0]
    rate = (len(recent) - 1) / max(1e-6, recent[-1][0] - recent[0][0]) if len(recent) > 1 else 0.0
    check(85.0 <= rate <= 115.0, f"rambo/state rate {rate:.1f} Hz over the last second")
    decoded = []
    try:
        decoded = [state_stream.decode(s[1]) for s in states[-300:]]
    except ValueError as e:
        check(False, f"rambo/state undecodable: {e}")
    modes = sorted({d.mode for d in decoded})
    check("POLICY" in modes, f"rambo/state never reported POLICY (modes {modes})")
    in_policy = [d for d in decoded if d.mode == "POLICY"]
    check(all(d.fresh and d.estimator_valid and not d.tripped for d in in_policy), "rambo/state POLICY samples not fresh/valid")
    check(all(0.15 < d.base_height < 0.5 for d in in_policy), "rambo/state POLICY height out of range")
    check(all(abs(np.linalg.norm(d.quat_wxyz) - 1.0) < 1e-3 for d in in_policy), "rambo/state quaternion not unit")
    check(any(any(v != 0.0 for v in d.action18) for d in in_policy), "rambo/state POLICY action18 all zero")
    ages = [d.wall - d.state_time for d in decoded]
    check(max(ages) < 0.05, f"rambo/state source age max {max(ages)*1e3:.1f} ms")
    return {"state_stream_rate_hz": rate, "state_stream_modes": modes, "state_stream_n": len(states)}


def spin_for(node, seconds):
    end = time.time() + seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.01)


def summarize(cmds, label):
    if not cmds:
        return {"phase": label, "count": 0}
    return {
        "phase": label,
        "count": len(cmds),
        "kp": float(np.median([c["kp"].max() for c in cmds])),
        "kd": float(np.median([c["kd"].max() for c in cmds])),
        "crc_ok": all(c["crc_ok"] for c in cmds),
        "q_finite": all(np.all(np.isfinite(c["q"])) for c in cmds),
        "tau_range": (float(min(c["tau"].min() for c in cmds)),
                      float(max(c["tau"].max() for c in cmds))),
    }


def check_policy(check, policy):
    check(policy["count"] > 100,
          f"POLICY: too few commands ({policy['count']}, expect ~100 Hz)")
    check(abs(policy["kp"] - 40.0) < 1e-6,
          f"POLICY: kp={policy['kp']} (expected 40)")
    check(abs(policy["kd"] - 1.0) < 1e-6,
          f"POLICY: kd={policy['kd']} (expected 1)")
    check(policy["crc_ok"], "POLICY: bad CRC")
    check(policy["q_finite"], "POLICY: non-finite joint targets")
    check(max(abs(v) for v in policy["tau_range"]) > 0.1,
          "POLICY: feedforward torque is all zero")


def run_quadruped(robot, check):
    results = []
    robot.cmds.clear()
    spin_for(robot, 1.0)
    damp = summarize(robot.cmds, "DAMP")
    results.append(damp)

    robot.cmds.clear()
    robot.request("STAND")
    spin_for(robot, 1.5)
    stand = summarize(robot.cmds, "STAND")
    stand_pose_ok = bool(robot.cmds and np.allclose(
        robot.cmds[-1]["q"], DEFAULT_Q_GO2, atol=1e-3))
    results.append(stand)

    robot.cmds.clear()
    # the node refuses POLICY until the stand-up completes;
    # retry like a real operator would instead of racing the 1 s stand
    for _ in range(4):
        robot.request("POLICY")
        spin_for(robot, 0.4)
    spin_for(robot, 1.6)
    policy = summarize(robot.cmds[-160:], "POLICY")
    results.append(policy)

    check(damp["count"] > 50, f"DAMP: too few commands ({damp['count']})")
    check(damp["kp"] == 0.0, f"DAMP: kp={damp['kp']} (expected 0)")
    check(abs(damp["kd"] - DAMPING_KD) < 1e-6,
          f"DAMP: kd={damp['kd']} (expected {DAMPING_KD})")
    check(damp["crc_ok"], "DAMP: bad CRC")
    check(stand["count"] > 50, f"STAND: too few commands ({stand['count']})")
    check(stand["kp"] > 0.0, "STAND: kp not raised")
    check(stand_pose_ok, "STAND: did not reach default pose")
    check(stand["crc_ok"], "STAND: bad CRC")
    check_policy(check, policy)
    print("stand_pose_ok:", stand_pose_ok)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="",
                    help="policy checkpoint (default: the task's checkpoint under policies/)")
    args = ap.parse_args()

    rclpy.init()
    robot = MockRobot()

    node_args = ["ros2", "run", "wam_go2_deploy", "controller", "--ros-args",
                 "-p", "task:=quadruped",
                 "-p", "auto_release_sport_mode:=false",
                 "-p", "wired_nic:=none",      # mock robot on loopback: no cable to check
                 "-p", "stand_duration:=1.0"]
    if args.checkpoint:
        node_args += ["-p", f"policy_checkpoint:={args.checkpoint}"]
    proc = subprocess.Popen(
        node_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True)  # own process group, so we can kill the tree

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    results = []
    try:
        spin_for(robot, 6.0)                       # node startup + policy load
        results = run_quadruped(robot, check)
        results.append(check_state_stream(robot, check))
        print("==== RESULTS ====")
        for d in results:
            print(d)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        out = proc.stdout.read() if proc.stdout else ""
        print("==== NODE LOG (tail) ====")
        print("\n".join(out.strip().splitlines()[-14:]))
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if failures:
        print("\n==== FAILURES ====")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
