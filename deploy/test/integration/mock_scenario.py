"""Shared helpers of the mock-robot scenarios (mock_wam_test.py): a fresh
controller node per scenario, a probe on the command topics, and the entry
sequencer (scripts/demo_sequence.py). Needs a sourced ROS 2 environment and
the built package; not a pytest module."""

import os
import signal
import subprocess
import sys
import time

from geometry_msgs.msg import Point, Twist, Vector3
from std_msgs.msg import String

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from mock_robot_test import DEFAULT_Q_GO2, spin_for  # noqa: E402  (adds REPO to sys.path)

SEQUENCER = os.path.join(REPO, "scripts", "demo_sequence.py")
BASE_ARGS = ["--gap", "0.5", "--stand-wait", "2.5", "--settle", "1.2",
             "--mode-timeout", "10", "--policy-timeout", "12"]


class Probe:
    """Extra subscriptions on the mock node: what was published on the
    command topics and how the controller's mode evolved, all wall-clock
    stamped. fsm also records the mock's own rambo/fsm_cmd publishes (a node
    receives its own messages)."""

    def __init__(self, node):
        self.modes, self.ee, self.vel, self.fsm, self.force = [], [], [], [], []
        node.create_subscription(String, "rambo/mode", self._on_mode, 10)
        node.create_subscription(Point, "rambo/ee_pos_cmd",
                                 lambda m: self.ee.append((time.time(), (m.x, m.y, m.z))), 10)
        node.create_subscription(
            Twist, "cmd_vel",
            lambda m: self.vel.append((time.time(), (m.linear.x, m.linear.y, m.angular.z))), 10)
        node.create_subscription(String, "rambo/fsm_cmd",
                                 lambda m: self.fsm.append((time.time(), m.data.strip().upper())), 10)
        node.create_subscription(
            Vector3, "rambo/ee_force_cmd",
            lambda m: self.force.append((time.time(), (m.x, m.y, m.z))), 10)

    def _on_mode(self, m):
        if not self.modes or self.modes[-1][1] != m.data:
            self.modes.append((time.time(), m.data))

    def clear(self):
        for lst in (self.modes, self.ee, self.vel, self.fsm, self.force):
            lst.clear()


def compress(seq):
    out = []
    for v in seq:
        if not out or out[-1] != v:
            out.append(v)
    return out


def spawn_controller(extra=()):
    """extra: further controller parameters ("name:=value"), e.g. command_handling:=sim."""
    args = ["ros2", "run", "wam_go2_deploy", "controller", "--ros-args",
            "-p", "task:=quadruped",
            "-p", "auto_release_sport_mode:=false",
            "-p", "wired_nic:=none",      # mock robot on loopback: no cable to check
            "-p", "stand_duration:=1.0",
            # the mock robot is a plain Python timer competing with the
            # sequencer / ros2 CLI start-ups on the same machine and can stall
            # for ~0.34 s, past the code default 0.1 s. The real launch uses
            # 0.25 (config/wam_go2.yaml).
            "-p", "lowstate_timeout:=0.5"]
    for param in extra:
        args += ["-p", param]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)


def stop_controller(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    return proc.stdout.read() if proc.stdout else ""


def run_while_spinning(robot, cmd, timeout, stdout):
    """Run a subprocess while keeping the mock robot spinning (the controller's
    lowstate watchdog would trip if /lowstate paused)."""
    p = subprocess.Popen(cmd, stdout=stdout, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + timeout
    while p.poll() is None and time.time() < deadline:
        spin_for(robot, 0.05)
    if p.poll() is None:
        p.kill()
        p.wait()
        return None
    return p.returncode


def spin_until(robot, cond, timeout):
    """Keep the mock robot spinning until cond() holds (True) or timeout (False)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        spin_for(robot, 0.02)
    return bool(cond())


def begin_scenario(label, robot, probe, extra=()):
    """Fresh mock state + a fresh controller node, spun up (policy loaded)."""
    print(f"\n==== {label} ====", flush=True)
    robot.q = DEFAULT_Q_GO2.copy()
    robot.cmds.clear()
    probe.clear()
    proc = spawn_controller(extra)
    spin_for(robot, 6.0)                       # node startup + policy load
    # Start-up hygiene (mock artefact): the node's torch import plus the
    # previous node's shutdown can starve this plain-Python 500 Hz mock long
    # enough to trip the node's LowState watchdog before the scenario starts
    # (gaps of ~0.34 s). The sequencer then - correctly - refuses to
    # start (exit 4, "safety latched"). Re-arm here so the scenario tests
    # what it is meant to test; a real Go2 streams LowState on its own CPU.
    spin_until(robot, lambda: bool(probe.modes), 10.0)
    if probe.modes and "(safety:" in probe.modes[-1][1]:
        print(f"  note: safety latched during start-up ({probe.modes[-1][1]!r}); "
              "REARM (mock artefact, not a scenario failure)", flush=True)
        robot.request("REARM")
        spin_until(robot, lambda: bool(probe.modes) and probe.modes[-1][1] == "DAMP", 3.0)
    probe.clear()
    robot.cmds.clear()
    return proc


def end_scenario(proc, summary):
    node_log = stop_controller(proc)
    print("---- controller log (tail) ----")
    print("\n".join("  " + ln for ln in node_log.strip().splitlines()[-12:]))
    print("---- summary ----")
    print(summary)
    return summary
