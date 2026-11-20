#!/usr/bin/env python3
"""End-to-end test of the demo automation (scripts/demo_sequence.py) against
the mock Go2 from mock_robot_test.py - no robot, no network.

Scenarios (--scenarios, default: all; a fresh controller node each):

  happy       for each demo in --demos (default box_push, handshake_stand,
              plate_walk, stand_fl_up; timed variants):
              python3 scripts/demo_sequence.py --demo <demo> --gap 0.5
                  --stand-wait 2.5 --settle 1.2 --hold 2.0 --walk-time 1.0
              gains        DAMP (kp 0) -> STAND (kp 60) -> POLICY (kp 40) -> DAMP
              compliance   handshake demos: while rambo/ee_compliance is True
                           the FL motors (Go2 idx 3,4,5) run kp == ee_compliant_kp
                           (kd == ee_compliant_kd) and the other nine kp 40; all
                           12 back to 40 afterwards. A controller build without
                           the topic fails with "NODE LACKS COMPLIANCE - rebuild".
              ee           after rambo/ee_pos_cmd the FL targets move > 0.1 rad
                           away from the default stand pose (ee_pos_cmd applied)
              cmd_vel      plate_walk: +vx, 0, -vx, 0 in that order
              exit code    the sequencer exits 0
  forever     plate_walk with its default open-ended walk; once cmd_vel is
              non-zero the mock publishes DAMP on rambo/fsm_cmd (= the
              operator's stop): exit 0, >= 5 zero Twists, nothing published
              afterwards, node damped.
  abort       handshake_stand --hold 30; once the sequencer has switched
              compliance ON the mock publishes rambo/fsm_cmd DAMP (= a DAMP
              request during a timed step / a safety trip). The sequencer
              must exit 2 within 3 s, publish
              >= 5 zero Twists and >= 3 False Bools after the mode change, and
              publish no ee_pos_cmd / non-zero cmd_vel after the abort.
  end_policy  plate_walk --end policy --walk-time 1.0: exit 0 while rambo/mode
              is still POLICY and the node keeps running (kp 40 frames after
              the sequencer is gone); the test then damps the node itself.
  sigint      handshake_stand --hold 30; SIGINT to the sequencer once
              compliance is ON -> exit 130 and a DAMP on rambo/fsm_cmd (plus
              the zero Twists / False Bools of the abort path); the node damps.

Needs a ROS 2 shell with the unitree_go messages and this package built and
sourced, plus the deploy Python environment (see README.md); NOT a pytest
(pytest.ini excludes integration/), run it by hand:

    python3 test/integration/mock_demo_test.py [--scenarios happy abort end_policy sigint]
                                               [--demos handshake_stand plate_walk]
                                               [--compliant-kp 10 --compliant-kd 0.5]

Prints ALL CHECKS PASSED and exits 0 when every check holds.
"""

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Twist, Vector3
from std_msgs.msg import Bool, String

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from mock_robot_test import DEFAULT_Q_GO2, MockRobot, spin_for  # noqa: E402  (adds REPO to sys.path)

from wam_go2.constants import RAMBO_TO_GO2, STAND_KP  # noqa: E402
from wam_go2.control import kinematics  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "scripts"))
import demo_sequence  # noqa: E402  (nominal ee per demo)

SEQUENCER = os.path.join(REPO, "scripts", "demo_sequence.py")
FL = [3, 4, 5]                               # Go2 motor indices of FL hip/thigh/calf
OTHERS = [i for i in range(12) if i not in FL]
POLICY_KP = 40.0
BASE_ARGS = ["--gap", "0.5", "--stand-wait", "2.5", "--settle", "1.2",
             "--mode-timeout", "10", "--policy-timeout", "12"]
SEQ_ARGS = BASE_ARGS + ["--hold", "2.0", "--walk-time", "1.0"]   # timed variants
EXPECTED_VX = {"plate_walk": [0.15, 0.0],            # timed walk, no return leg
               "box_push": [0.15, 0.0],              # same timed walk, pushing
               "stand_fl_up": [],
               "handshake_stand": []}
SCENARIOS = ("happy", "abort", "forever", "end_policy", "sigint")
ABORT_EXIT_WITHIN = 3.0      # [s] sequencer gone after the DAMP that threw it out
ABORT_SLACK = 0.3            # [s] after the abort burst nothing may be commanded
COMPLIANCE_WAIT = 45.0       # [s] stand + POLICY + settle + lift + gaps, worst case


class Probe:
    """Extra subscriptions on the mock node: what the sequencer published
    and how the controller's mode evolved, all wall-clock stamped.

    fsm also records the mock's own rambo/fsm_cmd publishes (a node receives
    its own messages); the scenarios only look at windows where the mock
    stays silent on that topic."""

    def __init__(self, node):
        self.comp, self.modes, self.ee, self.vel, self.fsm = [], [], [], [], []
        self.force = []          # rambo/ee_force_cmd (box_push only)
        node.create_subscription(Bool, "rambo/ee_compliance",
                                 lambda m: self.comp.append((time.time(), bool(m.data))), 10)
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
        for lst in (self.comp, self.modes, self.ee, self.vel, self.fsm):
            lst.clear()

    def compliance_on(self):
        return any(v for _, v in self.comp)


def compress(seq):
    out = []
    for v in seq:
        if not out or out[-1] != v:
            out.append(v)
    return out


def is_zero(v):
    return all(abs(x) < 1e-9 for x in v)


def spawn_controller(kp, kd, extra=()):
    """extra: further controller parameters ("name:=value"), e.g. mock_wam_test's command_handling:=sim."""
    args = ["ros2", "run", "wam_go2_deploy", "controller", "--ros-args",
            "-p", "task:=quadruped",
            "-p", "auto_release_sport_mode:=false",
            "-p", "wired_nic:=none",      # mock robot on loopback: no cable to check
            "-p", "stand_duration:=1.0",
            # the mock robot is a plain Python timer competing with the
            # sequencer / ros2 CLI start-ups on the same machine and can stall
            # for ~0.34 s, past the code default 0.1 s. The real launch uses
            # 0.25 (config/wam_go2.yaml).
            "-p", "lowstate_timeout:=0.5",
            "-p", f"ee_compliant_kp:={kp}",
            "-p", f"ee_compliant_kd:={kd}"]
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


def wait_exit(robot, p, timeout):
    """Spin until the sequencer exits: (returncode, wall time of exit), or
    (None, None) after killing a process that outlived the timeout."""
    if not spin_until(robot, lambda: p.poll() is not None, timeout):
        p.kill()
        p.wait()
        return None, None
    return p.returncode, time.time()


def start_sequencer(args, logf):
    return subprocess.Popen([sys.executable, SEQUENCER, *args], stdout=logf,
                            stderr=subprocess.STDOUT, text=True)


def controller_has_compliance(robot, tries=3):
    """Does the running node declare ee_compliant_kp? (ros2 param get)

    The CLI's discovery can lag a freshly started node ('Node not found'
    although it is up and publishing) - retry a few times before concluding."""
    out = ""
    for attempt in range(tries):
        with tempfile.TemporaryFile(mode="w+") as f:
            rc = run_while_spinning(
                robot, ["ros2", "param", "get", "/wam_go2_controller", "ee_compliant_kp"],
                timeout=20, stdout=f)
            f.seek(0)
            out = f.read().strip()
        if rc == 0 and "value is" in out:
            return True, out
        if "Node not found" not in out:
            break                       # a real answer (e.g. parameter not declared)
        spin_for(robot, 1.0)            # discovery lag: let the CLI daemon catch up
    return False, out


def begin_scenario(label, robot, probe, kp, kd, extra=()):
    """Fresh mock state + a fresh controller node, spun up (policy loaded)."""
    print(f"\n==== {label} ====", flush=True)
    robot.q = DEFAULT_Q_GO2.copy()
    robot.cmds.clear()
    probe.clear()
    proc = spawn_controller(kp, kd, extra)
    spin_for(robot, 6.0)                       # node startup + policy load
    # Start-up hygiene (mock artefact): the node's torch import plus the
    # previous node's shutdown can starve this plain-Python 500 Hz mock long
    # enough to trip the node's LowState watchdog before the demo starts
    # (gaps of ~0.34 s). The sequencer then - correctly - refuses to
    # start (exit 4, "safety latched"). Re-arm here so the scenario tests
    # what it is meant to test; a real Go2 streams LowState on its own CPU.
    spin_until(robot, lambda: bool(probe.modes), 10.0)
    if probe.modes and "(safety:" in probe.modes[-1][1]:
        print(f"  note: safety latched during start-up ({probe.modes[-1][1]!r}); "
              "REARM (mock artefact, not a demo failure)", flush=True)
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


def print_sequencer_output(seq_out):
    print("---- sequencer output ----")
    print("\n".join("  " + ln for ln in seq_out.strip().splitlines()))


def run_demo(demo, robot, probe, kp, kd, check):
    proc = begin_scenario(f"DEMO {demo}", robot, probe, kp, kd)
    summary = {"demo": demo}
    try:
        has_comp, param_out = controller_has_compliance(robot)
        summary["node_has_compliance"] = has_comp

        with tempfile.NamedTemporaryFile(mode="w+", prefix=f"demo_{demo}_",
                                         suffix=".log") as logf:   # printed below, then removed
            t0 = time.time()
            rc = run_while_spinning(
                robot, [sys.executable, SEQUENCER, "--demo", demo, *SEQ_ARGS],
                timeout=90, stdout=logf)
            logf.seek(0)
            seq_out = logf.read()
        spin_for(robot, 1.0)                        # let the final DAMP frames arrive
        summary["sequencer_rc"] = rc
        summary["sequencer_time_s"] = round(time.time() - t0, 1)
        print_sequencer_output(seq_out)
        check(rc == 0, f"{demo}: sequencer exit code {rc} (expected 0)")

        cmds = list(robot.cmds)
        modes = compress([m for _, m in probe.modes])
        summary["modes"] = modes
        check(modes == ["DAMP", "STAND", "POLICY", "DAMP"],
              f"{demo}: rambo/mode sequence {modes} (expected DAMP, STAND, POLICY, DAMP)")

        runs = compress([round(float(c["kp"].max()), 3) for c in cmds])
        summary["kp_runs"] = runs
        check(runs == [0.0, STAND_KP, POLICY_KP, 0.0],
              f"{demo}: /lowcmd kp sequence {runs} (expected [0, {STAND_KP}, {POLICY_KP}, 0])")
        check(all(c["crc_ok"] for c in cmds), f"{demo}: bad CRC in /lowcmd")

        t_policy = next((t for t, m in probe.modes if m == "POLICY"), None)
        t_end = next((t for t, m in probe.modes if t_policy and t > t_policy and m != "POLICY"),
                     time.time())
        pol = [c for c in cmds if t_policy and t_policy < c["t"] < t_end
               and abs(c["kp"].max() - POLICY_KP) < 1e-6]
        summary["policy_cmds"] = len(pol)
        check(len(pol) > 100, f"{demo}: too few POLICY commands ({len(pol)})")

        # ee_pos_cmd applied: FL targets leave the stand pose
        exp_ee = demo_sequence.DEMOS[demo]["ee"]
        check(bool(probe.ee) and np.allclose(probe.ee[0][1], exp_ee, atol=1e-6),
              f"{demo}: ee_pos_cmd {probe.ee[0][1] if probe.ee else None} (expected {exp_ee})")
        if probe.ee:
            t_ee = probe.ee[0][0]
            after = [c for c in pol if c["t"] > t_ee + 0.3]
            dev = max((float(np.abs(c["q"][FL] - DEFAULT_Q_GO2[FL]).max()) for c in after),
                      default=0.0)
            summary["fl_max_dev_rad"] = round(dev, 3)
            check(dev > 0.1, f"{demo}: FL joint targets stayed within {dev:.3f} rad of the "
                             "stand pose after ee_pos_cmd (expected > 0.1)")
            if after:
                q_rambo = after[-1]["q"][RAMBO_TO_GO2]
                foot = kinematics.foot_positions_b(q_rambo)[0]
                summary["fl_foot_b_last"] = [round(float(v), 3) for v in foot]
                # the pre-published target must have overridden the node's
                # POLICY-entry reset (default ee (0.1934, 0.142, 0.05) would
                # put the paw 15-35 cm away from every demo target; the mock
                # stands at the nominal 0.30 m, IK clamps for the folded pose
                # cost < 2 cm)
                target_b = np.array([exp_ee[0], exp_ee[1], exp_ee[2] - 0.30])
                miss = float(np.linalg.norm(foot - target_b))
                summary["fl_target_miss_m"] = round(miss, 3)
                check(miss < 0.08, f"{demo}: FL paw {np.round(foot, 3).tolist()} is {miss:.3f} m "
                                   f"from the commanded target {np.round(target_b, 3).tolist()} "
                                   "(the ee stream did not override the POLICY-entry default?)")
            n_ee = len(probe.ee)
            span = probe.ee[-1][0] - t_ee
            summary["ee_msgs"] = n_ee
            check(span > 0.5 and 10.0 < n_ee / span < 30.0,
                  f"{demo}: ee_pos_cmd rate {n_ee / max(span, 1e-3):.1f} Hz over {span:.1f} s "
                  "(expected ~20 Hz)")

        # velocity commands
        vxs = compress([round(v[0], 3) for _, v in probe.vel])
        summary["cmd_vel_vx"] = vxs
        check(vxs == EXPECTED_VX[demo], f"{demo}: cmd_vel vx sequence {vxs} "
                                        f"(expected {EXPECTED_VX[demo]})")
        if probe.vel and t_policy is not None:
            check(all(t_policy < t < t_end for t, _ in probe.vel),
                  f"{demo}: cmd_vel published outside POLICY")

        # compliance
        ons = [t for t, v in probe.comp if v]
        offs = [t for t, v in probe.comp if not v]
        if not demo.startswith("handshake"):
            check(not ons, f"{demo}: compliance must not be enabled")
            check(all(np.allclose(c["kp"], POLICY_KP) for c in pol),
                  f"{demo}: not every POLICY command has kp 40 on all 12 motors")
        else:
            check(len(ons) == 3 and len([t for t in offs if ons and t > ons[0]]) >= 3,
                  f"{demo}: rambo/ee_compliance True x{len(ons)} / False x{len(offs)} "
                  "(expected 3 each)")
            if not has_comp:
                check(False, f"{demo}: NODE LACKS COMPLIANCE - rebuild: the running "
                             "controller does not declare ee_compliant_kp (ros2 param get "
                             f"-> {param_out!r}); colcon build --packages-select "
                             "wam_go2_deploy in your ROS 2 workspace, then re-run")
            elif ons and offs:
                t_on = ons[0]
                t_off = min(t for t in offs if t > t_on)
                win = [c for c in pol if t_on + 0.4 < c["t"] < t_off]
                post = [c for c in pol if c["t"] > t_off + 0.4]
                summary["compliance_window_cmds"] = len(win)
                summary["post_compliance_cmds"] = len(post)
                check(len(win) > 10, f"{demo}: compliance window too short ({len(win)} cmds)")
                bad_fl = [c for c in win if not np.allclose(c["kp"][FL], kp)]
                check(not bad_fl, f"{demo}: FL kp during compliance = "
                                  f"{bad_fl[0]['kp'][FL].tolist() if bad_fl else None} "
                                  f"(expected ee_compliant_kp {kp}) in {len(bad_fl)}/{len(win)} cmds")
                bad_kd = [c for c in win if not np.allclose(c["kd"][FL], kd)]
                check(not bad_kd, f"{demo}: FL kd during compliance = "
                                  f"{bad_kd[0]['kd'][FL].tolist() if bad_kd else None} "
                                  f"(expected ee_compliant_kd {kd})")
                bad_others = [c for c in win if not np.allclose(c["kp"][OTHERS], POLICY_KP)]
                check(not bad_others, f"{demo}: non-FL kp changed during compliance "
                                      f"({len(bad_others)}/{len(win)} cmds)")
                check(len(post) > 10 and all(np.allclose(c["kp"], POLICY_KP) for c in post),
                      f"{demo}: kp not back to 40 on all 12 motors after compliance "
                      f"({len(post)} cmds)")
    finally:
        end_scenario(proc, summary)
    return summary


def _run_until_compliance(robot, probe, p, check, tag):
    """Spin until the sequencer has switched compliance ON (or died). Returns
    True when the trigger point was reached."""
    reached = spin_until(robot, lambda: probe.compliance_on() or p.poll() is not None,
                         COMPLIANCE_WAIT)
    if p.poll() is not None:
        check(False, f"{tag}: sequencer exited (rc {p.returncode}) before compliance ON")
        return False
    if not reached:
        check(False, f"{tag}: compliance never switched ON within {COMPLIANCE_WAIT:.0f} s")
        p.kill()
        p.wait()
        return False
    return True


def run_abort(robot, probe, kp, kd, check):
    """Controller leaves POLICY behind the sequencer's back (DAMP request /
    safety trip / watchdog - here a DAMP on rambo/fsm_cmd from the mock) -> exit 2,
    zero velocity + compliance OFF, no more commands, gone within 3 s."""
    tag = "abort"
    proc = begin_scenario("SCENARIO abort (DAMP while compliant)", robot, probe, kp, kd)
    summary = {"scenario": tag}
    try:
        with tempfile.NamedTemporaryFile(mode="w+", prefix="demo_abort_", suffix=".log") as logf:
            p = start_sequencer(["--demo", "handshake_stand", "--hold", "30", *BASE_ARGS], logf)
            t_damp = None
            if _run_until_compliance(robot, probe, p, check, tag):
                spin_for(robot, 0.3)             # a few compliant frames on the wire
                t_damp = time.time()
                robot.request("DAMP")            # = operator stop / safety trip
                rc, t_exit = wait_exit(robot, p, 45.0)
            else:
                rc, t_exit = p.returncode, None
            spin_for(robot, 1.0)                 # anything published after the abort?
            logf.seek(0)
            seq_out = logf.read()
        print_sequencer_output(seq_out)
        summary["sequencer_rc"] = rc
        check(rc == 2, f"{tag}: sequencer exit code {rc} (expected 2: controller left POLICY)")
        if t_damp is None:
            return summary

        summary["exit_after_damp_s"] = round(t_exit - t_damp, 2) if t_exit else None
        check(t_exit is not None and t_exit - t_damp <= ABORT_EXIT_WITHIN,
              f"{tag}: sequencer still running {summary['exit_after_damp_s']} s after the "
              f"DAMP (expected exit within {ABORT_EXIT_WITHIN} s)")

        modes = compress([m for _, m in probe.modes])
        summary["modes"] = modes
        check(modes == ["DAMP", "STAND", "POLICY", "DAMP"],
              f"{tag}: rambo/mode sequence {modes} (expected DAMP, STAND, POLICY, DAMP)")
        t_policy = next((t for t, m in probe.modes if m == "POLICY"), None)
        t_change = next((t for t, m in probe.modes
                         if t_policy and t > t_policy and m != "POLICY"), None)
        summary["mode_change_after_damp_s"] = (
            round(t_change - t_damp, 3) if t_change else None)
        check(t_change is not None and t_change >= t_damp,
              f"{tag}: no mode change out of POLICY after the DAMP request")

        # Everything the abort publishes must come AFTER the mode change. A
        # handshake_stand run publishes no cmd_vel and only True Bools before
        # it, so "after t_damp" (which precedes the mode change) counts exactly
        # the abort reaction without racing the probe's own callback order.
        pre_vel = [v for t, v in probe.vel if t <= t_damp]
        pre_false = [t for t, v in probe.comp if t <= t_damp and not v]
        check(not pre_vel and not pre_false,
              f"{tag}: {len(pre_vel)} cmd_vel / {len(pre_false)} False Bools before the "
              "DAMP - the abort reaction is not separable")
        zeros = [t for t, v in probe.vel if t > t_damp and is_zero(v)]
        falses = [t for t, v in probe.comp if t > t_damp and not v]
        summary["zero_twists_after"] = len(zeros)
        summary["false_bools_after"] = len(falses)
        check(len(zeros) >= 5, f"{tag}: {len(zeros)} zero Twists after the mode change "
                               "(expected >= 5)")
        check(len(falses) >= 3, f"{tag}: {len(falses)} False Bools after the mode change "
                                "(expected >= 3)")
        if t_change is not None:
            check(all(t > t_change for t in zeros + falses),
                  f"{tag}: abort burst timestamped before the rambo/mode change")

        # after the abort burst: no FL targets, no non-zero velocity
        t_abort = min(zeros) if zeros else (t_exit or time.time())
        late_ee = [t for t, _ in probe.ee if t > t_abort + ABORT_SLACK]
        late_vel = [v for t, v in probe.vel if t > t_abort + ABORT_SLACK and not is_zero(v)]
        summary["ee_msgs_before_abort"] = len([t for t, _ in probe.ee if t <= t_abort])
        summary["ee_msgs_after_abort"] = len(late_ee)
        check(summary["ee_msgs_before_abort"] > 5,
              f"{tag}: ee_pos_cmd was not being published before the abort (setup)")
        check(not late_ee, f"{tag}: {len(late_ee)} ee_pos_cmd published > {ABORT_SLACK} s "
                           "after the abort (publishing must stop)")
        check(not late_vel, f"{tag}: non-zero cmd_vel {late_vel[:2]} published after the abort")

        # the node itself survives the abort (only the sequencer stops)
        check(proc.poll() is None, f"{tag}: the controller node died")
        tail = [c for c in robot.cmds if c["t"] > t_damp + 0.5]
        check(bool(tail) and all(c["kp"].max() == 0.0 for c in tail),
              f"{tag}: /lowcmd not damped (kp 0) after the DAMP request")
    finally:
        end_scenario(proc, summary)
    return summary


def run_forever(robot, probe, kp, kd, check):
    """plate_walk with the default open-ended walk: a DAMP request on
    rambo/fsm_cmd (here from the mock) ENDS the demo -> exit 0, zero
    velocity published, no more commands, gone within 3 s."""
    tag = "forever"
    proc = begin_scenario("SCENARIO forever (walk until DAMP)", robot, probe, kp, kd)
    summary = {"scenario": tag}
    try:
        with tempfile.NamedTemporaryFile(mode="w+", prefix="demo_forever_", suffix=".log") as logf:
            p = start_sequencer(["--demo", "plate_walk", *BASE_ARGS], logf)   # walk-time 0
            walking = spin_until(
                robot, lambda: any(not is_zero(v) for _, v in probe.vel) or p.poll() is not None,
                COMPLIANCE_WAIT)
            t_damp = None
            if p.poll() is not None:
                check(False, f"{tag}: sequencer exited (rc {p.returncode}) before walking")
                rc, t_exit = p.returncode, None
            elif not walking:
                check(False, f"{tag}: no non-zero cmd_vel within {COMPLIANCE_WAIT:.0f} s")
                p.kill(); p.wait(); rc, t_exit = None, None
            else:
                spin_for(robot, 1.5)             # walking "for a while"
                still = [v for t, v in probe.vel if t > time.time() - 0.3]
                check(bool(still) and all(not is_zero(v) for v in still),
                      f"{tag}: velocity not held while walking ({still[-3:]})")
                t_damp = time.time()
                robot.request("DAMP")            # = operator stop
                rc, t_exit = wait_exit(robot, p, 45.0)
            spin_for(robot, 1.0)
            logf.seek(0)
            seq_out = logf.read()
        print_sequencer_output(seq_out)
        summary["sequencer_rc"] = rc
        check(rc == 0, f"{tag}: sequencer exit code {rc} (expected 0: a DAMP request ends the "
                       "open-ended walk)")
        if t_damp is None:
            return summary
        summary["exit_after_damp_s"] = round(t_exit - t_damp, 2) if t_exit else None
        check(t_exit is not None and t_exit - t_damp <= ABORT_EXIT_WITHIN,
              f"{tag}: sequencer still running {summary['exit_after_damp_s']} s after the DAMP")
        modes = compress([m for _, m in probe.modes])
        summary["modes"] = modes
        check(modes == ["DAMP", "STAND", "POLICY", "DAMP"],
              f"{tag}: rambo/mode sequence {modes} (expected DAMP, STAND, POLICY, DAMP)")
        zeros = [t for t, v in probe.vel if t > t_damp and is_zero(v)]
        summary["zero_twists_after"] = len(zeros)
        check(len(zeros) >= 5, f"{tag}: {len(zeros)} zero Twists after the DAMP (expected >= 5)")
        t_stop = min(zeros) if zeros else (t_exit or time.time())
        late_ee = [t for t, _ in probe.ee if t > t_stop + ABORT_SLACK]
        late_vel = [v for t, v in probe.vel if t > t_stop + ABORT_SLACK and not is_zero(v)]
        summary["ee_msgs_after_stop"] = len(late_ee)
        check(not late_ee, f"{tag}: {len(late_ee)} ee_pos_cmd published after the stop")
        check(not late_vel, f"{tag}: non-zero cmd_vel {late_vel[:2]} published after the stop")
        check(proc.poll() is None, f"{tag}: the controller node died")
        tail = [c for c in robot.cmds if c["t"] > t_damp + 0.5]
        check(bool(tail) and all(c["kp"].max() == 0.0 for c in tail),
              f"{tag}: /lowcmd not damped (kp 0) after the DAMP request")
    finally:
        end_scenario(proc, summary)
    return summary


def run_end_policy(robot, probe, kp, kd, check):
    """--end policy: the sequencer finishes with exit 0 and leaves the
    controller running in POLICY (ended later by a DAMP request)."""
    tag = "end_policy"
    proc = begin_scenario("SCENARIO end_policy (plate_walk --end policy)", robot, probe, kp, kd)
    summary = {"scenario": tag}
    try:
        with tempfile.NamedTemporaryFile(mode="w+", prefix="demo_end_policy_",
                                         suffix=".log") as logf:
            t0 = time.time()
            rc = run_while_spinning(
                robot, [sys.executable, SEQUENCER, "--demo", "plate_walk", "--end", "policy",
                        "--walk-time", "1.0", *BASE_ARGS],
                timeout=90, stdout=logf)
            t_exit = time.time()
            logf.seek(0)
            seq_out = logf.read()
        print_sequencer_output(seq_out)
        summary["sequencer_rc"] = rc
        summary["sequencer_time_s"] = round(t_exit - t0, 1)
        check(rc == 0, f"{tag}: sequencer exit code {rc} (expected 0)")

        alone = 1.5                              # node must keep running POLICY alone
        spin_for(robot, alone)
        modes = compress([m for _, m in probe.modes])
        summary["modes_at_exit"] = modes
        check(modes == ["DAMP", "STAND", "POLICY"],
              f"{tag}: rambo/mode sequence {modes} (expected DAMP, STAND, POLICY - no DAMP)")
        check(bool(probe.modes) and probe.modes[-1][1] == "POLICY",
              f"{tag}: last rambo/mode {probe.modes[-1][1] if probe.modes else None} "
              "(expected POLICY)")
        check(proc.poll() is None, f"{tag}: the controller node exited with the sequencer")
        after = [c for c in robot.cmds if c["t"] > t_exit]
        summary["policy_cmds_after_exit"] = len(after)
        summary["lowcmd_hz_after_exit"] = round(len(after) / alone, 1)
        # liveness, not rate: the node's 100 Hz timer skips ticks when the
        # machine is loaded (a policy step + QP per tick; ~25 Hz under a
        # concurrent colcon build) - the rate itself is mock_robot_test's job
        check(len(after) >= 20, f"{tag}: only {len(after)} /lowcmd frames in the {alone} s "
                                "after the sequencer exited (the node must keep running)")
        check(all(abs(c["kp"].max() - POLICY_KP) < 1e-6 for c in after),
              f"{tag}: /lowcmd kp {sorted(set(round(float(c['kp'].max()), 1) for c in after))} "
              "after the sequencer exited (expected only 40: the node must stay in POLICY)")
        check(not [t for t, c in probe.fsm if c == "DAMP"],
              f"{tag}: a DAMP was published on rambo/fsm_cmd (--end policy must not damp)")
        vxs = compress([round(v[0], 3) for _, v in probe.vel])
        summary["cmd_vel_vx"] = vxs
        check(vxs == EXPECTED_VX["plate_walk"],
              f"{tag}: cmd_vel vx sequence {vxs} (expected {EXPECTED_VX['plate_walk']})")

        # the test plays the operator's DAMP request
        robot.request("DAMP")
        spin_for(robot, 1.0)
        summary["modes_after_test_damp"] = compress([m for _, m in probe.modes])
        check(bool(probe.modes) and probe.modes[-1][1] == "DAMP",
              f"{tag}: node did not DAMP on the test's request "
              f"(mode {probe.modes[-1][1] if probe.modes else None})")
        check(bool(robot.cmds) and robot.cmds[-1]["kp"].max() == 0.0,
              f"{tag}: /lowcmd not damped after the test's DAMP")
    finally:
        end_scenario(proc, summary)
    return summary


def run_sigint(robot, probe, kp, kd, check):
    """Ctrl+C in the handshake window: exit 130, zero velocity + compliance
    OFF + DAMP published; the controller damps."""
    tag = "sigint"
    proc = begin_scenario("SCENARIO sigint (Ctrl+C while compliant)", robot, probe, kp, kd)
    summary = {"scenario": tag}
    try:
        with tempfile.NamedTemporaryFile(mode="w+", prefix="demo_sigint_", suffix=".log") as logf:
            p = start_sequencer(["--demo", "handshake_stand", "--hold", "30", *BASE_ARGS], logf)
            t_sig = None
            if _run_until_compliance(robot, probe, p, check, tag):
                spin_for(robot, 0.3)
                t_sig = time.time()
                p.send_signal(signal.SIGINT)     # the sequencer only, like Ctrl+C
                rc, t_exit = wait_exit(robot, p, 45.0)
            else:
                rc, t_exit = p.returncode, None
            spin_for(robot, 1.0)
            logf.seek(0)
            seq_out = logf.read()
        print_sequencer_output(seq_out)
        summary["sequencer_rc"] = rc
        check(rc == 130, f"{tag}: sequencer exit code {rc} (expected 130 after SIGINT)")
        if t_sig is None:
            return summary

        summary["exit_after_sigint_s"] = round(t_exit - t_sig, 2) if t_exit else None
        check(t_exit is not None and t_exit - t_sig <= ABORT_EXIT_WITHIN,
              f"{tag}: sequencer still running {summary['exit_after_sigint_s']} s after "
              f"SIGINT (expected exit within {ABORT_EXIT_WITHIN} s)")
        damps = [t for t, c in probe.fsm if t > t_sig and c == "DAMP"]
        summary["damp_cmds_after_sigint"] = len(damps)
        check(bool(damps), f"{tag}: no DAMP observed on rambo/fsm_cmd after SIGINT "
                           f"(fsm_cmd seen: {[c for _, c in probe.fsm]})")
        zeros = [t for t, v in probe.vel if t > t_sig and is_zero(v)]
        falses = [t for t, v in probe.comp if t > t_sig and not v]
        summary["zero_twists_after"] = len(zeros)
        summary["false_bools_after"] = len(falses)
        check(len(zeros) >= 5, f"{tag}: {len(zeros)} zero Twists after SIGINT (expected >= 5)")
        check(len(falses) >= 3, f"{tag}: {len(falses)} False Bools after SIGINT (expected >= 3)")
        t_abort = min(damps) if damps else t_sig
        late_ee = [t for t, _ in probe.ee if t > t_abort + ABORT_SLACK]
        check(not late_ee, f"{tag}: {len(late_ee)} ee_pos_cmd published > {ABORT_SLACK} s "
                           "after the DAMP was sent")

        modes = compress([m for _, m in probe.modes])
        summary["modes"] = modes
        check(modes == ["DAMP", "STAND", "POLICY", "DAMP"],
              f"{tag}: rambo/mode sequence {modes} (expected DAMP, STAND, POLICY, DAMP: "
              "the sequencer's DAMP must reach the node)")
        check(proc.poll() is None, f"{tag}: the controller node died")
        check(bool(robot.cmds) and robot.cmds[-1]["kp"].max() == 0.0,
              f"{tag}: /lowcmd not damped after the sequencer's DAMP")
    finally:
        end_scenario(proc, summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS,
                    help="which scenarios to run (default: all)")
    ap.add_argument("--demos", nargs="+",
                    default=["box_push", "handshake_stand", "plate_walk", "stand_fl_up"],
                    choices=sorted(demo_sequence.DEMOS),
                    help="demos for the happy scenario")
    ap.add_argument("--compliant-kp", type=float, default=10.0,
                    help="ee_compliant_kp passed to the node (default 10)")
    ap.add_argument("--compliant-kd", type=float, default=0.5,
                    help="ee_compliant_kd passed to the node (default 0.5)")
    args = ap.parse_args()

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
            print("  FAIL:", msg, flush=True)

    rclpy.init()
    robot = MockRobot()
    probe = Probe(robot)
    kp, kd = args.compliant_kp, args.compliant_kd
    try:
        for scenario in SCENARIOS:              # fixed order, whatever the CLI order
            if scenario not in args.scenarios:
                continue
            if scenario == "happy":
                for demo in args.demos:
                    run_demo(demo, robot, probe, kp, kd, check)
            elif scenario == "abort":
                run_abort(robot, probe, kp, kd, check)
            elif scenario == "forever":
                run_forever(robot, probe, kp, kd, check)
            elif scenario == "end_policy":
                run_end_policy(robot, probe, kp, kd, check)
            elif scenario == "sigint":
                run_sigint(robot, probe, kp, kd, check)
    finally:
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
