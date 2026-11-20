#!/usr/bin/env python3
"""Entry sequence of a WAM-Policy run on the RAMBO Go2 controller.

STAND -> FL to the entry target -> POLICY -> settle, then exit 0 and leave
POLICY running for the WAM-Policy bridge (wam_go2/nodes/wam_bridge_node.py).
Drives the running controller purely over ROS 2 topics (rambo/fsm_cmd,
rambo/ee_pos_cmd) and watches rambo/mode. Started by scripts/demo/wam_task.sh
after its bring-up.

Usage (ROS 2 environment sourced, controller already running):

    python3 scripts/demo_sequence.py --demo wam_entry --ee 0.1934 0.142 0.05
    python3 scripts/demo_sequence.py --demo wam_entry --print-plan      # no ROS

Steps: wait for rambo/mode -> STAND (if in DAMP) -> start publishing the FL
target (20 Hz) -> POLICY (re-sent every 0.5 s until entered; the node frees FL
1 s later and it lifts straight to the target) -> settle -> hand over.
Consecutive actions are separated by --gap seconds.

Exit codes:
    0    POLICY entered, handed over to the bridge
    1    unexpected error (best effort: zero commands, DAMP)
    2    the controller left STAND/POLICY on its own (a DAMP request, a safety
         trip, the tracking watchdog): zero commands were published, then we
         stopped commanding
    3    rambo/mode went silent > 2.5 s (controller died)
    4    could not start (no controller / no /lowstate, safety latched,
         STAND or POLICY not entered in time, unsupported initial mode)
    64   bad arguments
    130  Ctrl+C / SIGTERM / SIGHUP: zero commands and DAMP sent

build_plan() is pure (no ROS) and unit-tested in test/test_demo_plan.py.
"""

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from wam_go2.constants import (  # noqa: E402  (numpy only, no ROS)
    FL_POS_X,
    FL_POS_Y,
    FL_POS_Z,
)

# ROS is only needed by the executor; the plan builder must import without it
try:
    import rclpy
    from geometry_msgs.msg import Point, Twist, Vector3
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import String
    HAVE_ROS = True
except ImportError:  # unit tests / --print-plan without a sourced ROS env
    HAVE_ROS = False
    Node = object

DAMP = "DAMP"
STAND = "STAND"
POLICY = "POLICY"

# RAMBO's default command: every training episode of the WAM-Policy tasks starts from it
FL_ENTRY = (0.1934, 0.142, 0.05)
DEMOS = {
    "wam_entry": dict(
        ee=FL_ENTRY,
        en="enter POLICY with FL at the entry target and hand over to the WAM-Policy bridge"),
}
END_POLICY_HOLD = 1.0      # [s] keep the last commands published before handing over
STAND_CONFIRM_TIMEOUT = 3.0  # [s] rambo/mode must say STAND after the request
POLICY_REQUEST_PERIOD = 0.5
FL_FREE_AFTER = 1.0        # [s] node: all-stance first second of POLICY

STEP_KINDS = ("fsm", "ee", "wait_mode", "sleep", "log")


@dataclass(frozen=True)
class Step:
    """One plan entry. `kind` selects which of the optional fields matter."""
    kind: str                     # fsm | wait_mode | ee | sleep | log
    label: str                    # one line
    cmd: str = ""                 # fsm: request; wait_mode: re-sent every `repeat` s
    mode: Optional[str] = None    # wait_mode: target mode ("" = first message of any mode)
    timeout: float = 0.0          # wait_mode [s]
    repeat: float = 0.0           # wait_mode: re-send period of cmd [s] (0 = never)
    value: Any = None             # ee: (x,y,z)|None
    seconds: float = 0.0          # sleep [s]
    when: Optional[str] = None    # "initial_damp": run only if the first mode seen was DAMP
    fail_code: int = 4            # wait_mode: exit code on timeout


# ---------------------------------------------------------------------------
# CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", choices=sorted(DEMOS), default="wam_entry")
    ap.add_argument("--gap", type=float, default=3.0,
                    help="pause between consecutive actions [s] (default 3.0)")
    ap.add_argument("--ee", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="FL foot target in the projected-CoM frame [m] (default %s; "
                         "x %.4g-%.4g, y %g-%g, z %g-%g)"
                         % ((FL_ENTRY,) + FL_POS_X + FL_POS_Y + FL_POS_Z))
    ap.add_argument("--stand-wait", type=float, default=4.0,
                    help="wait after requesting STAND before POLICY [s] (default 4.0: "
                         "1.5 s gather + 2.0 s rise + margin)")
    ap.add_argument("--policy-timeout", type=float, default=12.0,
                    help="give up if POLICY is not entered within [s] (default 12)")
    ap.add_argument("--mode-timeout", type=float, default=15.0,
                    help="give up if no rambo/mode arrives within [s] (default 15)")
    ap.add_argument("--settle", type=float, default=1.5,
                    help="seconds after POLICY entry before handing over (>= %.1f: the "
                         "node keeps all four feet planted that long)" % FL_FREE_AFTER)
    ap.add_argument("--result-file", help="save exit code, controller mode and reason as JSON")
    ap.add_argument("--print-plan", action="store_true",
                    help="print the resolved plan and exit (no ROS needed)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# plan builder (pure)
def _check(cond, msg):
    if not cond:
        raise ValueError(msg)


def resolve_ee(args):
    """FL target: --ee if given, else the entry target."""
    ee = args.ee if getattr(args, "ee", None) is not None else DEMOS[args.demo]["ee"]
    return tuple(float(v) for v in ee)


def _validate(args, ee):
    _check(args.demo in DEMOS, f"unknown sequence {args.demo!r} (choices: {sorted(DEMOS)})")
    _check(args.settle >= FL_FREE_AFTER,
           f"--settle must be >= {FL_FREE_AFTER} s: the node keeps all four feet "
           "planted for the first second of POLICY (ee commands are only applied after)")
    _check(len(ee) == 3, "--ee needs X Y Z")
    for name, v, (lo, hi) in zip("xyz", ee, (FL_POS_X, FL_POS_Y, FL_POS_Z)):
        _check(lo <= v <= hi, f"--ee {name}={v} outside the FL range [{lo}, {hi}]")
    _check(args.gap >= 0.0, "--gap must be >= 0")
    _check(args.stand_wait >= 0.0, "--stand-wait must be >= 0")
    _check(args.policy_timeout > 0.0, "--policy-timeout must be > 0")
    _check(args.mode_timeout > 0.0, "--mode-timeout must be > 0")


def _gap(gap, when=None):
    return Step("sleep", f"pause {gap:.1f} s", seconds=gap, when=when)


def build_plan(args) -> List[Step]:
    """Ordered steps of the entry sequence. Pure: no ROS, no side effects.
    Raises ValueError on inconsistent parameters."""
    ee = resolve_ee(args)
    _validate(args, ee)
    gap = float(args.gap)
    x, y, z = ee
    return [
        Step("log", f"{args.demo}: {DEMOS[args.demo]['en']}"),
        Step("wait_mode", "wait for the controller (rambo/mode)",
             mode="", timeout=float(args.mode_timeout)),
        # only when the robot is lying in DAMP; skipped when already standing
        Step("fsm", "robot in DAMP: request STAND",
             cmd=STAND, when="initial_damp"),
        # re-sent every 0.5 s: a request published before the node's
        # subscription has matched our publisher is silently dropped by DDS
        Step("wait_mode", f"confirm STAND (re-sent every {POLICY_REQUEST_PERIOD:.1f} s)",
             mode=STAND, cmd=STAND,
             repeat=POLICY_REQUEST_PERIOD, timeout=STAND_CONFIRM_TIMEOUT, fail_code=4,
             when="initial_damp"),
        Step("sleep", f"wait {args.stand_wait:.1f} s for the stand-up (gather 1.5 s + rise 2.0 s)",
             seconds=float(args.stand_wait), when="initial_damp"),
        _gap(gap, when="initial_damp"),
        # The FL target is published BEFORE the POLICY request (20 Hz, kept
        # up for the whole run): the node resets the ee command on POLICY
        # entry and the next message restores it within 50 ms, so when FL is
        # freed 1 s later it lifts straight into the target - one FL motion
        # instead of default-low-then-lift.
        Step("ee", f"pre-publish the FL target ({x:.3f}, {y:.3f}, {z:.3f}) m at 20 Hz; FL lifts "
             "straight to it 1 s into POLICY", value=ee),
        Step("wait_mode", f"request POLICY (re-sent every {POLICY_REQUEST_PERIOD:.1f} s until "
             "entered)", mode=POLICY, cmd=POLICY,
             repeat=POLICY_REQUEST_PERIOD, timeout=float(args.policy_timeout)),
        Step("sleep", f"settle {args.settle:.1f} s in POLICY; FL is freed after 1 s and lifts "
             "to the target", seconds=float(args.settle)),
        _gap(gap),
        Step("log", "hand-over: POLICY keeps running, the WAM-Policy bridge (wam_bridge) "
             "commands from here"),
        Step("sleep", f"keep the last commands published for {END_POLICY_HOLD:.0f} s, "
             "then stop", seconds=END_POLICY_HOLD),
    ]


def plan_duration(plan, initial_damp=True):
    """Rough wall time of the sleeps in the plan [s] (waits not counted)."""
    return sum(s.seconds for s in plan if s.kind == "sleep"
               and (initial_damp or s.when != "initial_damp"))


# ---------------------------------------------------------------------------
# console
def say(msg):
    """Log one line; never raises (a closed terminal or log pipe must not skip a stop command)."""
    t = time.time()
    stamp = time.strftime("%H:%M:%S", time.localtime(t)) + ".%03d" % (int(t * 1000) % 1000)
    try:
        print(f"[{stamp}] {msg}", flush=True)
    except (OSError, ValueError):
        pass


def describe_step(step):
    extra = ""
    if step.kind == "sleep":
        extra = f"{step.seconds:.1f}s"
    elif step.kind == "fsm":
        extra = step.cmd
    elif step.kind == "wait_mode":
        extra = (step.mode or "any") + (f" (resend {step.cmd})" if step.cmd else "")
    elif step.kind == "ee":
        extra = str(step.value)
    return f"{step.kind:<10} {extra:<28} {step.label}"


def print_params(args, plan):
    ee = resolve_ee(args)
    say("resolved parameters:")
    say(f"  {args.demo}  ee=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}) m  gap={args.gap:.1f} s  "
        f"stand_wait={args.stand_wait:.1f} s  settle={args.settle:.1f} s  "
        f"policy_timeout={args.policy_timeout:.0f} s  mode_timeout={args.mode_timeout:.0f} s")
    say(f"  {len(plan)} steps, ~{plan_duration(plan):.0f} s from DAMP "
        f"(~{plan_duration(plan, initial_damp=False):.0f} s if already standing)")


# ---------------------------------------------------------------------------
# executor
class DemoSequencer(Node):
    """Runs a build_plan() plan against the live controller over topics.

    A 20 Hz timer publishes the current FL target (while set), advances the
    plan, and watches the controller: once STAND or POLICY was reached, any
    fall back (DAMP request, safety trip, tracking watchdog) stops the
    sequence at once (exit 2); a silent rambo/mode means the controller died
    (exit 3).
    """

    TICK = 0.05              # 20 Hz
    PUB_MATCH_TIMEOUT = 2.0  # [s] wait for the node to subscribe before the first command
    # rambo/mode silent -> controller died. 2.5 s: the node saves its 100 Hz
    # POLICY trace synchronously when leaving POLICY (up to ~1.4 s at the
    # 60 000-row cap) before the first DAMP mode message goes out.
    MODE_SILENCE = 2.5

    def __init__(self, plan, args):
        super().__init__("rambo_demo_sequencer")
        self._plan = list(plan)
        self._args = args
        self._mode = None            # latest rambo/mode string
        self._mode_time = None       # monotonic time of the latest message
        self._initial_mode = None    # first mode seen (decides the STAND steps)
        self._first_mode_time = None
        self._reached = None         # STAND / POLICY once the controller got there
        self._leaving = False        # DAMP requested: watchdog off
        self._ee = None              # current FL target, published at 20 Hz
        self._idx = 0
        self._step_start = None
        self._step_state = {}
        self._interrupted = None
        self.result = None           # exit code once finished
        self.result_reason = ""

        self._fsm_pub = self.create_publisher(String, "rambo/fsm_cmd", 10)
        self._ee_pub = self.create_publisher(Point, "rambo/ee_pos_cmd", 10)
        self._vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self._force_pub = self.create_publisher(Vector3, "rambo/ee_force_cmd", 10)
        self.create_subscription(String, "rambo/mode", self._on_mode, 10)
        self._timer = self.create_timer(self.TICK, self._tick)

    # ---- inputs
    def request_interrupt(self, what="Ctrl+C"):
        self._interrupted = what

    def _on_mode(self, msg):
        data = msg.data.strip()
        if data != self._mode:
            say(f"  rambo/mode: {self._mode or '-'} -> {data}")
        self._mode = data
        self._mode_time = time.monotonic()
        if self._initial_mode is None:
            self._initial_mode = data
        if data == POLICY:
            self._reached = POLICY
        elif data == STAND and self._reached is None:
            self._reached = STAND

    # ---- messages
    @staticmethod
    def _point(xyz):
        m = Point()
        m.x, m.y, m.z = (float(v) for v in xyz)
        return m

    def _send_fsm(self, cmd):
        self._fsm_pub.publish(String(data=cmd))

    # ---- 20 Hz
    def _tick(self):
        if self.result is not None:
            return  # finished: nothing is commanded any more
        now = time.monotonic()
        if self._interrupted:
            self.abort(130, f"{self._interrupted}: entry interrupted", damp=True)
            return
        if self._mode_time is not None:
            silent = now - self._mode_time
            if silent > self.MODE_SILENCE:
                self.abort(3, f"rambo/mode silent for {silent:.1f} s - controller died?")
                return
            if self._reached and not self._leaving and self._regressed():
                self.abort(2, f"controller left {self._reached}: {self._mode} "
                              "(DAMP request, safety latch or tracking watchdog)")
                return
        if self._ee is not None:
            self._ee_pub.publish(self._point(self._ee))
        self._advance(now)

    def _regressed(self):
        """The controller fell back from the state the sequence had reached."""
        m = self._mode or ""
        if m.startswith(DAMP) or "(safety:" in m:
            return True
        return self._reached == POLICY and m != POLICY

    def _advance(self, now):
        while self.result is None and self._idx < len(self._plan):
            step = self._plan[self._idx]
            if self._step_start is None:
                if step.when == "initial_damp" and self._initial_mode != DAMP:
                    say(f"step {self._idx + 1:02d}/{len(self._plan):02d} skipped (initial mode "
                        f"{self._initial_mode}, already standing): {step.label}")
                    self._idx += 1
                    continue
                self._step_start = now
                self._step_state = {}
                say(f"step {self._idx + 1:02d}/{len(self._plan):02d} {describe_step(step)}")
                self._begin(step, now)
            if not self._poll(step, now):
                return
            self._idx += 1
            self._step_start = None
        if self.result is None and self._idx >= len(self._plan):
            self.finish()

    def _begin(self, step, now):
        k = step.kind
        if k == "ee":
            self._ee = None if step.value is None else tuple(step.value)
            if self._ee is not None:
                if self._initial_mode == POLICY:
                    # normally the target is pre-published in STAND and FL
                    # lifts to it 1 s into POLICY; a controller that is
                    # already in POLICY moves FL to the new target right now
                    say("  note: the controller is already in POLICY - FL moves to the "
                        "target now")
                self._ee_pub.publish(self._point(self._ee))
        elif k == "wait_mode" and step.cmd:
            self._step_state["next_send"] = now

    def _poll(self, step, now):
        """True when the step is complete."""
        elapsed = now - self._step_start
        k = step.kind
        if k in ("log", "ee"):
            return True
        if k == "fsm":
            if (self._fsm_pub.get_subscription_count() == 0
                    and elapsed < self.PUB_MATCH_TIMEOUT):
                return False  # let discovery match the node's subscription first
            self._send_fsm(step.cmd)
            return True
        if k == "sleep":
            return elapsed >= step.seconds
        if k == "wait_mode":
            return self._poll_wait_mode(step, now, elapsed)
        raise RuntimeError(f"unknown step kind {k!r}")

    def _unmatched_publishers(self):
        pubs = [("rambo/fsm_cmd", self._fsm_pub), ("rambo/ee_pos_cmd", self._ee_pub),
                ("cmd_vel", self._vel_pub), ("rambo/ee_force_cmd", self._force_pub)]
        return [name for name, pub in pubs if pub.get_subscription_count() == 0]

    def _poll_wait_mode(self, step, now, elapsed):
        if step.mode == "":
            # first message of any mode
            if self._mode is None:
                if elapsed >= step.timeout:
                    self.abort(step.fail_code,
                               f"no rambo/mode within {step.timeout:.0f} s: controller not "
                               "running or no /lowstate")
                return False
            if "(safety:" in self._mode:
                self.abort(4, f"safety latched: {self._mode!r} - check the robot, then "
                              "re-arm with ros2 topic pub --once rambo/fsm_cmd "
                              "std_msgs/msg/String \"{data: REARM}\" (not done automatically)")
                return False
            if self._mode not in (DAMP, STAND, POLICY):
                self.abort(4, f"unsupported initial mode {self._mode!r}: the entry sequence "
                              "only supports the quadruped task's DAMP/STAND/POLICY")
                return False
            if self._first_mode_time is None:
                self._first_mode_time = now
            unmatched = self._unmatched_publishers()
            if unmatched and now - self._first_mode_time < self.PUB_MATCH_TIMEOUT:
                return False  # give DDS discovery a moment to match all endpoints
            for name in unmatched:
                say(f"  WARNING: no subscriber on {name}")
            say(f"  controller mode: {self._mode}"
                + ("" if self._mode == DAMP else " (already standing or running: STAND skipped)"))
            return True
        if self._mode == step.mode:
            return True
        if step.cmd and now >= self._step_state.get("next_send", float("inf")):
            self._send_fsm(step.cmd)
            self._step_state["next_send"] = now + max(step.repeat, self.TICK)
        if elapsed >= step.timeout:
            self.abort(step.fail_code,
                       f"timeout waiting for {step.mode} ({step.timeout:.0f} s), mode "
                       f"{self._mode!r}"
                       + (" - stand not complete? increase --stand-wait"
                          if step.mode == POLICY else "")
                       + (" - the STAND request was not taken (controller not in DAMP? "
                          "message lost?)" if step.mode == STAND else ""))
        return False

    # ---- termination
    def abort(self, code, reason, damp=False):
        """Stop commanding: zero velocity and zero FL force (5x each), no more
        FL targets; optionally request DAMP (Ctrl+C). Sets the exit code."""
        if self.result is not None:
            return
        say(f"ABORT (exit {code}): {reason}; rambo/mode = {self._mode!r}")
        self._ee = None
        for _ in range(5):
            self._vel_pub.publish(Twist())
        for _ in range(5):
            self._force_pub.publish(Vector3())
        if damp:
            self._leaving = True
            self._send_fsm(DAMP)
            say("zero commands and DAMP sent")
        else:
            say("zero commands sent, publishing stopped")
        self.result_reason = f"{reason}; rambo/mode = {self._mode!r}"
        self.result = code

    def finish(self):
        if "(safety:" in (self._mode or ""):
            self.abort(2, f"controller fault at the end of the entry: {self._mode}")
            return
        self._ee = None
        self.result_reason = f"entry completed; rambo/mode = {self._mode!r}"
        say("entry complete, POLICY running: the WAM-Policy bridge takes over")
        self.result = 0


# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    try:
        plan = build_plan(args)
    except ValueError as e:
        say(f"bad arguments: {e}")
        return 64
    print_params(args, plan)
    if args.print_plan:
        for i, s in enumerate(plan, 1):
            when = "  [only if initial mode is DAMP]" if s.when == "initial_damp" else ""
            print(f"  {i:02d} {describe_step(s)}{when}")
        return 0
    if not HAVE_ROS:
        say("rclpy not importable - source the ROS 2 environment first")
        return 64

    # own the signals: Ctrl+C must publish zero commands + DAMP, not kill spin
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = DemoSequencer(plan, args)

    def _on_signal(signum, _frame):
        node.request_interrupt(signal.Signals(signum).name)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):   # SIGHUP: terminal/SSH gone
        signal.signal(sig, _on_signal)
    say("safety: Ctrl+C here sends zero commands and DAMP; this program stops commanding "
        "if the controller leaves STAND/POLICY")
    code = 1
    try:
        while rclpy.ok() and node.result is None:
            rclpy.spin_once(node, timeout_sec=0.05)
        code = node.result if node.result is not None else 1
    except Exception as e:             # never leave a command latched behind us
        try:
            node.abort(1, f"unexpected error: {e!r}", damp=True)
        except Exception:
            pass
        code = 1
    finally:
        try:
            end = time.monotonic() + 0.3   # let the last publishes leave
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.05)
        except Exception:
            pass
        if args.result_file:
            with open(args.result_file, "w", encoding="utf-8") as output:
                json.dump(dict(demo=args.demo, exit_code=code, mode=node._mode,
                               reason=node.result_reason), output, ensure_ascii=False, indent=2)
                output.write("\n")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
