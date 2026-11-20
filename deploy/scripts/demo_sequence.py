#!/usr/bin/env python3
"""Scripted sequences for the RAMBO Go2 controller.

Drives the running controller purely over ROS 2 topics (rambo/fsm_cmd,
rambo/ee_pos_cmd, cmd_vel, rambo/ee_compliance, rambo/ee_force_cmd) and
watches rambo/mode. The real-robot task runs use only `--demo wam_entry`,
started by scripts/demo/wam_task.sh after its bring-up. The other demos are
scripted controller checks that can be run by hand against a running
controller.

Usage (ROS 2 environment sourced, controller already running):

    python3 scripts/demo_sequence.py --demo stand_fl_up            # 30 s, then DAMP
    python3 scripts/demo_sequence.py --demo plate_walk             # walk until stopped
    python3 scripts/demo_sequence.py --demo plate_walk --walk-time 10 --vx 0.1
    python3 scripts/demo_sequence.py --demo handshake_stand --hold 30
    python3 scripts/demo_sequence.py --demo plate_walk --print-plan   # no ROS

Demos (quadruped task; FL is RAMBO's manipulation leg):
    stand_fl_up      stand in place on three legs, FL folded up 15-20 cm,
                     ee (0.1934, 0.142, 0.20); hold --hold s (default 30,
                     0 = open-ended), then --end
    plate_walk       FL extended forward like the RAMBO paper's plate figure
                     (thigh forward-down, shin forward-up, paw ~20 cm off the
                     ground: ee (0.48, 0.142, 0.24)); walk straight at --vx,
                     open-ended (default) or for --walk-time s, then stop
                     (--return: walk back too)
    handshake_stand  FL raised; FL joints compliant (soft) for --hold s
                     (default 20, 0 = open-ended): shake the foot
    box_push         FL braced low against a box face, ee (0.44, 0.142, 0.16);
                     walk straight pushing it, open-ended or for --walk-time s
    wam_entry        STAND -> FL to the entry target (--ee, default RAMBO's default
                     command (0.1934, 0.142, 0.05), where every training episode
                     starts) -> POLICY -> settle, then exit 0 and leave POLICY
                     to the WAM-Policy bridge (--end is always policy)

Every demo: wait for rambo/mode -> STAND (if in DAMP) -> start publishing
the FL target (20 Hz) -> POLICY (re-sent every 0.5 s until entered; the node
frees FL 1 s later and it lifts straight to the target) -> settle -> demo
body -> --end (damp: request DAMP and wait for it; policy: leave POLICY
running).
Open-ended steps (an endless walk or push, --hold 0) end when the controller
leaves POLICY: press Ctrl+C in this terminal (the sequencer then sends zero
velocity, compliance off and DAMP) or publish DAMP on rambo/fsm_cmd (a plain
DAMP during an open-ended step is a normal exit 0).
Consecutive actions are separated by --gap seconds.

Exit codes:
    0    finished (also: an open-ended stand / walk / push / handshake ended
         by a DAMP request)
    1    unexpected error (best effort: zero velocity, compliance off, DAMP)
    2    the controller left STAND/POLICY on its own (a DAMP request during a
         timed step, a safety trip, the tracking watchdog): zero velocity +
         compliance off were published, then we stopped commanding
    3    rambo/mode went silent > 2.5 s (controller died)
    4    could not start (no controller / no /lowstate, safety latched,
         STAND or POLICY not entered in time, unsupported initial mode)
    5    the final DAMP was not reported within 5 s (re-sent every 0.5 s)
    64   bad arguments
    130  Ctrl+C / SIGTERM / SIGHUP: zero velocity, compliance off, DAMP sent

build_plan() is pure (no ROS) and unit-tested in test/test_demo_plan.py;
the rclpy executor is exercised end to end by
test/integration/mock_demo_test.py against the mock Go2.
"""

import argparse
import json
import math
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
    VEL_RANGE_X,
)

from wam_go2.control.force_command import push_motion_phases

# ROS is only needed by the executor; the plan builder must import without it
try:
    import rclpy
    from geometry_msgs.msg import Point, Twist, Vector3
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import Bool, String
    HAVE_ROS = True
except ImportError:  # unit tests / --print-plan without a sourced ROS env
    HAVE_ROS = False
    Node = object

DAMP = "DAMP"
STAND = "STAND"
POLICY = "POLICY"

# ---------------------------------------------------------------------------
# nominal demo values
# FL folded up, foot under the hip: the three-leg stand pose (measured on the
# robot: foot 0.15 m off the ground at base height 0.257, calf at the soft
# limit). The IK cannot reach z 0.20 exactly; the controller clamps the calf
# -> fully folded leg.
FL_UP = (0.1934, 0.142, 0.20)
# FL extended forward the way the RAMBO paper carries the plate: thigh
# forward-down, shin forward-up, paw ~29 cm ahead of the hip at about hip
# height (real base 0.26 m: IK thigh -0.67 / calf -1.66 rad, paw 24 cm off the
# ground). x stays 2 cm inside the trained command range [0.1934, 0.5].
# In MuJoCo it walks 10 s with a 0.5 kg plate, paw 20 cm up, and stays standing.
FL_PLATE = (0.48, 0.142, 0.24)
# FL braced low against a box face for box_push. NOT the plate pose: at the
# push pose det(J_FL) is 0.0061 (2.4x the plate pose) and cond(J) 2.8 (vs 6.6),
# because det goes to zero exactly when the commanded ee height equals the base
# height - the plate pose sits 5.5 cm from that surface, this one 13.5 cm. The
# leg still has 158 N of forward authority here, 5x the 30 N command clip.
FL_PUSH = (0.44, 0.142, 0.16)
# RAMBO's default command: every training episode of the WAM-Policy tasks starts from it
FL_ENTRY = (0.1934, 0.142, 0.05)
DEMOS = {
    "stand_fl_up": dict(
        ee=FL_UP, hold=30.0,
        en="stand on three legs with FL folded up"),
    "plate_walk": dict(
        ee=FL_PLATE, hold=0.0,
        en="FL extended forward carrying a plate; walk straight"),
    "handshake_stand": dict(
        ee=(0.32, 0.142, 0.20), hold=20.0,
        en="FL raised and compliant; shake the foot"),
    "box_push": dict(
        ee=FL_PUSH, hold=0.0,
        en="FL braced low against a box; walk straight pushing it"),
    # Entry for the WAM-Policy bridge (scripts/demo/wam_task.sh): STAND, FL to the
    # entry target (--ee, default the push pose), POLICY, settle - then this sequence
    # stops publishing and leaves POLICY running for wam_go2/nodes/wam_bridge_node.py
    # (--end is always policy).
    "wam_entry": dict(
        ee=FL_ENTRY, hold=0.0,
        en="enter POLICY with FL at the entry target and hand over to the WAM-Policy bridge"),
}
ZERO_VEL_HOLD = 0.5        # [s] zeros published after each walk leg (>= 0.5)
END_POLICY_HOLD = 1.0      # [s] keep the last commands published (--end policy)
END_DAMP_TIMEOUT = 5.0     # [s] DAMP must be reported within this after the request
STAND_CONFIRM_TIMEOUT = 3.0  # [s] rambo/mode must say STAND after the request
POLICY_REQUEST_PERIOD = 0.5
FL_FREE_AFTER = 1.0        # [s] node: all-stance first second of POLICY
COMPLIANCE_REPEATS = (0.0, 0.15, 0.30)   # Bool published 3x over 0.3 s
COUNTDOWN_PERIOD = 5.0     # [s] handshake hold: reminder every 5 s

COMMANDING_KINDS = ("fsm", "ee", "vel", "compliance", "ee_force")
STEP_KINDS = COMMANDING_KINDS + ("wait_mode", "sleep", "log")


@dataclass(frozen=True)
class Step:
    """One plan entry. `kind` selects which of the optional fields matter."""
    kind: str                     # fsm | wait_mode | ee | vel | compliance | sleep | log
    label: str                    # one line
    cmd: str = ""                 # fsm: request; wait_mode: re-sent every `repeat` s
    mode: Optional[str] = None    # wait_mode: target mode ("" = first message of any mode)
    timeout: float = 0.0          # wait_mode [s]
    repeat: float = 0.0           # wait_mode: re-send period of cmd [s] (0 = never)
    force: Any = None             # vel: simultaneous force update for push phases
    value: Any = None             # ee: (x,y,z)|None, vel: (vx,vy,wz), compliance: bool
    seconds: float = 0.0          # sleep [s]
    note: str = ""                # sleep: reminder printed every COUNTDOWN_PERIOD
    when: Optional[str] = None    # "initial_damp": run only if the first mode seen was DAMP
    expect_leave: bool = False    # fsm: leaves POLICY on purpose (watchdog disarmed)
    fail_code: int = 4            # wait_mode: exit code on timeout
    until_leave: bool = False     # sleep: open-ended, ends when the controller leaves
                                  # POLICY (a plain DAMP request: exit 0)


# ---------------------------------------------------------------------------
# CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:<16} {v['en']} (ee {v['ee']})"
                         for k, v in DEMOS.items()))
    ap.add_argument("--demo", required=True, choices=sorted(DEMOS))
    ap.add_argument("--gap", type=float, default=3.0,
                    help="pause between consecutive actions [s] (default 3.0)")
    ap.add_argument("--ee", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="FL foot target in the projected-CoM frame [m]; "
                         "default per demo (x %.4g-%.4g, y %g-%g, z %g-%g)"
                         % (FL_POS_X + FL_POS_Y + FL_POS_Z))
    ap.add_argument("--vx", type=float, default=0.15, help="walk speed [m/s] (default 0.15)")
    ap.add_argument("--wz", type=float, default=0.0,
                    help="yaw rate while walking [rad/s] (default 0 = walk straight: the node's "
                         "heading hold then keeps the IMU yaw it had when the walk started, "
                         "cancelling the three-leg gait's right drift of 1-2.4 deg/s). A non-zero "
                         "value (|wz| >= 0.02) is a steering command and switches the hold off for "
                         "as long as it is sent")
    ap.add_argument("--walk-time", type=float, default=0.0,
                    help="plate_walk / box_push: walk this long, then stop [s] (default 0 = keep "
                         "walking until the controller leaves POLICY: Ctrl+C here or a DAMP "
                         "request on rambo/fsm_cmd)")
    ap.add_argument("--return", dest="return_leg", action="store_true",
                    help="plate_walk with --walk-time: walk back afterwards")
    ap.add_argument("--hold", type=float, default=None,
                    help="stand_fl_up: stand this long (default 30); handshake_stand: compliant "
                         "window (default 20); 0 = open-ended (until the controller leaves POLICY)")
    ap.add_argument("--stand-wait", type=float, default=4.0,
                    help="wait after requesting STAND before POLICY [s] (default 4.0: "
                         "1.5 s gather + 2.0 s rise + margin)")
    ap.add_argument("--policy-timeout", type=float, default=12.0,
                    help="give up if POLICY is not entered within [s] (default 12)")
    ap.add_argument("--mode-timeout", type=float, default=15.0,
                    help="give up if no rambo/mode arrives within [s] (default 15)")
    ap.add_argument("--end", choices=("damp", "policy"), default="damp",
                    help="damp: request DAMP at the end (default); policy: leave "
                         "POLICY running (end it later with DAMP on rambo/fsm_cmd)")
    ap.add_argument("--settle", type=float, default=1.5,
                    help="seconds after POLICY entry before lifting FL (>= %.1f: the "
                         "node keeps all four feet planted that long)" % FL_FREE_AFTER)
    ap.add_argument("--result-file", help="save exit code, controller mode and reason as JSON")
    ap.add_argument("--print-plan", action="store_true",
                    help="print the resolved plan and exit (no ROS needed)")
    args = ap.parse_args(argv)
    if args.demo == "wam_entry":
        args.end = "policy"      # the WAM bridge, then the wrapper's cleanup, end the run
    return args


# ---------------------------------------------------------------------------
# plan builder (pure)
def _check(cond, msg):
    if not cond:
        raise ValueError(msg)


def resolve_ee(args):
    """FL target for the demo: --ee if given, else the nominal one."""
    ee = args.ee if getattr(args, "ee", None) is not None else DEMOS[args.demo]["ee"]
    return tuple(float(v) for v in ee)


def resolve_hold(args):
    """Hold window [s]: --hold if given, else the demo's default (0 = open-ended)."""
    hold = getattr(args, "hold", None)
    return float(DEMOS[args.demo]["hold"] if hold is None else hold)


def _validate(args, ee):
    _check(args.demo in DEMOS, f"unknown demo {args.demo!r} (choices: {sorted(DEMOS)})")
    _check(args.settle >= FL_FREE_AFTER,
           f"--settle must be >= {FL_FREE_AFTER} s: the node keeps all four feet "
           "planted for the first second of POLICY (ee commands are only applied after)")
    _check(len(ee) == 3, "--ee needs X Y Z")
    for name, v, (lo, hi) in zip("xyz", ee, (FL_POS_X, FL_POS_Y, FL_POS_Z)):
        _check(lo <= v <= hi, f"--ee {name}={v} outside the FL range [{lo}, {hi}]")
    _check(0.0 < args.vx <= VEL_RANGE_X[1],
           f"--vx must be in (0, {VEL_RANGE_X[1]}] m/s (the return leg uses -vx)")
    _check(abs(args.wz) <= 0.5, "--wz must be within +-0.5 rad/s (training range)")
    _check(args.gap >= 0.0, "--gap must be >= 0")
    _check(args.walk_time >= 0.0, "--walk-time must be >= 0 (0 = open-ended)")
    _check(resolve_hold(args) >= 0.0, "--hold must be >= 0 (0 = open-ended)")
    _check(args.stand_wait >= 0.0, "--stand-wait must be >= 0")
    _check(args.policy_timeout > 0.0, "--policy-timeout must be > 0")
    _check(args.mode_timeout > 0.0, "--mode-timeout must be > 0")
    _check(args.end in ("damp", "policy"), "--end must be damp or policy")


def _gap(gap, when=None):
    return Step("sleep", f"pause {gap:.1f} s", seconds=gap, when=when)


def _walk_leg(vx, walk_time, direction, wz=0.0):
    what = "walk forward" if vx > 0 else "walk back"
    trim = f", wz={wz:+.2f}" if wz else ""
    return [
        Step("vel", f"{what} vx={vx:+.2f} m/s{trim}, {walk_time:.1f} s", value=(vx, 0.0, wz)),
        Step("sleep", f"walking {walk_time:.1f} s", seconds=walk_time),
        Step("vel", "stop: zero velocity", value=(0.0, 0.0, 0.0)),
        Step("sleep", f"hold zero velocity {ZERO_VEL_HOLD:.1f} s", seconds=ZERO_VEL_HOLD),
    ]


def _until_leave(label):
    """Open-ended step: ends when the controller leaves POLICY (Ctrl+C here,
    or a DAMP request on rambo/fsm_cmd)."""
    return Step("sleep", label, seconds=float("inf"), note=label, until_leave=True)


def _let_go(gap):
    """The pause before the FL gains snap back: whoever holds the foot lets go."""
    return Step("sleep", f"pause {gap:.1f} s, let go of the foot",
                seconds=gap,
                note="FL stiffness (kp 40) comes back next - let go of the foot")


def _compliance(on):
    if on:
        return Step("compliance", "FL compliance ON (soft gains, the foot can be held)",
                    value=True)
    return Step("compliance", "FL compliance OFF (gains back to kp 40 / kd 1)", value=False)


def build_plan(args) -> List[Step]:
    """Ordered steps for args.demo. Pure: no ROS, no side effects.
    Raises ValueError on inconsistent parameters."""
    ee = resolve_ee(args)
    _validate(args, ee)
    gap = float(args.gap)
    d = DEMOS[args.demo]
    x, y, z = ee

    plan = [
        Step("log", f"demo {args.demo}: {d['en']}"),
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
    ]

    vx, walk, hold = float(args.vx), float(args.walk_time), resolve_hold(args)
    wz = float(args.wz)
    open_ended = False
    if args.demo == "stand_fl_up":
        if hold > 0.0:
            plan.append(Step("sleep", f"hold the three-leg stand {hold:.0f} s",
                             seconds=hold, note="standing on three legs"))
        else:
            plan.append(_until_leave("standing on three legs - Ctrl+C or DAMP on "
                                     "rambo/fsm_cmd ends it"))
            open_ended = True
    elif args.demo == "plate_walk":
        if walk > 0.0:
            plan += _walk_leg(vx, walk, +1, wz)
            if args.return_leg:
                plan.append(_gap(gap))
                plan += _walk_leg(-vx, walk, -1, -wz)
        else:
            trim = f", wz={wz:+.2f}" if wz else ""
            plan.append(Step("vel", f"walk straight vx={vx:+.2f} m/s{trim}",
                             value=(vx, 0.0, wz)))
            plan.append(_until_leave(f"walking straight at vx={vx:+.2f} m/s - Ctrl+C or DAMP "
                                     "on rambo/fsm_cmd ends it"))
            open_ended = True
    elif args.demo == "box_push":
        # Shared with the simulator: the push uses the leg's position stiffness
        # alone (zero FL force command), with the same stop boundary.
        for phase in push_motion_phases(vx, 0.0, walk, wz=wz):
            moving = phase.velocity[0] != 0.0
            label = (f"push the box vx={vx:+.2f} m/s, {phase.force[0]:g} N"
                     if moving else "stop: zero velocity, release force")
            plan.append(Step("vel", label, value=phase.velocity, force=phase.force))
            if math.isinf(phase.seconds):
                plan.append(_until_leave("pushing - Ctrl+C or DAMP on rambo/fsm_cmd ends it"))
                open_ended = True
            else:
                plan.append(Step("sleep", "pushing" if moving else "hold zero velocity",
                                 seconds=phase.seconds))
    elif args.demo == "handshake_stand":
        plan.append(_compliance(True))
        if hold > 0.0:
            plan += [
                Step("sleep", f"handshake window {hold:.0f} s",
                     seconds=hold, note="shake the FL foot now"),
                _let_go(gap),
                _compliance(False),
            ]
        else:
            plan.append(_until_leave("handshake, FL compliant - Ctrl+C or DAMP on "
                                     "rambo/fsm_cmd ends it"))
            open_ended = True

    if args.demo == "wam_entry":
        plan += [
            Step("log", "hand-over: POLICY keeps running, the WAM-Policy bridge (wam_bridge) "
                 "commands from here"),
            Step("sleep", f"keep the last commands published for {END_POLICY_HOLD:.0f} s, "
                 "then stop", seconds=END_POLICY_HOLD),
        ]
        return plan
    if open_ended:
        return plan     # leaving POLICY is the end; --end does not apply

    plan.append(_gap(gap))
    if args.end == "damp":
        plan += [
            Step("fsm", "end: request DAMP (the robot lies down softly)",
                 cmd=DAMP, expect_leave=True),
            Step("wait_mode", f"wait for DAMP (re-sent every {POLICY_REQUEST_PERIOD:.1f} s)",
                 mode=DAMP, cmd=DAMP,
                 repeat=POLICY_REQUEST_PERIOD, timeout=END_DAMP_TIMEOUT, fail_code=5),
        ]
    else:
        plan += [
            Step("log", "end: POLICY keeps running - end it with DAMP on rambo/fsm_cmd"),
            Step("sleep", f"keep the last commands published for {END_POLICY_HOLD:.0f} s, "
                 "then stop", seconds=END_POLICY_HOLD),
        ]
    return plan


def plan_duration(plan, initial_damp=True):
    """Rough wall time of the sleeps in the plan [s] (waits not counted)."""
    total = 0.0
    for s in plan:
        if s.when == "initial_damp" and not initial_damp:
            continue
        if s.kind == "sleep":
            if s.until_leave:
                return float("inf")
            total += s.seconds
        elif s.kind == "compliance":
            total += COMPLIANCE_REPEATS[-1]
    return total


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
        extra = "open-ended" if step.until_leave else f"{step.seconds:.1f}s"
    elif step.kind == "fsm":
        extra = step.cmd
    elif step.kind == "wait_mode":
        extra = (step.mode or "any") + (f" (resend {step.cmd})" if step.cmd else "")
    elif step.kind == "ee_force":
        extra = f"{step.value[0]:+.1f} N (x)"
    elif step.kind in ("ee", "vel", "compliance"):
        extra = str(step.value)
    return f"{step.kind:<10} {extra:<28} {step.label}"


def print_params(args, plan):
    ee = resolve_ee(args)
    say("resolved parameters:")
    hold = resolve_hold(args)
    walk = "open-ended" if args.walk_time <= 0 else f"{args.walk_time:.1f} s"
    say(f"  demo={args.demo}  ee=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}) m  vx={args.vx:.2f} m/s"
        f"  wz={args.wz:+.2f} rad/s  walk_time={walk}  return={args.return_leg}")
    say(f"  hold={'open-ended' if hold <= 0 else f'{hold:.1f} s'}  gap={args.gap:.1f} s  "
        f"stand_wait={args.stand_wait:.1f} s  settle={args.settle:.1f} s  "
        f"policy_timeout={args.policy_timeout:.0f} s  mode_timeout={args.mode_timeout:.0f} s"
        f"  end={args.end}")
    dur = plan_duration(plan)
    if dur == float("inf"):
        say(f"  {len(plan)} steps, open-ended: Ctrl+C here or DAMP on rambo/fsm_cmd ends it")
    else:
        say(f"  {len(plan)} steps, ~{dur:.0f} s from DAMP "
            f"(~{plan_duration(plan, initial_damp=False):.0f} s if already standing)")


# ---------------------------------------------------------------------------
# executor
class DemoSequencer(Node):
    """Runs a build_plan() plan against the live controller over topics.

    A 20 Hz timer publishes the current FL target (while set) and the current
    velocity (while POLICY is active), re-publishes compliance changes 3x,
    advances the plan, and watches the controller: once POLICY was entered,
    any other mode (DAMP request, safety trip, tracking watchdog) stops the
    demo at once (exit 2, or 0 for a plain DAMP during an open-ended step);
    a silent rambo/mode means the controller died (exit 3).
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
        self._policy_entered = False
        self._reached = None         # STAND / POLICY once the controller got there
        self._leaving = False        # DAMP requested by the plan: watchdog off
        self._ee = None              # current FL target, published at 20 Hz
        self._vel = None             # current (vx, vy, wz), published in POLICY
        self._force = None           # current FL force (fx, fy, fz) [N], published in POLICY
        self._comp_value = False
        self._comp_pending = []      # monotonic times of the remaining Bool publishes
        self._idx = 0
        self._step_start = None
        self._step_state = {}
        self._interrupted = None
        self.result = None           # exit code once finished
        self.result_reason = ""

        self._fsm_pub = self.create_publisher(String, "rambo/fsm_cmd", 10)
        self._ee_pub = self.create_publisher(Point, "rambo/ee_pos_cmd", 10)
        self._vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self._comp_pub = self.create_publisher(Bool, "rambo/ee_compliance", 10)
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
            self._policy_entered = True
            self._reached = POLICY
        elif data == STAND and self._reached is None:
            self._reached = STAND

    # ---- messages
    @staticmethod
    def _point(xyz):
        m = Point()
        m.x, m.y, m.z = (float(v) for v in xyz)
        return m

    @staticmethod
    def _twist(v):
        m = Twist()
        m.linear.x, m.linear.y, m.angular.z = (float(x) for x in v)
        return m

    @staticmethod
    def _vector3(v):
        m = Vector3()
        m.x, m.y, m.z = (float(x) for x in v)
        return m

    def _send_fsm(self, cmd):
        self._fsm_pub.publish(String(data=cmd))

    # ---- 20 Hz
    def _tick(self):
        if self.result is not None:
            return  # finished: nothing is commanded any more
        now = time.monotonic()
        if self._interrupted:
            self.abort(130, f"{self._interrupted}: demo interrupted", damp=True)
            return
        if self._mode_time is not None:
            silent = now - self._mode_time
            if silent > self.MODE_SILENCE:
                self.abort(3, f"rambo/mode silent for {silent:.1f} s - controller died?")
                return
            if self._reached and not self._leaving and self._regressed():
                step = self._plan[self._idx] if self._idx < len(self._plan) else None
                if (step is not None and step.until_leave and self._reached == POLICY
                        and self._mode == DAMP):
                    self.finish_by_operator()
                else:
                    self.abort(2, f"controller left {self._reached}: {self._mode} "
                                  "(DAMP request, safety latch or tracking watchdog)")
                return
        if self._ee is not None:
            self._ee_pub.publish(self._point(self._ee))
        if self._vel is not None and self._mode == POLICY:
            self._vel_pub.publish(self._twist(self._vel))
        if self._force is not None and self._mode == POLICY:
            # must keep streaming: the node zeroes every ee command on POLICY
            # entry (controller_node _to_policy) and, with the force watchdog
            # enabled, decays a stale one to zero
            self._force_pub.publish(self._vector3(self._force))
        while self._comp_pending and self._comp_pending[0] <= now:
            self._comp_pending.pop(0)
            self._comp_pub.publish(Bool(data=self._comp_value))
        self._advance(now)

    def _regressed(self):
        """The controller fell back from the state the demo had reached."""
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
        if k == "fsm":
            if step.expect_leave:
                self._leaving = True
        elif k == "ee":
            self._ee = None if step.value is None else tuple(step.value)
            if self._ee is not None:
                if self._initial_mode == POLICY:
                    # normally the target is pre-published in STAND and FL
                    # lifts to it 1 s into POLICY; a controller that is
                    # already in POLICY (e.g. left running by --end policy)
                    # moves FL to the new target right now
                    say("  note: the controller is already in POLICY - FL moves to the "
                        "target now")
                self._ee_pub.publish(self._point(self._ee))
        elif k == "vel":
            self._vel = tuple(step.value)
            if step.force is not None:
                self._force = tuple(step.force)
                if self._mode == POLICY:
                    self._force_pub.publish(self._vector3(self._force))
            if self._mode == POLICY:
                self._vel_pub.publish(self._twist(self._vel))
        elif k == "ee_force":
            self._force = tuple(step.value)
            if self._force_pub.get_subscription_count() == 0:
                say("  WARNING: no subscriber on rambo/ee_force_cmd - the push force has "
                    "no effect")
            if self._mode == POLICY:
                self._force_pub.publish(self._vector3(self._force))
        elif k == "compliance":
            self._comp_value = bool(step.value)
            self._comp_pending = [now + dt for dt in COMPLIANCE_REPEATS]
            if self._comp_pub.get_subscription_count() == 0:
                say("  WARNING: no subscriber on rambo/ee_compliance - rebuild the "
                    "controller; compliance has no effect")
        elif k == "sleep" and step.note:
            self._step_state["next_note"] = now
        elif k == "wait_mode" and step.cmd:
            self._step_state["next_send"] = now

    def _poll(self, step, now):
        """True when the step is complete."""
        elapsed = now - self._step_start
        k = step.kind
        if k in ("log", "ee", "vel", "ee_force"):
            return True
        if k == "fsm":
            if (self._fsm_pub.get_subscription_count() == 0
                    and elapsed < self.PUB_MATCH_TIMEOUT):
                return False  # let discovery match the node's subscription first
            self._send_fsm(step.cmd)
            return True
        if k == "compliance":
            return not self._comp_pending
        if k == "sleep":
            if step.note and now >= self._step_state.get("next_note", float("inf")):
                remaining = max(0.0, step.seconds - elapsed)
                say(f"  >>> {step.note}  (remaining {remaining:.0f} s)")
                self._step_state["next_note"] = now + COUNTDOWN_PERIOD
            return elapsed >= step.seconds
        if k == "wait_mode":
            return self._poll_wait_mode(step, now, elapsed)
        raise RuntimeError(f"unknown step kind {k!r}")

    def _unmatched_publishers(self):
        pubs = [("rambo/fsm_cmd", self._fsm_pub), ("rambo/ee_pos_cmd", self._ee_pub),
                ("cmd_vel", self._vel_pub), ("rambo/ee_compliance", self._comp_pub),
                ("rambo/ee_force_cmd", self._force_pub)]
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
                self.abort(4, f"unsupported initial mode {self._mode!r}: the demos only "
                              "support the quadruped task's DAMP/STAND/POLICY")
                return False
            if self._first_mode_time is None:
                self._first_mode_time = now
            unmatched = self._unmatched_publishers()
            if unmatched and now - self._first_mode_time < self.PUB_MATCH_TIMEOUT:
                return False  # give DDS discovery a moment to match all endpoints
            for name in unmatched:
                hint = (" - the controller lacks it, rebuild"
                        if name == "rambo/ee_compliance" else "")
                say(f"  WARNING: no subscriber on {name}{hint}")
            say(f"  controller mode: {self._mode}"
                + ("" if self._mode == DAMP else " (already standing or running: STAND skipped)"))
            return True
        if self._mode == step.mode or (step.mode == DAMP and (self._mode or "").startswith(DAMP)):
            return True     # "DAMP (safety: ...)" counts as DAMP for the end step
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
    def _stop_publishing(self):
        self._ee = None
        self._vel = None
        self._force = None
        self._comp_pending = []

    def abort(self, code, reason, damp=False):
        """Stop commanding: zero velocity (5x), compliance off (3x), no more
        FL targets; optionally request DAMP (Ctrl+C). Sets the exit code."""
        if self.result is not None:
            return
        say(f"ABORT (exit {code}): {reason}; rambo/mode = {self._mode!r}")
        self._stop_publishing()
        zero = Twist()
        for _ in range(5):
            self._vel_pub.publish(zero)
        for _ in range(3):
            self._comp_pub.publish(Bool(data=False))
        # the FL force has no staleness timeout in the node unless
        # ee_force_cmd_timeout is set, so simply stopping would leave the last
        # push latched for the rest of the POLICY run
        for _ in range(5):
            self._force_pub.publish(Vector3())
        if damp:
            self._leaving = True
            self._send_fsm(DAMP)
            say("zero velocity, compliance off and DAMP sent")
        else:
            say("zero velocity and compliance off sent, publishing stopped")
        self.result_reason = f"{reason}; rambo/mode = {self._mode!r}"
        self.result = code

    def finish_by_operator(self):
        """Only a plain DAMP request ends an open-ended step successfully."""
        if self.result is not None:
            return
        if self._mode != DAMP:
            self.abort(2, f"controller fault ended open-ended demo: {self._mode}")
            return
        say(f"a DAMP request ended the open-ended demo, rambo/mode = {self._mode!r}")
        self._stop_publishing()
        zero = Twist()
        for _ in range(5):
            self._vel_pub.publish(zero)
        for _ in range(3):
            self._comp_pub.publish(Bool(data=False))
        # the FL force has no staleness timeout in the node unless
        # ee_force_cmd_timeout is set, so simply stopping would leave the last
        # push latched for the rest of the POLICY run
        for _ in range(5):
            self._force_pub.publish(Vector3())
        self.result_reason = f"operator ended open-ended demo; rambo/mode = {self._mode!r}"
        self.result = 0

    def finish(self):
        if "(safety:" in (self._mode or ""):
            self.abort(2, f"controller fault at demo completion: {self._mode}")
            return
        self._stop_publishing()
        self.result_reason = f"scheduled demo completed; rambo/mode = {self._mode!r}"
        if self._args.end == "damp":
            say(f"demo complete, robot damped (rambo/mode = {self._mode!r})")
        else:
            say("demo complete, POLICY still running - end it with DAMP on rambo/fsm_cmd")
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

    # own the signals: Ctrl+C must publish zero velocity + DAMP, not kill spin
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = DemoSequencer(plan, args)

    def _on_signal(signum, _frame):
        node.request_interrupt(signal.Signals(signum).name)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):   # SIGHUP: terminal/SSH gone
        signal.signal(sig, _on_signal)
    say("safety: Ctrl+C here sends zero velocity, compliance off and DAMP; this program "
        "stops commanding if the controller leaves STAND/POLICY")
    code = 1
    try:
        while rclpy.ok() and node.result is None:
            rclpy.spin_once(node, timeout_sec=0.05)
        code = node.result if node.result is not None else 1
    except Exception as e:             # never leave a velocity latched behind us
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
