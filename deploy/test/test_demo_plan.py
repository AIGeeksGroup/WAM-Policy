"""ROS-free tests for scripts/demo_sequence.py build_plan() (the demo
automation plan; the rclpy executor is covered by
test/integration/mock_demo_test.py against the mock robot)."""

import math
import os
import subprocess
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
sys.path.insert(0, SCRIPTS)

import demo_sequence as ds  # noqa: E402

GAP = 2.5   # distinct from every other default duration in the plan
FL_UP = (0.1934, 0.142, 0.20)
FL_PLATE = (0.48, 0.142, 0.24)
HANDSHAKE = (0.32, 0.142, 0.20)
BOX_PUSH = (0.44, 0.142, 0.16)
ENTRY = (0.1934, 0.142, 0.05)            # RAMBO's default command, the start of every training episode


def plan(*argv, gap=GAP):
    return ds.build_plan(ds.parse_args(["--gap", str(gap), *argv]))


def kinds(p):
    return [s.kind for s in p]


def gap_sleep(seconds=GAP):
    return ("sleep", seconds)


def signature(p):
    """(kind, distinguishing value) per step, for exact sequence checks."""
    out = []
    for s in p:
        if s.kind == "sleep":
            out.append(("sleep", "until_leave" if s.until_leave else round(s.seconds, 6)))
        elif s.kind == "fsm":
            out.append(("fsm", s.cmd))
        elif s.kind == "wait_mode":
            out.append(("wait_mode", s.mode))
        elif s.kind in ("ee", "vel", "compliance"):
            out.append((s.kind, s.value))
        else:
            out.append((s.kind,))
    return out


STAND_PREFIX = [
    ("log",),
    ("wait_mode", ""),
    ("fsm", "STAND"),
    ("wait_mode", "STAND"),      # STAND must be confirmed (re-sent) before the wait
    ("sleep", 4.0),              # --stand-wait
    gap_sleep(),
]


def prefix(ee):
    """Common head of every plan: the FL target goes out BEFORE the POLICY
    request, so FL lifts straight to it when the node frees it 1 s into
    POLICY."""
    return STAND_PREFIX + [
        ("ee", ee),
        ("wait_mode", "POLICY"),
        ("sleep", 1.5),              # --settle
        gap_sleep(),
    ]


PREFIX = prefix(FL_UP)
END_DAMP = [gap_sleep(), ("fsm", "DAMP"), ("wait_mode", "DAMP")]
OPEN_ENDED = ("sleep", "until_leave")


def walk_leg(vx, walk_time=4.0):
    return [("vel", (vx, 0.0, 0.0)), ("sleep", walk_time), ("vel", (0.0, 0.0, 0.0)),
            ("sleep", ds.ZERO_VEL_HOLD)]


# ---------------------------------------------------------------------------
def test_step_kinds_are_known():
    for demo in ds.DEMOS:
        for s in plan("--demo", demo):
            assert s.kind in ds.STEP_KINDS
            assert s.label.strip(), f"{demo}: step without label {s}"


def test_demo_set():
    assert sorted(ds.DEMOS) == ["box_push", "handshake_stand", "plate_walk", "stand_fl_up", "wam_entry"]
    # the stand folds FL up; the plate walk extends the leg forward like the
    # RAMBO paper's plate figure
    assert ds.DEMOS["stand_fl_up"]["ee"] == FL_UP
    assert ds.DEMOS["plate_walk"]["ee"] == FL_PLATE
    assert ds.DEMOS["handshake_stand"]["ee"] == HANDSHAKE
    # box_push braces the paw low and near: at the plate pose det(J_FL) is only
    # 0.0025 and cond 6.6 because det goes to zero when the ee height reaches the
    # base height; at 0.16 m it is 0.0061 / 2.8
    assert ds.DEMOS["box_push"]["ee"] == BOX_PUSH
    assert ds.DEMOS["wam_entry"]["ee"] == ENTRY             # where every training episode starts
    lo, hi = (0.1934, 0.0, 0.0), (0.5, 0.2, 0.4)      # trained command range
    for d in ds.DEMOS.values():
        assert all(l <= v <= h for l, v, h in zip(lo, d["ee"], hi))


def test_stand_fl_up_sequence():
    p = plan("--demo", "stand_fl_up")
    expected = PREFIX + [("sleep", 30.0)] + END_DAMP
    assert signature(p) == expected
    hold = [s for s in p if s.kind == "sleep" and s.seconds == 30.0][0]
    assert "standing on three legs" in hold.note


def test_plate_walk_default_is_open_ended():
    p = plan("--demo", "plate_walk")
    expected = prefix(FL_PLATE) + [("vel", (0.15, 0.0, 0.0)), OPEN_ENDED]
    assert signature(p) == expected
    last = p[-1]
    assert last.until_leave and math.isinf(last.seconds) and "Ctrl+C" in last.note
    # nothing after the open-ended step: no DAMP, no gap
    assert not [s for s in p if s.kind == "fsm" and s.cmd == "DAMP"]
    assert ds.plan_duration(p) == float("inf")


def test_plate_walk_timed_and_return():
    p = plan("--demo", "plate_walk", "--walk-time", "10")
    expected = prefix(FL_PLATE) + walk_leg(0.15, 10.0) + END_DAMP
    assert signature(p) == expected
    p2 = plan("--demo", "plate_walk", "--walk-time", "4", "--return")
    expected2 = (prefix(FL_PLATE) + walk_leg(0.15) + [gap_sleep()]
                 + walk_leg(-0.15) + END_DAMP)
    assert signature(p2) == expected2
    assert [s.value[0] for s in p2 if s.kind == "vel"] == [0.15, 0.0, -0.15, 0.0]


def test_handshake_stand_sequence():
    p = plan("--demo", "handshake_stand")
    expected = (prefix(HANDSHAKE) + [("compliance", True), ("sleep", 20.0), gap_sleep(),
                                     ("compliance", False)]
                + END_DAMP)
    assert signature(p) == expected
    hold = [s for s in p if s.kind == "sleep" and s.seconds == 20.0][0]
    assert "shake" in hold.note
    # the pause before the gains snap back tells the person to let go
    let_go = p[signature(p).index(("compliance", False)) - 1]
    assert let_go.kind == "sleep" and let_go.seconds == GAP and "let go" in let_go.note


@pytest.mark.parametrize("demo", ["stand_fl_up", "handshake_stand"])
def test_hold_zero_is_open_ended(demo):
    p = plan("--demo", demo, "--hold", "0")
    assert signature(p)[-1] == OPEN_ENDED
    assert not [s for s in p if s.kind == "fsm" and s.cmd == "DAMP"]
    if demo == "handshake_stand":
        # compliance stays on until the controller leaves POLICY (the node
        # forces it off on exit)
        assert [s.value for s in p if s.kind == "compliance"] == [True]


@pytest.mark.parametrize("demo", sorted(ds.DEMOS))
def test_actions_are_separated_by_gap_sleeps(demo):
    p = plan("--demo", demo)
    steps = [s for s in p if s.kind != "log"]
    # two commanding steps never follow each other directly
    for a, b in zip(steps, steps[1:]):
        if a.kind in ds.COMMANDING_KINDS and b.kind in ds.COMMANDING_KINDS:
            pytest.fail(f"{demo}: {a.label!r} directly followed by {b.label!r}")
    # every phase start (a walk leg, compliance on/off, final DAMP) is
    # preceded by exactly a --gap pause. The ee step is not a motion: it only
    # starts the 20 Hz target stream before the POLICY request; the FL lift
    # happens 1 s into POLICY and is covered by --settle.
    for prev, s in zip(steps, steps[1:]):
        starts_phase = (
            (s.kind == "vel" and s.value[0] != 0.0)
            or s.kind == "compliance"          # on AND off (gains snap back)
            or (s.kind == "fsm" and s.cmd == "DAMP")
        )
        if starts_phase:
            assert prev.kind == "sleep" and prev.seconds == GAP, \
                f"{demo}: {s.label!r} not preceded by a {GAP}s gap (got {prev})"
    # the gap value is taken from the CLI
    p2 = plan("--demo", demo, gap=0.7)
    assert [s.seconds for s in p2 if s.kind == "sleep" and s.seconds == 0.7]
    assert not [s for s in p2 if s.kind == "sleep" and s.seconds == GAP]


def test_stand_steps_only_from_damp():
    p = plan("--demo", "plate_walk")
    conditional = [s for s in p if s.when == "initial_damp"]
    assert [s.kind for s in conditional] == ["fsm", "wait_mode", "sleep", "sleep"]
    assert conditional[0].cmd == "STAND"
    assert conditional[1].mode == "STAND" and conditional[1].fail_code == 4
    assert conditional[1].timeout == ds.STAND_CONFIRM_TIMEOUT
    assert conditional[1].cmd == "STAND" and conditional[1].repeat == 0.5   # DDS early-publish drop
    assert conditional[2].seconds == 4.0 and conditional[3].seconds == GAP
    assert all(s.when is None for s in p if s not in conditional)


def test_wait_mode_parameters():
    p = plan("--demo", "stand_fl_up", "--mode-timeout", "9", "--policy-timeout", "7")
    first = p[1]
    assert first.kind == "wait_mode" and first.mode == "" and first.timeout == 9.0
    pol = [s for s in p if s.kind == "wait_mode" and s.mode == "POLICY"][0]
    assert pol.cmd == "POLICY" and pol.repeat == 0.5 and pol.timeout == 7.0
    damp = [s for s in p if s.kind == "wait_mode" and s.mode == "DAMP"][0]
    assert damp.fail_code == 5 and damp.timeout > 0
    assert damp.cmd == "DAMP" and damp.repeat == 0.5    # DAMP is re-sent until reported
    fsm_damp = [s for s in p if s.kind == "fsm" and s.cmd == "DAMP"][0]
    assert fsm_damp.expect_leave is True
    assert not [s for s in p if s.kind == "fsm" and s.cmd == "STAND" and s.expect_leave]


def test_end_policy_has_no_damp():
    p = plan("--demo", "handshake_stand", "--end", "policy")
    assert not [s for s in p if s.kind == "fsm" and s.cmd == "DAMP"]
    assert not [s for s in p if s.kind == "wait_mode" and s.mode == "DAMP"]
    assert kinds(p)[-3:] == ["sleep", "log", "sleep"]
    assert p[-3].seconds == GAP
    assert p[-1].seconds == ds.END_POLICY_HOLD
    assert "rambo/fsm_cmd" in p[-2].label
    damp = plan("--demo", "handshake_stand")
    assert [s.cmd for s in damp if s.kind == "fsm"][-1] == "DAMP"


@pytest.mark.parametrize("demo,ee", [
    ("stand_fl_up", FL_UP),
    ("plate_walk", FL_PLATE),
    ("handshake_stand", HANDSHAKE),
    ("box_push", BOX_PUSH),
    ("wam_entry", ENTRY),
])
def test_default_ee_per_demo(demo, ee):
    p = plan("--demo", demo)
    ee_steps = [s for s in p if s.kind == "ee"]
    assert len(ee_steps) == 1
    assert ee_steps[0].value == pytest.approx(ee)


@pytest.mark.parametrize("demo", sorted(ds.DEMOS))
def test_ee_target_is_published_before_the_policy_request(demo):
    """The ee topic is already streaming when POLICY is requested, so FL lifts
    straight to the target when freed at 1 s. No default-low-then-lift step,
    no gap between the two (nothing moves)."""
    p = plan("--demo", demo)
    k = kinds(p)
    i_ee = k.index("ee")
    i_pol = [i for i, s in enumerate(p) if s.kind == "wait_mode" and s.mode == "POLICY"][0]
    assert i_pol == i_ee + 1
    assert p[i_pol].cmd == "POLICY"
    # the ee step is unconditional (also when the robot is already standing)
    assert p[i_ee].when is None
    # after POLICY: only the settle and one gap before the demo body
    assert p[i_pol + 1].kind == "sleep" and "settle" in p[i_pol + 1].label
    assert p[i_pol + 2].kind == "sleep" and p[i_pol + 2].seconds == GAP
    assert p[i_pol + 3].kind in ("sleep", "vel", "compliance", "log")   # the body (wam_entry: a hand-over log)


def test_ee_override_and_walk_parameters():
    p = plan("--demo", "plate_walk", "--ee", "0.25", "0.10", "0.30",
             "--vx", "0.2", "--walk-time", "2.2", "--return", "--settle", "2")
    assert [s.value for s in p if s.kind == "ee"] == [(0.25, 0.10, 0.30)]
    assert [s.value[0] for s in p if s.kind == "vel"] == [0.2, 0.0, -0.2, 0.0]
    assert [s.seconds for s in p if s.kind == "sleep" and s.seconds == 2.2] == [2.2, 2.2]
    settle = [s for s in p if s.kind == "sleep" and "settle" in s.label]
    assert len(settle) == 1 and settle[0].seconds == 2.0
    hs = plan("--demo", "handshake_stand", "--hold", "7")
    assert [s.seconds for s in hs if s.note] == [7.0, GAP]   # hold + let-go pause
    st = plan("--demo", "stand_fl_up", "--hold", "12")
    assert [s.seconds for s in st if s.kind == "sleep" and s.seconds == 12.0] == [12.0]


def test_wz_heading_trim():
    p = plan("--demo", "plate_walk", "--wz", "0.03")
    assert [s.value for s in p if s.kind == "vel"] == [(0.15, 0.0, 0.03)]
    p2 = plan("--demo", "plate_walk", "--walk-time", "4", "--return", "--wz", "0.03")
    assert [s.value for s in p2 if s.kind == "vel"] == [
        (0.15, 0.0, 0.03), (0.0, 0.0, 0.0), (-0.15, 0.0, -0.03), (0.0, 0.0, 0.0)]
    with pytest.raises(ValueError, match="wz"):
        plan("--demo", "plate_walk", "--wz", "0.6")


def test_settle_below_one_second_is_rejected():
    with pytest.raises(ValueError, match="settle"):
        plan("--demo", "plate_walk", "--settle", "0.5")
    with pytest.raises(ValueError, match="settle"):
        plan("--demo", "handshake_stand", "--settle", "0.99")
    plan("--demo", "handshake_stand", "--settle", "1.0")   # boundary accepted


@pytest.mark.parametrize("argv", [
    ["--ee", "0.1", "0.142", "0.12"],      # x below the FL range
    ["--ee", "0.30", "0.25", "0.12"],      # y above
    ["--ee", "0.30", "0.142", "0.5"],      # z above
    ["--vx", "0.6"],                       # above the training range
    ["--vx", "0"],
    ["--vx", "-0.1"],
    ["--walk-time", "-1"],
    ["--hold", "-1"],
    ["--gap", "-1"],
])
def test_out_of_range_parameters_are_rejected(argv):
    with pytest.raises(ValueError):
        plan("--demo", "plate_walk", *argv)


def test_bad_cli_exits():
    with pytest.raises(SystemExit):
        ds.parse_args(["--demo", "nope"])
    with pytest.raises(SystemExit):
        ds.parse_args(["--demo", "handshake_walk"])     # not a demo
    with pytest.raises(SystemExit):
        ds.parse_args([])


def test_plan_duration_counts_sleeps():
    p = plan("--demo", "handshake_stand")
    from_damp = ds.plan_duration(p)
    standing = ds.plan_duration(p, initial_damp=False)
    assert from_damp == pytest.approx(standing + 4.0 + GAP)
    # settle, gap, compliance on, hold, let-go gap, compliance off, final gap
    # (the ee target is pre-published: no gap of its own)
    assert standing == pytest.approx(1.5 + GAP + 0.3 + 20.0 + GAP + 0.3 + GAP)
    assert ds.plan_duration(plan("--demo", "plate_walk")) == float("inf")


def test_builder_importable_without_rclpy():
    """The plan builder must work when only numpy is available."""
    code = (
        "import sys\n"
        "for m in ('rclpy', 'rclpy.node', 'rclpy.signals', 'std_msgs', 'std_msgs.msg',"
        " 'geometry_msgs', 'geometry_msgs.msg'):\n"
        "    sys.modules[m] = None\n"
        f"sys.path.insert(0, {SCRIPTS!r})\n"
        "import demo_sequence as ds\n"
        "assert ds.HAVE_ROS is False\n"
        "p = ds.build_plan(ds.parse_args(['--demo', 'plate_walk']))\n"
        "print(len(p))\n"
        "sys.exit(ds.main(['--demo', 'handshake_stand', '--print-plan']))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "handshake_stand" in r.stdout and "compliance" in r.stdout


# ---------------------------------------------------------------------------
# wam_entry: enter POLICY and hand over to the WAM-Policy bridge
@pytest.mark.parametrize("end", ["damp", "policy"])
def test_wam_entry_leaves_policy_running_whatever_end_says(end):
    p = plan("--demo", "wam_entry", "--end", end)
    assert not [s for s in p if s.kind == "fsm" and s.cmd == "DAMP"]
    assert not [s for s in p if s.kind in ("vel", "compliance")]
    assert [s.kind for s in p[-2:]] == ["log", "sleep"] and p[-1].seconds == ds.END_POLICY_HOLD
    assert "WAM" in p[-2].label
    assert ds.parse_args(["--demo", "wam_entry", "--end", end]).end == "policy"


def test_wam_entry_prefix_is_the_common_bring_up():
    p = plan("--demo", "wam_entry")
    q = plan("--demo", "box_push")
    common = [(s.kind, s.cmd, s.mode) for s in p[1:-2]]
    assert common == [(s.kind, s.cmd, s.mode) for s in q[1:len(common) + 1]]
