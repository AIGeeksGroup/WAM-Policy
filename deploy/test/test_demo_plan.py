"""ROS-free tests for scripts/demo_sequence.py build_plan() (the entry
sequence of a WAM-Policy run; the rclpy executor is covered by
test/integration/mock_wam_test.py against the mock robot)."""

import os
import subprocess
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
sys.path.insert(0, SCRIPTS)

import demo_sequence as ds  # noqa: E402

GAP = 2.5   # distinct from every other default duration in the plan
ENTRY = (0.1934, 0.142, 0.05)            # RAMBO's default command, the start of every training episode


def plan(*argv, gap=GAP):
    return ds.build_plan(ds.parse_args(["--gap", str(gap), *argv]))


def signature(p):
    """(kind, distinguishing value) per step, for exact sequence checks."""
    out = []
    for s in p:
        if s.kind == "sleep":
            out.append(("sleep", round(s.seconds, 6)))
        elif s.kind == "fsm":
            out.append(("fsm", s.cmd))
        elif s.kind == "wait_mode":
            out.append(("wait_mode", s.mode))
        elif s.kind == "ee":
            out.append(("ee", s.value))
        else:
            out.append((s.kind,))
    return out


# ---------------------------------------------------------------------------
def test_the_entry_sequence():
    """STAND (from DAMP only), the FL target BEFORE the POLICY request so FL
    lifts straight to it when the node frees it 1 s into POLICY, settle, then
    hand over to the WAM-Policy bridge with POLICY still running."""
    assert sorted(ds.DEMOS) == ["wam_entry"]
    p = plan("--demo", "wam_entry")
    assert signature(p) == [
        ("log",),
        ("wait_mode", ""),
        ("fsm", "STAND"),
        ("wait_mode", "STAND"),      # STAND must be confirmed (re-sent) before the wait
        ("sleep", 4.0),              # --stand-wait
        ("sleep", GAP),
        ("ee", ENTRY),
        ("wait_mode", "POLICY"),
        ("sleep", 1.5),              # --settle
        ("sleep", GAP),
        ("log",),
        ("sleep", ds.END_POLICY_HOLD),
    ]
    assert not [s for s in p if s.kind == "fsm" and s.cmd == "DAMP"]
    assert "WAM" in p[-2].label
    assert all(s.kind in ds.STEP_KINDS and s.label.strip() for s in p)
    assert signature(ds.build_plan(ds.parse_args(["--gap", str(GAP)]))) == signature(p)  # the default


def test_stand_steps_only_from_damp():
    p = plan()
    conditional = [s for s in p if s.when == "initial_damp"]
    assert [s.kind for s in conditional] == ["fsm", "wait_mode", "sleep", "sleep"]
    assert conditional[0].cmd == "STAND"
    assert conditional[1].mode == "STAND" and conditional[1].fail_code == 4
    assert conditional[1].timeout == ds.STAND_CONFIRM_TIMEOUT
    assert conditional[1].cmd == "STAND" and conditional[1].repeat == 0.5   # DDS early-publish drop
    assert conditional[2].seconds == 4.0 and conditional[3].seconds == GAP
    assert all(s.when is None for s in p if s not in conditional)


def test_wait_mode_parameters():
    p = plan("--mode-timeout", "9", "--policy-timeout", "7")
    first = p[1]
    assert first.kind == "wait_mode" and first.mode == "" and first.timeout == 9.0
    pol = [s for s in p if s.kind == "wait_mode" and s.mode == "POLICY"][0]
    assert pol.cmd == "POLICY" and pol.repeat == 0.5 and pol.timeout == 7.0


def test_ee_override_and_settle():
    p = plan("--ee", "0.25", "0.10", "0.30", "--settle", "2")
    assert [s.value for s in p if s.kind == "ee"] == [(0.25, 0.10, 0.30)]
    settle = [s for s in p if s.kind == "sleep" and "settle" in s.label]
    assert len(settle) == 1 and settle[0].seconds == 2.0
    p2 = plan(gap=0.7)
    assert [s.seconds for s in p2 if s.kind == "sleep" and s.seconds == 0.7]
    assert not [s for s in p2 if s.kind == "sleep" and s.seconds == GAP]


def test_settle_below_one_second_is_rejected():
    with pytest.raises(ValueError, match="settle"):
        plan("--settle", "0.5")
    with pytest.raises(ValueError, match="settle"):
        plan("--settle", "0.99")
    plan("--settle", "1.0")   # boundary accepted


@pytest.mark.parametrize("argv", [
    ["--ee", "0.1", "0.142", "0.12"],      # x below the FL range
    ["--ee", "0.30", "0.25", "0.12"],      # y above
    ["--ee", "0.30", "0.142", "0.5"],      # z above
    ["--gap", "-1"],
    ["--stand-wait", "-1"],
    ["--policy-timeout", "0"],
])
def test_out_of_range_parameters_are_rejected(argv):
    with pytest.raises(ValueError):
        plan(*argv)


@pytest.mark.parametrize("argv", [["--demo", "nope"], ["--demo", "other"], ["--vx", "0.1"],
                                  ["--end", "damp"]])
def test_bad_cli_exits(argv):
    with pytest.raises(SystemExit):
        ds.parse_args(argv)


def test_plan_duration_counts_sleeps():
    p = plan()
    from_damp = ds.plan_duration(p)
    standing = ds.plan_duration(p, initial_damp=False)
    assert from_damp == pytest.approx(standing + 4.0 + GAP)
    assert standing == pytest.approx(1.5 + GAP + ds.END_POLICY_HOLD)


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
        "print(len(ds.build_plan(ds.parse_args([]))))\n"
        "sys.exit(ds.main(['--demo', 'wam_entry', '--print-plan']))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "wam_entry" in r.stdout and "hand-over" in r.stdout
