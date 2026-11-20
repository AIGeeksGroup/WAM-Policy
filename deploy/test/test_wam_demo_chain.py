"""scripts/demo/wam_task.sh + launch/wam_bridge.launch.py: the shell chain every task runs on the robot.

Pure bash with a fake `ros2` (no ROS graph, no robot). What is pinned:
  * every launch argument wam_bridge_start passes is one ros2 launch accepts (`name:=value`, value not empty:
    ros2launch rejects 'calib:=' and the launch then starts nothing), for all ten tasks, with the task's
    prompt intact; calib is passed only when WAM_CALIB is set;
  * the robot is not stood up unless the bridge is running: wam_bridge_ready waits for the bridge's start line
    and fails when the launch dies first or stays silent;
  * the bridge's own exit code (its summary.json) is the run's exit code (ros2 launch itself returns 0);
  * the bridge launch ends when the bridge ends (ego_stream would otherwise keep it, and the run, alive);
  * the mock rehearsal enters POLICY at the task's entry target;
  * the operator's Ctrl+C damps the robot before anything waits for the bridge or the sidecars.
"""
import json
import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "scripts" / "demo" / "wam_task.sh"
LAUNCH = REPO / "launch" / "wam_bridge.launch.py"
TASKS = {t["id"]: t for t in json.loads((REPO / "config" / "wam_tasks.json").read_text())["tasks"]}


def functions(*names):
    text = DEMO.read_text()
    out = []
    for name in names:
        m = re.search(rf"^{name}\(\) \{{[^\n]*\n.*?^\}}\n", text, re.S | re.M)
        assert m, f"{name} not found in {DEMO}"
        out.append(m.group(0))
    return "\n".join(out)


FAKE_ROS2 = textwrap.dedent('''\
    #!/usr/bin/env bash
    # stands in for ros2: records argv; FAKE_ROS2_MODE decides what the "launch" then does
    python3 -c 'import json, os, sys; json.dump(sys.argv[1:], open(os.environ["FAKE_ROS2_OUT"], "w"))' "$@"
    case "${FAKE_ROS2_MODE:-ready}" in
      ready)  echo "[wam_bridge-2] [INFO] [1.0] [wam_bridge]: waiting for rambo/state, ego and D435i sim frames, the controller in POLICY for 2.0 s"
              sleep "${FAKE_ROS2_SLEEP:-0.2}" ;;
      die)    echo "malformed launch argument 'calib:=', expected format '<name>:=<value>'" >&2; exit 1 ;;
      silent) sleep "${FAKE_ROS2_SLEEP:-30}" ;;
      summary)
              echo "[wam_bridge-2] [INFO] [1.0] [wam_bridge]: waiting for rambo/state, ego and D435i sim frames"
              d="$FAKE_OUTPUT_DIR/go2-real-x_123"; mkdir -p "$d"
              printf '{"schema": "wam-bridge-run-v1", "exit_code": %s, "stop_reason": "%s"}\\n' \\
                "$FAKE_BRIDGE_RC" "$FAKE_STOP_REASON" > "$d/summary.json"
              echo "wam_bridge: exit $FAKE_BRIDGE_RC, records in $d"; exit 0 ;;
    esac
''')


@pytest.fixture
def fake_ros2(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "ros2").write_text(FAKE_ROS2)
    (bindir / "ros2").chmod(0o755)
    (tmp_path / "logs").mkdir()
    return {"bin": bindir, "out": tmp_path / "ros2_argv.json", "logs": tmp_path / "logs", "output": tmp_path / "bridge"}


def run_bash(body, fake_ros2, env=None, timeout=60, extra_functions=()):
    script = textwrap.dedent('''\
        set -euo pipefail
        # never leave a fake `ros2 launch` behind: a later pre-flight would take it for a stray launch
        trap 'if [[ -n "${WAM_BRIDGE_PID:-}" ]]; then kill -- "-$WAM_BRIDGE_PID" 2>/dev/null || true; fi' EXIT
        demo_say() { echo "[demo] $*"; }
        demo_die() { echo "[demo] DIE: $*"; exit 1; }
        demo_warn() { echo "[demo] WARNING: $*"; }
        DEMO_VIEW_CPUSET=""
        WAM_SHM_PROFILE=/dev/null
    ''') + functions("wam_bridge_start", "wam_bridge_ready", "wam_bridge_summary", "wam_bridge_wait",
                     *extra_functions) + "\n" + body
    base = {"PATH": f"{fake_ros2['bin']}:{os.environ['PATH']}", "HOME": os.environ.get("HOME", "/tmp"),
            "FAKE_ROS2_OUT": str(fake_ros2["out"]), "DEMO_LOG_DIR": str(fake_ros2["logs"]),
            "WAM_HOST": "127.0.0.1", "WAM_PORT": "8862",
            "WAM_INSTRUCTION": TASKS["push_box"]["instruction"], "WAM_TASK_ID": "push_box",
            "WAM_TASK_VERSION": "mujoco-push_box-runtime-v1", "WAM_SESSION": "go2-real-push_box",
            "WAM_MAX_CHUNKS": "1000", "WAM_MAX_WALL_S": "0", "WAM_SAVE_IMAGES": "0", "WAM_GO2_LATENCY_MS": "0",
            "WAM_CALIB": "", "WAM_OUTPUT_DIR": str(fake_ros2["output"]), "WAM_START_TIMEOUT_S": "90",
            "WAM_SETTLE_S": "2.0", "WAM_BRIDGE_READY_S": "30", "WAM_BRIDGE_T0": "0",
            "FAKE_OUTPUT_DIR": str(fake_ros2["output"])}
    base.update(env or {})
    return subprocess.run(["bash", "-c", script], env=base, capture_output=True, text=True, timeout=timeout)


def ros2launch_accepts(argument):
    """The check of ros2launch 0.26 (Jazzy) parse_launch_arguments, used when ros2launch is not importable."""
    try:
        from ros2launch.api import parse_launch_arguments
    except ImportError:
        count = argument.count(":=")
        return not (count == 0 or argument.startswith(":=") or (count == 1 and argument.endswith(":=")))
    try:
        parse_launch_arguments([argument])
        return True
    except RuntimeError:
        return False


def test_the_check_rejects_what_ros2launch_rejects():                       # positive control of the helper
    assert not ros2launch_accepts("calib:=") and not ros2launch_accepts("calib")
    assert ros2launch_accepts("calib:=/tmp/c.json") and ros2launch_accepts("instruction:=Press down, then push.")


@pytest.mark.parametrize("task", sorted(TASKS))
def test_every_launch_argument_is_accepted_for_every_task(fake_ros2, task):
    env = {"WAM_INSTRUCTION": TASKS[task]["instruction"], "WAM_TASK_ID": task,
           "WAM_TASK_VERSION": TASKS[task]["task_profile_version"], "WAM_SESSION": f"go2-real-{task}"}
    r = run_bash('wam_bridge_start; wait "$WAM_BRIDGE_PID"', fake_ros2, env)
    assert r.returncode == 0, r.stdout + r.stderr
    argv = json.loads(fake_ros2["out"].read_text())
    assert argv[:3] == ["launch", "wam_go2_deploy", "wam_bridge.launch.py"]
    bad = [a for a in argv[3:] if not ros2launch_accepts(a)]
    assert not bad, f"ros2 launch would refuse {bad}"
    args = dict(a.split(":=", 1) for a in argv[3:])
    assert args["instruction"] == TASKS[task]["instruction"]
    assert args["task_id"] == task and args["task_version"] == TASKS[task]["task_profile_version"]
    assert "calib" not in args                                               # WAM_CALIB unset: not passed at all


def test_calib_is_passed_when_set(fake_ros2, tmp_path):
    calib = tmp_path / "calib.json"
    calib.write_text("{}")
    r = run_bash('wam_bridge_start; wait "$WAM_BRIDGE_PID"', fake_ros2, {"WAM_CALIB": str(calib)})
    assert r.returncode == 0, r.stdout + r.stderr
    args = dict(a.split(":=", 1) for a in json.loads(fake_ros2["out"].read_text())[3:])
    assert args["calib"] == str(calib)


def test_the_bridge_must_be_up_before_the_robot_stands(fake_ros2):
    gate = 'wam_bridge_start; if wam_bridge_ready; then echo GATE=open; else echo GATE=closed; fi'
    r = run_bash(gate, fake_ros2, {"FAKE_ROS2_MODE": "ready", "FAKE_ROS2_SLEEP": "3"})
    assert "GATE=open" in r.stdout, r.stdout + r.stderr
    r = run_bash(gate, fake_ros2, {"FAKE_ROS2_MODE": "die"})                # the launch refused its arguments
    assert "GATE=closed" in r.stdout and "exit" in r.stdout, r.stdout + r.stderr
    r = run_bash(gate, fake_ros2, {"FAKE_ROS2_MODE": "silent", "WAM_BRIDGE_READY_S": "2"})
    assert "GATE=closed" in r.stdout, r.stdout + r.stderr


def test_main_gates_the_entry_sequence_on_the_bridge():
    main = re.search(r"^main\(\) \{\n.*?^\}\n", DEMO.read_text(), re.S | re.M).group(0)
    start, gate, entry = main.index("wam_bridge_start"), main.index("wam_bridge_ready"), main.index("demo_sequence.py")
    assert start < gate < entry
    assert re.search(r"wam_bridge_ready \|\| demo_die", main)


@pytest.mark.parametrize("bridge_rc,reason", [(2, "controller_left_policy"), (7, "service_error"), (0, "chunk_budget")])
def test_the_bridge_exit_code_comes_from_its_summary(fake_ros2, bridge_rc, reason):
    r = run_bash('wam_bridge_start; rc=0; wam_bridge_wait || rc=$?; echo "RC=$rc"', fake_ros2,
                 {"FAKE_ROS2_MODE": "summary", "FAKE_BRIDGE_RC": str(bridge_rc), "FAKE_STOP_REASON": reason})
    assert f"RC={bridge_rc}" in r.stdout, r.stdout + r.stderr
    assert reason in r.stdout


def test_without_a_summary_the_launch_exit_code_is_kept(fake_ros2):
    r = run_bash('wam_bridge_start; rc=0; wam_bridge_wait || rc=$?; echo "RC=$rc"', fake_ros2, {"FAKE_ROS2_MODE": "die"})
    assert "RC=1" in r.stdout, r.stdout + r.stderr


def test_an_old_summary_is_not_taken_for_this_run(fake_ros2):
    old = fake_ros2["output"] / "go2-real-old_1"
    old.mkdir(parents=True)
    (old / "summary.json").write_text('{"exit_code": 7, "stop_reason": "old"}')
    os.utime(old / "summary.json", (1, 1))
    r = run_bash('wam_bridge_start; rc=0; wam_bridge_wait || rc=$?; echo "RC=$rc"', fake_ros2, {"FAKE_ROS2_MODE": "die"})
    assert "RC=1" in r.stdout, r.stdout + r.stderr


def test_the_launch_ends_when_the_bridge_ends():
    """ego_stream never exits on its own: without this the run waits forever after the bridge's exit."""
    import ast
    tree = ast.parse(LAUNCH.read_text())
    bridge = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "bridge")
    call = bridge.value
    on_exit = next((k.value for k in call.keywords if k.arg == "on_exit"), None)
    assert on_exit is not None, "the wam_bridge Node needs on_exit=[Shutdown(...)]"
    assert "Shutdown" in ast.unparse(on_exit)
    assert re.search(r"from launch\.actions import [^\n]*\bShutdown\b", LAUNCH.read_text())


def test_the_mock_rehearsal_enters_at_the_task_entry_target():
    body = functions("wam_service_check")
    cmd = body[body.index("python3 -m wam_go2.wam.mock_service"):]
    cmd = cmd[:cmd.index("\n", cmd.index("--instruction"))]                # the mock's argument line
    assert "--rgb-profile" not in cmd                                       # one image profile
    assert "--ee $WAM_EE" in cmd, cmd
    assert cmd.index("--ee $WAM_EE") < cmd.index("$WAM_MOCK_ARGS"), "user mock arguments must stay last (override)"


def test_a_stuck_policy_service_ends_the_episode_within_a_minute():
    """A stuck server must not hold the robot in place for minutes: at most 2 x 30 s per request."""
    yaml_text = (REPO / "config" / "wam_bridge.yaml").read_text()
    value = float(re.search(r"^\s*rpc_timeout_s:\s*([0-9.]+)", yaml_text, re.M).group(1))
    assert value == 30.0
    node = (REPO / "wam_go2" / "nodes" / "wam_bridge_node.py").read_text()
    assert 'd("rpc_timeout_s", 30.0)' in node and "retries=1" in node          # <= 2 x 30 s


def test_a_bridge_that_died_without_its_summary_is_not_a_clean_exit(fake_ros2):
    """ros2 launch returns 0: a bridge killed before it wrote its summary.json (SIGKILL, a native crash) must not
    be recorded as exit 0 ('budget reached')."""
    script = 'wam_bridge_start; rc=0; wam_bridge_wait || rc=$?; echo "RC=$rc"'
    r = run_bash(script, fake_ros2, {"FAKE_ROS2_MODE": "ready", "FAKE_ROS2_SLEEP": "0.2"})    # exits 0, no summary
    assert "RC=0" not in r.stdout and re.search(r"RC=[1-9]", r.stdout), r.stdout + r.stderr
    assert "summary" in r.stdout


def test_the_gate_stays_closed_when_the_bridge_dies_right_after_its_start_line(fake_ros2):
    gate = 'wam_bridge_start; if wam_bridge_ready; then echo GATE=open; else echo GATE=closed; fi'
    r = run_bash(gate, fake_ros2, {"FAKE_ROS2_MODE": "ready", "FAKE_ROS2_SLEEP": "0"})        # line, then gone at once
    assert "GATE=closed" in r.stdout, r.stdout + r.stderr


def test_the_cleanup_cannot_be_cut_short_by_a_hangup():
    """A closed terminal sends SIGHUP twice: the second one must not abort the shutdown (DAMP, controller stop)."""
    text = DEMO.read_text()
    cleanup = re.search(r"^wam_cleanup\(\) \{\n.*?^\}\n", text, re.S | re.M).group(0)
    assert cleanup.index("trap '' INT TERM HUP PIPE") < cleanup.index("trap - EXIT"), "signals off before the EXIT trap"
    main = re.search(r"^main\(\) \{\n.*?^\}\n", text, re.S | re.M).group(0)
    assert re.search(r"trap '[^']*exit 130' INT TERM HUP", main), main[:400]
    common = (REPO / "scripts" / "demo" / "common.sh").read_text()
    cleanup = re.search(r"^demo_cleanup\(\) \{\n.*?^\}\n", common, re.S | re.M).group(0)
    assert cleanup.index("trap '' INT TERM HUP PIPE") < cleanup.index("trap - EXIT")


def test_ctrl_c_damps_before_waiting_for_the_bridge():
    """Ctrl+C is the operator's stop: DAMP is published right after the bridge is told to stop, before the
    cleanup waits for the bridge, the sidecars or the controller."""
    cleanup = re.search(r"^wam_cleanup\(\) \{\n.*?^\}\n", DEMO.read_text(), re.S | re.M).group(0)
    signal, damp = cleanup.index('kill -INT "$WAM_BRIDGE_PID"'), cleanup.index("demo_request_damp")
    wait = cleanup.index('wam_stop "$WAM_BRIDGE_PID"')                  # INT, TERM, KILL, each bounded
    assert signal < damp < wait < cleanup.index("wam_sidecar_stop") < cleanup.index("demo_cleanup")
    common = (REPO / "scripts" / "demo" / "common.sh").read_text()
    request = re.search(r"^demo_request_damp\(\) \{\n.*?^\}\n", common, re.S | re.M).group(0)
    assert request.index("demo_publish_damp") < request.index("demo_mode")      # publish first, then confirm


def test_the_bridge_launch_has_one_ego_source(fake_ros2):
    """The ego camera is the robot's /frontvideostream through ego_stream; there is no other source to choose."""
    r = run_bash('wam_bridge_start; wait "$WAM_BRIDGE_PID"', fake_ros2)
    assert r.returncode == 0, r.stdout + r.stderr
    args = dict(a.split(":=", 1) for a in json.loads(fake_ros2["out"].read_text())[3:])
    assert not {"ego_source", "ego_only"} & set(args)
    text = LAUNCH.read_text()
    assert "ego_source" not in text and "ego_only" not in text
