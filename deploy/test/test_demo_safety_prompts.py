"""scripts/demo/common.sh: the pre-flight's stray-launch check and the safety confirmation.

  * a leftover `ros2 launch` of THIS package (wam_go2.launch.py - the controller - or wam_bridge.launch.py) stops
    the pre-flight; an editor with the file open does not;
  * the "Press Enter to start" confirmation only takes an Enter pressed at that prompt: Enter presses typed ahead
    are discarded first, so they cannot stand the robot up;
  * under run_wam.sh (fd 3 = the terminal, stdout = the run's log) the prompt still reaches the terminal.
"""
import os
import pty
import select
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMMON = REPO / "scripts" / "demo" / "common.sh"


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DEMO_", "WAM_"))}
    env["COMMON_SH"] = str(COMMON)
    env["WAM_NO_CONFIG"] = "1"
    env.update(extra)
    return env


def _bash(script, env=None, timeout=60):
    full = 'set -euo pipefail\nsource "$COMMON_SH"\n' + script
    return subprocess.run(["bash", "-c", full], env=env or _env(), capture_output=True, text=True, timeout=timeout)


def _stray(launch_file):
    # the launch file name is assembled at run time so this test's own command line never contains it
    a, b = launch_file.split(".", 1)
    return f'''
NAME="{a}"".{b}"
( exec -a "/usr/bin/python3 /opt/ros/jazzy/bin/ros2 launch wam_go2_deploy $NAME" sleep 20 ) &
STRAY=$!
sleep 0.3
trap 'kill $STRAY 2>/dev/null || true' EXIT
demo_preflight_processes
echo "PREFLIGHT_PASSED"
'''


def test_a_stray_controller_launch_stops_the_preflight():
    r = _bash(_stray("wam_go2.launch.py"))
    assert "PREFLIGHT_PASSED" not in r.stdout, r.stdout + r.stderr
    assert "still running" in r.stdout + r.stderr and "wam_go2.launch.py" in r.stdout + r.stderr


def test_a_stray_bridge_launch_stops_the_preflight():
    r = _bash(_stray("wam_bridge.launch.py"))
    assert "PREFLIGHT_PASSED" not in r.stdout, r.stdout + r.stderr
    assert "still running" in r.stdout + r.stderr


def test_without_strays_the_preflight_passes():
    r = _bash('demo_preflight_processes\necho "PREFLIGHT_PASSED"')
    assert "PREFLIGHT_PASSED" in r.stdout, r.stdout + r.stderr


def _read_until(fd, needle, timeout):
    out, deadline = b"", time.time() + timeout
    while time.time() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.1)
        if ready:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
            if needle and needle in out:
                break
    return out


def test_enter_typed_ahead_does_not_answer_the_safety_confirmation():
    script = 'set -euo pipefail\nsource "$COMMON_SH"\nsleep 1\ndemo_confirm\necho CONFIRMED_BY_ENTER\n'
    pid, fd = pty.fork()
    if pid == 0:                                                   # child: bash on the pty (stdin is a tty)
        os.execvpe("bash", ["bash", "-c", script], _env())
    try:
        os.write(fd, b"\n\n")                                      # two Enter presses before the prompt exists
        out = _read_until(fd, b"Press Enter to start", 10)
        assert b"Press Enter to start" in out, out
        out = _read_until(fd, b"CONFIRMED_BY_ENTER", 1.5)          # nobody pressed Enter at the prompt yet
        assert b"CONFIRMED_BY_ENTER" not in out, "a typed-ahead Enter answered the safety confirmation"
        os.write(fd, b"\n")                                        # the operator's Enter at the prompt
        out = _read_until(fd, b"CONFIRMED_BY_ENTER", 10)
        assert b"CONFIRMED_BY_ENTER" in out, out
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)
        os.close(fd)


def test_an_editor_with_the_launch_file_open_does_not_block_the_preflight():
    """Only a running `ros2 launch` of this package is a stray."""
    script = '''
NAME="wam_bridge"".launch.py"
( exec -a "less /home/x/deploy/launch/$NAME" sleep 20 ) &
STRAY=$!
sleep 0.3
trap 'kill $STRAY 2>/dev/null || true' EXIT
demo_preflight_processes
echo "PREFLIGHT_PASSED"
'''
    r = _bash(script)
    assert "PREFLIGHT_PASSED" in r.stdout, r.stdout + r.stderr


def test_the_prompt_reaches_the_terminal_when_stdout_is_the_log(tmp_path):
    log = tmp_path / "run.log"
    script = (f'set -euo pipefail\nsource "$COMMON_SH"\nexec 3>&1\nexec >>"{log}" 2>&1\n'
              'demo_confirm\necho CONFIRMED_BY_ENTER >&3\n')
    pid, fd = pty.fork()
    if pid == 0:
        os.execvpe("bash", ["bash", "-c", script], _env())
    try:
        out = _read_until(fd, b"Press Enter to start", 10)
        assert b"Ctrl+C stops the robot" in out and b"Press Enter to start" in out, out
        os.write(fd, b"\n")
        out = _read_until(fd, b"CONFIRMED_BY_ENTER", 10)
        assert b"CONFIRMED_BY_ENTER" in out, out
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)
        os.close(fd)
    assert "operator confirmed" in log.read_text()


def test_there_is_no_hidden_prompt():
    text = COMMON.read_text(encoding="utf-8")
    assert "read -r -s" not in text and "password" not in text.lower()
