"""Ctrl+C is the operator's stop: the real signal paths of run_wam.sh -> scripts/demo/wam_task.sh.

The run is real (run_wam.sh, wam_task.sh's wam_cleanup / wam_stop, common.sh's output and demo_cleanup) under a
pseudo-terminal; only the robot is replaced: a stand-in "controller" process and a demo_request_damp that writes a
marker instead of publishing DAMP. What is pinned:
  * Ctrl+C damps and stops the controller even when the terminal output goes through a pipe whose reader the same
    Ctrl+C kills (`tasks/push_box.sh | tee session.txt`);
  * two hang-ups in quick succession (a closed terminal) cannot skip the shutdown;
  * with WAM_VERBOSE=1 (run.log also shown through tee) Ctrl+C still ends the run as "interrupted" with the
    shutdown logged.
"""
import os
import pty
import re
import select
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TASK_SH = REPO / "scripts" / "demo" / "wam_task.sh"


def functions(*names):
    text = TASK_SH.read_text()
    out = []
    for name in names:
        m = re.search(rf"^{name}\(\) \{{[^\n]*\n.*?^\}}\n", text, re.S | re.M)
        assert m, name
        out.append(m.group(0))
    return "\n".join(out)


ROBOT_SIDE = textwrap.dedent('''\
    #!/usr/bin/env bash
    # wam_task.sh with the robot replaced: its real cleanup, a stand-in controller, a DAMP marker
    set -euo pipefail
    source "$COMMON_SH"
    eval "$WAM_FUNCS"
    demo_request_damp() { echo damp >> "$MARK"; }
    WAM_BRIDGE_PID=""
    WAM_SIDECAR_PIDS=()
    setsid env --default-signal=INT,QUIT sleep 60 </dev/null >/dev/null 2>&1 3>&- &
    DEMO_LAUNCH_PID=$!
    DEMO_LAUNCH_PGID=$DEMO_LAUNCH_PID
    echo "$DEMO_LAUNCH_PID" > "$MARK.controller"
    TRAPS
    demo_stage "[3/5] Robot bring-up: network, controller, cameras"
    demo_stage "[4/5] Policy running (Ctrl+C to stop the robot)"
    while :; do sleep 0.1; done
''')


def main_traps():
    """The trap lines of wam_task.sh's main(), exactly as they are."""
    main = re.search(r"^main\(\) \{\n.*?^\}\n", TASK_SH.read_text(), re.S | re.M).group(0)
    lines = [ln.strip() for ln in main.splitlines() if ln.strip().startswith("trap ")]
    assert any("wam_cleanup EXIT" in ln for ln in lines) and len(lines) >= 2, lines
    return "\n".join(lines)


@pytest.fixture
def run_env(tmp_path):
    robot = tmp_path / "robot_side.sh"
    robot.write_text(ROBOT_SIDE.replace("TRAPS", main_traps()))
    robot.chmod(0o755)
    build = tmp_path / "build.sh"
    build.write_text("#!/usr/bin/env bash\necho build\n")
    build.chmod(0o755)
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)
    mark = tmp_path / "mark"
    env = {**{k: v for k, v in os.environ.items() if not k.startswith(("WAM_", "DEMO_"))},
           "WAM_NO_CONFIG": "1", "WAM_MOCK": "1", "WAM_PORT": "18911", "DEMO_NIC": "lo",
           "DEMO_ROBOT_IP": "192.0.2.1", "DEMO_LAPTOP_IP": "192.0.2.10",
           "DEPLOY_VENV": str(tmp_path / "venv"), "WAM_RUN_ROOT": str(tmp_path / "result"),
           "WAM_RUN_BUILD": str(build), "WAM_RUN_TASK": str(robot), "COMMON_SH": str(REPO / "scripts/demo/common.sh"),
           "WAM_FUNCS": functions("wam_stop", "wam_sidecar_stop", "wam_cleanup"), "MARK": str(mark)}
    return {"env": env, "mark": mark, "tmp": tmp_path}


def _read_until(fd, needle, timeout):
    out, end = b"", time.time() + timeout
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.1)
        if r:
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


def _run_in_terminal(command, env, keys_after_stage4, timeout=30):
    """Run `command` (bash) on a pseudo-terminal; once stage 4 shows, send the given action; -> (rc, output)."""
    pid, fd = pty.fork()
    if pid == 0:
        # a shell started by a terminal has default signal dispositions (Python ignores SIGPIPE, and a shell
        # cannot un-ignore a signal it was started with)
        for sig in (signal.SIGPIPE, signal.SIGINT, signal.SIGHUP, signal.SIGTERM):
            signal.signal(sig, signal.SIG_DFL)
        os.chdir(REPO)
        os.execvpe("bash", ["bash", "-c", command], env)
    try:
        out = _read_until(fd, b"[4/5]", timeout)
        assert b"[4/5]" in out, out.decode(errors="replace")
        time.sleep(0.3)
        keys_after_stage4(pid, fd)
        out += _read_until(fd, None, timeout)
        _, status = os.waitpid(pid, 0)
        return os.waitstatus_to_exitcode(status), out.decode(errors="replace")
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _controller_stopped(mark):
    pid = int(Path(str(mark) + ".controller").read_text())
    for _ in range(50):
        if not Path(f"/proc/{pid}").exists() or "zombie" in Path(f"/proc/{pid}/status").read_text().lower():
            return True
        time.sleep(0.1)
    os.kill(pid, signal.SIGKILL)
    return False


def ctrl_c(pid, fd):
    os.write(fd, b"\x03")                          # the terminal's Ctrl+C: SIGINT to the foreground group


def test_ctrl_c_damps_even_when_the_output_pipe_dies_with_it(run_env):
    session = run_env["tmp"] / "session.txt"
    rc, out = _run_in_terminal(f'./tasks/push_box.sh | tee "{session}"', run_env["env"], ctrl_c)
    assert run_env["mark"].read_text().startswith("damp"), "Ctrl+C did not request DAMP"
    assert _controller_stopped(run_env["mark"]), "the controller was left running"


def test_two_quick_hangups_cannot_skip_the_shutdown(run_env):
    """A race: without the fix about 1 run in 15 lost its shutdown; 12 runs with gaps of 1-8 ms."""
    gaps = iter([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008] * 2)

    def two_hangups(pid, fd):
        os.killpg(pid, signal.SIGHUP)
        time.sleep(next(gaps))
        os.killpg(pid, signal.SIGHUP)
    for _ in range(12):
        if run_env["mark"].exists():
            run_env["mark"].unlink()
        rc, out = _run_in_terminal("./tasks/push_box.sh", run_env["env"], two_hangups)
        assert run_env["mark"].exists() and run_env["mark"].read_text().startswith("damp"), out
        assert _controller_stopped(run_env["mark"])


def test_ctrl_c_with_verbose_output_ends_as_interrupted(run_env):
    env = {**run_env["env"], "WAM_VERBOSE": "1"}
    rc, out = _run_in_terminal("./tasks/push_box.sh", env, ctrl_c)
    assert rc == 130, out
    assert "Done: interrupted (exit 130)" in out
    assert run_env["mark"].read_text().startswith("damp") and _controller_stopped(run_env["mark"])
    (log,) = list((run_env["tmp"] / "result" / "push_box").glob("*/run.log"))
    text = log.read_text()
    assert "interrupted" in text and "stopping the controller" in text, text[-2000:]
