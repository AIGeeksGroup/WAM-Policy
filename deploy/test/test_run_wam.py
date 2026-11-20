"""run_wam.sh and tasks/<task>.sh: one command per task, a terse terminal, everything in the run's directory.

Pure bash with fakes for the build, the policy server and the robot side (no ROS, no GPU, no robot).
What is pinned:
  * --list / no task / an unknown task / a bad WAM_POLICY start nothing;
  * the terminal shows only the header, the stage headers (between dashed lines) and the result; the rest
    goes to run.log;
  * the robot side gets the task's prompt, the run's directories and WAM_MOCK, plus the extra arguments;
  * the policy server is started for the task, waited for, and stopped after the run; a port that already
    answers is never used; a server that dies or never answers ends the run before the robot moves;
  * the listener on the port must belong to the server started here (ss, or /proc without ss);
  * the robot side's exit code is the run's;
  * every task in config/wam_tasks.json has an executable tasks/<task>.sh that runs exactly that task.
"""
import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "run_wam.sh"
TASKS = {t["id"]: t for t in json.loads((REPO / "config" / "wam_tasks.json").read_text())["tasks"]}

FAKE_BUILD = "#!/usr/bin/env bash\necho '[build] fake build'\nexit \"${FAKE_BUILD_RC:-0}\"\n"
FAKE_TASK = textwrap.dedent('''\
    #!/usr/bin/env bash
    # the robot side: records what it got, prints its stage lines to fd 3 like wam_task.sh
    python3 - "$@" <<'EOF'
    import json, os, sys
    keys = ("WAM_TASK_ID", "WAM_INSTRUCTION", "WAM_OUTPUT_DIR", "DEMO_VIEW_DIR", "DEMO_LOG_DIR", "WAM_MOCK", "WAM_PORT")
    json.dump({"argv": sys.argv[1:], "env": {k: os.environ.get(k) for k in keys}}, open(os.environ["FAKE_TASK_OUT"], "w"))
    EOF
    rule=------------------------------------------------------------
    { printf '%s\\n[3/5] bring-up\\n%s\\n[4/5] running\\n%s\\n[5/5] shutdown\\n' "$rule" "$rule" "$rule" >&3; } 2>/dev/null
    echo "robot side: details for the log only"
    exit "${FAKE_TASK_RC:-0}"
''')
FAKE_SERVER = textwrap.dedent('''\
    #!/usr/bin/env bash
    # the policy server: records its task, then listens on WAM_PORT (or dies / stays silent)
    echo "$1" > "$FAKE_SERVER_OUT"
    echo "log dir: ${WAM_SERVER_LOG_DIR:-}" >> "$FAKE_SERVER_OUT"
    case "${FAKE_SERVER_MODE:-listen}" in
      die) echo "Traceback: the model did not load" >&2; exit 1 ;;
      silent) exec sleep 60 ;;
    esac
    sleep "${FAKE_SERVER_DELAY:-0.5}"
    exec python3 -c 'import socket, sys
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", int(sys.argv[1]))); s.listen(5)
    while True:
        c, _ = s.accept(); c.close()' "$WAM_PORT"
''')


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fakes(tmp_path):
    for name, text in (("build.sh", FAKE_BUILD), ("task.sh", FAKE_TASK), ("server.sh", FAKE_SERVER)):
        (tmp_path / name).write_text(text)
        (tmp_path / name).chmod(0o755)
    port = free_port()
    venv = tmp_path / "venv" / "bin"                                   # a deploy venv stand-in
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)

    def run(*args, env=None, timeout=60):
        base = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "WAM_NO_CONFIG": "1", "WAM_PORT": str(port),
                "WAM_RUN_ROOT": str(tmp_path / "result"), "WAM_RUN_BUILD": str(tmp_path / "build.sh"),
                "WAM_RUN_TASK": str(tmp_path / "task.sh"), "WAM_RUN_POLICY_SERVER": str(tmp_path / "server.sh"),
                "FAKE_TASK_OUT": str(tmp_path / "task.json"), "FAKE_SERVER_OUT": str(tmp_path / "server.txt"),
                "WAM_SERVER_WAIT_S": "20", "DEMO_NIC": "lo", "DEPLOY_VENV": str(tmp_path / "venv"),
                "DEMO_ROBOT_IP": "192.0.2.1", "DEMO_LAPTOP_IP": "192.0.2.10"}
        base.update(env or {})
        return subprocess.run([str(RUN), *args], env=base, capture_output=True, text=True, timeout=timeout)

    run.tmp, run.port = tmp_path, port
    return run


def run_dir(fakes, task="push_box"):
    dirs = sorted((fakes.tmp / "result" / task).iterdir())
    assert len(dirs) == 1, dirs
    return dirs[0]


def test_list_and_usage_start_nothing(fakes):
    r = fakes("--list")
    assert r.returncode == 0 and all(t in r.stdout for t in TASKS)
    r = fakes()
    assert r.returncode == 64 and "push_box" in r.stdout
    assert not (fakes.tmp / "result").exists() and not (fakes.tmp / "task.json").exists()


def test_unknown_task_and_bad_policy_are_refused(fakes):
    r = fakes("no_such_task")
    assert r.returncode == 64 and "unknown task" in r.stdout
    for env in ({"WAM_POLICY": "sft"}, {"WAM_POLICY": "mock"}, {"WAM_MOCK": "yes"}):
        r = fakes("push_box", env=env)
        assert r.returncode == 64 and list(env)[0] in r.stdout, r.stdout
    assert not (fakes.tmp / "result").exists() and not (fakes.tmp / "server.txt").exists()


def test_mock_run_shows_only_the_stages_and_hands_the_task_over(fakes):
    r = fakes("open_room_door", "--settle", "3", env={"WAM_MOCK": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    lines = r.stdout.splitlines()
    rule = "-" * 60
    text = [line for line in lines if line != rule]
    assert lines[0] == rule and lines[-1] == rule
    assert text[0] == "WAM-Policy | task open_room_door | policy mock (rehearsal, no model)"
    assert text[1] == '"' + TASKS["open_room_door"]["instruction"] + '"' and text[2].startswith("Run: ")
    stages = [i for i, line in enumerate(lines) if line.startswith("[")]
    assert [lines[i][:5] for i in stages] == ["[1/5]", "[2/5]", "[3/5]", "[4/5]", "[5/5]"]
    assert all(lines[i - 1] == rule for i in stages), "every stage starts after a dashed line"
    assert text[-2] == "Done: chunk budget reached (exit 0)" and text[-1].startswith("Results: ")
    assert len(text) <= 10, r.stdout                                    # header (3), 5 stages, result (2)
    assert all(ord(c) < 128 for c in r.stdout), "the terminal output is English only"
    assert "fake build" not in r.stdout and "details for the log only" not in r.stdout
    out = run_dir(fakes, "open_room_door")
    log = (out / "run.log").read_text()
    assert "fake build" in log and "details for the log only" in log
    got = json.loads((fakes.tmp / "task.json").read_text())
    assert got["argv"] == ["--settle", "3"]
    env = got["env"]
    assert env["WAM_TASK_ID"] == "open_room_door" and env["WAM_INSTRUCTION"] == TASKS["open_room_door"]["instruction"]
    assert env["WAM_MOCK"] == "1" and env["WAM_PORT"] == str(fakes.port)
    assert env["WAM_OUTPUT_DIR"] == str(out / "bridge") and env["DEMO_VIEW_DIR"] == str(out / "view")
    assert env["DEMO_LOG_DIR"] == str(out / "logs")
    record = json.loads((out / "run.json").read_text())
    assert record["task"] == "open_room_door" and record["mock"] is True and record["exit_code"] == 0
    assert not (fakes.tmp / "server.txt").exists()                     # no model in a rehearsal


def test_the_server_is_started_for_the_task_waited_for_and_stopped(fakes):
    r = fakes("push_box", env={"FAKE_SERVER_DELAY": "1.5"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Policy server ready (" in r.stdout
    server = (fakes.tmp / "server.txt").read_text().splitlines()
    out = run_dir(fakes)
    assert server[0] == "push_box" and server[1] == f"log dir: {out / 'server'}"
    assert json.loads((fakes.tmp / "task.json").read_text())["env"]["WAM_MOCK"] == "0"
    with socket.socket() as s:                                          # stopped after the run
        assert s.connect_ex(("127.0.0.1", fakes.port)) != 0


def test_a_port_that_already_answers_is_never_used(fakes):
    with socket.socket() as s:
        s.bind(("127.0.0.1", fakes.port))
        s.listen(1)
        for mock in ("0", "1"):
            r = fakes("push_box", env={"WAM_MOCK": mock})
            assert r.returncode == 3 and "already in use" in r.stdout, r.stdout + r.stderr
    assert not (fakes.tmp / "server.txt").exists() and not (fakes.tmp / "task.json").exists()


def test_a_server_that_dies_or_never_answers_ends_the_run_before_the_robot(fakes):
    r = fakes("push_box", env={"FAKE_SERVER_MODE": "die"})
    assert r.returncode == 3 and "exited before it was ready" in r.stdout
    assert "the model did not load" in r.stdout                        # the log's tail is shown
    assert not (fakes.tmp / "task.json").exists()
    r = fakes("press_button", env={"FAKE_SERVER_MODE": "silent", "WAM_SERVER_WAIT_S": "2"})
    assert r.returncode == 3 and "not ready within 2 s" in r.stdout
    assert not (fakes.tmp / "task.json").exists()


def test_the_listener_must_belong_to_the_server_started_here(fakes, tmp_path):
    fake_ss = tmp_path / "ss"
    fake_ss.write_text('#!/usr/bin/env bash\necho \'LISTEN 0 5 127.0.0.1:1 0.0.0.0:* users:(("python3",pid=1,fd=3))\'\n')
    fake_ss.chmod(0o755)
    r = fakes("push_box", env={"WAM_RUN_SS": str(fake_ss)})
    assert r.returncode == 3 and "cannot confirm" in r.stdout, r.stdout + r.stderr
    assert not (fakes.tmp / "task.json").exists()


def test_without_ss_the_listener_is_found_through_proc(fakes):
    r = fakes("push_box", env={"WAM_RUN_SS": "/nonexistent/ss"})
    assert r.returncode == 0, r.stdout + r.stderr
    r = fakes("lift_basket", env={"WAM_RUN_SS": "/nonexistent/ss", "WAM_RUN_PROC_NET": "/nonexistent/tcp"})
    assert r.returncode == 3 and "cannot confirm" in r.stdout          # nothing identifiable: fail closed


@pytest.mark.parametrize("rc,meaning", [(2, "the controller left POLICY"), (130, "interrupted"), (6, "failed")])
def test_the_robot_sides_exit_code_is_the_runs(fakes, rc, meaning):
    r = fakes("push_box", env={"WAM_MOCK": "1", "FAKE_TASK_RC": str(rc)})
    done = [line for line in r.stdout.splitlines() if line.startswith("Done: ")]
    assert r.returncode == rc and done == [f"Done: {meaning}" + done[0][len(f"Done: {meaning}"):]], r.stdout
    assert done[0].endswith(f"(exit {rc})")
    assert ("Log: " in r.stdout) == (rc == 6), "the log is pointed at only when something failed"
    assert json.loads((run_dir(fakes) / "run.json").read_text())["exit_code"] == rc


@pytest.mark.parametrize("env,what", [({"DEMO_NIC": ""}, "DEMO_NIC is not set"),
                                      ({"DEMO_ROBOT_IP": ""}, "DEMO_ROBOT_IP is not set"),
                                      ({"DEMO_LAPTOP_IP": "laptop"}, "DEMO_LAPTOP_IP is not set"),
                                      ({"DEMO_NIC": "no_such_nic0"}, "not found"),
                                      ({"DEPLOY_VENV": "/nonexistent"}, "deploy venv not found"),
                                      ({"WAM_HEADING_HOLD": "yes"}, "true or false"),
                                      ({"WAM_MAX_CHUNKS": "1e3"}, "positive integer"),
                                      ({"WAM_MAX_WALL_S": "-1"}, "non-negative number"),
                                      ({"WAM_CALIB": "/nonexistent.json"}, "not readable")])
def test_bad_settings_stop_before_the_model_loads(fakes, env, what):
    r = fakes("push_box", env=env)
    assert r.returncode == 64 and what in r.stdout, r.stdout
    assert not (fakes.tmp / "server.txt").exists() and not (fakes.tmp / "result").exists()


def test_a_failed_build_stops_before_the_robot(fakes):
    r = fakes("push_box", env={"WAM_MOCK": "1", "FAKE_BUILD_RC": "1"})
    assert r.returncode == 1 and "build failed" in r.stdout
    assert not (fakes.tmp / "task.json").exists()


@pytest.mark.parametrize("task", sorted(TASKS))
def test_every_task_has_its_one_command(task):
    path = REPO / "tasks" / f"{task}.sh"
    assert path.is_file() and os.access(path, os.X_OK)
    text = path.read_text()
    assert f'/run_wam.sh" {task} "$@"' in text and TASKS[task]["instruction"] in text
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    assert sorted(p.stem for p in (REPO / "tasks").glob("*.sh")) == sorted(TASKS)


def test_every_script_in_the_chain_is_executable():
    for rel in ("run_wam.sh", "scripts/build.sh", "scripts/demo/wam_task.sh", "scripts/wam/policy_server.sh",
                "scripts/demo_sequence.py"):
        assert os.access(REPO / rel, os.X_OK), rel
        if rel.endswith(".sh"):
            assert subprocess.run(["bash", "-n", str(REPO / rel)]).returncode == 0, rel
