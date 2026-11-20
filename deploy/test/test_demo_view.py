"""scripts/demo/common.sh: the camera view sidecar (two windows, one video each).

Pure bash + a fake viewer, no ROS, no network, no cameras. What is pinned:
  * nothing is pinned unless DEMO_CPUSET / DEMO_VIEW_CPUSET say so;
  * demo_view_start launches the viewer with the expected arguments (default --res 360; no layout or fps
    choice), niced and optionally pinned, in its own session, with SIGINT/SIGQUIT not ignored;
    demo_view_stop ends it (SIGINT, then TERM, then KILL);
  * nothing the view does can fail the run: DEMO_VIEW=0, a missing or immediately-dying viewer, a viewer
    dying mid-run, an unreadable calibration file, an unwritable output dir and a missing display all leave
    the calling `set -euo pipefail` shell alive with status 0;
  * a leftover viewer from an earlier run is stopped by demo_preflight_processes (before the data-path
    byte-rate check) and by demo_view_start;
  * DEMO_VIEW_DIR (a run's result/<task>/<time>/view/) holds both videos and every record;
  * the integration order in wam_task.sh: the view starts after the data-path and controller checks and
    before the Enter confirmation, and is stopped after the controller in demo_cleanup.
"""
import json
import os
import shutil
import subprocess
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
COMMON = REPO / "scripts/demo/common.sh"
TASK_SH = REPO / "scripts/demo/wam_task.sh"

FAKE_VIEWER = textwrap.dedent('''\
    import json, os, signal, sys, time
    out = os.environ["FAKE_VIEW_OUT"]
    mode = os.environ.get("FAKE_VIEW_MODE", "normal")
    info = {"argv": sys.argv[1:], "nice": os.nice(0), "affinity": sorted(os.sched_getaffinity(0)),
            "sid_is_own": os.getsid(0) == os.getpid(), "pid": os.getpid(),
            "sigint_at_entry": str(signal.getsignal(signal.SIGINT)), "sigquit_at_entry": str(signal.getsignal(signal.SIGQUIT)),
            "profile": os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE")}
    with open(out, "w") as f:
        json.dump(info, f)
    if mode == "die":
        sys.exit(3)
    if mode == "die_later":                      # outlive the 1 s liveness check, then die mid-"demo"
        time.sleep(2.0)
        sys.exit(4)
    def on_int(*_):
        line = os.environ.get("FAKE_VIEW_PAIRING")   # the real viewer's last stdout line at exit
        if line:
            print(line, flush=True)
        with open(out + ".stopped", "w") as f:
            f.write("INT")
        sys.exit(0)
    def on_term(*_):
        with open(out + ".stopped", "w") as f:
            f.write("TERM")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal.SIG_IGN if mode in ("ignore_int", "ignore_all") else on_int)
    signal.signal(signal.SIGTERM, signal.SIG_IGN if mode == "ignore_all" else on_term)
    open(out + ".ready", "w").close()           # handlers installed: tests wait for this instead of a fixed sleep
    time.sleep(60)
''')


def _bash(script: str, env=None, timeout=60):
    """Run `script` in bash with the wrappers' strict mode, common.sh sourced first.

    DEMO_VIEW* variables of the caller's environment are dropped so every test
    starts from the defaults it sets itself.
    """
    full = 'set -euo pipefail\nsource "$COMMON_SH"\n' + script
    merged = {k: v for k, v in os.environ.items() if not k.startswith("DEMO_")}
    merged["COMMON_SH"] = str(COMMON)
    merged["WAM_NO_CONFIG"] = "1"
    if env:
        merged.update(env)
    return subprocess.run(["bash", "-c", full], env=merged, capture_output=True, text=True, timeout=timeout)


def test_defaults_point_at_the_viewer_and_pin_nothing():
    r = _bash('echo "BIN=$DEMO_VIEW_BIN RES=$DEMO_VIEW_RES DIR=[$DEMO_VIEW_DIR] VIEW=$DEMO_VIEW CPUSET=[$DEMO_VIEW_CPUSET] '
              'DEMO_CPUSET=[$DEMO_CPUSET]"')
    assert r.returncode == 0, r.stderr
    assert f"BIN={REPO / 'scripts/x5/ego_d435i_view.py'} RES=360 DIR=[] VIEW=1 CPUSET=[] DEMO_CPUSET=[]" in r.stdout
    assert (REPO / "scripts/x5/ego_d435i_view.py").is_file()


@pytest.fixture
def fake_view(tmp_path):
    bin_path = tmp_path / "fake_view.py"
    bin_path.write_text(FAKE_VIEWER)
    out = tmp_path / "fake_view.json"
    repo = tmp_path / "repo"
    repo.mkdir()
    return dict(bin=str(bin_path), out=out, repo=repo)


def _env(fake_view, **extra):
    env = {"DEMO_VIEW_BIN": fake_view["bin"], "FAKE_VIEW_OUT": str(fake_view["out"]), "FAKE_REPO": str(fake_view["repo"]),
           "DEMO_VIEW_CPUSET": "", "DISPLAY": ":99"}
    env.update(extra)
    return env


def _start_stop_script(sleep_s=0.5, extra_before=""):
    return f'''
DEMO_REPO="$FAKE_REPO"
DEMO_LOG_DIR="$FAKE_REPO/data/logs"
{extra_before}
demo_view_start fake_demo
echo "PID=[$DEMO_VIEW_PID] VIDEO=[$DEMO_VIEW_VIDEO] LOG=[$DEMO_VIEW_LOG]"
sleep {sleep_s}
demo_view_stop
echo "AFTER=[$DEMO_VIEW_PID]"
'''


def _info(fake_view):
    return json.loads(fake_view["out"].read_text())


def test_view_starts_with_defaults_niced_pinned_own_session_and_stops_on_sigint(fake_view):
    env = _env(fake_view, DEMO_VIEW_CPUSET="0")
    t0 = time.monotonic()
    r = _bash(_start_stop_script(), env)
    assert r.returncode == 0, r.stderr + r.stdout
    assert time.monotonic() - t0 < 10, "stop must not wait for the fake's 60 s sleep"
    info = _info(fake_view)
    argv = info["argv"]
    assert argv[:2] == ["--res", "360"], "default: the 360p /frontvideostream"
    assert "--fps" not in argv and "--layout" not in argv
    rec = argv[argv.index("--record") + 1]
    assert rec.startswith(str(fake_view["repo"] / "data/videos/demo_view/demo_fake_demo_")) and rec.endswith("_view.mp4")
    assert argv[argv.index("--out-dir") + 1] == str(fake_view["repo"] / "data/logs/demo_view")
    assert "--no-window" not in argv and "--calib" not in argv
    assert info["nice"] >= 10, "viewer must be niced below the controller"
    assert info["affinity"] == [0], "DEMO_VIEW_CPUSET must pin the viewer"
    assert info["sid_is_own"], "viewer must live in its own session (terminal Ctrl+C must not reach it)"
    assert "SIG_IGN" not in info["sigint_at_entry"] and "SIG_IGN" not in info["sigquit_at_entry"], \
        "env --default-signal must reset the dispositions a background job inherits"
    assert (fake_view["out"].parent / "fake_view.json.stopped").read_text() == "INT"
    assert "PID=[]" not in r.stdout and "AFTER=[]" in r.stdout
    assert "camera view stopped" in r.stdout and "uncalibrated" in r.stdout
    assert (fake_view["repo"] / "data/logs/demo_view").is_dir() and (fake_view["repo"] / "data/videos/demo_view").is_dir()


def test_res_knob(fake_view):
    r = _bash(_start_stop_script(), _env(fake_view, DEMO_VIEW_RES="720"))
    assert r.returncode == 0, r.stderr + r.stdout
    argv = _info(fake_view)["argv"]
    assert argv[:2] == ["--res", "720"] and "--fps" not in argv


def test_by_default_the_view_is_not_pinned_and_raises_no_cpu_warning(fake_view):
    env = _env(fake_view)
    del env["DEMO_VIEW_CPUSET"]                                           # default: no pinning
    r = _bash(_start_stop_script(), env)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "cpuset none" in r.stdout
    assert _info(fake_view)["affinity"] == sorted(os.sched_getaffinity(0))


def test_stop_escalates_to_sigterm_when_sigint_is_ignored(fake_view):
    t0 = time.monotonic()
    r = _bash(_start_stop_script(), _env(fake_view, FAKE_VIEW_MODE="ignore_int"))
    assert r.returncode == 0, r.stderr + r.stdout
    assert time.monotonic() - t0 < 12
    assert (fake_view["out"].parent / "fake_view.json.stopped").read_text() == "TERM"


def test_stop_escalates_to_sigkill_when_everything_is_ignored(fake_view):
    t0 = time.monotonic()
    r = _bash(_start_stop_script(), _env(fake_view, FAKE_VIEW_MODE="ignore_all"))
    assert r.returncode == 0, r.stderr + r.stdout
    assert time.monotonic() - t0 < 12
    assert not (fake_view["out"].parent / "fake_view.json.stopped").exists()
    pid = _info(fake_view)["pid"]
    assert not Path(f"/proc/{pid}").exists() or "zombie" in Path(f"/proc/{pid}/status").read_text().lower()
    assert "AFTER=[]" in r.stdout


def test_view_off_by_env_starts_nothing(fake_view):
    r = _bash(_start_stop_script(sleep_s=0), _env(fake_view, DEMO_VIEW="0"))
    assert r.returncode == 0, r.stderr + r.stdout
    assert not fake_view["out"].exists()
    assert "DEMO_VIEW=0" in r.stdout and "PID=[]" in r.stdout


def test_dying_viewer_is_one_warning_not_a_failure(fake_view):
    r = _bash(_start_stop_script(sleep_s=0), _env(fake_view, FAKE_VIEW_MODE="die"))
    assert r.returncode == 0, r.stderr + r.stdout
    assert "camera viewer exited" in r.stdout and "rc=3" in r.stdout and "PID=[]" in r.stdout


def test_viewer_dying_mid_run_is_reported_at_stop(fake_view):
    r = _bash(_start_stop_script(sleep_s=3), _env(fake_view, FAKE_VIEW_MODE="die_later"))
    assert r.returncode == 0, r.stderr + r.stdout
    pid_line = [ln for ln in r.stdout.splitlines() if ln.startswith("PID=[")][0]
    assert not pid_line.startswith("PID=[]"), "it was alive after the 1 s liveness check"
    assert "camera viewer stopped during the run" in r.stdout and "rc=4" in r.stdout
    assert "camera view stopped, video" not in r.stdout, "no video claim for a viewer that died"
    assert "AFTER=[]" in r.stdout


def test_missing_viewer_binary_is_one_warning(fake_view):
    env = {"DEMO_VIEW_BIN": str(fake_view["repo"] / "nope.py"), "FAKE_REPO": str(fake_view["repo"])}
    r = _bash(_start_stop_script(sleep_s=0), env)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "viewer not found" in r.stdout and "PID=[]" in r.stdout


def test_unwritable_output_dir_is_one_warning(fake_view, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    env = _env(fake_view, FAKE_REPO=str(blocker / "repo"))
    r = _bash(_start_stop_script(sleep_s=0), env)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "cannot create" in r.stdout and "PID=[]" in r.stdout and not fake_view["out"].exists()


def test_no_display_records_only_and_calib_file_is_checked(fake_view, tmp_path):
    calib = tmp_path / "calib.json"
    calib.write_text("{}")
    env = _env(fake_view, DISPLAY="", WAYLAND_DISPLAY="", DEMO_VIEW_CALIB=str(calib))
    r = _bash(_start_stop_script(), env)
    assert r.returncode == 0, r.stderr + r.stdout
    argv = _info(fake_view)["argv"]
    assert "--no-window" in argv and argv[argv.index("--calib") + 1] == str(calib)
    assert "records without windows" in r.stdout
    status = [ln for ln in r.stdout.splitlines() if "camera view: windows Ego and D435i" in ln][0]
    assert f"calib {calib}" in status and status.count(str(calib)) == 1 and "uncalibrated" not in status
    # unreadable calibration file: warning, uncalibrated view, the run goes on
    env["DEMO_VIEW_CALIB"] = str(tmp_path / "missing.json")
    r = _bash(_start_stop_script(), env)
    assert r.returncode == 0, r.stderr + r.stdout
    argv = _info(fake_view)["argv"]
    assert "--calib" not in argv and "is not readable" in r.stdout
    status = [ln for ln in r.stdout.splitlines() if "camera view: windows Ego and D435i" in ln][0]
    assert "uncalibrated" in status and "missing.json" not in status


def test_extra_args_are_split_on_whitespace_without_globbing(fake_view):
    r = _bash(_start_stop_script(), _env(fake_view, DEMO_VIEW_EXTRA="--flash-thresh 30 --x5-ip 10.0.0.*"))
    assert r.returncode == 0, r.stderr + r.stdout
    argv = _info(fake_view)["argv"]
    assert argv[-4:] == ["--flash-thresh", "30", "--x5-ip", "10.0.0.*"], "no pathname expansion of the extra args"


LEFTOVER_SPELLINGS = ["python3", "$(command -v python3)", "python3 -u"]


def _wait_ready(marker):
    return f'for i in $(seq 50); do [[ -e "{marker}" ]] && break; sleep 0.1; done; [[ -e "{marker}" ]]'


@pytest.mark.parametrize("interp", LEFTOVER_SPELLINGS)
def test_leftover_viewer_is_stopped_by_the_preflight_and_by_start(fake_view, tmp_path, interp):
    # a "leftover" must match the real viewer's name: copy the fake under that name
    leftover_dir = tmp_path / "leftover"
    leftover_dir.mkdir()
    leftover = leftover_dir / "ego_d435i_view.py"
    shutil.copy(fake_view["bin"], leftover)
    env = _env(fake_view, FAKE_VIEW_OUT=str(tmp_path / "leftover.json"))
    ready = tmp_path / "leftover.json.ready"
    script = f'''
DEMO_REPO="$FAKE_REPO"
DEMO_LOG_DIR="$FAKE_REPO/data/logs"
setsid {interp} "{leftover}" --leftover > /dev/null 2>&1 &
LEFT=$!
{_wait_ready(ready)}
kill -0 "$LEFT"
demo_preflight_processes
sleep 0.3
if kill -0 "$LEFT" 2>/dev/null; then echo "LEFTOVER_ALIVE"; else echo "LEFTOVER_GONE"; fi
'''
    r = _bash(script, env)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "left over from an earlier run" in r.stdout and "LEFTOVER_GONE" in r.stdout
    assert (tmp_path / "leftover.json.stopped").read_text() == "INT"
    # demo_view_start also stops a leftover before starting its own
    (tmp_path / "leftover.json.stopped").unlink()
    ready.unlink()
    script2 = f'''
DEMO_REPO="$FAKE_REPO"
DEMO_LOG_DIR="$FAKE_REPO/data/logs"
setsid {interp} "{leftover}" --leftover > /dev/null 2>&1 &
LEFT=$!
{_wait_ready(ready)}
FAKE_VIEW_OUT="{fake_view['out']}" demo_view_start fake_demo
sleep 0.3
if kill -0 "$LEFT" 2>/dev/null; then echo "LEFTOVER_ALIVE"; else echo "LEFTOVER_GONE"; fi
demo_view_stop
'''
    r = _bash(script2, env)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "left over from an earlier run" in r.stdout and "LEFTOVER_GONE" in r.stdout and "camera view stopped" in r.stdout


def test_leftover_pattern_matches_interpreters_only():
    import re
    text = COMMON.read_text(encoding="utf-8")
    m = re.search(r"local pat='([^']+)'", text)
    assert m, "demo_view_stop_stray must define its pattern in a single-quoted local"
    pat = re.compile(m.group(1))
    for cmd in ["python3 /home/x/scripts/x5/ego_d435i_view.py --res 360", "python3 scripts/x5/ego_d435i_view.py",
                "python3.10 ego_d435i_view.py", "/usr/bin/python3 scripts/x5/ego_d435i_view.py",
                "python3 -u scripts/x5/ego_d435i_view.py", "/opt/miniconda3/bin/python3 -u /abs/ego_d435i_view.py"]:
        assert pat.search(cmd), cmd
    for cmd in ["python3 -m pytest test/test_ego_d435i_view.py", "python3 test/test_ego_d435i_view.py",
                "/usr/share/code/code /home/x/scripts/x5/ego_d435i_view.py", "less ego_d435i_view.py",
                "python3 /abs/sleeper.py /abs/ego_d435i_view.py", "python3 my_ego_d435i_view.py",
                "python3 ego_d435i_view.py.bak", "bash -c source common.sh"]:
        assert not pat.search(cmd), cmd


def test_integration_order():
    text = TASK_SH.read_text(encoding="utf-8")
    main = text[text.index("main() {"):]
    assert main.index("demo_preflight_processes") < main.index("demo_preflight_datapath") \
        < main.index("demo_check_controller") < main.index('demo_view_start "$WAM_TASK_ID"') < main.index("demo_confirm")
    common = COMMON.read_text(encoding="utf-8")
    pre = common[common.index("demo_preflight_processes() {"):common.index("demo_release_sport_mode() {")]
    assert "demo_view_stop_stray" in pre, "a leftover viewer must be stopped before the data-path byte-rate check"
    assert pre.index("[r]os2 launch .*wam_(go2|bridge)\\.launch\\.py") < pre.index("demo_view_stop_stray"), \
        "stop the leftover only after the stray-controller checks"
    cleanup = common[common.index("demo_cleanup() {"):]
    assert cleanup.index('wait "$DEMO_LAUNCH_PID"') < cleanup.index("demo_view_stop") < cleanup.index('exit "$rc"')
    assert 'DEMO_VIEW="${DEMO_VIEW:-1}"' in common
    r = subprocess.run(["bash", "-n", str(COMMON)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_view_tells_the_terminal_one_line(fake_view, tmp_path):
    """Under run_wam.sh fd 3 is the terminal: the view adds one line there, the details go to the log."""
    script = _start_stop_script().replace("demo_view_start fake_demo", 'demo_view_start fake_demo 3>"$TERM_OUT"')
    r = _bash(script, _env(fake_view, TERM_OUT=str(tmp_path / "term.txt")))
    assert r.returncode == 0, r.stderr + r.stdout
    lines = (tmp_path / "term.txt").read_text().splitlines()
    assert len(lines) == 1 and "Camera windows open" in lines[0]


# The run names the viewer's pair log at start and logs the viewer's one-line pairing report
# ("pairing: ...", its last stdout line) after the run.
def test_start_names_the_pair_log_and_stop_prints_the_viewers_pairing_line(fake_view):
    line = ("pairing: 410/420 matched (2.4 % unmatched), |delta| p95 15.1 ms, receive_time_approx, "
            "common-event offset +44 ms (n=3); /x/demo_view_pairs.csv")
    r = _bash(_start_stop_script(), _env(fake_view, FAKE_VIEW_PAIRING=line))
    assert r.returncode == 0, r.stderr + r.stdout
    log_dir = fake_view["repo"] / "data/logs/demo_view"
    pairs = [ln.split("pair log: ")[1].strip() for ln in r.stdout.splitlines() if "pair log: " in ln]
    assert len(pairs) == 1 and pairs[0].startswith(str(log_dir / "demo_fake_demo_")) and pairs[0].endswith("_view_pairs.csv")
    video = [ln for ln in r.stdout.splitlines() if ln.startswith("PID=[")][0].split("VIDEO=[")[1].split("]")[0]
    assert Path(pairs[0]).name == Path(video).stem + "_pairs.csv"      # the viewer derives it from the --record stem
    assert f"  {line}" in r.stdout


def test_stop_without_a_pairing_line_prints_nothing_extra(fake_view):
    r = _bash(_start_stop_script(), _env(fake_view))
    assert r.returncode == 0, r.stderr + r.stdout
    assert "pairing:" not in r.stdout and "camera view stopped" in r.stdout


def test_the_ego_source_knob_passes_through(fake_view):
    r = _bash(_start_stop_script(), _env(fake_view))
    assert r.returncode == 0, r.stderr + r.stdout
    argv = _info(fake_view)["argv"]
    assert "--layout" not in argv and argv[argv.index("--ego-source") + 1] == "frontvideo"
    assert "windows Ego and D435i" in r.stdout
    r = _bash(_start_stop_script(), _env(fake_view, DEMO_VIEW_EGO_SOURCE="topic"))
    assert r.returncode == 0, r.stderr + r.stdout
    argv = _info(fake_view)["argv"]
    assert "--layout" not in argv and argv[argv.index("--ego-source") + 1] == "topic"


def test_view_dir_holds_both_videos_and_every_record(fake_view, tmp_path):
    # run_wam.sh passes result/<task>/<time>/view/ as DEMO_VIEW_DIR
    view_dir = tmp_path / "result" / "push_box" / "20260101_120000" / "view"
    script = '''
DEMO_REPO="$FAKE_REPO"
DEMO_LOG_DIR="$FAKE_REPO/data/logs"
demo_view_start fake_demo
echo "VIDEO=[$DEMO_VIEW_VIDEO] EGO=[$DEMO_VIEW_VIDEO_EGO] D435I=[$DEMO_VIEW_VIDEO_D435I] LOG=[$DEMO_VIEW_LOG] PAIRS=[$DEMO_VIEW_PAIRS]"
printf x > "$DEMO_VIEW_VIDEO_EGO"                 # the viewer's ego video; no D435i video
sleep 0.5
demo_view_stop
'''
    r = _bash(script, _env(fake_view, DEMO_VIEW_DIR=str(view_dir)))
    assert r.returncode == 0, r.stderr + r.stdout
    argv = _info(fake_view)["argv"]
    assert argv[argv.index("--record") + 1] == str(view_dir / "camera.mp4")
    assert argv[argv.index("--out-dir") + 1] == str(view_dir)
    assert (f"VIDEO=[{view_dir / 'camera.mp4'}] EGO=[{view_dir / 'camera_ego.mp4'}] D435I=[{view_dir / 'camera_d435i.mp4'}] "
            f"LOG=[{view_dir / 'camera.log'}] PAIRS=[{view_dir / 'camera_pairs.csv'}]") in r.stdout
    assert f"videos: {view_dir / 'camera_ego.mp4'}, {view_dir / 'camera_d435i.mp4'}" in r.stdout
    assert f"camera view stopped, video: {view_dir / 'camera_ego.mp4'}" in r.stdout
    assert f"no video {view_dir / 'camera_d435i.mp4'}" in r.stdout
    assert view_dir.is_dir() and not (fake_view["repo"] / "data").exists(), "nothing in data/videos/demo_view"


def test_stop_prints_the_grid_line_before_the_pairing_line(fake_view):
    line = "grid 50 Hz: 100 ticks (2.0 s); D435i new 100.0 %"
    r = _bash(_start_stop_script(), _env(fake_view, FAKE_VIEW_PAIRING=line))
    assert r.returncode == 0, r.stderr + r.stdout
    assert "  " + line in r.stdout
