#!/usr/bin/env python3
"""The real `ros2 launch wam_go2_deploy wam_bridge.launch.py` with the exact arguments of
scripts/demo/wam_task.sh (wam_bridge_start / wam_bridge_ready / wam_bridge_wait) - no robot, no streams.

mock_wam_test starts the bridge with `ros2 run`; this test covers the launch path every task takes on
the robot: ros2 launch (Jazzy) refuses an empty 'calib:=' argument, and ego_stream never exits on its
own, so the launch must end with the bridge.

Scenarios (ROS domain 43, loopback-only FastDDS, a 3 s start timeout, no camera / controller streams):
  bridge      ego_stream + wam_bridge: the arguments are accepted, the bridge prints its start line
              (the stand-up gate opens), gives up after 3 s (exit 4, could_not_start), the launch ends
              with it - ego_stream stopped - and wam_bridge_wait returns the bridge's 4 from its
              summary.json
  calib       the same with a calib file given in WAM_CALIB
  wrong_arg   a launch argument ros2 launch refuses: the gate closes at once (the robot would not stand up)

Needs a sourced ROS 2 env and the built package (like the other integration tests):

    python3 test/integration/mock_bridge_launch_test.py

Prints ALL BRIDGE LAUNCH CHECKS PASSED and exits 0 when every check holds.
"""
import os
import re
import subprocess
import sys
import tempfile
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMO = os.path.join(REPO, "scripts", "demo", "wam_task.sh")
LAST_ARG = '"settle_s:=$WAM_SETTLE_S")'          # the last launch argument of wam_bridge_start
FAILURES = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILURES.append(msg)


def functions(*names):
    text = open(DEMO, encoding="utf-8").read()
    out = []
    for name in names:
        m = re.search(rf"^{name}\(\) \{{[^\n]*\n.*?^\}}\n", text, re.S | re.M)
        assert m, name
        out.append(m.group(0))
    return "\n".join(out)


def profile():
    text = open(os.path.join(REPO, "config", "fastdds_wired_shm.xml"), encoding="utf-8").read()
    path = os.path.join(tempfile.gettempdir(), f"wam_bridge_launch_test_{os.getpid()}.xml")
    open(path, "w", encoding="utf-8").write(text.replace("@LAPTOP_IP@", "127.0.0.1"))
    return path


def run(name, extra_env=None, bad_arg=False):
    tmp = tempfile.mkdtemp(prefix=f"wam_bridge_launch_{name}_")
    os.makedirs(os.path.join(tmp, "data", "logs"))
    body = functions("wam_bridge_start", "wam_bridge_ready", "wam_bridge_summary", "wam_bridge_wait")
    if bad_arg:                                       # one argument ros2 launch refuses
        assert LAST_ARG in body, "wam_bridge_start changed: update LAST_ARG"
        body = body.replace(LAST_ARG, LAST_ARG[:-1] + ' "bogus:=")')
    script = ("set -euo pipefail\n"
              'demo_say() { echo "[demo] $*"; }\n'
              'demo_warn() { echo "[demo] WARNING: $*"; }\n'
              'DEMO_VIEW_CPUSET=""\n' + body + "\n"
              't0=$(date +%s.%N)\n'
              "wam_bridge_start\n"
              'if wam_bridge_ready; then echo "GATE=open"; else echo "GATE=closed"; exit 0; fi\n'
              "rc=0; wam_bridge_wait || rc=$?\n"
              'echo "RC=$rc ELAPSED=$(python3 -c "import time,sys; print(round(time.time()-float(sys.argv[1]),1))" "$t0")"\n')
    env = dict(os.environ, ROS_DOMAIN_ID="43", DEMO_REPO=tmp, DEMO_LOG_DIR=os.path.join(tmp, "data", "logs"),
               WAM_SHM_PROFILE=profile(), WAM_HOST="127.0.0.1", WAM_PORT="8799",
               WAM_INSTRUCTION="Press down the lever handle, then push the door open.", WAM_TASK_ID="open_room_door",
               WAM_TASK_VERSION="mujoco-open_room_door-runtime-v1", WAM_SESSION=f"launch-test-{name}",
               WAM_MAX_CHUNKS="1000", WAM_MAX_WALL_S="0", WAM_SAVE_IMAGES="0", WAM_GO2_LATENCY_MS="0", WAM_CALIB="",
               WAM_OUTPUT_DIR=os.path.join(tmp, "bridge"), WAM_START_TIMEOUT_S="3", WAM_SETTLE_S="2.0",
               WAM_BRIDGE_READY_S="30", WAM_BRIDGE_T0="0")
    env.pop("ROS_LOCALHOST_ONLY", None)
    qlm = os.path.join(REPO, "third_party", "QLM-Bench", "src")     # as wam_task.sh wam_check_qlm exports it
    env["PYTHONPATH"] = qlm + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(extra_env or {})
    t = time.time()
    r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=180)
    out = r.stdout + r.stderr
    log = ""
    logs = [f for f in os.listdir(os.path.join(tmp, "data", "logs")) if f.startswith("wam_bridge_")]
    if logs:
        log = open(os.path.join(tmp, "data", "logs", logs[0]), encoding="utf-8", errors="replace").read()
    return out, log, time.time() - t, tmp


def leftovers():
    r = subprocess.run(["pgrep", "-fa", "[l]ib/wam_go2_deploy/(ego_stream|wam_bridge)"], capture_output=True, text=True)
    return r.stdout.strip()


def main():
    if leftovers():
        print("ABORT: a bridge / ego_stream is already running:\n" + leftovers())
        return 3
    print("== bridge (ego_stream + wam_bridge)")
    out, log, wall, tmp = run("bridge")
    check("malformed launch argument" not in out + log, "ros2 launch accepted every argument")
    check("GATE=open" in out, "the bridge's start line opened the stand-up gate")
    check("ego_stream" in log, "ego_stream was started")
    m = re.search(r"RC=(\d+) ELAPSED=([0-9.]+)", out)
    check(bool(m) and m.group(1) == "4", f"wam_bridge_wait returned the bridge's exit 4 (could_not_start): {m.group(0) if m else out[-300:]}")
    check(bool(m) and float(m.group(2)) < 45, f"the launch ended with the bridge ({m.group(2) if m else '?'} s, start timeout 3 s)")
    check("could_not_start" in out, "the summary's stop_reason was reported")
    time.sleep(1.0)
    check(not leftovers(), "no ego_stream / wam_bridge left running")

    print("== calib (a calibration file in WAM_CALIB)")
    calib = os.path.join(tempfile.gettempdir(), f"wam_calib_{os.getpid()}.json")
    open(calib, "w").write("{}")
    out, log, wall, tmp = run("calib", extra_env={"WAM_CALIB": calib})
    check("malformed launch argument" not in out + log, "ros2 launch accepted every argument incl. calib")
    check("GATE=open" in out and "ego_stream" in log, "the gate opened; ego_stream was started")
    m = re.search(r"RC=(\d+)", out)
    check(bool(m) and m.group(1) == "4", f"exit 4 from the summary: {m.group(0) if m else out[-300:]}")
    time.sleep(1.0)
    check(not leftovers(), "nothing left running")

    print("== wrong_arg (an argument ros2 launch refuses)")
    out, log, wall, tmp = run("wrong_arg", bad_arg=True)
    check("GATE=closed" in out, "the gate closed: the robot would not be stood up")
    check(wall < 20, f"... at once ({wall:.1f} s)")
    check(not leftovers(), "nothing left running")

    if FAILURES:
        print("==== FAILURES ====\n" + "\n".join(" - " + f for f in FAILURES))
        return 1
    print("ALL BRIDGE LAUNCH CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
