#!/usr/bin/env python3
"""End-to-end test of the WAM bridge against the mock robot, the real controller
node, synthetic camera streams and the protocol-faithful mock policy server.

    export ROS_DOMAIN_ID=42 && unset FASTRTPS_DEFAULT_PROFILES_FILE ROS_LOCALHOST_ONLY
    python3 test/integration/mock_wam_test.py [--scenarios happy operator_stop ego_loss]

Needs a sourced ROS 2 env, the built package (wam_bridge / controller executables)
and the QLM-Bench live package (QLM_BENCH_SRC = the src/ of a QLM-Bench checkout).
The camera frames need shared memory between the
local processes: this test writes a FastDDS profile with SHM + UDP on 127.0.0.1
and exports it to every process it starts.

The bridge is started BEFORE the entry sequence, as scripts/demo/wam_task.sh does:
it waits for the streams, POLICY + settle and the sequence's exit (its start-up burst
next to a controller in POLICY can trip the 50 ms control-loop guard). The
controller runs with torch capped at 4 threads like under the demo scripts. The mock
policy server runs in its own process (`python3 -m wam_go2.wam.mock_service --record`,
as the real server and the demo's WAM_MOCK=1 do): its 30 MB feedback messages in the
test process would starve the mock robot's 500 Hz LowState timer (a 53 ms gap trips
the controller's source-stale guard); the events it accepted are read from the record file.

Scenarios (a fresh controller node each):
  happy          wam_entry (STAND -> FL target -> POLICY), then the bridge runs 3 chunks
                 of the mock policy (vx 0.10): exit 0, 48 commands on the 20 ms grid,
                 cmd_vel 0.10 during chunks and 0 in the holds, 3 feedback commits with
                 four samples each, the controller never left POLICY
  operator_stop  the test requests DAMP during the second chunk: the bridge exits 2
                 after a terminal prefix; nothing is published after the stop
  ego_loss       the ego stream stops after the first chunk: exit 6 (observation gap),
                 terminal feedback, the robot held (zero velocity)
  wrong_size     the D435i sim stream at 1280x720 (d435i_sim_camera.py --scale 1.0, the full export):
                 the bridge exits 4 at its start gate naming the size; the server is never called
  wrong_prompt   the mock serves another instruction: reset refused, the bridge exits 7 with
                 the server's reason; nothing was commanded
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
import rclpy

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from mock_robot_test import MockRobot, spin_for  # noqa: E402  (adds REPO to sys.path)
from mock_demo_test import (BASE_ARGS, SEQUENCER, Probe, begin_scenario, compress, end_scenario,  # noqa: E402
                            run_while_spinning, spin_until)

QLM_SRC = os.environ.get("QLM_BENCH_SRC", "")
if QLM_SRC and os.path.isdir(os.path.join(QLM_SRC, "qlm_bench")):
    sys.path.insert(0, QLM_SRC)
    os.environ["PYTHONPATH"] = QLM_SRC + os.pathsep + os.environ.get("PYTHONPATH", "")
from wam_go2.wam import mock_service as ms  # noqa: E402

SCENARIOS = ("happy", "operator_stop", "ego_loss", "wrong_size", "wrong_prompt")
from wam_go2.wam import settings  # noqa: E402

# the controller parameter scripts/demo/wam_task.sh adds for every WAM run: the commands taken unclipped,
# as the QLM-Bench MuJoCo simulation of the training data did
WAM_CONTROLLER = ("command_handling:=sim",)

INSTRUCTION = settings.INSTRUCTION          # push_box's instruction
KP, KD = 15.0, 0.8


def shm_profile_for_tests():
    """config/fastdds_wired_shm.xml with the loopback address instead of the wired IP."""
    src = os.path.join(REPO, "config", "fastdds_wired_shm.xml")
    text = open(src, encoding="utf-8").read().replace("@LAPTOP_IP@", "127.0.0.1")
    path = os.path.join(tempfile.gettempdir(), f"rambo_fastdds_shm_mock_{os.getpid()}.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


class Cameras:
    """The synthetic ego / D435i sim streams in their own process (synthetic_cameras.py), both
    640x360 as deployed (ego_stream ~14 fps, d435i_sim_camera.py --scale 0.5 at 60 fps):
    publishing camera frames from the mock robot's node would starve its 500 Hz LowState."""

    def __init__(self):
        self.control = tempfile.mkdtemp(prefix="wam_cams_")
        # one process per stream: a 70 ms timer next to 2.8 MB publishes jitters past the ego tolerance
        self.procs = [self._start(only, "640x360") for only in ("ego", "d435i")]
        self._ego_on = True

    def _start(self, only, d435i_size):
        return subprocess.Popen([sys.executable, os.path.join(HERE, "synthetic_cameras.py"),
                                 "--control-dir", self.control, "--only", only,
                                 "--ego-period", "0.070", "--d435i-period", f"{1 / 60}",
                                 "--d435i-size", d435i_size], start_new_session=True)

    def d435i_size(self, size):
        """Restart the D435i stream at another size ("1280x720" = the sim stream at --scale 1.0)."""
        self._stop(self.procs[1])
        self.procs[1] = self._start("d435i", size)

    @property
    def ego_on(self):
        return self._ego_on

    @ego_on.setter
    def ego_on(self, value):
        flag = os.path.join(self.control, "ego_off")
        if value:
            if os.path.exists(flag):
                os.remove(flag)
        else:
            open(flag, "w").close()
        self._ego_on = bool(value)

    def stop(self):
        for proc in self.procs:
            self._stop(proc)

    @staticmethod
    def _stop(proc):
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


class ServiceProcess:
    """The mock policy server in its own process, fresh per scenario; ``events()`` /
    ``state()`` come from its --record file."""

    def __init__(self, port, vx=0.10, ee=(0.44, 0.142, 0.16), latency_s=0.3, instruction=INSTRUCTION):
        fd, self.record = tempfile.mkstemp(prefix="wam_mock_service_", suffix=".jsonl")
        os.close(fd)
        self.log = tempfile.NamedTemporaryFile(mode="w+", prefix="wam_mock_service_", suffix=".log")
        self.proc = subprocess.Popen([sys.executable, "-m", "wam_go2.wam.mock_service", "--host", "127.0.0.1",
                                      "--port", str(port), "--instruction", instruction, "--vx", f"{vx}",
                                      "--ee", *[f"{v}" for v in ee], "--latency", f"{latency_s}",
                                      "--record", self.record],
                                     cwd=REPO, stdout=self.log, stderr=subprocess.STDOUT, text=True,
                                     start_new_session=True)
        deadline = time.time() + 15.0
        while time.time() < deadline and self.proc.poll() is None:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                return
            except OSError:
                time.sleep(0.1)
        self.log.seek(0)
        raise RuntimeError(f"mock policy server did not come up on 127.0.0.1:{port}: {self.log.read()[-500:]}")

    def events(self):
        return ms.read_events(self.record)

    def state(self):
        events = self.events()
        return events[-1]["state"] if events else "NEW"

    def stop(self):
        if self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait()
        self.log.close()
        try:
            os.remove(self.record)
        except OSError:
            pass


def start_bridge(output_dir, max_chunks, logf, extra=()):
    args = ["ros2", "run", "wam_go2_deploy", "wam_bridge", "--ros-args",
            "-p", "host:=127.0.0.1", "-p", f"port:={PORT}", "-p", "session_id:=mock-wam",
            "-p", f"instruction:={INSTRUCTION}", "-p", f"max_chunks:={max_chunks}",
            "-p", f"output_dir:={output_dir}", "-p", "start_timeout_s:=60.0", "-p", "rpc_timeout_s:=30.0",
            *extra]
    return subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT, text=True, start_new_session=True)


def stop_bridge(p):
    if p.poll() is None:
        os.killpg(os.getpgid(p.pid), signal.SIGINT)
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            p.wait()
    return p.returncode


def read_events(output_dir):
    runs = sorted(os.listdir(output_dir))
    if not runs:
        return [], None
    run = os.path.join(output_dir, runs[-1])
    events = []
    with open(os.path.join(run, "events.jsonl"), encoding="utf-8") as f:
        for line in f:
            events.append(json.loads(line))
    summary = json.load(open(os.path.join(run, "summary.json"), encoding="utf-8"))
    return events, summary


def enter_policy(robot, logf):
    rc = run_while_spinning(robot, [sys.executable, SEQUENCER, "--demo", "wam_entry", *BASE_ARGS],
                            timeout=90, stdout=logf)
    if rc != 0:
        logf.seek(0)
        print("---- wam_entry sequencer output ----")
        print("\n".join("  " + ln for ln in logf.read().strip().splitlines()[-30:]))
    return rc


def run_happy(robot, probe, cams, service, check):
    tag = "happy"
    proc = begin_scenario("SCENARIO happy (wam_entry, then 3 chunks of the mock policy)", robot, probe, KP, KD, extra=WAM_CONTROLLER)
    summary = {"scenario": tag}
    out = tempfile.mkdtemp(prefix="wam_bridge_happy_")
    try:
        probe.vel.clear()
        with tempfile.NamedTemporaryFile(mode="w+", prefix="wam_bridge_", suffix=".log") as blog:
            t0 = time.time()
            p = start_bridge(out, 3, blog)        # first, as the demo does: it waits for POLICY + settle + the sequence's exit
            with tempfile.NamedTemporaryFile(mode="w+", prefix="wam_entry_", suffix=".log") as logf:
                rc = enter_policy(robot, logf)
            check(rc == 0, f"{tag}: wam_entry sequencer exit {rc}")
            spin_until(robot, lambda: p.poll() is not None, 120.0)
            rc = stop_bridge(p)
            blog.seek(0)
            bridge_out = blog.read()
        print("---- bridge output (tail) ----")
        print("\n".join("  " + ln for ln in bridge_out.strip().splitlines()[-25:]))
        summary["bridge_rc"] = rc
        summary["bridge_time_s"] = round(time.time() - t0, 1)
        check(rc == 0, f"{tag}: bridge exit {rc} (expected 0)")
        events, run_summary = read_events(out)
        kinds = [e["event"] for e in events]
        summary["chunks"] = run_summary and run_summary.get("chunks")
        check(run_summary is not None and run_summary.get("chunks") == 3, f"{tag}: chunks {summary['chunks']}")
        check(kinds.count("executed") == 3 and kinds.count("feedback") == 3, f"{tag}: events {compress(kinds)}")
        samples = [e for e in events if e["event"] == "sample"]
        check(len(samples) == 1 + 12, f"{tag}: {len(samples)} samples (expected cold start + 12)")
        offs = [abs(e["ego"]["offset_s"]) for e in samples] or [float("nan")]
        summary["ego_offset_max_ms"] = round(1e3 * max(offs), 1)
        check(max(offs) <= 0.040 + 1e-3, f"{tag}: ego sample offset {max(offs) * 1e3:.1f} ms")
        holds = [e["duration_s"] for e in events if e["event"] == "hold"]
        summary["holds_s"] = [round(h, 3) for h in holds]
        # server side: three feedback commits with four samples each at the right ticks
        fb = [e for e in service.events() if e["event"] == "feedback_commit"]
        check(len(fb) == 3 and all(len(e["samples"]) == 4 for e in fb), f"{tag}: server feedback commits {len(fb)}")
        last_ticks = [s["tick"] for s in fb[-1]["samples"]] if fb else None
        check(last_ticks == [360, 400, 440, 480], f"{tag}: last sample ticks {last_ticks}")
        # commands: 48 on a 20 ms grid at vx 0.10, zeros in the holds, controller never left POLICY
        vels = [(t, v) for t, v in probe.vel if t >= t0]
        moving = [t for t, v in vels if abs(v[0] - 0.10) < 1e-6]
        zeros = [t for t, v in vels if v == (0.0, 0.0, 0.0)]
        summary["cmd_vel_moving"], summary["cmd_vel_zero"] = len(moving), len(zeros)
        check(len(moving) == 48, f"{tag}: {len(moving)} moving cmd_vel (expected 48)")
        check(len(zeros) >= 4, f"{tag}: {len(zeros)} zero cmd_vel (holds)")
        if len(moving) >= 16:
            gaps = np.diff(moving[:16])
            summary["cmd_gap_ms"] = [round(float(np.median(gaps)) * 1e3, 2), round(float(gaps.max()) * 1e3, 2)]
            check(abs(np.median(gaps) - 0.020) < 0.003 and gaps.max() < 0.040,
                  f"{tag}: command spacing median {np.median(gaps) * 1e3:.1f} max {gaps.max() * 1e3:.1f} ms")
        modes = compress([m for _, m in probe.modes])
        summary["modes"] = modes
        check(modes[-1:] == ["POLICY"] and "DAMP" not in modes[1:], f"{tag}: modes {modes}")
        robot.request("DAMP")
        spin_for(robot, 1.0)
    finally:
        end_scenario(proc, summary)
    return summary


def run_operator_stop(robot, probe, cams, service, check):
    tag = "operator_stop"
    proc = begin_scenario("SCENARIO operator_stop (DAMP during the second chunk)", robot, probe, KP, KD, extra=WAM_CONTROLLER)
    summary = {"scenario": tag}
    out = tempfile.mkdtemp(prefix="wam_bridge_opstop_")
    try:
        probe.vel.clear()
        with tempfile.NamedTemporaryFile(mode="w+", prefix="wam_bridge_", suffix=".log") as blog:
            p = start_bridge(out, 10, blog)
            with tempfile.NamedTemporaryFile(mode="w+", prefix="wam_entry_", suffix=".log") as logf:
                rc = enter_policy(robot, logf)
            check(rc == 0, f"{tag}: wam_entry sequencer exit {rc}")
            # wait for the second chunk to start (17th moving command), then DAMP
            spin_until(robot, lambda: len([1 for _, v in probe.vel if abs(v[0] - 0.10) < 1e-6]) >= 20, 60.0)
            t_damp = time.time()
            robot.request("DAMP")
            spin_until(robot, lambda: p.poll() is not None, 60.0)
            rc = stop_bridge(p)
            blog.seek(0)
            bridge_out = blog.read()
        print("---- bridge output (tail) ----")
        print("\n".join("  " + ln for ln in bridge_out.strip().splitlines()[-15:]))
        summary["bridge_rc"] = rc
        check(rc == 2, f"{tag}: bridge exit {rc} (expected 2: controller left POLICY)")
        events, run_summary = read_events(out)
        check(run_summary is not None and run_summary.get("stop_reason") == "controller_left_policy",
              f"{tag}: stop reason {run_summary and run_summary.get('stop_reason')}")
        kinds = [e["event"] for e in service.events()]
        check(service.state() == "ENDED" and "terminal" in kinds, f"{tag}: server state {service.state()}, events {kinds}")
        late = [v for t, v in probe.vel if t > t_damp + 0.5 and v != (0.0, 0.0, 0.0)]
        check(not late, f"{tag}: {len(late)} nonzero cmd_vel after the stop")
        summary["completed"] = run_summary and run_summary.get("commands_completed")
    finally:
        end_scenario(proc, summary)
    return summary


def run_ego_loss(robot, probe, cams, service, check):
    tag = "ego_loss"
    proc = begin_scenario("SCENARIO ego_loss (ego stream stops after the first chunk)", robot, probe, KP, KD, extra=WAM_CONTROLLER)
    summary = {"scenario": tag}
    out = tempfile.mkdtemp(prefix="wam_bridge_egoloss_")
    try:
        probe.vel.clear()
        with tempfile.NamedTemporaryFile(mode="w+", prefix="wam_bridge_", suffix=".log") as blog:
            p = start_bridge(out, 10, blog)
            with tempfile.NamedTemporaryFile(mode="w+", prefix="wam_entry_", suffix=".log") as logf:
                rc = enter_policy(robot, logf)
            check(rc == 0, f"{tag}: wam_entry sequencer exit {rc}")
            spin_until(robot, lambda: len([1 for _, v in probe.vel if abs(v[0] - 0.10) < 1e-6]) >= 20, 60.0)
            cams.ego_on = False
            spin_until(robot, lambda: p.poll() is not None, 60.0)
            cams.ego_on = True
            rc = stop_bridge(p)
            blog.seek(0)
            bridge_out = blog.read()
        print("---- bridge output (tail) ----")
        print("\n".join("  " + ln for ln in bridge_out.strip().splitlines()[-15:]))
        summary["bridge_rc"] = rc
        check(rc == 6, f"{tag}: bridge exit {rc} (expected 6: observation gap)")
        events, run_summary = read_events(out)
        reason = run_summary and run_summary.get("stop_reason", "")
        check(str(reason).startswith("observation_gap:ego_frame"), f"{tag}: stop reason {reason}")
        check(service.state() == "ENDED", f"{tag}: service state {service.state()}")
        modes = compress([m for _, m in probe.modes])
        check(modes[-1:] == ["POLICY"], f"{tag}: the controller should still be in POLICY (holding): {modes}")
        robot.request("DAMP")
        spin_for(robot, 1.0)
    finally:
        end_scenario(proc, summary)
    return summary


def run_wrong_size(robot, probe, cams, service, check):
    tag = "wrong_size"
    proc = begin_scenario("SCENARIO wrong_size (D435i sim stream at 1280x720, the bridge expects 640x360)",
                          robot, probe, KP, KD, extra=WAM_CONTROLLER)
    summary = {"scenario": tag}
    out = tempfile.mkdtemp(prefix="wam_bridge_wrongsize_")
    try:
        cams.d435i_size("1280x720")
        spin_for(robot, 2.0)
        with tempfile.NamedTemporaryFile(mode="w+", prefix="wam_bridge_", suffix=".log") as blog:
            p = start_bridge(out, 3, blog, extra=("-p", "start_timeout_s:=6.0"))
            spin_until(robot, lambda: p.poll() is not None, 60.0)
            rc = stop_bridge(p)
            blog.seek(0)
            bridge_out = blog.read()
        print("---- bridge output (tail) ----")
        print("\n".join("  " + ln for ln in bridge_out.strip().splitlines()[-8:]))
        summary["bridge_rc"] = rc
        check(rc == 4, f"{tag}: bridge exit {rc} (expected 4: could not start)")
        check("task_centric frames are 1280x720" in bridge_out and "640x360" in bridge_out,
              f"{tag}: the start line does not name the frame size")
        check(service.events() == [], f"{tag}: the server was called: {service.events()[:2]}")
    finally:
        cams.d435i_size("640x360")
        spin_for(robot, 2.0)
        end_scenario(proc, summary)
    return summary


def run_wrong_prompt(robot, probe, cams, service, check):
    tag = "wrong_prompt"
    proc = begin_scenario("SCENARIO wrong_prompt (the mock serves another instruction)", robot, probe, KP, KD, extra=WAM_CONTROLLER)
    summary = {"scenario": tag}
    out = tempfile.mkdtemp(prefix="wam_bridge_wrongprompt_")
    try:
        probe.vel.clear()
        with tempfile.NamedTemporaryFile(mode="w+", prefix="wam_bridge_", suffix=".log") as blog:
            p = start_bridge(out, 3, blog)
            with tempfile.NamedTemporaryFile(mode="w+", prefix="wam_entry_", suffix=".log") as logf:
                rc = enter_policy(robot, logf)
            check(rc == 0, f"{tag}: wam_entry sequencer exit {rc}")
            spin_until(robot, lambda: p.poll() is not None, 60.0)
            rc = stop_bridge(p)
            blog.seek(0)
            bridge_out = blog.read()
        print("---- bridge output (tail) ----")
        print("\n".join("  " + ln for ln in bridge_out.strip().splitlines()[-8:]))
        summary["bridge_rc"] = rc
        check(rc == 7, f"{tag}: bridge exit {rc} (expected 7: the policy server refused the reset)")
        check("Instruction differs" in bridge_out, f"{tag}: the server's reason is not in the bridge log")
        check(service.state() == "NEW", f"{tag}: server state {service.state()} (the reset must be refused)")
        moving = [v for _, v in probe.vel if abs(v[0]) > 1e-9]
        check(not moving, f"{tag}: {len(moving)} nonzero cmd_vel published")
        robot.request("DAMP")
        spin_for(robot, 1.0)
    finally:
        end_scenario(proc, summary)
    return summary


PORT = 8766


def main():
    global PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    PORT = args.port
    if os.environ.get("ROS_DOMAIN_ID", "0") == "0":
        print("refusing to run on ROS domain 0 (a real robot may be connected); export ROS_DOMAIN_ID=42",
              file=sys.stderr)
        return 2
    profile = shm_profile_for_tests()
    os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = profile
    os.environ.pop("ROS_LOCALHOST_ONLY", None)
    # The demo scripts run the controller with torch capped at 4 threads (common.sh: 24 OpenMP
    # workers plus a background CPU burst can stall the 100 Hz loop by 50 ms). The bridge /
    # camera / sequencer start-ups here are such bursts: same cap for every child.
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    print(f"FastDDS profile for all processes: {profile}")

    print(f"mock policy server: its own process on 127.0.0.1:{PORT} per scenario (vx 0.10, 0.3 s latency per request)")

    rclpy.init()
    robot = MockRobot()
    probe = Probe(robot)
    cams = Cameras()
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
            print("  CHECK FAILED:", msg, flush=True)

    results = []
    try:
        spin_for(robot, 1.0)
        runners = {"happy": run_happy, "operator_stop": run_operator_stop, "ego_loss": run_ego_loss,
                   "wrong_size": run_wrong_size, "wrong_prompt": run_wrong_prompt}
        for name in args.scenarios:
            service = ServiceProcess(PORT, instruction="Push the box into the target area." if name == "wrong_prompt"
                                     else INSTRUCTION)
            try:
                results.append(runners[name](robot, probe, cams, service, check))
            finally:
                service.stop()
    finally:
        cams.stop()
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print("\n==== RESULTS ====")
    for r in results:
        print(r)
    if failures:
        print("\n==== FAILURES ====")
        for f in failures:
            print(" -", f)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
