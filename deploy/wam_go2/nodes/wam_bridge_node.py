"""wam_bridge: the real-robot client of the WAM-Policy policy server.

Runs one episode of the qlm-live-v1 protocol (wam_go2/wam/episode.py) against
a policy server on the laptop (``python -m wam_policy.inference.serve`` of this
repository, or the mock, ``ros2 run wam_go2_deploy wam_mock_service``), with the
RAMBO controller already in POLICY (the task wrapper scripts/demo/wam_task.sh
brings it there).

Inputs (subscriptions only):
  rambo/state                      the controller's state stream (wam_go2/state_stream.py)
  /camera/go2/ego/image_raw        ego_stream: rgb8 640x360 (~14 fps), sent as is (the 640x360 profile),
                                   header.stamp = capture time on the laptop clock, frame_id tf:<counter>
  /camera/d435i/sim/color/image_raw  the real-to-sim D435i stream (scripts/x5/d435i_sim_camera.py --scale 0.5:
                                   640x360 in the sim camera's model)
Outputs: cmd_vel, rambo/ee_pos_cmd, rambo/ee_force_cmd (always zero) at the
protocol's 20 ms command grid; zero velocity + the last FL target while the
policy computes. It never sends FSM requests: the wrapper requests DAMP when
the bridge exits (Ctrl+C in the wrapper's terminal stops the episode).

Records go to ``output_dir/<session>_<epoch>/``: events.jsonl (every RPC, chunk,
sample, hold and stop with wall times), summary.json, and with save_images the
PNG of every sample pair. Exit codes: 0 budget reached, 1 execution error,
2 the controller left POLICY, 4 could not start (no streams / not in POLICY),
6 observation gap, 7 policy server failure, 8 prediction refused, 64 bad
parameters or QLM-Bench not importable, 130 interrupted.

Start: the node is launched BEFORE the entry sequence (scripts/demo/wam_task.sh
does) and waits up to ``start_timeout_s`` for every source, the controller in
POLICY and settled for ``settle_s``, and no other cmd_vel publisher (the sequence
has exited) - node_support.start_gate. Its own Python start-up therefore never
coincides with POLICY (a start-up CPU burst next to the controller can trip the
50 ms control-loop guard).

The big local image streams need the shared-memory FastDDS profile
(CAMERA_DDS_SHM=1, config/fastdds_wired_shm.xml) for this node and the camera
nodes alike.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time

import rclpy
from geometry_msgs.msg import Point, Twist, Vector3
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String

from .. import state_stream
from ..realtime import freeze_startup_heap
from ..wam import node_support as ns
from ..wam import settings
from ..wam.observation import FrameRecord, ObservationBuffers

DEFAULT_INSTRUCTION = settings.INSTRUCTION


class RosRuntime:
    """episode.Runtime on top of the node's buffers and publishers."""

    def __init__(self, node: "WamBridgeNode"):
        self.node = node
        self.lock = node.lock
        self.buffers = node.buffers
        self.last_ee = None
        self.commands_published = 0
        self.interrupt_flag = threading.Event()
        self.events_path = None
        self._events_file = None
        self.events_written = 0

    # clock ----------------------------------------------------------------
    def now(self) -> float:
        return time.time()

    def sleep_until(self, t: float) -> None:
        # Runs on the driver thread, never inside a ROS callback: the executor spins in
        # its own thread, and the 20 ms command schedule is wall-clock by design.
        while True:
            dt = t - time.time()
            if dt <= 0.0:
                return
            time.sleep(min(dt, 0.002))

    # commands ---------------------------------------------------------------
    def command(self, native9) -> None:
        twist = Twist()
        if native9 is not None:
            twist.linear.x, twist.linear.y, twist.angular.z = float(native9[0]), float(native9[1]), float(native9[2])
            self.last_ee = [float(native9[3]), float(native9[4]), float(native9[5])]
        self.node.vel_pub.publish(twist)
        if self.last_ee is not None:
            p = Point()
            p.x, p.y, p.z = self.last_ee
            self.node.ee_pub.publish(p)
        self.node.force_pub.publish(Vector3())
        self.commands_published += 1

    # controller state -------------------------------------------------------
    def step_ok(self):
        with self.lock:
            sample, received = self.node.last_state, self.node.last_state_received
        if sample is None:
            return False, "no rambo/state received"
        age = time.time() - received
        if age > self.node.state_timeout_s:
            return False, f"rambo/state silent for {age * 1e3:.0f} ms"
        if sample.tripped:
            return False, f"controller safety latched in {sample.mode}"
        if sample.mode != "POLICY":
            return False, f"controller mode {sample.mode}"
        if not sample.fresh:
            return False, "controller reports a stale LowState"
        return True, ""

    def ready(self, grid_wall, tolerances):
        with self.lock:
            return self.buffers.ready(grid_wall, tolerances)

    def resolve(self, **kwargs):
        with self.lock:
            sample = self.buffers.resolve(**kwargs)
        if self.node.save_images:
            self.node.save_sample_images(sample)
        return sample

    def last_action18(self):
        with self.lock:
            sample = self.node.last_state
        return None if sample is None else list(sample.action18)

    def latest_ego_time(self):
        with self.lock:
            return self.buffers.latest_ego_time()

    # records ------------------------------------------------------------------
    def open_records(self, path):
        self.events_path = path
        self._events_file = open(path, "a", encoding="utf-8")

    def record(self, event, payload):
        row = {"wall": time.time(), "event": event, **payload}
        if event == "execute":                      # 16 x 9 numbers per chunk are enough; keep rows compact
            row = dict(row, actions=[[round(float(v), 5) for v in a] for a in payload.get("actions", [])])
        if self._events_file is not None:
            self._events_file.write(json.dumps(row, allow_nan=False, default=str) + "\n")
            self._events_file.flush()
            self.events_written += 1
        if event in ("hold", "executed", "rejected", "error", "feedback_terminal", "end_failed", "policy_reset"):
            self.node.get_logger().info(f"{event}: {json.dumps(payload, default=str)[:300]}")

    def close_records(self):
        if self._events_file is not None:
            self._events_file.close()
            self._events_file = None

    def interrupted(self):
        return self.interrupt_flag.is_set()


class WamBridgeNode(Node):
    def __init__(self):
        super().__init__("wam_bridge")
        d = self.declare_parameter
        d("host", "127.0.0.1"); d("port", 8862); d("rpc_timeout_s", 30.0)
        d("session_id", "go2-real"); d("episode_id", ""); d("reset_epoch", 0)
        d("instruction", DEFAULT_INSTRUCTION); d("task_id", "push_box"); d("task_version", "mujoco-push_box-runtime-v1")
        d("policy_seed", 42); d("max_chunks", 1000); d("max_wall_s", 0.0); d("sample_wait_s", 0.30)
        d("ego_tolerance_s", 0.040); d("d435i_tolerance_s", 0.020); d("state_tolerance_s", 0.015)
        d("bound_vx", 0.75); d("bound_vy", 0.75); d("bound_wz", 0.75)
        d("bound_ee_x", [0.0, 0.65]); d("bound_ee_y", [-0.15, 0.35]); d("bound_ee_z", [-0.05, 0.50])
        d("ego_topic", "/camera/go2/ego/image_raw"); d("d435i_topic", "/camera/d435i/sim/color/image_raw")
        d("d435i_stamp_offset_ms", 0.0); d("state_timeout_s", 0.06); d("keep_s", 3.0)
        d("start_timeout_s", 15.0); d("settle_s", 2.0); d("output_dir", "~/.ros/wam_bridge"); d("save_images", False)
        g = lambda name: self.get_parameter(name).value  # noqa: E731
        self.params = {name: g(name) for name in (
            "host", "port", "rpc_timeout_s", "session_id", "episode_id", "reset_epoch", "instruction", "task_id",
            "task_version", "policy_seed", "max_chunks", "max_wall_s", "sample_wait_s", "ego_tolerance_s",
            "d435i_tolerance_s", "state_tolerance_s", "bound_vx", "bound_vy", "bound_wz", "bound_ee_x", "bound_ee_y",
            "bound_ee_z", "ego_topic", "d435i_topic", "d435i_stamp_offset_ms", "state_timeout_s", "keep_s",
            "start_timeout_s", "settle_s", "output_dir", "save_images")}
        self.state_timeout_s = float(self.params["state_timeout_s"])
        self.d435i_offset_s = float(self.params["d435i_stamp_offset_ms"]) * 1e-3
        self.save_images = bool(self.params["save_images"])
        self.lock = threading.Lock()
        self.buffers = ObservationBuffers(keep_s=float(self.params["keep_s"]))
        self.size_problems = {}        # camera -> why its last frame was refused (shown by the start gate)
        self.last_state = None
        self.last_state_received = None
        self.last_mode = None
        self.counters = {"state": 0, "state_bad": 0, "ego": 0, "ego_bad": 0, "d435i": 0, "d435i_bad": 0}
        self.images_dir = None
        self.vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.ee_pub = self.create_publisher(Point, "rambo/ee_pos_cmd", 10)
        self.force_pub = self.create_publisher(Vector3, "rambo/ee_force_cmd", 10)
        self.create_subscription(Float64MultiArray, state_stream.TOPIC, self._on_state, 50)
        self.create_subscription(String, "rambo/mode", self._on_mode, 10)
        self.create_subscription(Image, str(self.params["ego_topic"]), self._on_ego, qos_profile_sensor_data)
        self.create_subscription(Image, str(self.params["d435i_topic"]), self._on_d435i, qos_profile_sensor_data)

    # subscriptions -----------------------------------------------------------
    def _on_state(self, msg):
        try:
            sample = state_stream.decode(msg.data)
        except ValueError:
            self.counters["state_bad"] += 1
            return
        with self.lock:
            self.last_state, self.last_state_received = sample, time.time()
            self.buffers.add_state(sample.state_record())
            self.counters["state"] += 1

    def _on_mode(self, msg):
        self.last_mode = msg.data.strip()

    def _on_image(self, msg, camera, offset_s):
        try:
            rgb = ns.image_to_rgb(height=msg.height, width=msg.width, step=msg.step, encoding=msg.encoding, data=msg.data)
            rec = FrameRecord(capture_time=ns.stamp_to_float(msg.header.stamp.sec, msg.header.stamp.nanosec) - offset_s,
                              frame_id=ns.frame_id_from_header(msg.header.frame_id, msg.header.stamp.sec,
                                                               msg.header.stamp.nanosec),
                              image=rgb, receive_time=time.time())
            with self.lock:
                self.buffers.add_frame(camera, rec)
                self.counters[{"ego": "ego", "task_centric": "d435i"}[camera]] += 1
        except ValueError:
            self.counters[{"ego": "ego_bad", "task_centric": "d435i_bad"}[camera]] += 1
            problem = ns.frame_size_problem(msg.width, msg.height, camera)
            if problem:
                self.size_problems[camera] = problem

    def _on_ego(self, msg):
        self._on_image(msg, "ego", 0.0)

    def _on_d435i(self, msg):
        self._on_image(msg, "task_centric", self.d435i_offset_s)

    # helpers ---------------------------------------------------------------------
    def streams_report(self):
        with self.lock:
            newest = self.buffers.newest_times()
            mode = None if self.last_state is None else self.last_state.mode
        now = time.time()
        return {k: (None if v is None else round(now - v, 3)) for k, v in newest.items()} | {"mode": mode,
                                                                                            "counters": dict(self.counters)}

    def wait_for_streams(self, timeout_s: float, settle_s: float = 0.0, interrupted=lambda: False) -> str:
        """'' once the bridge may take over (node_support.start_gate: every source recent, the
        controller in POLICY and settled, no other cmd_vel publisher), else the last reason -
        or "interrupted". Logs what it is waiting for whenever that changes."""
        deadline = time.time() + timeout_s
        policy_since = None
        reason, logged = "not started", None
        while time.time() < deadline:
            if interrupted():
                return "interrupted"
            with self.lock:
                newest = self.buffers.newest_times()
                sample = self.last_state
            now = time.time()
            in_policy = sample is not None and sample.mode == "POLICY" and not sample.tripped
            policy_since = None if not in_policy else (now if policy_since is None else policy_since)
            others = max(0, self.count_publishers(self.vel_pub.topic_name) - 1)
            reason = ns.start_gate(newest=newest, mode=None if sample is None else sample.mode,
                                   tripped=sample is not None and bool(sample.tripped), now=now,
                                   policy_since=policy_since, settle_s=settle_s, other_cmd_vel_publishers=others,
                                   size_problems=dict(self.size_problems))
            if not reason:
                return ""
            if reason != logged:
                self.get_logger().info(f"waiting: {reason}")
                logged = reason
            time.sleep(0.1)          # main thread polling the buffers; the executor spins elsewhere
        return reason

    def save_sample_images(self, sample):
        if self.images_dir is None:
            return
        try:
            import cv2
            for name, image in (("go2_ego", sample.ego.image), ("d435i_rgb_task", sample.d435i.image)):
                cv2.imwrite(os.path.join(self.images_dir, f"{sample.physics_step:06d}_{name}.png"), image[:, :, ::-1])
        except Exception as exc:
            self.get_logger().warn(f"sample images not saved: {exc}")


def main(args=None):
    try:
        from qlm_bench.live import RPCClient
    except ImportError as error:
        print(f"wam_bridge: the QLM-Bench live package is not importable ({error}): put the src/ of a QLM-Bench "
              "checkout on PYTHONPATH (QLM_BENCH_SRC in configs/site.env) and install its dependencies "
              "(requirements.txt).", file=sys.stderr)
        return 64
    from ..wam.episode import EpisodeAbort, EpisodeDriver
    rclpy.init(args=args, signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    node = WamBridgeNode()
    runtime = RosRuntime(node)
    p = node.params
    try:
        bounds = ns.bounds_from_params(p)
        tolerances = ns.tolerances_from_params(p)
        session_id, episode_id, epoch = ns.episode_identity(session_id=str(p["session_id"]), episode_id=str(p["episode_id"]),
                                                            reset_epoch=int(p["reset_epoch"]), now=time.time())
    except ValueError as exc:
        node.get_logger().error(f"bad parameters: {exc}")
        node.destroy_node(); rclpy.try_shutdown()
        return 64
    out_dir = os.path.join(os.path.expanduser(str(p["output_dir"])), f"{session_id}_{epoch}")
    os.makedirs(out_dir, exist_ok=True)
    if node.save_images:
        node.images_dir = os.path.join(out_dir, "images")
        os.makedirs(node.images_dir, exist_ok=True)
    runtime.open_records(os.path.join(out_dir, "events.jsonl"))
    summary = {"schema": "wam-bridge-run-v1", "parameters": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                                                              for k, v in p.items()},
               "rgb_profile": settings.RGB_PROFILE,
               "session_id": session_id, "episode_id": episode_id, "reset_epoch": epoch, "output_dir": out_dir,
               "started_at": time.time(), "exit_code": None}

    def write_summary():
        summary["streams"] = node.streams_report()
        summary["commands_published"] = runtime.commands_published
        summary["events_written"] = runtime.events_written
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

    def on_signal(signum, _frame):
        runtime.interrupt_flag.set()
        node.get_logger().warn(f"{signal.Signals(signum).name}: stopping after the current step (robot holds)")
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)

    # the start-up heap out of the cyclic GC before the episode (wam_go2/realtime.py), so a full collection
    # never stalls the 20 ms command grid
    frozen, collect_ms = freeze_startup_heap()
    node.get_logger().info(f"GC: start-up heap frozen ({frozen} objects, collect {collect_ms:.1f} ms)")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    stop_spinning = threading.Event()

    def spin():                       # joinable: a daemon thread still inside spin() at exit aborts the process
        while rclpy.ok() and not stop_spinning.is_set():
            executor.spin_once(timeout_sec=0.05)
    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()
    code = 1
    started = False                   # nothing is ever published unless the start gate passed
    try:
        node.get_logger().info(f"waiting for rambo/state, ego and D435i sim frames, the controller in POLICY for "
                               f"{float(p['settle_s']):.1f} s and the entry sequence to finish "
                               f"(up to {float(p['start_timeout_s']):.0f} s) ...")
        reason = node.wait_for_streams(float(p["start_timeout_s"]), settle_s=float(p["settle_s"]),
                                       interrupted=runtime.interrupted)
        if reason == "interrupted":
            node.get_logger().warn("interrupted before the start; nothing was commanded")
            summary["stop_reason"] = "interrupted"
            code = 130
            return code
        if reason:
            node.get_logger().error(f"cannot start: {reason}; streams: {node.streams_report()}")
            summary["stop_reason"] = "could_not_start: " + reason
            code = 4
            return code
        started = True
        client = RPCClient(str(p["host"]), int(p["port"]), timeout_s=float(p["rpc_timeout_s"]), retries=1)
        driver = EpisodeDriver(runtime, client, session_id=session_id, episode_id=episode_id, reset_epoch=epoch,
                               instruction=str(p["instruction"]), task_id=str(p["task_id"]),
                               task_version=str(p["task_version"]), policy_seed=int(p["policy_seed"]),
                               service_timeout_s=float(p["rpc_timeout_s"]), tolerances=tolerances, bounds=bounds,
                               sample_wait_s=float(p["sample_wait_s"]), max_chunks=int(p["max_chunks"]),
                               max_wall_s=None if float(p["max_wall_s"]) <= 0.0 else float(p["max_wall_s"]))
        node.get_logger().info(f"episode {episode_id} epoch {epoch}: reset (cold start) ...")
        try:
            reply = driver.reset_episode()
        except EpisodeAbort as abort:
            node.get_logger().error(f"reset failed: {abort}")
            summary["stop_reason"] = abort.reason
            summary["detail"] = abort.detail
            code = {"observation_gap": 6, "rpc_error": 7}.get(abort.reason.split(":")[0], 1)
            return code
        node.get_logger().info(f"policy ready: {json.dumps(reply, default=str)[:200]}; running chunks "
                               f"(max {int(p['max_chunks'])}); Ctrl+C in the wrapper ends it")
        result = driver.run()
        summary.update({"stop_reason": result.reason, "detail": result.detail, "chunks": result.chunks,
                        "commands_completed": result.commands_completed, "ticks": result.ticks,
                        "wall_s": result.wall_s, "holds_s": result.holds_s,
                        "hold_s_max": max(result.holds_s) if result.holds_s else None})
        node.get_logger().info(f"episode over: {result.reason} ({result.detail}) after {result.chunks} chunks, "
                               f"{result.commands_completed} commands, {result.wall_s:.1f} s; holds max "
                               f"{(max(result.holds_s) if result.holds_s else 0.0):.2f} s")
        code = result.exit_code
        return code
    finally:
        if started:
            runtime.command(None)     # never leave the robot commanded
        summary["exit_code"] = code
        summary["ended_at"] = time.time()
        try:
            write_summary()
        finally:
            runtime.close_records()
            stop_spinning.set()
            spinner.join(timeout=3.0)
            executor.shutdown(timeout_sec=1.0)
            node.destroy_node()
            if rclpy.ok():
                rclpy.try_shutdown()
            print(f"wam_bridge: exit {code}, records in {out_dir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
