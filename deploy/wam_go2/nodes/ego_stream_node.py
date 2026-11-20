"""ego_stream: the Go2 front camera as a timed sensor_msgs/Image stream.

Subscribes to the firmware's raw ``/frontvideostream`` (one resolution, default
360p: what this firmware streams), decodes the H.264 access units and publishes
every frame as ``rgb8`` on ``/camera/go2/ego/image_raw`` with ``header.stamp`` =
the frame's time on the laptop clock from the arrival-envelope time model
(wam_go2/ego_stream.py) minus the calibrated Go2 latency, and
``header.frame_id`` = ``tf:<time_frame>`` (the stream's frame counter, the
strictly increasing sensor id the policy protocol wants). Until the time model
has enough messages (~2 s) nothing is published. A JSON status line goes to
``/camera/go2/ego/status`` once a second (counters, the time model, whether the
latency is calibrated).

The 640x360x3 frames at ~14 fps (10 MB/s) need the shared-memory FastDDS
profile between local processes (``CAMERA_DDS_SHM=1``), like the D435i sim
stream. Subscriber only on robot topics; publishes nothing the robot reads.

    ros2 run wam_go2_deploy ego_stream --ros-args -p go2_latency_ms:=44.0
"""
from __future__ import annotations

import array
import collections
import json
import os
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ..ego_stream import EgoTimeModel, Go2Decoder, parse_frontvideo, to_rgb


def latency_from_calib(path: str):
    """go2_ego_latency_ms from a sync_check.py / pair_report.py --write-calib JSON, or None."""
    with open(path, "rb") as f:
        cal = json.loads(f.read()).get("model_input", {}).get("calibration", {})
    value = cal.get("go2_ego_latency_ms")
    return None if value is None else float(value)


class EgoStreamNode(Node):
    def __init__(self):
        super().__init__("ego_stream", start_parameter_services=False)
        self.declare_parameter("res", 360)
        self.declare_parameter("go2_latency_ms", 0.0)
        self.declare_parameter("calib", "")
        self.declare_parameter("topic", "/camera/go2/ego/image_raw")
        self.declare_parameter("status_topic", "/camera/go2/ego/status")
        self.res = int(self.get_parameter("res").value)
        latency_ms = float(self.get_parameter("go2_latency_ms").value)
        calib = str(self.get_parameter("calib").value)
        self.calibration = "parameter" if latency_ms != 0.0 else "none"
        if calib:
            value = latency_from_calib(os.path.expanduser(calib))
            if value is None:
                self.get_logger().warn(f"calib {calib}: no go2_ego_latency_ms; ego times stay receive-time estimates")
            else:
                latency_ms = value
                self.calibration = f"calib:{os.path.basename(calib)}"
        self.model = EgoTimeModel(latency_ms * 1e-3)
        self.decoder = Go2Decoder()
        self.pub = self.create_publisher(Image, str(self.get_parameter("topic").value), qos_profile_sensor_data)
        self.status_pub = self.create_publisher(String, str(self.get_parameter("status_topic").value), 10)
        from unitree_go.msg import Go2FrontVideoData
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=30, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(Go2FrontVideoData, "/frontvideostream", self._on_frontvideo, qos, raw=True)
        self.lock = threading.Lock()
        self.queue = collections.deque(maxlen=8)
        self.event = threading.Event()
        self.counters = {"messages": 0, "other_res": 0, "decoded": 0, "published": 0, "no_time_model": 0,
                         "decode_errors": 0, "dropped": 0}
        self.last_published = None
        self._stopping = False
        self._thread = threading.Thread(target=self._decode_loop, daemon=True)
        self._thread.start()
        self.create_timer(1.0, self._status)
        self.get_logger().info(f"ego_stream: /frontvideostream {self.res}p -> {self.pub.topic_name} rgb8; "
                               f"Go2 latency {latency_ms:.1f} ms ({self.calibration})")

    def _on_frontvideo(self, raw):
        arrival = time.time()
        try:
            tf, res, data = parse_frontvideo(bytes(raw))
        except Exception:
            return
        self.counters["messages"] += 1
        if res != self.res:
            self.counters["other_res"] += 1
            return
        with self.lock:
            self.model.add(tf, arrival)
        if len(self.queue) == self.queue.maxlen:
            self.counters["dropped"] += 1
        self.queue.append((tf, arrival, data))
        self.event.set()

    def stop(self, timeout_s: float = 1.0):
        """End the decode thread (before the context is shut down under it)."""
        self._stopping = True
        self.event.set()
        self._thread.join(timeout=timeout_s)

    def _decode_loop(self):
        while rclpy.ok() and not self._stopping:
            self.event.wait(0.5)
            while self.queue and not self._stopping:
                tf, arrival, data = self.queue.popleft()
                try:
                    frames = self.decoder.decode(data, tf)
                except Exception:
                    self.counters["decode_errors"] += 1
                    continue
                for ftf, bgr in frames:
                    self.counters["decoded"] += 1
                    with self.lock:
                        t = self.model.time_of(ftf)
                    if t is None:
                        self.counters["no_time_model"] += 1
                        continue
                    self._publish(ftf, t, bgr)
            self.event.clear()

    def _publish(self, tf, t, bgr):
        rgb = to_rgb(bgr)
        msg = Image()
        msg.header.stamp.sec = int(t)
        msg.header.stamp.nanosec = int(round((t - int(t)) * 1e9)) % 1_000_000_000
        msg.header.frame_id = f"tf:{tf}"
        msg.height, msg.width = rgb.shape[0], rgb.shape[1]
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = rgb.shape[1] * 3
        msg.data = array.array("B", rgb.tobytes())   # rosidl's fast path; bytes are checked element by element (~27 ms)
        try:
            self.pub.publish(msg)
        except Exception:                # the context was shut down under this thread (Ctrl+C): end quietly
            self._stopping = True
            return
        self.counters["published"] += 1
        self.last_published = (tf, t, time.time())

    def _status(self):
        with self.lock:
            report = self.model.report()
        status = {"res": self.res, "counters": dict(self.counters), "time_model": report,
                  "calibration": self.calibration, "decoder_errors": self.decoder.errors,
                  "keyframe_seen": self.decoder.keyframe_seen, "last_published": self.last_published}
        self.status_pub.publish(String(data=json.dumps(status, allow_nan=False, default=str)))


def main(args=None):
    from rclpy.executors import ExternalShutdownException
    rclpy.init(args=args)
    node = EgoStreamNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):   # Ctrl+C / launch shutdown: a clean exit 0
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
