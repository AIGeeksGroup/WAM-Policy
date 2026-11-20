#!/usr/bin/env python3
"""Republish the real D435i colour stream as the SIMULATION D435i camera (Real = Sim).

The simulation export in config/d435i_rgb_simulation/ is the D435i model (1280x720, fx = fy 924.28,
principal point (640, 360), no distortion, base-frame mount (0.30, 0, 0.34) m pitched 65 deg down).
The node runs on the laptop and resamples every real colour frame into that camera at half its
resolution by default (--scale 0.5: 640x360, K's first two rows halved = the dual-rgb-360p-v1 profile the
WAM-Policy models take) and publishes

  <ns>/sim/color/image_raw    the sim-camera image, same header (stamp, frame_id) and encoding
  <ns>/sim/color/camera_info  the sim camera's intrinsics with the same header
  <ns>/sim/color/valid_mask   mono8, 255 = seen by the real sensor, 0 = black fill (latched; re-sent
                              when the real calibration changes)

Pixels the real sensor does not see are black, never filled in; with a typical camera that is the
right 9 columns and bottom 23 rows (3.9 %) of the 1280x720 export, about 5 columns and 12 rows at the
640x360 default. A real stream smaller than the output camera is upsampled and logged as such (the
X5 runs CAMERA_PROFILE=640x360x60: the same size as the 640x360 output).

Subscribe-only towards the camera; it publishes images, never robot commands. Run it with the same
wired DDS environment as the other camera tools:

  CAMERA_DDS_SHM=1 source scripts/x5/laptop_camera_env.bash   # local consumers need shared memory too
  nice -n 10 python3 scripts/x5/d435i_sim_camera.py [--ns /camera/d435i] [--sim-json path]

It pins no CPU; OpenCV is limited to one thread.

The operator view of a run shows the raw picture; this stream is the model's D435i input (the
WAM bridge subscribes to <ns>/sim/color/image_raw).
"""

from __future__ import annotations

import argparse
import array
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from wam_go2 import sim_camera as sc  # noqa: E402

CHANNELS = {"rgb8": 3, "bgr8": 3, "mono8": 1}


def output_topics(ns):
    return {"image": f"{ns}/sim/color/image_raw", "info": f"{ns}/sim/color/camera_info",
            "mask": f"{ns}/sim/color/valid_mask"}


def sim_info_fields(sim):
    return sc.camera_info_fields(sim)


def msg_payload(img):
    """sensor_msgs/Image data for array `img`: an array.array('B'). rosidl's uint8[] setter checks
    every element of a bytes value in Python (107 ms per 1280x720 rgb8 frame); an array.array is
    taken as is (1 ms)."""
    return array.array("B", np.ascontiguousarray(img).tobytes())


def image_array(msg):
    """H x W (x C) uint8 view of a sensor_msgs/Image with an 8-bit encoding (honours row padding)."""
    ch = CHANNELS[msg.encoding]
    buf = np.frombuffer(msg.data, dtype=np.uint8)[:msg.height * msg.step]
    buf = buf.reshape(msg.height, msg.step)[:, :msg.width * ch]
    return buf.reshape(msg.height, msg.width, ch) if ch > 1 else buf


class Republisher:
    """ROS-free core: camera_info -> remap, image -> sim image; counts what it skips."""

    def __init__(self, sim, log=print, allow_distortion=False):
        self.sim = sim
        self.remapper = sc.SimRemapper(sim)
        self.log = log
        self.allow_distortion = bool(allow_distortion)
        self.stats = {"in": 0, "out": 0, "skipped_no_info": 0, "skipped_encoding": 0, "skipped_size": 0,
                      "skipped_malformed": 0, "bad_info": 0, "rebuilds": 0, "errors": 0}
        self.remap_ms = []
        self._mask_pending = False
        self._warned = set()
        self._info_ok = False

    def _warn_once(self, key, text):
        if key not in self._warned:
            self._warned.add(key)
            self.log(text)

    def on_info(self, msg):
        """Real colour camera_info; returns True when the remap was (re)built."""
        try:
            real = sc.camera_from_info(width=msg.width, height=msg.height, k=msg.k, d=msg.d,
                                       distortion_model=msg.distortion_model,
                                       allow_distortion=self.allow_distortion)
        except ValueError as e:
            self.stats["bad_info"] += 1
            self._info_ok = False
            self._warn_once(("info", str(e)), f"d435i_sim_camera: unusable camera_info, frames skipped: {e}")
            return False
        self._info_ok = True
        if not self.remapper.update(real):
            return False
        self.stats["rebuilds"] += 1
        self._mask_pending = True
        c = self.remapper.coverage
        self.log(f"d435i_sim_camera: real {real.width}x{real.height} fx {real.K[0, 0]:.1f} fy {real.K[1, 1]:.1f} "
                 f"cx {real.K[0, 2]:.1f} cy {real.K[1, 2]:.1f} ({real.distortion_model}) -> sim "
                 f"{self.sim.width}x{self.sim.height} fx {self.sim.K[0, 0]:.2f}: valid {100 * c['valid_fraction']:.1f} % "
                 f"(black: left {c['left_cols']} / right {c['right_cols']} cols, top {c['top_rows']} / "
                 f"bottom {c['bottom_rows']} rows)")
        if c["upsampled"]:
            self.log(f"d435i_sim_camera: WARNING the real stream is {real.width}x{real.height}, smaller than the sim "
                     f"camera's {self.sim.width}x{self.sim.height}: upsampled, not real {self.sim.width}x{self.sim.height} "
                     f"detail (X5 camera.conf CAMERA_PROFILE=640x360x60, scripts/x5/camera.conf.example)")
        return True

    def take_mask(self):
        """The validity mask once after each rebuild (to publish), else None."""
        if not self._mask_pending:
            return None
        self._mask_pending = False
        return self.remapper.mask_u8()

    def on_image(self, msg):
        """(sim image array, encoding) for a real colour frame, or None when it has to be skipped."""
        self.stats["in"] += 1
        if not self._info_ok or self.remapper.real is None:
            self.stats["skipped_no_info"] += 1
            return None
        if msg.encoding not in CHANNELS:
            self.stats["skipped_encoding"] += 1
            self._warn_once(("enc", msg.encoding), f"d435i_sim_camera: encoding {msg.encoding!r} not supported "
                            f"({', '.join(CHANNELS)}), frames skipped")
            return None
        real = self.remapper.real
        if (msg.width, msg.height) != (real.width, real.height):
            self.stats["skipped_size"] += 1
            self._warn_once(("size", msg.width, msg.height),
                            f"d435i_sim_camera: frame {msg.width}x{msg.height} but camera_info "
                            f"{real.width}x{real.height}, frames skipped until they agree")
            return None
        if msg.step < msg.width * CHANNELS[msg.encoding] or len(msg.data) < msg.height * msg.step:
            self.stats["skipped_malformed"] += 1
            self._warn_once(("malformed", msg.width, msg.height, msg.step),
                            f"d435i_sim_camera: malformed frame ({msg.width}x{msg.height} {msg.encoding}, "
                            f"step {msg.step}, {len(msg.data)} bytes), frames skipped")
            return None
        t0 = time.perf_counter()
        out = self.remapper.apply(image_array(msg))
        self.remap_ms.append((time.perf_counter() - t0) * 1e3)
        if len(self.remap_ms) > 2000:
            del self.remap_ms[:1000]
        self.stats["out"] += 1
        return out, msg.encoding

    def summary(self):
        s = self.stats
        c = self.remapper.coverage
        ms = (f", remap p50 {np.median(self.remap_ms):.1f} / max {np.max(self.remap_ms):.1f} ms"
              if self.remap_ms else "")
        cov = f", valid {100 * c['valid_fraction']:.1f} %" if c else ""
        return (f"d435i_sim_camera: in {s['in']} out {s['out']} skipped (no info {s['skipped_no_info']}, "
                f"encoding {s['skipped_encoding']}, size {s['skipped_size']}, malformed {s['skipped_malformed']}), "
                f"bad camera_info {s['bad_info']}, errors {s['errors']}, rebuilds {s['rebuilds']}{cov}{ms}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", default="/camera/d435i", help="realsense namespace (default /camera/d435i)")
    ap.add_argument("--sim-json", default=str(sc.SIM_JSON), help="simulation camera export (default: this repo's)")
    ap.add_argument("--stats-every", type=float, default=10.0, help="seconds between status lines (0 = off)")
    ap.add_argument("--allow-opencv-distortion", action="store_true",
                    help="accept nonzero distortion coefficients as OpenCV forward (plumb_bob) ones; "
                         "realsense colour intrinsics are inverse Brown-Conrady, so they are refused by default")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="output at this fraction of the sim camera's 1280x720 (default 0.5 = 640x360, the policy's "
                         "dual-rgb-360p-v1 profile: same field of view, K's first two rows halved)")
    return ap.parse_args(argv)


def sim_camera(args):
    """The output camera: the stored sim export, scaled by --scale."""
    return sc.scaled(sc.load_sim_camera(args.sim_json), args.scale)


def main():
    args = parse_args()

    import cv2
    cv2.setNumThreads(1)          # one core per frame; never spread over the controller's cores

    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    sim = sim_camera(args)
    rclpy.init()
    node = Node("d435i_sim_camera", enable_rosout=False, start_parameter_services=False)
    log = lambda text: print(text, flush=True)  # noqa: E731
    rep = Republisher(sim, log=log, allow_distortion=args.allow_opencv_distortion)
    topics = output_topics(args.ns)
    latest_only = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
    latched = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub_img = node.create_publisher(Image, topics["image"], qos_profile_sensor_data)
    pub_info = node.create_publisher(CameraInfo, topics["info"], qos_profile_sensor_data)
    pub_mask = node.create_publisher(Image, topics["mask"], latched)
    f = sim_info_fields(sim)

    def guarded(callback):
        """Keep the stream alive: count and report an unexpected exception once per kind."""
        def run(msg):
            try:
                callback(msg)
            except Exception as exc:  # noqa: BLE001 - a republisher must not die on one bad message
                rep.stats["errors"] += 1
                rep._warn_once(("error", type(exc).__name__, str(exc)),
                               f"d435i_sim_camera: {type(exc).__name__}: {exc} (message skipped)")
        return run

    def on_info(msg):
        rep.on_info(msg)

    def on_image(msg):
        r = rep.on_image(msg)
        mask = rep.take_mask()
        if mask is not None:
            m = Image(header=msg.header, height=mask.shape[0], width=mask.shape[1], encoding="mono8",
                      is_bigendian=0, step=mask.shape[1])
            m.data = msg_payload(mask)
            pub_mask.publish(m)
        if r is None:
            return
        img, encoding = r
        out = Image(header=msg.header, height=img.shape[0], width=img.shape[1], encoding=encoding, is_bigendian=0,
                    step=img.shape[1] * (img.shape[2] if img.ndim == 3 else 1))
        out.data = msg_payload(img)
        pub_img.publish(out)
        info = CameraInfo(header=msg.header, width=f["width"], height=f["height"],
                          distortion_model=f["distortion_model"])
        info.d, info.k, info.r, info.p = f["d"], f["k"], f["r"], f["p"]
        pub_info.publish(info)

    node.create_subscription(CameraInfo, f"{args.ns}/color/camera_info", guarded(on_info), qos_profile_sensor_data)
    node.create_subscription(Image, f"{args.ns}/color/image_raw", guarded(on_image), latest_only)
    if args.stats_every > 0:
        node.create_timer(args.stats_every, lambda: log(rep.summary()))
    log(f"d435i_sim_camera: {args.ns}/color -> {topics['image']} (sim camera from {args.sim_json})")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        log(rep.summary())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
