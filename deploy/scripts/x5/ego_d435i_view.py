#!/usr/bin/env python3
"""Live view of the two cameras: the Go2 front ("ego") camera and the D435i colour, each in its own
window and recorded to its own video, on one 50 Hz capture grid.

WINDOWS AND VIDEOS: window "Ego" and window "D435i", side by side on the screen, each the camera's picture
at 640x360 shown 1:1 (a 640x360 stream is not scaled at all; another size is fitted whole, never cropped)
under a label bar with the stream's size and measured rate ("Ego 640x360 14.2 fps", "sync 50 Hz"). The
capture is resampled onto a --grid-hz (50 Hz = 20 ms) grid (wam_go2/grid_sync.py): every decided tick
writes ONE frame to each video, the bare 640x360 picture without the bar (a camera without a frame for
the tick: a "waiting" panel), so both videos run at the grid rate in real time and frame i of the ego
video is the same tick as frame i of the D435i video; <stem>_grid.csv has one row per tick with each
camera's frame, offset, new / repeat / missing status and that video frame index. Videos: <record
without .mp4>_ego.mp4 and _d435i.mp4.

TIME: chrony puts the laptop and the X5 on one clock, so D435i stamps (header.stamp, X5 clock) are on the
laptop time base. --ego-source frontvideo (default) decodes Unitree's /frontvideostream
(unitree_go/Go2FrontVideoData, which does not match the official msg definition on this firmware; the
wire layout is
    uint64 time_frame | uint32 resolution (360|720) | uint8[] H.264 access unit
one message per resolution, 2 per frame). `time_frame` is a frame counter in disguise (6667/6666 per
frame = 1/15 s in 10 us units while frames arrive at 14.23 fps), so it is used only as an index; the frame
time on the laptop clock is a low-delay envelope fit of arrival time vs time_frame (residual +-4 ms) MINUS
the capture->encode->DDS latency L, unknown until measured. --ego-source topic reads the same stream as
published by the ego_stream node (sensor_msgs/Image, frame_id tf:<time_frame>, header.stamp = its own
arrival model minus its own calibrated latency). Two measurements run in the background (results in the
summary JSON; nothing is drawn on the pictures):
  * MOTION: each stream's frame-difference energy (stamped at the midpoint of the two frames it spans)
    is cross-correlated (as timesync.gyro_lag, sub-frame interpolation) once both share >= 10 s;
  * FLASH: a brightness jump seen by both lenses (phone torch); each jump is stamped at the midpoint
    between the previous and the first bright frame (+-P/2 per event); raw = D435i stamp - Go2 arrival
    model = eD - L.
Key c applies the measured L (hand-wave median + eD if there is one, else eD - flash raw median); r clears
the flashes. D435i capture = header.stamp - eD (stamp minus exposure midpoint, from sync_check.py --calib).

PAIRING (wam_go2/camera_pairing.py): every received frame is recorded in <stem>_frames.csv (raw device
time, laptop receive time, frame id, time source). Every Go2 ego frame takes the nearest unused D435i
frame within --slop-ms (20) of its common time, otherwise it is marked UNMATCHED with the reason;
<stem>_pairs.csv has one row per Go2 ego frame. The summary JSON's "time_sync" block is the per-session
report (time sources, pairing counts, delta statistics, the grid statistics); the last stdout lines
"grid 50 Hz: ..." and "pairing: ..." are what the demo scripts print. scripts/x5/pair_report.py
aggregates the common-event lags of several sessions.

Laptop only, wired DDS, read-only (publishes nothing, never touches control):
    source scripts/x5/laptop_camera_env.bash
    python3 scripts/x5/ego_d435i_view.py                       # two windows: q quit, s snapshot (both), c apply Go2 latency, r reset flashes
    python3 scripts/x5/ego_d435i_view.py --no-window --duration 30
    --ego-source frontvideo|topic   the ego stream (default frontvideo; topic = the ego_stream node's image topic)
    --res 720|360      /frontvideostream resolution to decode (default 720)
    --go2-latency-ms X start with a calibrated Go2 latency (frontvideo only)
    --d435i-stamp-offset-ms X   D435i colour header.stamp minus exposure midpoint (sync_check.py)
    --calib JSON       take both constants from a sync_check.py JSON (model_input.calibration)
    --flash-thresh N   brightness jump (0-255 grey) that counts as a flash (default 20)
    --record PATH      recording name: videos <PATH without .mp4>_ego.mp4 and _d435i.mp4 (default
                       <out-dir>/ego_d435i_<time>_ego.mp4 / _d435i.mp4); --no-record disables them
    --grid-hz X        capture grid rate = the videos' rate (default 50)
    --slop-ms X        pairing window |D435i - Go2 common time| (default 20)
    --out-dir DIR      where the summary JSON and the CSVs go (default data/logs/camera_checks_<date>/)
The summary JSON is <out-dir>/<stem>.json and the records <out-dir>/<stem>_frames.csv / _pairs.csv /
_grid.csv, <stem> = the basename of --record without .mp4 (else ego_d435i_<time>); snapshots (s key:
<stem>_<n>_ego.png and _d435i.png, what the windows show) go next to the videos. The demo sidecar
(scripts/demo/common.sh) passes a WAM run's result/<task>/<time>/view/camera.mp4 (videos camera_ego.mp4,
camera_d435i.mp4, everything else next to them).
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import struct
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from wam_go2 import camera_pairing as cp  # noqa: E402
from wam_go2 import grid_sync as gs  # noqa: E402
from wam_go2 import timesync as ts  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TF_UNIT_S = 1e-5            # nominal unit of time_frame (10 us); the fitted slope is what is used
GO2_PERIOD_S = 0.0703       # nominal frame periods (Go2 ego 14.23 fps, D435i 30 fps): the flash quantisation +-P/2
D435I_PERIOD_S = 0.0334     # on the banner, until the detector has measured a few intervals itself
MOTION_MIN_SPAN_S = 10.0    # no hand-wave lag before both motion series share this much time (shorter windows give false ok lags)
FALLBACK_S = 1.0            # no grid tick decided for this long -> the windows show the latest pictures (not recorded)
# each camera in its own window at 640x360 shown 1:1 (a 640x360 stream is not scaled at all) under a label bar
# with its size and measured rate; the capture is resampled onto a 50 Hz grid (wam_go2/grid_sync.py): one frame
# per tick in each camera's own video
GRID_PANEL_W, GRID_PANEL_H, GRID_LABEL_H = 640, 360, 34
EGO_TOPIC_TIME_SOURCE = "ego_stream_topic_model"     # --ego-source topic: header.stamp of the ego_stream node
EGO_GRID_TOL_S = 0.040                               # ego grid tolerance (frames repeat)


def ego_grid_mode(tol_ms=None):
    """(repeat, tolerance s) of the ego stream on the grid: always repeat, 40 ms (or --grid-ego-tol-ms).

    The ~14 fps ego stream brings a new frame every ~3.5 ticks of the 50 Hz grid. With repeat the
    nearest frame within 40 ms is shown again on the ticks in between and recorded as such ("repeat"
    in _grid.csv, never as a new capture); a stream as fast as the grid still gets a new frame on every
    tick. "missing" (the waiting panel) is then a real gap (no frame within 40 ms of the tick)."""
    return True, (float(tol_ms) * 1e-3 if tol_ms is not None else EGO_GRID_TOL_S)


def hms(t):
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"


def fit_box(img, w, h):
    """Scale img to fit inside w x h keeping its aspect ratio and pad the rest black (centred).
    A copy when img already is w x h."""
    import cv2
    ih, iw = img.shape[:2]
    if (iw, ih) == (w, h):
        return img.copy()
    s = min(w / iw, h / ih)
    nw, nh = max(1, min(w, int(round(iw * s)))), max(1, min(h, int(round(ih * s))))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    if (nw, nh) == (w, h):
        return small
    out = np.zeros((h, w) + img.shape[2:], img.dtype)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


def grid_label(name, wh, fps):
    """"Ego 640x360 14.2 fps": the stream's frame size as received and its measured frame rate."""
    if wh is None:
        return f"{name} (no frames)"
    rate = "--" if fps is None or not fps > 0 else f"{fps:.1f}"
    return f"{name} {wh[0]}x{wh[1]} {rate} fps"


class FrameSizeGuard:
    """cv2.VideoWriter (FFmpeg backend) silently DROPS every frame whose size differs from the size
    it was opened with: write() returns None and nothing is logged. Renderer.panel() keeps 640x360;
    this is the backstop in front of each writer: a frame of another size is fitted to the writer size
    (fit_box) with ONE warning, so no frame is ever lost silently."""

    def __init__(self, size):
        self.size = (int(size[0]), int(size[1]))        # (width, height) the writer was opened with
        self.warned = False

    def __call__(self, frame):
        h, w = frame.shape[:2]
        if (w, h) == self.size:
            return frame
        if not self.warned:
            print(f"WARNING: frame {w}x{h} != video {self.size[0]}x{self.size[1]}; "
                  "resizing to the video size (warned once)")
            self.warned = True
        return fit_box(frame, *self.size)


# ------------------------------------------------------------------ decoding
class Go2Decoder:
    """H.264 access units (Annex B, straight from /frontvideostream) -> BGR frames,
    with the message's time_frame carried through the decoder as the packet pts."""

    def __init__(self):
        import av
        import av.logging
        av.logging.set_level(av.logging.FATAL)
        self.av = av
        self.codec = av.CodecContext.create("h264", "r")     # pts is passed through untouched (a decoder has no time_base)
        self.av_error = getattr(av, "FFmpegError", None) or av.error.FFmpegError   # PyAV >= 14 dropped av.AVError
        self.keyframe_seen = False
        self.errors = 0

    def decode(self, data: bytes, tf: int):
        pkt = self.av.Packet(data)
        pkt.pts = int(tf); pkt.dts = int(tf)
        try:
            frames = self.codec.decode(pkt)
        except self.av_error:                 # no reference yet (P-frame before the first IDR) etc.
            self.errors += 1
            return []
        out = []
        for fr in frames:
            if fr.key_frame:
                self.keyframe_seen = True
            out.append((int(fr.pts) if fr.pts is not None else int(tf), fr.to_ndarray(format="bgr24")))
        return out


def parse_frontvideo(raw: bytes):
    """CDR-LE: 4-byte encapsulation, uint64 time_frame, uint32 resolution, uint32 len, bytes."""
    tf, res, ln = struct.unpack_from("<QII", raw, 4)
    return tf, res, raw[20:20 + ln]


# --------------------------------------------------------------- flash logic
class FlashDetector:
    """Rising-edge detector on the mean brightness of the image centre.

    The torch comes on somewhere between the previous frame and the first bright one, so the
    event is stamped at the MIDPOINT of that interval (as the motion energies are): the per-event
    error is then a symmetric +-P/2 instead of a one-sided 0..P (a +P/2 bias, +35 ms Go2 / +17 ms
    D435i, when the first bright frame's own time is used). After a dropped frame or a decode gap
    the interval is wider, but its midpoint is still the unbiased estimate (the lost frames may
    have been dark or bright), so there is deliberately no frame_time - P/2 fallback: in a
    frame-quantised simulation that fallback biases the flash latency by +6 ms at 10 % Go2 frame
    loss (+12 ms at 25 %), the midpoint by < 3 ms. `key` (the Go2 time_frame) is carried the same
    way: after an event `mid_key` is the key of the event instant (None without keys). period()
    (median of the recent frame intervals, else the nominal period_s) is the P the banner quotes."""

    def __init__(self, thresh=20.0, history=8, period_s=None):
        self.thresh = thresh
        self.hist = collections.deque(maxlen=history)
        self.on = False
        self.base_at_on = None
        self.last_value = 0.0
        self.period_s = period_s
        self.dts = collections.deque(maxlen=15)          # recent frame intervals (s)
        self.prev_time = None
        self.prev_key = None
        self.mid_key = None

    def period(self):
        """Frame period: median of the recent intervals, else the nominal one."""
        return float(np.median(self.dts)) if len(self.dts) >= 3 else self.period_s

    def _event_at(self, frame_time, key):
        if self.prev_time is None:                       # no previous frame: the frame itself
            return frame_time, key
        mid_key = None if key is None or self.prev_key is None else 0.5 * (self.prev_key + key)
        return 0.5 * (self.prev_time + frame_time), mid_key

    def update(self, img, frame_time, key=None):
        h, w = img.shape[:2]
        crop = img[h // 4: 3 * h // 4: 6, w // 4: 3 * w // 4: 6]
        v = float(crop.mean())
        self.last_value = v
        event = None
        if not self.on:
            if len(self.hist) >= 4 and v > np.median(self.hist) + self.thresh:
                self.on = True; self.base_at_on = float(np.median(self.hist))
                event, self.mid_key = self._event_at(frame_time, key)
            else:
                self.hist.append(v)
        else:
            if v < self.base_at_on + self.thresh / 2:
                self.on = False; self.hist.append(v)
        if self.prev_time is not None and frame_time > self.prev_time:
            self.dts.append(frame_time - self.prev_time)
        self.prev_time, self.prev_key = frame_time, key
        return event


# -------------------------------------------------------------------- streams
class Streams:
    def __init__(self, res, flash_thresh, go2_latency_s, d435i_offset_s=0.0, slop_s=0.020,
                 calibration_version="none", go2_calibrated=False, grid=None, ego_source="frontvideo"):
        import cv2
        self.cv2 = cv2
        self.lock = threading.Lock()
        self.res = res
        self.d_offset_s = d435i_offset_s                       # D435i capture = header.stamp - d_offset_s
        self._prev_t = {"go2": None, "d435i": None}            # frame time of the previous decoded frame (for midpoints)
        self.dec = Go2Decoder()
        self.go2_pairs = collections.deque(maxlen=900)         # (tf, arrival)
        self.go2_frames = collections.deque(maxlen=6)          # (tf, arrival, img)
        self.go2_fit = None
        self.go2_latency_s = go2_latency_s
        self.go2_arrivals = collections.deque(maxlen=60)
        self.d_frames = collections.deque(maxlen=90)           # (stamp, arrival, img)
        self.d_lat_ms = collections.deque(maxlen=90)
        self.flash_go2 = FlashDetector(flash_thresh, period_s=GO2_PERIOD_S)
        self.flash_d = FlashDetector(flash_thresh, period_s=D435I_PERIOD_S)
        self.ev_go2, self.ev_d = [], []                        # unpaired event times
        self.flash_pairs = []                                  # dicts
        self.calib_index = None                                # len(flash_pairs) at the last c (banner residual: later flashes only;
                                                               # an index, not a wall time, so a laptop clock step cannot hide flashes)
        self.flash_mark = {"go2": 0.0, "d435i": 0.0}
        self.motion_go2 = collections.deque(maxlen=900)        # (uncalibrated model time, frame-diff energy)
        self.motion_d = collections.deque(maxlen=1800)         # (stamp, energy)
        self._prev_small = {"go2": None, "d435i": None}
        self.motion_lag = None                                 # latest ts.GyroLag on the motion signals
        self.motion_hist = []
        self._last_motion = 0.0
        self.msgs = {"go2": 0, "d435i": 0}
        self.msgs_by_res = {360: 0, 720: 0}                      # every /frontvideostream message, before the --res filter
        self._last_fit = 0.0
        # per-frame records, bounded-window pairing, sync labels (wam_go2/camera_pairing.py)
        self.pairer = cp.BoundedPairer(slop_s=slop_s)
        self.ego_source = ego_source                           # "frontvideo" (decoded here) | "topic" (ego_stream node)
        self.grid = grid                                       # the capture grid (gs.GridSampler), or None
        self.shapes = {}                                       # stream -> (width, height) as received
        self._topic_t = collections.OrderedDict()              # topic ego: time_frame -> header.stamp
        self.sync_method = cp.SYNC_CAPTURE if go2_calibrated else cp.SYNC_RECEIVE
        self.calibration_version = calibration_version
        self.rec_rows = []                                     # frame records waiting for the main loop to write
        self.ego_msgs, self.d_msgs = [], []                    # (raw time, arrival) of every frame: the time-source report
        self._ego_ids = collections.OrderedDict()              # time_frame -> (frame_id, arrival) of the recent messages
        self._ego_count = self._d_count = 0
        self.decode_dropped = 0                                # access units lost to a full decode queue (backlog)
        self.modes = {"ego": None, "d435i": None}              # camera mode as received (in the summary)
        # every calibration in force during the session, and the decided Go2 frames counted under the one each
        # frame got its common time with: a c key mid-session must not relabel the pairs made before it
        self.calibrations = {calibration_version: {"sync_method": self.sync_method, "go2_latency_ms": go2_latency_s * 1e3,
                                                   "d435i_stamp_offset_ms": d435i_offset_s * 1e3}}
        self.cal_stats = {}
        self.decode_q = collections.deque(maxlen=8)            # (tf, arrival, data); decode off the DDS thread
        self.decode_ev = threading.Event()
        threading.Thread(target=self._decode_loop, daemon=True).start()

    # Go2 --------------------------------------------------------------
    def on_frontvideo(self, raw):
        arrival = time.time()
        b = bytes(raw)
        try:
            tf, res, data = parse_frontvideo(b)
        except struct.error:
            return
        self.msgs_by_res[res] = self.msgs_by_res.get(res, 0) + 1
        if res != self.res:
            return
        self.msgs["go2"] += 1
        self._note_ego_msg(tf, arrival)
        with self.lock:
            self.go2_pairs.append((tf, arrival)); self.go2_arrivals.append(arrival)
            if arrival - self._last_fit > 1.0 and len(self.go2_pairs) >= 30:
                tfs = np.array([p[0] for p in self.go2_pairs], dtype=np.int64)
                arr = np.array([p[1] for p in self.go2_pairs])
                try:                                           # iters=2 keeps ~25 % of a few hundred samples on the envelope
                    self.go2_fit = ts.fit_tick_clock(tfs, arr, iters=2, nominal_s_per_tick=TF_UNIT_S)
                except (ValueError, np.linalg.LinAlgError):
                    pass
                self._last_fit = arrival
        if len(self.decode_q) == self.decode_q.maxlen:         # the deque drops the oldest access unit: a backlog
            self.decode_dropped += 1
        self.decode_q.append((tf, arrival, data)); self.decode_ev.set()

    def _note_ego_msg(self, tf, arrival):
        """Per-frame record of a /frontvideostream message at the decoded resolution."""
        tf = int(tf)
        with self.lock:
            fid = self._ego_count; self._ego_count += 1
            self._ego_ids[tf] = (fid, arrival)
            while len(self._ego_ids) > 64:
                self._ego_ids.popitem(last=False)
            self.ego_msgs.append((tf, arrival))
            self.rec_rows.append(cp.frame_row(cp.Frame("ego", fid, tf, arrival, None)))

    def _decode_loop(self):
        while True:
            self.decode_ev.wait(0.5)
            while self.decode_q:
                tf, arrival, data = self.decode_q.popleft()
                for ftf, img in self.dec.decode(data, tf):
                    self._on_go2_frame(ftf, arrival, img)
            self.decode_ev.clear()

    def _on_go2_frame(self, ftf, arrival, img):
        """One decoded Go2 frame: display buffer, pairing (anchor), flash and motion measurements."""
        with self.lock:
            self.go2_frames.append((ftf, arrival, img))
            t_model = self.go2_time(ftf)
            t_raw = self.go2_time(ftf, calibrated=False)
            fid, rx = self._ego_ids.get(int(ftf), (-1, arrival))
            if self.modes["ego"] is None:
                self.modes["ego"] = f"{img.shape[1]}x{img.shape[0]} H.264 /frontvideostream {self.res}p"
            self.pairer.add_ego(cp.Frame("ego", fid, int(ftf), rx, t_model,
                                         {"img": img, "sync": (self.sync_method, self.calibration_version)}))
            self.shapes["ego"] = (img.shape[1], img.shape[0])
            if self.grid is not None:                          # with the calibration this frame was timed with
                self.grid.add_ego(cp.Frame("ego", fid, int(ftf), rx, t_model,
                                           {"img": img, "sync": (self.sync_method, self.calibration_version)}))
        if t_model is not None:
            ev = self.flash_go2.update(img, t_model, key=ftf)
            if ev is not None:                     # time_frame of the event instant (rounding: 5 us)
                self._flash("go2", ev, int(round(self.flash_go2.mid_key)))
            e = self._motion_energy("go2", img)
            prev_t, self._prev_t["go2"] = self._prev_t["go2"], t_raw
            if e is not None and prev_t is not None:
                with self.lock:
                    self.motion_go2.append((0.5 * (prev_t + t_raw), e))   # frame difference belongs to the interval midpoint

    def go2_time(self, tf, calibrated=True):
        """Go2 frame time on the laptop clock: arrival-envelope model (minus latency)."""
        if self.ego_source == "topic":
            t = self._topic_t.get(int(tf))
            if t is None:
                return None
            return t - self.go2_latency_s if calibrated else t
        if self.go2_fit is None:
            return None
        t = float(self.go2_fit.time_of(np.array([tf], dtype=np.int64))[0])
        return t - self.go2_latency_s if calibrated else t

    # D435i ------------------------------------------------------------
    def on_color(self, msg):
        arrival = time.time()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            img = img[:, :, ::-1]
        img = np.ascontiguousarray(img)
        self.msgs["d435i"] += 1
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        with self.lock:
            self.d_frames.append((stamp, arrival, img)); self.d_lat_ms.append((arrival - stamp) * 1e3)
            fid = self._d_count; self._d_count += 1
            if self.modes["d435i"] is None:
                self.modes["d435i"] = f"{msg.width}x{msg.height} {msg.encoding}"
            self.d_msgs.append((stamp_ns, arrival))
            self.rec_rows.append(cp.frame_row(cp.Frame("d435i", fid, stamp_ns, arrival, None)))
            self.pairer.add_d435i(cp.Frame("d435i", fid, stamp_ns, arrival, stamp - self.d_offset_s, img))
            self.shapes["d435i"] = (img.shape[1], img.shape[0])
            if self.grid is not None:
                self.grid.add_d435i(cp.Frame("d435i", fid, stamp_ns, arrival, stamp - self.d_offset_s, img))
        ev = self.flash_d.update(img, stamp)
        if ev is not None:
            self._flash("d435i", ev, None)
        e = self._motion_energy("d435i", img)
        prev_t, self._prev_t["d435i"] = self._prev_t["d435i"], stamp
        if e is not None and prev_t is not None:
            with self.lock:
                self.motion_d.append((0.5 * (prev_t + stamp), e))

    # Go2 ego from the ego_stream node's sensor_msgs/Image topic ----------------------------------
    def on_ego_image(self, msg):
        """--ego-source topic: header.stamp is ego_stream's frame time on the laptop clock (its arrival
        model minus its own calibrated latency); frame_id "tf:<time_frame>"."""
        arrival = time.time()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        step = getattr(msg, "step", msg.width * 3) or msg.width * 3
        rows = np.frombuffer(msg.data, dtype=np.uint8)[:msg.height * step].reshape(msg.height, step)
        img = rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            img = img[:, :, ::-1]
        img = np.ascontiguousarray(img)
        fid_text = str(getattr(msg.header, "frame_id", ""))
        tf = int(fid_text[3:]) if fid_text.startswith("tf:") and fid_text[3:].isdigit() else None
        self.msgs["go2"] += 1
        with self.lock:
            fid = self._ego_count; self._ego_count += 1
            key = tf if tf is not None else stamp_ns
            self._topic_t[key] = stamp
            while len(self._topic_t) > 256:
                self._topic_t.popitem(last=False)
            self.go2_arrivals.append(arrival)
            self.go2_frames.append((key, arrival, img))
            self.ego_msgs.append((stamp_ns, arrival))
            sync = (self.sync_method, self.calibration_version)
            t_model = stamp - self.go2_latency_s
            self.rec_rows.append(cp.frame_row(cp.Frame("ego", fid, stamp_ns, arrival, None), EGO_TOPIC_TIME_SOURCE))
            if self.modes["ego"] is None:
                self.modes["ego"] = f"{msg.width}x{msg.height} {msg.encoding} topic (ego_stream)"
            self.shapes["ego"] = (msg.width, msg.height)
            self.pairer.add_ego(cp.Frame("ego", fid, stamp_ns, arrival, t_model, {"img": img, "sync": sync}))
            if self.grid is not None:                          # with the calibration this frame was timed with
                self.grid.add_ego(cp.Frame("ego", fid, stamp_ns, arrival, t_model, {"img": img, "sync": sync}))
        ev = self.flash_go2.update(img, t_model, key=key)
        if ev is not None:
            self._flash("go2", ev, key)
        e = self._motion_energy("go2", img)
        prev_t, self._prev_t["go2"] = self._prev_t["go2"], stamp
        if e is not None and prev_t is not None:
            with self.lock:
                self.motion_go2.append((0.5 * (prev_t + stamp), e))

    def poll_grid(self, now, flush=False):
        """The grid ticks decided now (every remaining one when flush) and the pending frame rows."""
        with self.lock:
            ticks = [] if self.grid is None else (self.grid.flush(now) if flush else self.grid.poll(now))
            rows, self.rec_rows = self.rec_rows, []
        return ticks, rows

    def latest_ego_img(self):
        with self.lock:
            return self.go2_frames[-1][2] if self.go2_frames else None

    # motion energy / cross-correlation -------------------------------
    def _motion_energy(self, who, img):
        """Mean absolute frame difference of a 160-px-wide grey thumbnail (0..255)."""
        g = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
        small = self.cv2.resize(g, (160, int(160 * g.shape[0] / g.shape[1])), interpolation=self.cv2.INTER_AREA).astype(np.int16)
        prev = self._prev_small[who]; self._prev_small[who] = small
        if prev is None or prev.shape != small.shape:
            return None
        return float(np.abs(small - prev).mean())

    def update_motion_lag(self, window_s=25.0, max_lag_s=0.6, min_span_s=MOTION_MIN_SPAN_S):
        """Cross-correlate the two motion-energy signals (D435i stamps = A, Go2
        uncalibrated arrival model = B): lag > 0 = Go2 arrival model minus D435i
        header.stamp for the same motion = go2_latency - d435i_stamp_offset.
        Nothing is computed until both series share min_span_s of the window:
        shorter windows (~4 s, 60 / 120 samples) give false ok lags."""
        now = time.time()
        if now - self._last_motion < 3.0:
            return
        self._last_motion = now
        with self.lock:
            g = [(t, e) for t, e in self.motion_go2 if t is not None and now - t < window_s]
            d = [(t, e) for t, e in self.motion_d if now - t < window_s]
        if len(g) < 60 or len(d) < 120:
            return
        tg = np.array([x[0] for x in g]); eg = np.array([x[1] for x in g])
        td = np.array([x[0] for x in d]); ed = np.array([x[1] for x in d])
        if min(tg.max(), td.max()) - max(tg.min(), td.min()) < min_span_s:
            return
        res = ts.gyro_lag(td, ed, tg, eg, max_lag_s=max_lag_s, grid_hz=1000.0, highpass_s=1.5,
                          min_motion_rms=0.8, min_corr=0.5)
        self.motion_lag = res
        if res.ok:
            self.motion_hist.append({"time": now, "lag_ms": res.lag_s * 1e3, "corr": res.peak_corr,
                                     "rms_d435i": res.motion_rms_a, "rms_go2": res.motion_rms_b})

    # flash pairing ----------------------------------------------------
    def _flash(self, who, t_event, tf):
        with self.lock:
            self.flash_mark[who] = time.time()
            mine, other = (self.ev_go2, self.ev_d) if who == "go2" else (self.ev_d, self.ev_go2)
            # match with an unpaired event of the other stream within 1.5 s
            for i, (t_o, tf_o) in enumerate(other):
                if abs(t_o - t_event) < 1.5:
                    other.pop(i)
                    t_go2, tf_go2, t_d = (t_event, tf, t_o) if who == "go2" else (t_o, tf_o, t_event)
                    go2_model = self.go2_time(tf_go2, calibrated=False) if tf_go2 is not None else None
                    raw_ms = (t_d - go2_model) * 1e3 if go2_model is not None else float("nan")
                    # capture T: Go2 arrival model = T + L, D435i stamp = T + eD, so raw = eD - L and the
                    # residual of the displayed capture estimates (stamp - eD) - (model - L) = raw + L - eD
                    self.flash_pairs.append({"time": time.time(), "go2_tf": tf_go2, "go2_model_time": go2_model,
                                             "d435i_stamp": t_d, "raw_ms": raw_ms,
                                             "residual_ms": raw_ms + (self.go2_latency_s - self.d_offset_s) * 1e3})
                    return
            mine.append((t_event, tf))
            # drop stale unpaired events
            mine[:] = [e for e in mine if time.time() - e[0] < 10.0]

    def flash_median_raw(self):
        vals = [p["raw_ms"] for p in self.flash_pairs if np.isfinite(p["raw_ms"])]
        return float(np.median(vals)) if vals else None

    def flash_residual_ms(self):
        """(median true residual in ms, n) of the displayed capture estimates at the flashes,
        raw + (go2_latency - d435i_offset), over the flash pairs recorded AFTER the last c only
        (a flash calibration makes its own pairs read 0 by construction); (None, 0) if none yet."""
        with self.lock:
            new = self.flash_pairs if self.calib_index is None else self.flash_pairs[self.calib_index:]
            vals = [p["raw_ms"] for p in new if np.isfinite(p["raw_ms"])]
            if not vals:
                return None, 0
            return float(np.median(vals)) + (self.go2_latency_s - self.d_offset_s) * 1e3, len(vals)

    def apply_calibration(self):
        """The c key: install the measured Go2 latency L; returns the message to print, or None
        when there is nothing to apply. Capture T -> Go2 arrival model T + L, D435i stamp T + eD.
        MOTION (preferred): lag = Go2 arrival model - D435i stamp = L - eD  ->  L = median(lag) + eD.
        FLASH (fallback):   raw = D435i stamp - Go2 arrival model = eD - L  ->  L = eD - median(raw)."""
        with self.lock:
            if self.motion_hist:
                self.go2_latency_s = float(np.median([h["lag_ms"] for h in self.motion_hist])) * 1e-3 + self.d_offset_s
                msg = f"Go2 latency set from motion cross-correlation median + D435i stamp offset: {self.go2_latency_s * 1e3:.0f} ms"
            else:
                med = self.flash_median_raw()
                if med is None:
                    return None
                self.go2_latency_s = self.d_offset_s - med * 1e-3
                msg = f"Go2 latency set from D435i stamp offset - flash median: {self.go2_latency_s * 1e3:.0f} ms"
            self.calib_index = len(self.flash_pairs)
            self.sync_method = cp.SYNC_CAPTURE
            self.calibration_version = (f"c-key@{hms(time.time())}:L={self.go2_latency_s * 1e3:.1f}ms,"
                                        f"eD={self.d_offset_s * 1e3:.1f}ms")
            self.calibrations[self.calibration_version] = {"sync_method": self.sync_method,
                                                           "go2_latency_ms": self.go2_latency_s * 1e3,
                                                           "d435i_stamp_offset_ms": self.d_offset_s * 1e3}
        return msg

    def clear_flashes(self):
        """The r key: forget every flash pair and unpaired event. The post-calibration marker is
        an index into flash_pairs, so it must follow the list: after a clear every new pair is
        newer than the last c (0), or there is still no calibration (None)."""
        with self.lock:
            self.flash_pairs.clear(); self.ev_go2.clear(); self.ev_d.clear()
            if self.calib_index is not None:
                self.calib_index = 0

    # pairing for display ---------------------------------------------
    def current_pair(self):
        with self.lock:
            if not self.go2_frames:
                return None, None, None
            tf, arr, img = self.go2_frames[-1]
            t_go2 = self.go2_time(tf)
            best = None
            if t_go2 is not None and self.d_frames:
                best = min(self.d_frames, key=lambda f: abs(f[0] - self.d_offset_s - t_go2))
            elif self.d_frames:
                best = self.d_frames[-1]
            return (tf, arr, img, t_go2), best, (t_go2 is not None)

    def poll_pairs(self, now, flush=False):
        """The main loop's call per tick: the Go2 frames whose D435i partner can no longer change (every
        pending one when flush) and the frame record rows to write. Each decision is counted under the
        calibration its Go2 frame got its common time with, so a c key mid-session never relabels the
        pairs made before it."""
        with self.lock:
            decisions = self.pairer.flush() if flush else self.pairer.poll(now)
            rows, self.rec_rows = self.rec_rows, []
            for dec in decisions:
                pl = dec.ego.payload if isinstance(dec.ego.payload, dict) else {}
                method, version = pl.get("sync", (self.sync_method, self.calibration_version))
                acc = self.cal_stats.setdefault(version, {"sync_method": method, "decided": 0, "matched": 0, "deltas_ms": [],
                                                          "ego_frame_ids": [dec.ego.frame_id, dec.ego.frame_id]})
                acc["decided"] += 1
                acc["ego_frame_ids"][1] = dec.ego.frame_id
                if dec.matched:
                    acc["matched"] += 1
                    acc["deltas_ms"].append(dec.delta_s * 1e3)
        return decisions, rows

    def latest_d435i_img(self):
        with self.lock:
            return self.d_frames[-1][2] if self.d_frames else None

    def rates(self):
        with self.lock:
            g = list(self.go2_arrivals); d = [f[1] for f in self.d_frames]
        fg = (len(g) - 1) / (g[-1] - g[0]) if len(g) > 5 and g[-1] > g[0] else 0.0
        fd = (len(d) - 1) / (d[-1] - d[0]) if len(d) > 5 and d[-1] > d[0] else 0.0
        return fg, fd


# ------------------------------------------------------------------ rendering
class Renderer:
    """The two windows and the two videos. Each camera on its own at 640x360: a 640x360 frame is used 1:1
    (not scaled, nothing drawn on it), another size is fitted whole (fit_box), never cropped. The WINDOW
    has a label bar above the picture with the stream's size and measured rate; the VIDEO frame is the
    picture alone."""

    def __init__(self):
        import cv2
        self.cv2 = cv2
        self._cache = {}

    def _waiting_panel(self, w, h):
        key = ("wait", w, h)
        panel = self._cache.get(key)
        if panel is None:
            cv2 = self.cv2
            panel = np.zeros((h, w, 3), np.uint8)
            txt = "waiting for frames"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            cv2.putText(panel, txt, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
            self._cache[key] = panel
        return panel

    def panel(self, img):
        """The 640x360 picture of one camera (its video frame): the frame itself when it is 640x360, else
        the whole frame fitted and padded; a 'waiting' panel without a frame."""
        if img is None:
            return self._waiting_panel(GRID_PANEL_W, GRID_PANEL_H)
        if img.shape[:2] == (GRID_PANEL_H, GRID_PANEL_W):
            return img
        return fit_box(img, GRID_PANEL_W, GRID_PANEL_H)

    def _bar(self, label, sync):
        key = ("bar", label, sync)
        bar = self._cache.get(key)
        if bar is None:
            cv2 = self.cv2
            if len(self._cache) > 128:
                self._cache.clear()
            bar = np.full((GRID_LABEL_H, GRID_PANEL_W, 3), 32, np.uint8)
            cv2.putText(bar, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            if sync:
                (tw, _), _ = cv2.getTextSize(sync, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.putText(bar, sync, (GRID_PANEL_W - tw - 12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 1, cv2.LINE_AA)
            self._cache[key] = bar
        return bar

    def window(self, img, label, sync=""):
        """One camera's window: the label bar ("Ego 640x360 14.2 fps" ... "sync 50 Hz") above its picture.
        640 x 394."""
        return np.concatenate([self._bar(label, sync), self.panel(img)], axis=0)


# ------------------------------------------------------------- pairs, records
def sync_of(dec, st):
    """(sync_method, calibration_version) in force when this Go2 frame got its common time."""
    pl = dec.ego.payload if isinstance(dec.ego.payload, dict) else {}
    return pl.get("sync", (st.sync_method, st.calibration_version))


class GridSink:
    """One video frame PER CAMERA per decided grid tick (wam_go2/grid_sync.py): the ego picture to the ego
    video, the D435i picture to the D435i video (container rate = the grid rate, so both play back in real
    time and frame i of one is the same 20 ms tick as frame i of the other), and one row in <stem>_grid.csv
    with that video frame index. The two windows show the last tick under their label bars. Without a
    decided tick for FALLBACK_S the latest pictures are shown (not written), so a dead camera never
    freezes the view."""

    def __init__(self, rend, write_video=None, write_row=None, grid_hz=50.0):
        self.rend = rend
        self.write_video = write_video or (lambda ego, d435i: None)
        self.write_row = write_row or (lambda row: None)
        self.grid_hz = grid_hz
        self.last = None
        self.last_tick_wall = None
        self.video_frames = 0

    def labels(self, st):
        fg, fd = st.rates()
        return (grid_label("Ego", st.shapes.get("ego"), fg), grid_label("D435i", st.shapes.get("d435i"), fd),
                f"sync {self.grid_hz:g} Hz")

    def windows(self, ego_img, d_img, labels):
        """(ego window, D435i window)."""
        return self.rend.window(ego_img, labels[0], labels[2]), self.rend.window(d_img, labels[1], labels[2])

    @staticmethod
    def _ego(tick, st):
        """(picture, (sync_method, calibration_version)) of a tick: the calibration its ego frame was timed
        with (a later c key must not relabel it), else the one in force."""
        if tick.ego is None:
            return None, (st.sync_method, st.calibration_version)
        pl = tick.ego.payload
        if isinstance(pl, dict):
            return pl.get("img"), pl.get("sync", (st.sync_method, st.calibration_version))
        return pl, (st.sync_method, st.calibration_version)

    def handle(self, st, ticks, now):
        if ticks:
            labels = self.labels(st)
            for tick in ticks:
                ego_img, sync = self._ego(tick, st)
                d_img = None if tick.d435i is None else tick.d435i.payload
                idx = self.write_video(self.rend.panel(ego_img), self.rend.panel(d_img))
                self.write_row(gs.grid_row(tick, idx, *sync))
                if idx is not None:
                    self.video_frames += 1
            self.last, self.last_tick_wall = self.windows(ego_img, d_img, labels), now   # shown: the last tick
        elif self.last is None or self.last_tick_wall is None or now - self.last_tick_wall >= FALLBACK_S:
            self.last = self.windows(st.latest_ego_img(), st.latest_d435i_img(), self.labels(st))
        return self.last


class SessionRecords:
    """<out-dir>/<stem>_frames.csv (every received frame) and <stem>_pairs.csv (every decided Go2 ego
    frame, matched or not); flushed every loop. A write problem is ONE warning, never a stop."""

    def __init__(self, out_dir, stem, grid=False):
        self.frames_path = os.path.join(out_dir, f"{stem}_frames.csv")
        self.pairs_path = os.path.join(out_dir, f"{stem}_pairs.csv")
        self.grid_path = os.path.join(out_dir, f"{stem}_grid.csv") if grid else None
        self.ok, self._files = True, []
        self.n_frames = self.n_pairs = self.n_grid = 0
        try:
            ff, pf = open(self.frames_path, "w", newline=""), open(self.pairs_path, "w", newline="")
            self._files = [ff, pf]
            self._fw, self._pw = csv.writer(ff), csv.writer(pf)
            self._fw.writerow(cp.FRAMES_HEADER); self._pw.writerow(cp.PAIRS_HEADER)
            if grid:
                gf = open(self.grid_path, "w", newline="")
                self._files.append(gf)
                self._gw = csv.writer(gf)
                self._gw.writerow(gs.GRID_HEADER)
        except OSError as e:
            self._fail(e)

    def grid(self, row):
        if self.ok and self.grid_path:
            try:
                self._gw.writerow(row); self.n_grid += 1
            except (OSError, ValueError) as e:
                self._fail(e)

    def _fail(self, e):
        if self.ok:
            print(f"WARNING: frame / pair records disabled ({self.frames_path}): {e}")
        self.ok = False

    def frames(self, rows):
        if self.ok and rows:
            try:
                self._fw.writerows(rows); self.n_frames += len(rows)
            except (OSError, ValueError) as e:
                self._fail(e)

    def pair(self, row):
        if self.ok:
            try:
                self._pw.writerow(row); self.n_pairs += 1
            except (OSError, ValueError) as e:
                self._fail(e)

    def flush(self):
        if self.ok:
            try:
                for f in self._files:
                    f.flush()
            except (OSError, ValueError) as e:
                self._fail(e)

    def close(self):
        for f in self._files:
            try:
                f.close()
            except OSError:
                pass


def ego_calibration_for_source(source, go2_latency_ms, d435i_offset_ms, version, go2_calibrated):
    """The Go2 latency of --calib / --go2-latency-ms calibrates the /frontvideostream arrival model (H.264
    messages over DDS, ~44 ms). With --ego-source topic the ego_stream node has already subtracted its own
    Go2 latency (its go2_latency_ms / calib parameters) from the stamps, so the viewer's is NOT subtracted
    again and the pairs are not labelled calibrated by it; the D435i offset applies to both."""
    if source != "topic" or not go2_latency_ms:
        return {"go2_latency_ms": float(go2_latency_ms), "d435i_stamp_offset_ms": float(d435i_offset_ms),
                "calibration_version": version, "go2_calibrated": bool(go2_calibrated), "note": ""}
    return {"go2_latency_ms": 0.0, "d435i_stamp_offset_ms": float(d435i_offset_ms),
            "calibration_version": f"{version}|go2_latency_not_applied_to_topic",
            "go2_calibrated": False,
            "note": (f"--ego-source topic: the Go2 latency {go2_latency_ms:.1f} ms of the frontvideo calibration is not "
                     "applied (ego_stream applies its own go2_latency_ms / calib to the stamps); the "
                     f"D435i stamp offset {d435i_offset_ms:.1f} ms is.")}


def calibration_from_file(path):
    """--calib: a sync_check.py JSON or a pair_report.py --write-calib JSON (both carry
    model_input.calibration). Returns (L_ms | None, eD_ms | None, calibration_version, go2_calibrated)."""
    with open(path, "rb") as f:
        raw = f.read()
    cal = json.loads(raw).get("model_input", {}).get("calibration", {})
    L, eD = cal.get("go2_ego_latency_ms"), cal.get("d435i_stamp_minus_exposure_ms")
    version = f"calib:{os.path.basename(path)}@{hashlib.sha256(raw).hexdigest()[:12]}"
    return (None if L is None else float(L)), (None if eD is None else float(eD)), version, L is not None


def sync_summary(st):
    """The per-session report: time sources, pairing, common-event validation, queues. The sync
    method / calibration are reported per calibration actually used ("calibrations", in order of use);
    the session-level label is "mixed" when the calibration changed mid-session (c key), and
    validation.motion describes the pairs of the last calibration."""
    with st.lock:
        ego, dm = list(st.ego_msgs), list(st.d_msgs)
        lags = [h["lag_ms"] for h in st.motion_hist]
        pairing = st.pairer.stats.as_dict(n_d435i_frames=len(dm))
        l_now, ed_now = st.go2_latency_s * 1e3, st.d_offset_s * 1e3
        cur_method, cur_version = st.sync_method, st.calibration_version
        cal_stats = {v: dict(a, deltas_ms=list(a["deltas_ms"]), ego_frame_ids=list(a["ego_frame_ids"])) for v, a in st.cal_stats.items()}
        calibrations = {v: dict(c) for v, c in st.calibrations.items()}
        n_flash, modes = len(st.flash_pairs), dict(st.modes)
    cals = []
    for v, a in cal_stats.items():
        c = calibrations.get(v, {})
        l_ms, ed_ms = c.get("go2_latency_ms", l_now), c.get("d435i_stamp_offset_ms", ed_now)
        cals.append({"calibration_version": v, "sync_method": a["sync_method"], "go2_latency_ms": l_ms,
                     "d435i_stamp_offset_ms": ed_ms, "decided": a["decided"], "matched": a["matched"],
                     "unmatched": a["decided"] - a["matched"], "ego_frame_ids": a["ego_frame_ids"],
                     "delta_ms": cp.delta_stats(a["deltas_ms"]),
                     "motion": cp.motion_validation(lags, l_ms, ed_ms, deltas_ms=a["deltas_ms"])})
    if cals:
        methods = {c["sync_method"] for c in cals}
        method = cals[0]["sync_method"] if len(methods) == 1 else "mixed"
        version = cals[0]["calibration_version"] if len(cals) == 1 else "mixed"
        motion = dict(cals[-1]["motion"], calibration_version=cals[-1]["calibration_version"])
    else:                                       # nothing decided: the calibration in force
        method, version = cur_method, cur_version
        motion = dict(cp.motion_validation(lags, l_now, ed_now), calibration_version=cur_version)
    return {"sync_method": method, "calibration_version": version,
            "calibrations": cals,
            "final_go2_latency_ms": l_now, "final_d435i_stamp_offset_ms": ed_now, "camera_modes": modes,
            "time_sources": {"ego": (topic_ego_report([e[0] for e in ego], [e[1] for e in ego]) if st.ego_source == "topic"
                                     else cp.ego_time_source_report([e[0] for e in ego], [e[1] for e in ego])),
                             "d435i": cp.d435i_time_source_report([d[0] for d in dm], [d[1] for d in dm]),
                             "common_clock": "laptop CLOCK_REALTIME; the X5 is a chrony client of the laptop "
                                             "(sntp_x5_minus_laptop in this JSON)"},
            "pairing": pairing,
            "validation": {"motion": motion, "flash_pairs": n_flash, "flash_median_raw_ms": st.flash_median_raw()},
            "queues": {"decode_dropped": st.decode_dropped, "pairer_max_pending": pairing["max_pending"]}}


def topic_ego_report(stamp_ns, receive_s):
    """--ego-source topic: the ego_stream stamps (laptop clock) and their arrivals, like the D435i report."""
    rep = cp.d435i_time_source_report(stamp_ns, receive_s)
    rep["meaning"] = ("header.stamp from the ego_stream node: its arrival-envelope model of /frontvideostream "
                      "minus its own calibrated Go2 latency (a receive-time estimate while uncalibrated).")
    return rep


def pairing_line(s, pairs_path=None):
    """One line for the operator (the demo scripts print it after the demo)."""
    p, v = s["pairing"], s["validation"]["motion"]
    if not p["decided"]:
        line = "pairing: no Go2 ego frame decided (camera streams missing?)"
    else:
        d = p["delta_ms"]
        cals = s.get("calibrations") or []
        method = ("mixed: " + " then ".join(f"{c['sync_method']} {c['matched']}/{c['decided']}" for c in cals)
                  if len(cals) > 1 else s["sync_method"])
        line = (f"pairing: {p['matched']}/{p['decided']} matched ({100 * p['unmatched_ratio']:.1f} % unmatched), "
                + (f"|delta| p95 {d['p95_abs']:.1f} ms, " if d.get("n") else "") + method)
        line += (f", common-event offset {v['pairing_offset_ms']:+.0f} ms (n={v['n']})"
                 + (" for the last calibration" if len(cals) > 1 else "") if v["n"]
                 else ", no common-event lag (no motion seen by both cameras)")
    return line + (f"; {pairs_path}" if pairs_path else "")


# ----------------------------------------------------------------------- main
def video_paths_for(record, out_dir, stamp_tag, no_record=False):
    """(stem, {"ego": path, "d435i": path}) of a session. --record PATH names the recording: the two videos
    are <PATH without .mp4>_ego.mp4 and _d435i.mp4, the stem (summary JSON, CSVs, snapshots) is its
    basename; without --record both use ego_d435i_<time> in <out-dir>. --no-record: no videos."""
    base = os.path.splitext(record)[0] if record else os.path.join(out_dir, f"ego_d435i_{stamp_tag}")
    paths = {} if no_record else {"ego": f"{base}_ego.mp4", "d435i": f"{base}_d435i.mp4"}
    return os.path.basename(base), paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", type=int, default=720, choices=(360, 720))
    ap.add_argument("--ns", default="/camera/d435i")
    ap.add_argument("--go2-latency-ms", type=float, default=0.0)
    ap.add_argument("--d435i-stamp-offset-ms", type=float, default=0.0,
                    help="D435i colour header.stamp minus exposure midpoint (sync_check.py 'stamp_minus_exposure')")
    ap.add_argument("--calib", default=None, help="sync_check.py JSON; takes model_input.calibration.{go2_ego_latency_ms,d435i_stamp_minus_exposure_ms}")
    ap.add_argument("--flash-thresh", type=float, default=20.0)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N s (0 = until q)")
    ap.add_argument("--record", default=None,
                    help="recording name: videos <PATH without .mp4>_ego.mp4 and _d435i.mp4 (default <out-dir>/ego_d435i_<time>_*.mp4)")
    ap.add_argument("--no-record", action="store_true", help="do not write the videos")
    ap.add_argument("--grid-hz", type=float, default=50.0, help="capture grid rate = video rate (default 50 = 20 ms)")
    ap.add_argument("--grid-ego-tol-ms", type=float, default=None,
                    help="ego frame within this of a tick (default 40; the nearest frame of the ~14.2 fps stream "
                         "repeats over the ticks until a newer one, marked repeat)")
    ap.add_argument("--ego-source", choices=("frontvideo", "topic"), default="frontvideo",
                    help="frontvideo (default): decode Unitree's /frontvideostream here; topic: sensor_msgs/Image from the "
                         "ego_stream node (the same stream decoded, stamps already on the laptop clock)")
    ap.add_argument("--ego-topic", default="/camera/go2/ego/image_raw", help="--ego-source topic: the image topic")
    ap.add_argument("--slop-ms", type=float, default=20.0, help="pairing window |D435i - Go2 common time| in ms (default 20)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--x5-ip", default=os.environ.get("X5_IP", ""), help="the X5's address for the SNTP check (default $X5_IP)")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(HERE, "..", "..", "data", "logs", f"camera_checks_{time.strftime('%Y-%m-%d')}")
    os.makedirs(out_dir, exist_ok=True)

    import cv2
    import rclpy
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from unitree_go.msg import Go2FrontVideoData
    sys.path.insert(0, HERE)
    import sync_check  # SNTP helper

    calib_src = "args"
    calib_version, go2_calibrated = "none", bool(args.go2_latency_ms)
    if args.go2_latency_ms or args.d435i_stamp_offset_ms:
        calib_version = f"args:L={args.go2_latency_ms:.1f}ms,eD={args.d435i_stamp_offset_ms:.1f}ms"
    if args.calib:
        L, eD, calib_version, from_file = calibration_from_file(args.calib)
        if L is not None:
            args.go2_latency_ms = L
        if eD is not None:
            args.d435i_stamp_offset_ms = eD
        go2_calibrated = go2_calibrated or from_file
        calib_src = args.calib
        print(f"calibration from {args.calib}: Go2 latency {args.go2_latency_ms:.0f} ms, D435i stamp offset {args.d435i_stamp_offset_ms:.0f} ms")
    cal = ego_calibration_for_source(args.ego_source, args.go2_latency_ms, args.d435i_stamp_offset_ms,
                                     calib_version, go2_calibrated)
    if cal["note"]:
        print(cal["note"])
    args.go2_latency_ms, args.d435i_stamp_offset_ms = cal["go2_latency_ms"], cal["d435i_stamp_offset_ms"]
    calib_version, go2_calibrated = cal["calibration_version"], cal["go2_calibrated"]
    repeat, ego_tol = ego_grid_mode(args.grid_ego_tol_ms)
    sampler = gs.GridSampler(rate_hz=args.grid_hz, ego_tol_s=ego_tol, ego_repeat=repeat)
    st = Streams(args.res, args.flash_thresh, args.go2_latency_ms * 1e-3, args.d435i_stamp_offset_ms * 1e-3,
                 slop_s=args.slop_ms * 1e-3, calibration_version=calib_version, go2_calibrated=go2_calibrated,
                 grid=sampler, ego_source=args.ego_source)
    rclpy.init()
    node = Node("ego_d435i_view", enable_rosout=False, start_parameter_services=False)   # no publishers but the two subscriptions
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=30, reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE)
    if args.ego_source == "topic":
        node.create_subscription(Image, args.ego_topic, st.on_ego_image, qos)
    else:
        node.create_subscription(Go2FrontVideoData, "/frontvideostream", st.on_frontvideo, qos, raw=True)
    node.create_subscription(Image, f"{args.ns}/color/image_raw", st.on_color, qos)
    ex = SingleThreadedExecutor(); ex.add_node(node)
    def spin():
        try:
            ex.spin()
        except ExternalShutdownException:       # context shut down at exit (SIGINT / SIGTERM)
            pass
        except Exception:                       # a callback bug stops BOTH subscriptions: log it and leave
            import traceback                    # through the orderly teardown instead of showing a frozen view
            traceback.print_exc()
            rclpy.try_shutdown()
    spin_thread = threading.Thread(target=spin, daemon=True); spin_thread.start()

    sntp = {"v": None}
    def sntp_loop():
        while True:
            sntp["v"] = sync_check.sntp_offset(args.x5_ip, tries=3, timeout=0.5)
            time.sleep(15)
    threading.Thread(target=sntp_loop, daemon=True).start()

    rend = Renderer()
    t0 = time.time(); frames = 0
    stamp_tag = time.strftime("%Y%m%d_%H%M%S")
    stem, video_paths = video_paths_for(args.record, out_dir, stamp_tag, args.no_record)
    snap_dir = os.path.dirname(os.path.abspath(args.record)) if args.record else out_dir   # snapshots next to the videos
    period = 0.01                                           # poll at 100 Hz, write every grid tick
    next_tick = time.time()
    written = 0                                             # grid ticks written (each: one frame in each video)
    stop_reason = "running"                                 # written into the summary: q | duration | context_shutdown | keyboard_interrupt
    video_fps = args.grid_hz
    records = SessionRecords(out_dir, stem, grid=True)      # never raises: a write problem is one warning
    writers, guards = {}, {}                                # camera -> cv2.VideoWriter / FrameSizeGuard

    def open_writer(cam, frame):
        w = cv2.VideoWriter(video_paths[cam], cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (frame.shape[1], frame.shape[0]))
        if not w.isOpened():
            print(f"WARNING: cannot open video writer for {video_paths[cam]}"); video_paths.pop(cam)
            return None
        writers[cam], guards[cam] = w, FrameSizeGuard((frame.shape[1], frame.shape[0]))
        return w

    def write_video(ego, d435i):                            # one call per grid tick -> its frame index (None: no video)
        nonlocal written
        for cam, frame in (("ego", ego), ("d435i", d435i)):
            if cam not in video_paths:
                continue
            w = writers.get(cam)
            if w is None:
                w = open_writer(cam, frame)
            if w is not None:
                w.write(guards[cam](frame))
        if not video_paths:
            return None
        written += 1
        return written - 1

    gsink = GridSink(rend, write_video=write_video, write_row=records.grid, grid_hz=args.grid_hz)
    win_ego, win_d = "Ego", "D435i"
    if not args.no_window:
        for i, win in enumerate((win_ego, win_d)):
            cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)       # 1:1 pixels
            cv2.moveWindow(win, i * (GRID_PANEL_W + 20), 0) # next to each other, not on top of each other
    try:
        while True:
            wait = next_tick - time.time()
            if wait > 0:
                time.sleep(wait)
            next_tick += period
            if time.time() - next_tick > 1.0:           # fell far behind: resync instead of bursting
                next_tick = time.time()
            if not rclpy.ok():                          # SIGTERM / external shutdown: leave through the orderly teardown
                stop_reason = "context_shutdown"
                break
            now = time.time()
            st.update_motion_lag()
            decisions, rows = st.poll_pairs(now)        # the Go2 frames whose D435i partner can no longer change
            records.frames(rows)
            for dec in decisions:                       # the per-ego-frame pairs stay recorded
                records.pair(cp.pair_row(dec, None, *sync_of(dec, st)))
            ticks, rows = st.poll_grid(now)
            records.frames(rows)
            shown = gsink.handle(st, ticks, now)        # one frame per camera video + one grid row per 20 ms tick
            records.flush()
            frames += 1
            key = -1
            if not args.no_window:
                cv2.imshow(win_ego, shown[0]); cv2.imshow(win_d, shown[1])
                key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_reason = "q"
                break
            if key == ord("s") or (args.no_window and args.duration and now - t0 >= args.duration):
                for cam, img in zip(("ego", "d435i"), shown):
                    path = os.path.join(snap_dir, f"{stem}_{frames:05d}_{cam}.png")
                    cv2.imwrite(path, img); print(f"snapshot: {path}")
            if key == ord("c"):
                msg = st.apply_calibration()
                print(msg or "c: nothing measured yet (no hand-wave lag, no D435i offset, no flash) - nothing applied")
            if key == ord("r"):
                st.clear_flashes()
            if args.duration and now - t0 >= args.duration:
                stop_reason = "duration"
                break
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    finally:
        try:                                            # Go2 frames still waiting: decide them with what is buffered
            rest, rows = st.poll_pairs(time.time(), flush=True)
            records.frames(rows)
            for dec in rest:
                records.pair(cp.pair_row(dec, None, *sync_of(dec, st)))
            ticks, rows = st.poll_grid(time.time(), flush=True)
            records.frames(rows)
            gsink.handle(st, ticks, time.time())
        except Exception as e:                          # the records must never stop the orderly teardown
            print(f"WARNING: final pairing flush failed: {e!r}")
        records.close()
        for w in writers.values():
            w.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        ts_sum = sync_summary(st)
        ts_sum.update({"layout": "two_windows", "frames_csv": records.frames_path if records.ok else None,
                       "pairs_csv": records.pairs_path if records.ok else None,
                       "records": {"frames": records.n_frames, "pairs": records.n_pairs},
                       "video_frames": gsink.video_frames})
        ts_sum["grid"] = dict(st.grid.stats.as_dict(), csv=records.grid_path if records.ok else None,
                              ego_source=args.ego_source, panel=f"{GRID_PANEL_W}x{GRID_PANEL_H}")
        summary = {"time": time.strftime("%Y-%m-%d %H:%M:%S %z"), "res": args.res, "rendered_frames": frames,
                   "videos": dict(video_paths) or None, "video_frames": written, "video_fps": video_fps,
                   "layout": "two_windows", "wall_s": time.time() - t0,
                   "msgs": st.msgs, "decode_errors": st.dec.errors, "go2_fit": st.go2_fit.as_dict() if st.go2_fit else None,
                   "go2_latency_ms": st.go2_latency_s * 1e3, "d435i_stamp_offset_ms": st.d_offset_s * 1e3,
                   "calibration_source": calib_src, "flash_pairs": st.flash_pairs,
                   "flash_median_raw_ms": st.flash_median_raw(), "sntp_x5_minus_laptop": sntp["v"],
                   "motion_lag_latest": st.motion_lag.as_dict() if st.motion_lag else None,
                   "motion_lag_hist": st.motion_hist,
                   "motion_lag_median_ms": float(np.median([h["lag_ms"] for h in st.motion_hist])) if st.motion_hist else None,
                   "d435i_recv_minus_stamp_ms_median": float(np.median(st.d_lat_ms)) if st.d_lat_ms else None,
                   "msgs_by_res": st.msgs_by_res, "stop_reason": stop_reason,
                   "process": {"pid": os.getpid(), "argv": sys.argv[1:], "nice": os.nice(0),
                               "affinity": sorted(os.sched_getaffinity(0)), "own_session": os.getsid(0) == os.getpid(),
                               "fastdds_profile": os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE"),
                               "cpu_time_s": os.times().user + os.times().system},
                   "time_sync": ts_sum}
        path = os.path.join(out_dir, f"{stem}.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=float)
        print(json.dumps({k: v for k, v in summary.items() if k not in ("flash_pairs", "motion_lag_hist", "time_sync")},
                         indent=1, default=float))
        print(f"flash pairs: {len(st.flash_pairs)}  summary: {path}")
        for cam, vp in video_paths.items():
            print(f"video {cam}: {vp}  ({written} frames @ {video_fps:g} fps = {written / video_fps:.1f} s; wall {time.time() - t0:.1f} s)")
        print(gs.grid_line(ts_sum["grid"]) + (f"; {records.grid_path}" if records.ok else ""))
        print(pairing_line(ts_sum, ts_sum["pairs_csv"]))
        # orderly teardown: stop the executor and join its thread BEFORE rclpy.shutdown, otherwise the
        # DDS threads are torn down under the spinning executor ("terminate called without an active exception")
        ex.shutdown(timeout_sec=2.0)
        spin_thread.join(timeout=3.0)
        node.destroy_node()
        if rclpy.ok():                          # SIGINT: rclpy's own handler has already shut the context down
            rclpy.shutdown()


if __name__ == "__main__":
    main()
