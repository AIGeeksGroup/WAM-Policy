"""Pair Go2 ego and D435i colour frames inside a bounded time window, and report
where every time value came from.

Pure Python/numpy, no ROS: used by scripts/x5/ego_d435i_view.py (live, every
demo) and scripts/x5/pair_report.py (offline, several sessions); unit-tested in
test/test_camera_pairing.py. Not thread-safe: the viewer calls it under its lock.

How the pairing works:

1. Per-frame records - raw device/driver time, laptop receive time, frame id,
   time source; nothing is replaced by ``now()`` after decoding
   (:class:`Frame`, :data:`FRAMES_HEADER`).
2. One common time base = the laptop CLOCK_REALTIME (the X5 is a chrony client
   of the laptop). ``common = a * device_time + b``:
   * Go2 ego: ``time_frame`` is a frame counter, not a clock (+6667/6666 per
     frame = 1/15 s in 10 us units while frames really come at ~14.2 fps;
     :func:`ego_time_source_report` re-checks this every session). a and b come
     from the low-delay envelope of many arrivals (``timesync.fit_tick_clock``),
     never from one frame, minus the pipeline latency L. With L uncalibrated
     the result is a RECEIVE-time estimate and the pairs are labelled
     ``receive_time_approx``, not capture-time pairs.
   * D435i: ``header.stamp`` = librealsense global time (camera hardware clock
     mapped onto the X5 system clock), minus the stamp-to-exposure offset eD
     (:func:`d435i_time_source_report`).
3. :class:`BoundedPairer`: a bounded queue on the laptop. Each decoded Go2 ego
   frame (the slower stream, the anchor) takes the nearest unused D435i frame
   with ``|delta| <= slop`` (default 20 ms); otherwise it is marked UNMATCHED
   with the reason. message_filters.ApproximateTimeSynchronizer is not usable
   here: /frontvideostream has no header and is decoded from raw CDR.
4. Samples: ``pair_id`` / ``pair_time`` (= the ego frame's common time),
   raw + receive + common time of both frames, ``delta_ms``, ``sync_method``,
   ``calibration_version`` (:data:`PAIRS_HEADER`). Raw times are never
   overwritten; no frame is used twice and nothing is duplicated to fill gaps.
5. Validation with a common visual event: the robot's own motion (both cameras
   are on the body) or a hand-wave gives lag = L - eD by motion-energy cross-
   correlation (the viewer); :func:`motion_validation` turns it into the
   pairs' true capture skew, :func:`aggregate_motion_lags` /
   :func:`relative_calibration` into a calibration across sessions.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

EGO_TF_UNITS_PER_FRAME = 1e5 / 15      # time_frame step per frame: 1/15 s in 10 us units (6666.67)
SYNC_CAPTURE = "capture_time"          # Go2 latency L calibrated by a common event: capture-time pairs
SYNC_RECEIVE = "receive_time_approx"   # L uncalibrated: Go2 side is a receive-time estimate
UNMATCHED_REASONS = ("no_time_model", "no_d435i_frames", "outside_slop", "d435i_used")
TIME_SOURCE = {"ego": "go2_time_frame", "d435i": "d435i_header_stamp_ns"}
FRAMES_HEADER = ["stream", "frame_id", "raw_time", "time_source", "receive_time"]
PAIRS_HEADER = ["pair_id", "status", "pair_time",
                "ego_frame_id", "ego_raw_time", "ego_receive_time", "ego_common_time",
                "d435i_frame_id", "d435i_raw_time", "d435i_receive_time", "d435i_common_time",
                "delta_ms", "video_frame", "sync_method", "calibration_version"]


@dataclass
class Frame:
    stream: str                    # "ego" | "d435i"
    frame_id: int                  # per-stream receive index of this session
    raw_time: int                  # ego: time_frame as received; d435i: header.stamp in ns (X5 clock)
    receive_time: float            # laptop CLOCK_REALTIME in the subscription callback (s)
    common_time: float | None      # comparable time on the laptop clock (s); None: no Go2 time model yet
    payload: object = None         # the image (never written to the records)


@dataclass
class PairDecision:
    ego: Frame
    d435i: Frame | None            # the paired frame, or for an unmatched one the nearest rejected candidate
    matched: bool
    reason: str                    # "" when matched, else one of UNMATCHED_REASONS
    delta_s: float | None          # d435i.common_time - ego.common_time
    pair_id: str | None            # "pair_000000", ... for matched decisions only


def _stats(x, abs_keys=True) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    out = {"n": int(x.size), "median": float(np.median(x)), "min": float(x.min()), "max": float(x.max())}
    if abs_keys:
        a = np.abs(x)
        out.update({"p95_abs": float(np.percentile(a, 95)), "max_abs": float(a.max())})
    return out


def delta_stats(x) -> dict:
    """n / median / min / max / p95_abs / max_abs of a list of time differences (ms)."""
    return _stats(x)


class PairStats:
    """Running counts for the report; the duplicate check is independent of the pairer's own bookkeeping."""

    def __init__(self, slop_s: float):
        self.slop_s = slop_s
        self.decided = 0
        self.matched = 0
        self.unmatched = collections.Counter()
        self.deltas_ms = []
        self.duplicates = 0
        self.max_pending = 0
        self._paired_ids = set()

    def record(self, dec: PairDecision) -> None:
        self.decided += 1
        if dec.matched:
            self.matched += 1
            self.deltas_ms.append(dec.delta_s * 1e3)
            if dec.d435i.frame_id in self._paired_ids:
                self.duplicates += 1
            self._paired_ids.add(dec.d435i.frame_id)
        else:
            self.unmatched[dec.reason] += 1

    def as_dict(self, n_d435i_frames: int | None = None) -> dict:
        unmatched = sum(self.unmatched.values())
        return {"slop_ms": self.slop_s * 1e3, "anchor": "ego",
                "decided": self.decided, "matched": self.matched, "unmatched": unmatched,
                "unmatched_ratio": unmatched / self.decided if self.decided else None,
                "unmatched_by_reason": dict(self.unmatched),
                "delta_ms": _stats(self.deltas_ms),
                "duplicate_d435i_in_pairs": self.duplicates,
                "d435i_frames": n_d435i_frames,
                "d435i_unpaired": None if n_d435i_frames is None else n_d435i_frames - self.matched,
                "max_pending": self.max_pending}


class BoundedPairer:
    """Online pairing with a bounded D435i buffer.

    An ego frame is decided, in arrival order, as soon as no closer D435i frame can still come:
    the newest buffered D435i frame is past ``common + slop`` (frames arrive in stamp order), or
    the ego frame has waited ``max_wait_s``, or the D435i stream has been silent ``silence_s``
    (so a dead X5 never delays the ego view), or it has no common time (no Go2 time model yet).
    """

    def __init__(self, slop_s: float = 0.020, max_wait_s: float = 0.35, keep_s: float = 1.5,
                 silence_s: float = 1.0, max_keep: int = 90):
        self.slop_s, self.max_wait_s, self.keep_s, self.silence_s, self.max_keep = slop_s, max_wait_s, keep_s, silence_s, max_keep
        self.d435i_buffer = collections.deque()
        self.pending = collections.deque()
        self.stats = PairStats(slop_s)
        self._used = set()             # matched D435i ids still in the buffer
        self._last_d_rx = None

    @property
    def pending_count(self) -> int:
        return len(self.pending)

    def add_d435i(self, frame: Frame) -> None:
        buf = self.d435i_buffer
        buf.append(frame)
        self._last_d_rx = frame.receive_time if self._last_d_rx is None else max(self._last_d_rx, frame.receive_time)
        while buf and (len(buf) > self.max_keep or buf[0].common_time < frame.common_time - self.keep_s):
            self._used.discard(buf.popleft().frame_id)

    def add_ego(self, frame: Frame) -> None:
        self.pending.append(frame)
        self.stats.max_pending = max(self.stats.max_pending, len(self.pending))

    def _ready(self, e: Frame, now: float) -> bool:
        if e.common_time is None:
            return True
        if self._last_d_rx is None or now - self._last_d_rx >= self.silence_s:
            return True
        if self.d435i_buffer and self.d435i_buffer[-1].common_time >= e.common_time + self.slop_s:
            return True
        return now - e.receive_time >= self.max_wait_s

    def poll(self, now: float) -> list:
        out = []
        while self.pending and self._ready(self.pending[0], now):
            out.append(self._decide(self.pending.popleft()))
        return out

    def flush(self) -> list:
        """Decide every pending ego frame with what is buffered (shutdown)."""
        out = []
        while self.pending:
            out.append(self._decide(self.pending.popleft()))
        return out

    def _decide(self, e: Frame) -> PairDecision:
        if e.common_time is None:
            dec = PairDecision(e, None, False, "no_time_model", None, None)
        elif not self.d435i_buffer:
            dec = PairDecision(e, None, False, "no_d435i_frames", None, None)
        else:
            t = e.common_time
            key = lambda f: (abs(f.common_time - t), f.common_time)   # ties: the earlier frame
            nearest = min(self.d435i_buffer, key=key)
            free = [f for f in self.d435i_buffer if f.frame_id not in self._used]
            best = min(free, key=key) if free else None
            if best is not None and abs(best.common_time - t) <= self.slop_s:
                self._used.add(best.frame_id)
                dec = PairDecision(e, best, True, "", best.common_time - t, f"pair_{self.stats.matched:06d}")
            else:
                in_window_used = nearest.frame_id in self._used and abs(nearest.common_time - t) <= self.slop_s
                dec = PairDecision(e, nearest, False, "d435i_used" if in_window_used else "outside_slop",
                                   nearest.common_time - t, None)
        self.stats.record(dec)
        return dec


def true_skew_s(delta_s: float, lag_s: float, l_cfg_s: float, ed_cfg_s: float) -> float:
    """True capture skew (D435i capture - ego capture) of a pair made with L_cfg / eD_cfg, given
    the common-event lag = Go2 arrival model - D435i stamp for the same motion = L - eD:
    delta = (Td - Te) + (eD - eD_cfg) - (L - L_cfg)  =>  Td - Te = delta + lag - (L_cfg - eD_cfg)."""
    return delta_s + lag_s - (l_cfg_s - ed_cfg_s)


def ego_time_source_report(tf, receive_s) -> dict:
    """Time-source check of the Go2 ego stream: is time_frame a clock or a frame counter, were messages
    lost, did the time base reset? ``tf``/``receive_s``: every message at the decoded resolution."""
    tf = np.asarray(tf, dtype=np.int64)
    rx = np.asarray(receive_s, dtype=np.float64)
    n = int(tf.size)
    if n < 3:
        return {"n": n, "verdict": "too_few_frames"}
    steps = np.diff(tf)
    fwd = steps > 0
    k = np.rint(steps / EGO_TF_UNITS_PER_FRAME)
    whole = fwd & (k >= 1) & (np.abs(steps - k * EGO_TF_UNITS_PER_FRAME) <= 1.5)
    single, multi = whole & (k == 1), whole & (k >= 2)
    n_fwd = int(fwd.sum())
    counter_span = float(steps[fwd].sum()) * 1e-5
    rx_span = float(rx[-1] - rx[0])
    rate = (n - 1) / rx_span if rx_span > 0 else None
    is_counter = n_fwd > 0 and whole.sum() >= 0.99 * n_fwd and single.sum() >= 0.9 * n_fwd
    out = {"n": n, "single_frame_steps": int(single.sum()), "multi_frame_steps": int(multi.sum()),
           "lost_frames": int((k[multi] - 1).sum()), "zero_steps": int((steps == 0).sum()),
           "backward_steps": int((steps < 0).sum()), "other_steps": int((fwd & ~whole).sum()),
           "single_frame_fraction": float(single.sum()) / n_fwd if n_fwd else None,
           "receive_rate_hz": rate, "counter_span_s": counter_span, "receive_span_s": rx_span,
           "receive_over_counter": rx_span / counter_span if counter_span > 0 else None,
           "verdict": "frame_counter" if is_counter else "irregular"}
    if is_counter:
        out["meaning"] = (f"time_frame advances 1/15 s (6667/6666 x 10 us) per frame while frames arrive at "
                          f"{rate:.2f} fps: a frame counter, not a clock. Ego common time = arrival-envelope fit "
                          "a*time_frame+b minus the pipeline latency L; with L uncalibrated it is a receive-time estimate.")
    else:
        out["meaning"] = "time_frame steps are not whole 1/15 s frames: re-check its meaning before using it"
    return out


def d435i_time_source_report(stamp_ns, receive_s) -> dict:
    """Time-source check of the D435i colour stream: frame period, lost / repeated / backward stamps,
    and receive - stamp (transport + any X5-laptop clock offset)."""
    s = np.asarray(stamp_ns, dtype=np.int64)
    rx = np.asarray(receive_s, dtype=np.float64)
    n = int(s.size)
    if n < 3:
        return {"n": n}
    steps_ms = np.diff(s) / 1e6
    pos = steps_ms[steps_ms > 0]
    period = float(np.median(pos)) if pos.size else None
    gaps = steps_ms > 1.5 * period if period else np.zeros(steps_ms.size, bool)
    span_s = (s[-1] - s[0]) / 1e9
    rx_span = float(rx[-1] - rx[0])
    return {"n": n, "period_ms": period,
            "stamp_rate_hz": (n - 1) / span_s if span_s > 0 else None,
            "receive_rate_hz": (n - 1) / rx_span if rx_span > 0 else None,
            "gaps": int(gaps.sum()), "lost_frames": int((np.rint(steps_ms[gaps] / period) - 1).sum()) if period else 0,
            "zero_steps": int((steps_ms == 0).sum()), "backward_steps": int((steps_ms < 0).sum()),
            "recv_minus_stamp_ms": _stats((rx - s / 1e9) * 1e3, abs_keys=False),
            "meaning": ("header.stamp = librealsense global time: the camera hardware clock mapped onto the X5 "
                        "system clock, which chrony keeps on the laptop clock. A device-side frame time, not the "
                        "exposure midpoint (the X5 driver publishes no exposure metadata): D435i common time = "
                        "stamp - eD.")}


def motion_validation(lags_ms, l_cfg_ms: float, ed_cfg_ms: float, deltas_ms=None) -> dict:
    """Validation with a common visual event seen by both cameras (the robot's own motion, a hand-wave)
    gives lag = Go2 arrival model - D435i stamp = L - eD per correlation window. ``pairing_offset_ms``
    = how far the pairs made with L_cfg / eD_cfg are from one capture instant (> 0: the D435i frame
    was captured later); ``true_skew_ms`` applies it to the pairs' deltas."""
    lags = np.asarray(list(lags_ms), dtype=np.float64)
    lags = lags[np.isfinite(lags)]
    if lags.size == 0:
        return {"n": 0, "meaning": "no common-event lag measured (no motion seen by both cameras)"}
    med = float(np.median(lags))
    cfg = l_cfg_ms - ed_cfg_ms
    offset = med - cfg
    out = {"n": int(lags.size),
           "lag_ms": {"min": float(lags.min()), "p25": float(np.percentile(lags, 25)), "median": med,
                      "p75": float(np.percentile(lags, 75)), "max": float(lags.max())},
           "configured_l_minus_ed_ms": cfg, "pairing_offset_ms": offset,
           "suggested_go2_latency_ms": med + ed_cfg_ms, "true_skew_ms": None,
           "meaning": "lag = Go2 arrival model - D435i stamp for the same motion = L - eD; "
                      "true skew of a pair = delta + lag - (L_cfg - eD_cfg)"}
    if deltas_ms is not None and len(deltas_ms):
        out["true_skew_ms"] = _stats(np.asarray(deltas_ms, dtype=np.float64) + offset)
    return out


def aggregate_motion_lags(sessions) -> dict:
    """Long-term re-test: ``sessions`` = [(name, [lag_ms, ...]), ...] in time order."""
    rows, allv = [], []
    for name, lags in sessions:
        v = [float(x) for x in lags if x is not None and np.isfinite(x)]
        rows.append({"name": name, "n": len(v), "median_ms": float(np.median(v)) if v else None})
        allv += v
    meds = [r["median_ms"] for r in rows if r["median_ms"] is not None]
    a = np.asarray(allv, dtype=np.float64)
    return {"sessions": rows, "n_windows": int(a.size),
            "median_ms": float(np.median(a)) if a.size else None,
            "p25_ms": float(np.percentile(a, 25)) if a.size else None,
            "p75_ms": float(np.percentile(a, 75)) if a.size else None,
            "min_ms": float(a.min()) if a.size else None, "max_ms": float(a.max()) if a.size else None,
            "session_median_spread_ms": float(max(meds) - min(meds)) if meds else None}


def relative_calibration(agg: dict, ed_cfg_ms: float = 0.0) -> dict:
    """A --calib file for ego_d435i_view.py from aggregated camera-camera lags. Only L - eD is
    measured, so it calibrates the PAIRING (relative); absolute capture times still need eD
    (scripts/x5/sync_check.py, D435i colour motion vs its gyro)."""
    if not agg.get("n_windows"):
        raise ValueError("no common-event lag in the given sessions: nothing to calibrate from")
    return {"kind": "relative_camera_motion",
            "sessions": [r["name"] for r in agg["sessions"] if r["n"]],
            "n_windows": agg["n_windows"], "session_median_spread_ms": agg["session_median_spread_ms"],
            "lag_ms": {k: agg[k] for k in ("min_ms", "p25_ms", "median_ms", "p75_ms", "max_ms")},
            "model_input": {"calibration": {
                "go2_ego_latency_ms": agg["median_ms"] + ed_cfg_ms,
                "d435i_stamp_minus_exposure_ms": ed_cfg_ms,
                "note": "relative: L - eD from camera-camera motion cross-correlation; pairs become capture-time "
                        "pairs, absolute capture times still need eD from sync_check.py"}}}


def _fmt(t):
    return "" if t is None else f"{t:.6f}"


def frame_row(f: Frame, time_source: str | None = None) -> list:
    return [f.stream, f.frame_id, f.raw_time, time_source or TIME_SOURCE[f.stream], _fmt(f.receive_time)]


def pair_row(dec: PairDecision, video_frame, sync_method: str, calibration_version: str) -> list:
    e, d = dec.ego, dec.d435i
    return [dec.pair_id or "", "matched" if dec.matched else f"unmatched:{dec.reason}",
            _fmt(e.common_time) if dec.matched else "",
            e.frame_id, e.raw_time, _fmt(e.receive_time), _fmt(e.common_time),
            "" if d is None else d.frame_id, "" if d is None else d.raw_time,
            "" if d is None else _fmt(d.receive_time), "" if d is None else _fmt(d.common_time),
            "" if dec.delta_s is None else f"{dec.delta_s * 1e3:.3f}",
            "" if video_frame is None else video_frame, sync_method, calibration_version]
