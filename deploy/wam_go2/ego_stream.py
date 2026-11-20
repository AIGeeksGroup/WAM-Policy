"""The Go2 front ("ego") camera stream as timed RGB frames, for the ego_stream node.

``/frontvideostream`` (unitree_go/Go2FrontVideoData) does not match the official
message on this firmware; its wire layout is
``uint64 time_frame | uint32 resolution (360|720) | uint8[] H.264 access unit``
and ``time_frame`` is a frame counter in disguise (it advances 1/15 s per frame
while frames arrive at ~14.2 fps). The frame time on the laptop clock is the
low-delay envelope fit of arrival time against ``time_frame`` minus the
capture-to-DDS latency measured with a common event (scripts/x5/sync_check.py,
ego_d435i_view.py's hand-wave or flash calibration; one measurement gave
+43.7 ms relative to the D435i stamps).

``parse_frontvideo`` and ``Go2Decoder`` are verbatim copies of the field-verified
code in scripts/x5/ego_d435i_view.py; a test keeps them identical. The time model
is the viewer's, as a class.
"""
from __future__ import annotations

import collections
import struct
from typing import Optional

import numpy as np

from . import timesync as ts

TF_UNIT_S = 1e-5            # nominal unit of time_frame (10 us); the fitted slope is what is used
GO2_PERIOD_S = 0.0703       # nominal frame period (14.23 fps)


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


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(bgr[:, :, ::-1])


class EgoTimeModel:
    """Frame time on the laptop clock from (time_frame, arrival) pairs.

    ``time_of(tf)`` = arrival-envelope fit (timesync.fit_tick_clock, refitted every
    ``refit_s`` once ``min_samples`` messages are known) minus ``latency_s``.
    ``calibrated`` is False while the latency is the uncalibrated zero: then the
    times are receive-time estimates, labelled so in records.
    """

    def __init__(self, latency_s: float = 0.0, *, min_samples: int = 30, refit_s: float = 1.0, keep: int = 900):
        self.latency_s = float(latency_s)
        self.min_samples = int(min_samples)
        self.refit_s = float(refit_s)
        self.pairs = collections.deque(maxlen=int(keep))
        self.fit: Optional[ts.TickFit] = None
        self._last_fit = None
        self.fit_errors = 0

    @property
    def calibrated(self) -> bool:
        return self.latency_s != 0.0

    def add(self, tf: int, arrival: float) -> None:
        self.pairs.append((int(tf), float(arrival)))
        if len(self.pairs) >= self.min_samples and (self._last_fit is None or arrival - self._last_fit > self.refit_s):
            tfs = np.array([p[0] for p in self.pairs], dtype=np.int64)
            arr = np.array([p[1] for p in self.pairs])
            try:                                           # iters=2 keeps ~25 % of a few hundred samples on the envelope
                self.fit = ts.fit_tick_clock(tfs, arr, iters=2, nominal_s_per_tick=TF_UNIT_S)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                self.fit_errors += 1
            self._last_fit = arrival

    def time_of(self, tf: int, calibrated: bool = True) -> Optional[float]:
        if self.fit is None:
            return None
        t = float(self.fit.time_of(np.array([int(tf)], dtype=np.int64))[0])
        if not np.isfinite(t):
            return None
        return t - self.latency_s if calibrated else t

    def report(self) -> dict:
        return {"latency_s": self.latency_s, "calibrated": self.calibrated, "samples": len(self.pairs),
                "fit": None if self.fit is None else self.fit.as_dict(), "fit_errors": self.fit_errors}
