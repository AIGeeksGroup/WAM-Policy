"""Both cameras resampled onto one 50 Hz capture-time grid (640x360, time-synchronised).

Neither camera has a 50 fps mode (D435i colour 640x360: 60/30/15/6 fps; the Go2 front camera
~14.2 fps through /frontvideostream), so the 50 fps capture is a 20 ms grid on the laptop clock: for
every tick each camera contributes the frame whose capture time is nearest the tick, within a
tolerance. Pure Python, no ROS; used by the viewer (scripts/x5/ego_d435i_view.py: two windows, one
video per camera, one frame per tick), unit-tested in test/test_grid_sync.py.

Rules (the bounded-window pairing of camera_pairing.py applied to a fixed rate):
* D435i: a frame serves at most one tick (``new``); tolerance at most half a period (10 ms at
  50 Hz), so a 60 fps stream gives a new, distinct frame on every tick.
* Go2 ego: the nearest frame within 40 ms (``ego_repeat=True``), ``new`` on its first tick, ``repeat``
  afterwards - never presented as a fresh capture. At ~14.2 fps a frame spans ~3.5 ticks (the viewer
  always uses this mode: ego_d435i_view.ego_grid_mode). ``ego_repeat=False`` (a frame at most once,
  tolerance at most half a period) suits an ego stream as fast as the grid.
* No frame within the tolerance: ``missing``; nothing is filled in or interpolated.
* A tick is decided once no closer frame can still arrive (every live stream is past
  tick + tolerance), after ``max_wait_s`` (a late stream is then missing for that tick), or with a
  stream silent for ``silence_s``; the grid never runs past the newest frame of any stream.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass

from .camera_pairing import Frame, _fmt, _stats

STREAMS = ("ego", "d435i")
GRID_HEADER = ["tick", "grid_time",
               "ego_status", "ego_frame_id", "ego_raw_time", "ego_common_time", "ego_offset_ms",
               "d435i_status", "d435i_frame_id", "d435i_raw_time", "d435i_common_time", "d435i_offset_ms",
               "video_frame", "sync_method", "calibration_version"]


@dataclass
class GridTick:
    index: int                      # tick number: time = index / rate (laptop CLOCK_REALTIME)
    time: float
    ego: Frame | None
    ego_status: str                 # new | repeat | missing
    ego_offset_s: float | None      # frame common time - tick time
    d435i: Frame | None
    d435i_status: str               # new | missing
    d435i_offset_s: float | None


class _Stream:
    def __init__(self, name, tol_s, repeat):
        self.name, self.tol_s, self.repeat = name, tol_s, repeat
        self.buf = collections.deque()
        self.used = set()
        self.last_id = None                 # frame of the previous decided tick
        self.first_rx = self.last_rx = None
        self.first_t = self.newest_t = None
        self.n_frames = 0
        self.no_time_model = 0
        self.counts = collections.Counter()
        self.offsets_ms = []
        self.ids_new = []

    def add(self, f: Frame):
        if f.common_time is None:
            self.no_time_model += 1
            return
        self.n_frames += 1
        self.first_rx = f.receive_time if self.first_rx is None else self.first_rx
        self.last_rx = f.receive_time if self.last_rx is None else max(self.last_rx, f.receive_time)
        self.first_t = f.common_time if self.first_t is None else min(self.first_t, f.common_time)
        self.newest_t = f.common_time if self.newest_t is None else max(self.newest_t, f.common_time)
        if self.buf and f.common_time < self.buf[-1].common_time:       # out of order: keep the buffer sorted
            items = sorted(list(self.buf) + [f], key=lambda x: x.common_time)
            self.buf = collections.deque(items)
        else:
            self.buf.append(f)

    def silent(self, now, silence_s, start_rx):
        ref = self.last_rx if self.last_rx is not None else start_rx
        return ref is not None and now - ref >= silence_s

    def pick(self, t):
        """(frame | None, status) for tick time t."""
        key = lambda f: (abs(f.common_time - t), f.common_time)           # ties: the earlier frame
        within = [f for f in self.buf if abs(f.common_time - t) <= self.tol_s + 1e-9]
        if self.repeat:
            # the nearest frame, whatever it is: "new" on its first tick, "repeat" on the later ones
            # (never a farther, not yet shown frame early - that would move it off its capture time)
            if not within:
                return None, "missing"
            best = min(within, key=key)
            if best.frame_id in self.used:
                return best, "repeat"
            self.used.add(best.frame_id)
            return best, "new"
        free = [f for f in within if f.frame_id not in self.used]
        if not free:
            return None, "missing"
        best = min(free, key=key)
        self.used.add(best.frame_id)
        return best, "new"

    def trim(self, before):
        while self.buf and self.buf[0].common_time < before:
            self.used.discard(self.buf.popleft().frame_id)

    def record(self, f, status, offset_s):
        self.counts[status] += 1
        if f is not None:
            self.offsets_ms.append(offset_s * 1e3)
            self.last_id = f.frame_id
        else:
            self.last_id = None
        if status == "new":
            self.ids_new.append(f.frame_id)

    def report(self, ticks):
        span = (self.newest_t - self.first_t) if self.n_frames > 1 else 0.0
        dup = len(self.ids_new) - len(set(self.ids_new))
        return {"tol_ms": self.tol_s * 1e3, "repeat_allowed": self.repeat, "frames": self.n_frames,
                "fps": (self.n_frames - 1) / span if span > 0 else None,
                "new": self.counts["new"], "repeat": self.counts["repeat"], "missing": self.counts["missing"],
                "new_ratio": self.counts["new"] / ticks if ticks else None,
                "offset_ms": _stats(self.offsets_ms), "duplicates": dup, "no_time_model": self.no_time_model}


class GridStats:
    def __init__(self, sampler):
        self._s = sampler

    def as_dict(self) -> dict:
        s = self._s
        return {"rate_hz": s.rate_hz, "period_ms": s.period_s * 1e3, "ticks": s.n_ticks,
                "first_tick": s.first_index, "last_tick": None if s.next_index is None else s.next_index - 1,
                "max_wait_ms": s.max_wait_s * 1e3,
                "ego": s.streams["ego"].report(s.n_ticks), "d435i": s.streams["d435i"].report(s.n_ticks)}


class GridSampler:
    def __init__(self, rate_hz: float = 50.0, ego_tol_s: float = 0.040, d435i_tol_s: float | None = None,
                 ego_repeat: bool = True, max_wait_s: float = 0.40, silence_s: float = 1.0, keep_s: float = 2.0):
        if not rate_hz > 0:
            raise ValueError("rate_hz must be positive")
        self.rate_hz = float(rate_hz)
        self.period_s = 1.0 / self.rate_hz
        d_tol = 0.5 * self.period_s if d435i_tol_s is None else float(d435i_tol_s)
        if d_tol > 0.5 * self.period_s + 1e-9:
            raise ValueError("d435i_tol_s above half a grid period would let one frame serve two ticks")
        if not ego_tol_s > 0:
            raise ValueError("ego_tol_s must be positive")
        if not ego_repeat and ego_tol_s > 0.5 * self.period_s + 1e-9:
            raise ValueError("without repeats ego_tol_s must be at most half a grid period")
        self.max_wait_s, self.silence_s, self.keep_s = float(max_wait_s), float(silence_s), float(keep_s)
        self.streams = {"ego": _Stream("ego", float(ego_tol_s), bool(ego_repeat)),
                        "d435i": _Stream("d435i", d_tol, False)}
        self.next_index = None
        self.first_index = None
        self.n_ticks = 0
        self.stats = GridStats(self)

    def add_ego(self, f: Frame) -> None:
        self.streams["ego"].add(f)

    def add_d435i(self, f: Frame) -> None:
        self.streams["d435i"].add(f)

    # ------------------------------------------------------------------ deciding ticks
    def _start_rx(self):
        rx = [s.first_rx for s in self.streams.values() if s.first_rx is not None]
        return min(rx) if rx else None

    def _start(self, now):
        have = [s for s in self.streams.values() if s.n_frames]
        if not have:
            return False
        if len(have) < len(self.streams) and not any(
                s.silent(now, self.silence_s, self._start_rx()) for s in self.streams.values() if not s.n_frames):
            return False                                   # wait for the other stream (or its silence)
        t0 = max(s.first_t for s in have)
        self.next_index = math.ceil(t0 * self.rate_hz - 1e-4)
        self.first_index = self.next_index
        return True

    def _time(self, index):
        return index * self.period_s

    def _horizon(self, now, flush):
        live = [s for s in self.streams.values()
                if s.n_frames and (flush or not s.silent(now, self.silence_s, self._start_rx()))]
        if flush:
            return min((s.newest_t + s.tol_s for s in live), default=None)
        have = [s for s in self.streams.values() if s.n_frames]
        return max((s.newest_t + s.tol_s for s in have), default=None)

    def _ready(self, t, now):
        if now - t >= self.max_wait_s:
            return True
        for s in self.streams.values():
            if s.silent(now, self.silence_s, self._start_rx()):
                continue
            if s.newest_t is None or s.newest_t < t + s.tol_s:
                return False
        return True

    def _decide(self, index):
        t = self._time(index)
        picks = {}
        for name, s in self.streams.items():
            f, status = s.pick(t)
            off = None if f is None else f.common_time - t
            s.record(f, status, off if off is not None else 0.0)
            picks[name] = (f, status, off)
            s.trim(t - self.keep_s)
        self.n_ticks += 1
        (e, es, eo), (d, ds, do) = picks["ego"], picks["d435i"]
        return GridTick(index, t, e, es, eo, d, ds, do)

    def poll(self, now: float) -> list:
        if self.next_index is None and not self._start(now):
            return []
        out = []
        horizon = self._horizon(now, flush=False)
        while horizon is not None and self._time(self.next_index) <= horizon + 1e-9 \
                and self._ready(self._time(self.next_index), now):
            out.append(self._decide(self.next_index))
            self.next_index += 1
        return out

    def flush(self, now: float | None = None) -> list:
        """Shutdown: decide every tick up to where all streams still had frames."""
        if self.next_index is None and not self._start(float("inf") if now is None else now):
            return []
        out = []
        horizon = self._horizon(float("inf") if now is None else now, flush=True)
        while horizon is not None and self._time(self.next_index) <= horizon + 1e-9:
            out.append(self._decide(self.next_index))
            self.next_index += 1
        return out


def grid_row(tick: GridTick, video_frame, sync_method: str, calibration_version: str) -> list:
    def side(f, status, off):
        if f is None:
            return [status, "", "", "", ""]
        return [status, f.frame_id, f.raw_time, _fmt(f.common_time), f"{off * 1e3:.3f}"]
    return ([tick.index, _fmt(tick.time)] + side(tick.ego, tick.ego_status, tick.ego_offset_s)
            + side(tick.d435i, tick.d435i_status, tick.d435i_offset_s)
            + ["" if video_frame is None else video_frame, sync_method, calibration_version])


def grid_line(rep: dict) -> str:
    """One operator line, printed by the viewer at the end (the demo scripts show it)."""
    def part(name, label):
        r = rep[name]
        fps = "" if r["fps"] is None else f"{r['fps']:.1f} fps, "
        off = r["offset_ms"]
        p95 = f"|offset| p95 {off['p95_abs']:.1f} ms, " if off.get("n") else ""
        ratio = 0.0 if r["new_ratio"] is None else 100.0 * r["new_ratio"]
        return f"{label} new {ratio:.1f} % ({fps}{p95}repeat {r['repeat']}, missing {r['missing']})"
    secs = rep["ticks"] / rep["rate_hz"] if rep["rate_hz"] else 0.0
    return f"grid {rep['rate_hz']:g} Hz: {rep['ticks']} ticks ({secs:.1f} s); {part('d435i', 'D435i')}; {part('ego', 'ego')}"

