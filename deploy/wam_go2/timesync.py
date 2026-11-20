"""Offline time-alignment helpers for recordings that mix hosts and clocks
(laptop wall clock, Go2 MCU ``tick``, RDK X5 / D435i stamps). Pure numpy/scipy,
no ROS: used by ``scripts/x5/sync_check.py`` (the pre-recording sync check), the
camera viewer and the ego stream; unit-tested in ``test/test_timesync.py``.

Three clocks meet in a Go2 + X5 recording:

* the laptop wall clock (CLOCK_REALTIME) - the recording reference;
* the Go2 MCU ``tick`` in ``/lowstate`` (uint32 ms, free running, no wall
  clock) - :func:`fit_tick_clock` maps it onto the laptop clock;
* the X5 wall clock, disciplined by chrony from the laptop, which stamps the
  D435i frames and IMU samples - :func:`gyro_lag` measures the residual shift
  between the robot gyro (on the fitted tick clock) and the camera gyro.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import signal

TICK_WRAP = 1 << 32


def unwrap_tick(tick_ms) -> np.ndarray:
    """uint32 ms counter -> monotonically increasing float64 ms."""
    t = np.asarray(tick_ms, dtype=np.int64)
    if t.size == 0:
        return t.astype(np.float64)
    d = np.diff(t)
    d = np.where(d < -(TICK_WRAP // 2), d + TICK_WRAP, d)
    return np.concatenate(([t[0]], t[0] + np.cumsum(d))).astype(np.float64)


@dataclass
class TickFit:
    """recv_s ~= slope * tick_ms + offset_s for the low-delay envelope."""
    slope: float                 # seconds per tick; nominal_s_per_tick (1e-3 for the ms tick) = nominal rate
    offset_s: float
    rate_ppm: float              # MCU tick rate error relative to the receiver clock
    n: int
    n_used: int                  # samples that survived the envelope selection
    tick0: float                 # first (unwrapped) tick, for reference
    span_s: float
    resid_ms: dict = field(default_factory=dict)   # recv - fit, all samples: min/median/p99/max

    def time_of(self, tick_ms) -> np.ndarray:
        return self.slope * unwrap_tick(tick_ms) + self.offset_s

    def as_dict(self) -> dict:
        return asdict(self)


def fit_tick_clock(tick_ms, recv_s, iters: int = 4, keep_frac: float = 0.5,
                   nominal_s_per_tick: float = 1e-3) -> TickFit:
    """Map the Go2 tick onto the receiver's wall clock.

    Model: ``recv = slope*tick + offset + delay`` with ``delay >= 0`` and
    jittery (robot -> switch -> NIC -> DDS -> callback). Plain least squares is
    pulled up by the delay tail, so the fit is iterated towards the low-delay
    edge: fit, keep the ``keep_frac`` samples with the smallest residuals,
    refit. The residual statistics are reported for ALL samples against the
    final fit, so they describe the delay variation a consumer will see.
    ``nominal_s_per_tick`` only scales ``rate_ppm`` (1e-3 for the Go2 LowState
    tick in ms; 1e-5 for the front camera's ``time_frame`` in 10 us units).
    """
    tick = unwrap_tick(tick_ms)
    recv = np.asarray(recv_s, dtype=np.float64)
    if tick.size != recv.size or tick.size < 10:
        raise ValueError("need >= 10 paired samples")
    x0 = tick[0]
    x = tick - x0                                   # condition the fit
    keep = np.ones(x.size, dtype=bool)
    a = b = 0.0
    for _ in range(max(1, iters)):
        a, b = np.polyfit(x[keep], recv[keep], 1)
        r = recv - (a * x + b)
        thresh = np.percentile(r[keep], keep_frac * 100.0)
        new_keep = r <= thresh
        if new_keep.sum() < 10:
            break
        keep = new_keep
    # final envelope fit and shift the offset onto the low-delay edge
    a, b = np.polyfit(x[keep], recv[keep], 1)
    r = recv - (a * x + b)
    offset_s = b - a * x0
    return TickFit(
        slope=float(a), offset_s=float(offset_s), rate_ppm=float((a / nominal_s_per_tick - 1.0) * 1e6),
        n=int(x.size), n_used=int(keep.sum()), tick0=float(x0), span_s=float(recv[-1] - recv[0]),
        resid_ms={"min": float(r.min() * 1e3), "median": float(np.median(r) * 1e3),
                  "p99": float(np.percentile(r, 99) * 1e3), "max": float(r.max() * 1e3)},
    )


@dataclass
class GyroLag:
    ok: bool
    lag_s: float                 # > 0: stream B is stamped LATER than A for the same motion
    peak_corr: float
    motion_rms_a: float          # rad/s after slow-component removal
    motion_rms_b: float
    overlap_s: float
    reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _norm(w) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    return np.linalg.norm(w, axis=1) if w.ndim == 2 else np.abs(w)


def _highpass(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or win >= x.size:
        return x - x.mean()
    kernel = np.ones(win) / win
    slow = np.convolve(x, kernel, mode="same")
    return x - slow


def gyro_lag(t_a, w_a, t_b, w_b, max_lag_s: float = 0.2, grid_hz: float = 2000.0,
             highpass_s: float = 1.0, min_motion_rms: float = 0.05, min_corr: float = 0.5,
             min_motion_rms_b: float | None = None) -> GyroLag:
    """Residual time shift between two gyros on the same rigid body.

    Also used for any two nonnegative motion signals that share the motion
    (frame-difference energy of a camera against a gyro, or against another
    camera): ``w_*`` may be 1-D. ``min_motion_rms_b`` sets a separate motion
    threshold for stream B when its unit differs from A's (grey levels vs
    rad/s); it defaults to ``min_motion_rms``.

    ``|omega|`` is compared, so the (unknown) mounting rotation between the two
    IMUs and a constant gyro bias on either do not matter (bias only enters
    through a slowly varying term, which the high-pass removes). Both series
    are resampled onto a common ``grid_hz`` grid, high-passed (minus a
    ``highpass_s`` moving average), normalised and cross-correlated within
    ``+-max_lag_s``; the peak is refined by parabolic interpolation.

    Edge trimming: the moving average is zero-padded, so in the first and last
    half window it falls to about half the local mean and leaves a ramp in the
    high-passed series. ``|omega|`` and frame-difference energy always have a
    positive mean, and the ramps sit at the same grid indices in A and B
    whatever the true shift, so they would count as "motion" and pull the
    peak toward lag 0 (a constant 1 rad/s on both streams correlated to 1.0 at
    lag 0). Those samples are dropped before the motion RMS, the
    normalisation and the correlation; the global mean is deliberately NOT
    subtracted instead (that biases the lag when the motion does not fill the
    window). Because the trim removes one ``highpass_s`` of data, the overlap
    (after the ``max_lag_s`` margins) must be at least
    ``highpass_s + max(2 s, 4 * max_lag_s)`` - 3.0 s for the gyro-vs-gyro
    defaults, 3.9 s at the energy call sites (1.5 s high-pass, 0.6 s lag).

    Requires real motion: when the overlap is shorter than that, when either
    stream's RMS after high-pass is below ``min_motion_rms`` or when the peak
    correlation is below ``min_corr`` the result is flagged ``ok=False``
    instead of returning noise.
    """
    t_a = np.asarray(t_a, dtype=np.float64); t_b = np.asarray(t_b, dtype=np.float64)
    na, nb = _norm(w_a), _norm(w_b)
    if t_a.size < 10 or t_b.size < 10:
        return GyroLag(False, 0.0, 0.0, 0.0, 0.0, 0.0, "too few samples")
    lo, hi = max(t_a[0], t_b[0]) + max_lag_s, min(t_a[-1], t_b[-1]) - max_lag_s
    need = highpass_s + max(2.0, 4.0 * max_lag_s)   # the edge trim below removes highpass_s of it
    if hi - lo < need:
        return GyroLag(False, 0.0, 0.0, 0.0, 0.0, max(0.0, hi - lo), f"overlap shorter than {need:.1f} s")
    dt = 1.0 / grid_hz
    grid = np.arange(lo, hi, dt)
    win = int(round(highpass_s * grid_hz))
    ga = _highpass(np.interp(grid, t_a, na), win)
    gb = _highpass(np.interp(grid, t_b, nb), win)
    if 1 < win < grid.size:
        # drop the samples whose averaging window ran off either end (zero padding);
        # np.convolve 'same' averages x[i - win//2 .. i + (win-1)//2]
        h0, h1 = win // 2, (win - 1) // 2
        ga, gb = ga[h0:grid.size - h1], gb[h0:grid.size - h1]
    rms_a, rms_b = float(np.sqrt(np.mean(ga**2))), float(np.sqrt(np.mean(gb**2)))
    thr_b = min_motion_rms if min_motion_rms_b is None else min_motion_rms_b
    if rms_a < min_motion_rms or rms_b < thr_b:
        return GyroLag(False, 0.0, 0.0, rms_a, rms_b, float(hi - lo),
                       f"insufficient motion (rms a={rms_a:.3f} b={rms_b:.3f} < {min_motion_rms:g}/{thr_b:g})")
    ga = (ga - ga.mean()) / (ga.std() + 1e-12)
    gb = (gb - gb.mean()) / (gb.std() + 1e-12)
    # c[k] = sum_l gb[l] * ga[l - lag_k]: the peak sits at lag > 0 when B is delayed w.r.t. A
    c = signal.correlate(gb, ga, mode="full", method="fft") / ga.size
    lags = signal.correlation_lags(gb.size, ga.size, mode="full")
    max_k = int(round(max_lag_s * grid_hz))
    sel = np.abs(lags) <= max_k
    c, lags = c[sel], lags[sel]
    i = int(np.argmax(c))
    peak = float(c[i])
    lag = float(lags[i])
    if 0 < i < c.size - 1:                          # parabolic refinement
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom < 0:
            lag += 0.5 * (y0 - y2) / denom
    lag_s = lag * dt
    if peak < min_corr:
        return GyroLag(False, lag_s, peak, rms_a, rms_b, float(hi - lo),
                       f"peak correlation {peak:.2f} < {min_corr}: streams do not share the motion (camera not on the body?)")
    return GyroLag(True, lag_s, peak, rms_a, rms_b, float(hi - lo))


def offset_stats_ms(stamp_s, recv_s) -> dict:
    """recv - stamp in ms (transport latency + clock offset between stamper and receiver)."""
    d = (np.asarray(recv_s, dtype=np.float64) - np.asarray(stamp_s, dtype=np.float64)) * 1e3
    if d.size == 0:
        return {"n": 0}
    return {"n": int(d.size), "min": float(d.min()), "median": float(np.median(d)),
            "p99": float(np.percentile(d, 99)), "max": float(d.max())}


# --------------------------------------------------------------------------
# Model-input view: the policy consumes TWO RGB streams (Go2 ego + D435i
# colour) and the executed action history, nothing else. Its data contract wants both views captured at
# the SAME instant on an 80 ms grid (12.5 Hz), actions as 20 ms holds, and
# capture timestamps that are measured, never synthesised from frame indices.
# The helpers below turn per-frame capture-time estimates into the numbers
# that decide whether a real recording can be resampled onto that grid.

def midpoint_times(frame_times_s) -> np.ndarray:
    """Time to attach to a frame-DIFFERENCE signal.

    The motion energy between frames k-1 and k belongs to the middle of the
    two exposures, not to frame k. Cross-correlating an energy series stamped
    at frame k against a gyro (or against another camera with a different
    frame rate) would otherwise carry a bias of half a frame period - 35 ms for
    the 14.2 fps Go2 stream against 17 ms for the 30 fps D435i.
    """
    t = np.asarray(frame_times_s, dtype=np.float64)
    if t.size < 2:
        return np.empty(0)
    return 0.5 * (t[1:] + t[:-1])


def _stats_ms(x) -> dict:
    x = np.asarray(x, dtype=np.float64) * 1e3
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "min": float(x.min()), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def model_grid_pairing(ego_capture_s, task_capture_s, period_s: float = 0.080,
                       phase_s: float | None = None) -> dict:
    """Resample two real camera streams onto the model's fixed grid and report
    what that costs in time error.

    ``ego_capture_s`` / ``task_capture_s``: estimated capture instants of the
    Go2 ego frames and the D435i colour frames on ONE clock (seconds).
    For every grid instant ``g`` (period ``period_s``, first at ``phase_s`` or,
    by default, at the first ego frame captured at or after the first D435i
    frame, so that slot 0's ego pick always has a D435i frame at or before it)
    two ego selections are evaluated:

    * ``causal``  - the latest ego frame captured at or before ``g``;
      staleness = g - t_ego  (>= 0, at most one ego period without drops)
    * ``nearest`` - the ego frame closest to ``g`` either side (what an
      offline dataset builder may use);  |g - t_ego|

    and for each selected ego frame the D435i frame captured closest to it:
    skew = t_task - t_ego. ``duplicates`` counts grid slots that would reuse
    the previous slot's ego frame (the stream is slower than the grid or had a
    gap), ``ego_gaps`` counts slots whose selected ego frame is more than one
    grid period old. All statistics are in ms.

    Both selections work on CAPTURE times only, so they are OFFLINE
    (dataset-building) figures. The ``causal`` staleness is the capture-grid
    sampling offset (0 to one ego period, whatever the pipeline latency); it
    ignores transport and decode. An online consumer at grid instant ``g`` only
    has frames already received and decoded: its ego age there is
    L (ego pipeline latency) + delivery/decode + this offset, and the paired
    D435i frame must likewise already have arrived.
    """
    ego = np.sort(np.asarray(ego_capture_s, dtype=np.float64))
    task = np.sort(np.asarray(task_capture_s, dtype=np.float64))
    if ego.size < 2 or task.size < 2 or period_s <= 0:
        return {"ok": False, "reason": "need >= 2 frames per stream and period > 0"}
    if phase_s is None:
        # first slot = first ego frame at or after the first D435i frame: an earlier
        # start leaves slot 0's causal ego pick older than every D435i frame, the
        # nearest-D435i index clips to task[0] and slot 0 pairs up to one ego period off
        k = int(np.searchsorted(ego, task[0], side="left"))
        if k == ego.size:
            return {"ok": False, "reason": "no ego frame at or after the first D435i frame (streams do not overlap)"}
        start = max(ego[0], ego[k])
    else:
        start = phase_s
    stop = min(ego[-1], task[-1])
    if stop - start < 2 * period_s:
        return {"ok": False, "reason": "streams overlap for less than two grid periods"}
    grid = np.arange(start, stop, period_s)
    grid = grid[grid >= max(ego[0], task[0])]
    if grid.size < 2:
        return {"ok": False, "reason": "fewer than two grid slots"}
    out = {"ok": True, "period_ms": period_s * 1e3, "slots": int(grid.size),
           "ego_frames": int(ego.size), "task_frames": int(task.size),
           "ego_period_ms": float(np.median(np.diff(ego)) * 1e3),
           "task_period_ms": float(np.median(np.diff(task)) * 1e3)}
    for mode in ("causal", "nearest"):
        if mode == "causal":
            # capture-grid sampling offset (offline figure): latest ego frame CAPTURED at
            # or before g; transport and decode are ignored, so an online consumer's ego
            # age at g is L + delivery + this staleness, not this staleness alone
            idx = np.searchsorted(ego, grid, side="right") - 1
        else:
            right = np.clip(np.searchsorted(ego, grid, side="left"), 0, ego.size - 1)
            left = np.clip(right - 1, 0, ego.size - 1)
            idx = np.where(np.abs(ego[right] - grid) < np.abs(ego[left] - grid), right, left)
        idx = np.clip(idx, 0, ego.size - 1)
        t_ego = ego[idx]
        stale = grid - t_ego
        # D435i frame nearest to the selected ego capture
        r = np.clip(np.searchsorted(task, t_ego, side="left"), 0, task.size - 1)
        l = np.clip(r - 1, 0, task.size - 1)
        j = np.where(np.abs(task[r] - t_ego) < np.abs(task[l] - t_ego), r, l)
        skew = task[j] - t_ego
        dup = int(np.count_nonzero(np.diff(idx) == 0))
        gaps = int(np.count_nonzero(np.abs(stale) > period_s))
        out[mode] = {"ego_staleness_ms": _stats_ms(np.abs(stale) if mode == "nearest" else stale),
                     "task_minus_ego_ms": _stats_ms(skew), "abs_skew_ms": _stats_ms(np.abs(skew)),
                     "duplicates": dup, "ego_gaps": gaps}
    return out
