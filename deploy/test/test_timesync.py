"""ROS-free tests for wam_go2/timesync.py: Go2 tick -> wall-clock fit and the
robot-vs-camera gyro cross-correlation used by scripts/x5/sync_check.py."""
import numpy as np
import pytest

from wam_go2 import timesync as ts


def _ticks(n=30000, start=123456, wrap_at=None):
    t = start + 2 * np.arange(n, dtype=np.int64)          # 500 Hz, 2 ms per tick
    if wrap_at is not None:
        t = t - (start + 2 * (n // 2)) + wrap_at           # cross the uint32 wrap mid-run
    return t % ts.TICK_WRAP


def _receive(tick_unwrapped_ms, rate_ppm=50.0, offset=1.79e9, rng=None):
    rng = rng or np.random.default_rng(0)
    slope = 1e-3 * (1 + rate_ppm * 1e-6)
    delay = 0.0003 + rng.exponential(0.0002, tick_unwrapped_ms.size)   # >= 0.3 ms, jittery tail
    spikes = rng.random(tick_unwrapped_ms.size) < 0.002
    delay[spikes] += 0.005                                              # occasional 5 ms hiccups
    return slope * tick_unwrapped_ms + offset + delay, slope, delay


def test_unwrap_tick_crosses_uint32():
    t = _ticks(n=1000, wrap_at=ts.TICK_WRAP - 4)
    u = ts.unwrap_tick(t)
    assert np.all(np.diff(u) == 2)


def test_tick_fit_recovers_rate_and_low_delay_offset():
    tick = _ticks()
    recv, slope, delay = _receive(ts.unwrap_tick(tick))
    fit = ts.fit_tick_clock(tick, recv)
    assert fit.n == tick.size and 10 <= fit.n_used <= fit.n
    assert abs(fit.rate_ppm - 50.0) < 3.0
    # the fit sits on the low-delay edge: residuals are ~delay - min(delay)
    assert -0.15 < fit.resid_ms["min"] < 0.15
    assert fit.resid_ms["max"] > 4.0                    # the spikes are visible, not averaged away
    # time_of reproduces the reference clock to well under a millisecond
    t_hat = fit.time_of(tick)
    err_ms = (t_hat - (slope * ts.unwrap_tick(tick) + 1.79e9 + 0.0003)) * 1e3
    assert np.abs(err_ms).max() < 0.3


def test_tick_fit_across_wrap():
    tick = _ticks(n=20000, wrap_at=ts.TICK_WRAP - 2000)
    recv, _, _ = _receive(ts.unwrap_tick(tick), rate_ppm=-20.0)
    fit = ts.fit_tick_clock(tick, recv)
    assert abs(fit.rate_ppm + 20.0) < 3.0
    assert np.all(np.diff(fit.time_of(tick)) > 0)


def test_tick_fit_nominal_unit_scales_rate_ppm():
    # front-camera style counter: 10 us units, nominal 15 fps -> 6667 per frame
    tf = 6667 * np.arange(300, dtype=np.int64)
    recv = 1e9 + tf * 10.54e-6                          # measured cadence 14.23 fps
    fit = ts.fit_tick_clock(tf, recv, nominal_s_per_tick=1e-5)
    assert abs(fit.slope - 10.54e-6) < 1e-9
    assert abs(fit.rate_ppm - 54000) < 200               # +5.4 % against the nominal 10 us tick


def test_tick_fit_rejects_short_input():
    with pytest.raises(ValueError):
        ts.fit_tick_clock([1, 2, 3], [0.0, 0.002, 0.004])


def _motion(t, rng):
    """Rich rotation-rate signal (rad/s) shared by both IMUs."""
    w = np.zeros((t.size, 3))
    for axis in range(3):
        for f in (0.7, 1.9, 3.3, 5.1):
            w[:, axis] += rng.uniform(0.1, 0.4) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return w


def _rot(rng):
    q = rng.normal(size=(3, 3))
    r, _ = np.linalg.qr(q)
    return r


@pytest.mark.parametrize("true_lag", [0.0173, -0.0088, 0.0])
def test_gyro_lag_recovers_shift_with_rotation_bias_and_rates(true_lag):
    rng = np.random.default_rng(1)
    dur = 25.0
    t_a = np.arange(0, dur, 1 / 500.0)                              # Go2 IMU in LowState, 500 Hz
    t_b = np.arange(0.3, dur - 0.2, 1 / 200.0)                      # D435i gyro, 200 Hz
    # stream B reports the motion of physical time (t_b - true_lag) at stamp t_b -> stamped LATER
    w_a = _motion(t_a, rng) + rng.normal(0, 0.01, (t_a.size, 3))
    w_b = _motion(t_b - true_lag, np.random.default_rng(1)) @ _rot(rng).T + np.array([0.02, -0.03, 0.01])
    w_b += rng.normal(0, 0.02, w_b.shape)
    res = ts.gyro_lag(t_a, w_a, t_b, w_b)
    assert res.ok, res.reason
    assert abs(res.lag_s - true_lag) < 0.0006
    assert res.peak_corr > 0.8


def test_gyro_lag_flags_no_motion():
    t = np.arange(0, 10, 0.002)
    still = np.tile([0.01, -0.02, 0.0], (t.size, 1))
    res = ts.gyro_lag(t, still, t, still)
    assert not res.ok and "insufficient motion" in res.reason


def test_gyro_lag_flags_unrelated_motion():
    t = np.arange(0, 20, 0.002)
    a = _motion(t, np.random.default_rng(4))
    b = _motion(t, np.random.default_rng(5))
    res = ts.gyro_lag(t, a, t, b)
    assert not res.ok and "peak correlation" in res.reason


GYRO_KW = dict(max_lag_s=0.2, grid_hz=2000.0, highpass_s=1.0, min_motion_rms=0.05, min_corr=0.5)
# energy call sites (sync_check.analyse_model_input b/c): grid 1000 Hz, high-pass 1.5 s, lag search 0.6 s
ENERGY_KW = dict(max_lag_s=0.6, grid_hz=1000.0, highpass_s=1.5, min_motion_rms=0.05, min_corr=0.5,
                 min_motion_rms_b=0.8)


def test_gyro_lag_constant_rate_carries_no_timing():
    """Constant |omega| = 1 rad/s on both streams, B stamped 150 ms later. A constant has
    no timing information, so no lag may come out of it. Without the edge trim, the
    zero-padded moving average leaves a ramp at both window ends, identical in A and B,
    which passes the motion gate and correlates to 1.0 at lag 0."""
    t_a = np.arange(0, 20, 1 / 500.0)
    t_b = np.arange(0, 20, 1 / 200.0) + 0.150
    res = ts.gyro_lag(t_a, np.tile([0.0, 0.0, 1.0], (t_a.size, 1)),
                      t_b, np.tile([0.0, 0.0, 1.0], (t_b.size, 1)))
    assert not res.ok and "insufficient motion" in res.reason, res


def test_gyro_lag_steady_turn_is_not_motion():
    """A robot turning at a constant 0.8 rad/s (mount rotation, bias, sensor noise) has
    no timing information either and must be refused, not reported as lag 0."""
    rng = np.random.default_rng(0)
    q = _rot(rng)
    t_a = np.arange(0, 20, 1 / 500.0)
    t_b = np.arange(0.3, 19.8, 1 / 200.0) + 0.150
    w_a = np.tile([0.0, 0.0, 0.8], (t_a.size, 1)) + rng.normal(0, 0.01, (t_a.size, 3))
    w_b = np.tile([0.0, 0.0, 0.8], (t_b.size, 1)) @ q.T + np.array([0.02, -0.03, 0.01])
    w_b += rng.normal(0, 0.02, w_b.shape)
    res = ts.gyro_lag(t_a, w_a, t_b, w_b, **GYRO_KW)
    assert not res.ok, res


def test_gyro_lag_refuses_overlap_shorter_than_highpass_plus_lag_span():
    """The edge trim removes one high-pass window of data, so the overlap guard is
    highpass_s + max(2 s, 4 * max_lag_s) = 3.9 s at the energy call sites; a 3.5 s
    overlap of shared motion must be refused instead of returning a short-window lag."""
    def run(overlap_s):
        dur = overlap_s + 2 * ENERGY_KW["max_lag_s"]          # hi - lo = dur - 2 * max_lag
        t_a = np.linspace(0.0, dur, int(round(dur * 500)) + 1)  # Go2 IMU 500 Hz, spans exactly [0, dur]
        t_b = np.arange(-0.2, dur + 0.2, 1 / 14.2)              # ego energy, 14.2 fps, covers [0, dur]
        w_a = _motion(t_a, np.random.default_rng(11))
        e_b = 8.0 * np.linalg.norm(_motion(t_b - 0.1, np.random.default_rng(11)), axis=1) + 0.1
        return ts.gyro_lag(t_a, w_a, t_b, e_b, **ENERGY_KW)
    res = run(3.5)
    assert abs(res.overlap_s - 3.5) < 0.01
    assert not res.ok and "overlap shorter than 3.9 s" in res.reason, res
    assert "overlap shorter" not in run(4.2).reason


def _slow_rocking(t, rng):
    """Slow hand rocking (0.3-1.7 Hz, 0.05-0.2 rad/s per component)."""
    w = np.zeros((t.size, 3))
    for axis in range(3):
        for f in (0.3, 0.6, 1.1, 1.7):
            w[:, axis] += rng.uniform(0.05, 0.2) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return w


def test_gyro_lag_slow_motion_with_large_mean_is_not_pulled_to_zero():
    """Slow rocking on top of a steady 1 rad/s turn, B stamped 150 ms later, 30 s window,
    12 fixed seeds. The edge ramps of the zero-padded high-pass sit at the same grid
    indices in both streams and pull the lag toward 0 without the trim (mean error
    -7.1 ms, worst -22.5 ms); with the trim the mean error is -0.3 ms, worst 2.4 ms."""
    true_lag = 0.150
    turn = np.array([0.0, 0.0, 1.0])
    errs = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        t_a = np.arange(0, 30.0, 1 / 500.0)
        t_b = np.arange(0.3, 29.8, 1 / 200.0)
        w_a = _slow_rocking(t_a, np.random.default_rng(100 + seed)) + turn + rng.normal(0, 0.01, (t_a.size, 3))
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        w_b = (_slow_rocking(t_b - true_lag, np.random.default_rng(100 + seed)) + turn) @ q.T
        w_b += np.array([0.02, -0.03, 0.01]) + rng.normal(0, 0.02, w_b.shape)
        res = ts.gyro_lag(t_a, w_a, t_b, w_b, **GYRO_KW)
        assert res.ok, (seed, res.reason)
        errs.append((res.lag_s - true_lag) * 1e3)
    assert abs(np.mean(errs)) < 2.0, errs


def test_offset_stats():
    s = ts.offset_stats_ms([10.0, 11.0], [10.002, 11.004])
    assert s["n"] == 2 and abs(s["min"] - 2.0) < 1e-9 and abs(s["max"] - 4.0) < 1e-9


# ---------------------------------------------------------------- model grid
def test_midpoint_times_removes_half_period_bias():
    t = np.arange(0, 1, 0.0703)                          # 14.2 fps frames
    m = ts.midpoint_times(t)
    assert m.size == t.size - 1
    assert np.allclose(m, t[:-1] + 0.03515)
    assert ts.midpoint_times([1.0]).size == 0


def test_energy_lag_with_midpoints_is_unbiased_across_frame_rates():
    """Two cameras of different frame rate see the same |omega|; the frame-difference
    energy stamped at the interval midpoint must give lag ~ 0, stamped at the
    later frame it carries half the period difference (about 18 ms)."""
    rng = np.random.default_rng(7)
    dur = 30.0
    t_fine = np.arange(0, dur, 0.001)
    w = np.abs(_motion(t_fine, rng)[:, 0]) + 0.3      # scalar rotation-rate magnitude
    def energy(frame_t):
        # energy of frame k = mean |omega| over (t_{k-1}, t_k], the physical content of a frame difference
        e = []
        for a, b in zip(frame_t[:-1], frame_t[1:]):
            sel = (t_fine > a) & (t_fine <= b)
            e.append(w[sel].mean())
        return np.asarray(e)
    t_ego = np.arange(0.011, dur, 0.0703)
    t_d = np.arange(0.004, dur, 0.03334)
    e_ego, e_d = energy(t_ego), energy(t_d)
    good = ts.gyro_lag(ts.midpoint_times(t_d), e_d, ts.midpoint_times(t_ego), e_ego,
                       max_lag_s=0.3, grid_hz=1000.0, highpass_s=1.5, min_motion_rms=0.01)
    assert good.ok, good.reason
    assert abs(good.lag_s) < 0.004
    biased = ts.gyro_lag(t_d[1:], e_d, t_ego[1:], e_ego,
                         max_lag_s=0.3, grid_hz=1000.0, highpass_s=1.5, min_motion_rms=0.01)
    assert biased.ok and 0.012 < biased.lag_s < 0.026   # ~ (70.3 - 33.3) / 2 ms


def test_model_grid_pairing_real_rates_fit_the_80ms_grid():
    rng = np.random.default_rng(2)
    ego = 0.02 + np.cumsum(np.full(400, 0.0703) + rng.normal(0, 0.002, 400))      # 14.2 fps
    task = 0.005 + np.cumsum(np.full(900, 0.03334) + rng.normal(0, 0.0005, 900))   # 30 fps
    rep = ts.model_grid_pairing(ego, task, period_s=0.08)
    assert rep["ok"] and rep["slots"] > 300
    c, n = rep["causal"], rep["nearest"]
    assert c["duplicates"] == 0 and c["ego_gaps"] == 0                # 70 ms frames always refresh an 80 ms slot
    assert 0.0 <= c["ego_staleness_ms"]["min"] and c["ego_staleness_ms"]["max"] < 75.0
    assert n["ego_staleness_ms"]["max"] < 40.0                        # nearest either side: <= half an ego period
    assert c["abs_skew_ms"]["max"] < 18.0 and n["abs_skew_ms"]["max"] < 18.0   # D435i nearest to ego: <= half a D435i period


def test_model_grid_pairing_counts_duplicates_and_gaps():
    ego = np.concatenate([np.arange(0, 5, 0.0703), np.arange(5.5, 10, 0.0703)])   # 0.5 s hole
    task = np.arange(0, 10, 0.03334)
    rep = ts.model_grid_pairing(ego, task, period_s=0.08)
    assert rep["causal"]["duplicates"] >= 5 and rep["causal"]["ego_gaps"] >= 5
    fast = ts.model_grid_pairing(np.arange(0, 10, 0.0703), task, period_s=0.05)    # grid faster than the ego stream
    assert fast["causal"]["duplicates"] > 50
    assert not ts.model_grid_pairing([0.0, 0.1], [0.0], 0.08)["ok"]


def test_model_grid_pairing_slot0_has_a_d435i_frame_before_the_ego_pick():
    """The ego stream starts 924 ms before the D435i stream. A grid starting at the first
    D435i frame would give slot 0 a causal ego frame older than every D435i frame (the
    D435i index clips to 0 and slot 0 pairs up to one ego period apart - a false pairing
    FAIL for most calibrated (L, eD)). The grid starts at the first ego frame at or after
    the first D435i frame, so every slot pairs within half a D435i period
    (33.36 / 2 = 16.7 ms)."""
    ego0 = 100.0 + 0.07026 * np.arange(430)                  # 14.2 fps, ~30 s
    task0 = 100.924 + 0.03336 * np.arange(870)               # 30 fps, starts 0.924 s later
    for L in (0.0, 0.05, 0.10, 0.15, 0.20):                  # ego capture = arrival fit - L
        for eD in (0.0, 0.02, 0.04):                         # D435i capture = stamp - eD
            rep = ts.model_grid_pairing(ego0 - L, task0 - eD, period_s=0.08)
            assert rep["ok"], (L, eD, rep)
            assert rep["causal"]["abs_skew_ms"]["max"] <= 16.7 + 0.2, (L, eD, rep["causal"])
    # disjoint streams (every ego frame before the first D435i frame) are refused, not an IndexError
    late = ts.model_grid_pairing(np.arange(0, 1, 0.07), np.arange(2, 3, 0.0333), 0.08)
    assert not late["ok"] and late["reason"]
