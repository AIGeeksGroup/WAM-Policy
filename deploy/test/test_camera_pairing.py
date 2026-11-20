"""wam_go2/camera_pairing.py: bounded-window Go2 ego <-> D435i pairing and the per-session report
(per-frame records, common time, 20 ms window, unmatched frames marked, pair_id samples,
common-event validation). Pure numpy, no ROS, no cameras.

Synthetic rig used below: capture instants T; Go2 arrival = T + L + jitter, its time_frame a frame
counter (+6667/6666 per frame, as the firmware); D435i header.stamp = T + eD, arrival = stamp + 51 ms.
"""
import numpy as np
import pytest

from wam_go2 import camera_pairing as cp
from wam_go2 import timesync as ts

EGO_FPS = 14.23
TF0 = 433_012_921


def ego_counter(n, tf0=TF0):
    steps = [(6667, 6666)[i % 2] for i in range(n - 1)]
    return np.concatenate(([tf0], tf0 + np.cumsum(steps))).astype(np.int64)


def E(fid, t, rx=None, tf=None):
    return cp.Frame("ego", fid, int(TF0 + 6667 * fid if tf is None else tf),
                    t + 0.003 if rx is None else rx, t)


def D(fid, t, rx=None):
    return cp.Frame("d435i", fid, int(round(t * 1e9)), t + 0.051 if rx is None else rx, t)


# ----------------------------------------------------------------- the window
def test_ego_frame_pairs_with_the_nearest_d435i_frame_inside_the_window():
    p = cp.BoundedPairer(slop_s=0.020)
    for k in range(10):
        p.add_d435i(D(k, 100.0 + k / 30))
    p.add_ego(E(0, 100.110))                 # nearest: k=3 at 100.100 (-10 ms); k=4 at 100.133 (+23 ms)
    out = p.poll(now=100.2)
    assert len(out) == 1
    dec = out[0]
    assert dec.matched and dec.reason == "" and dec.d435i.frame_id == 3
    assert dec.delta_s == pytest.approx(-0.010, abs=1e-9)
    assert dec.pair_id == "pair_000000"


def test_frame_outside_the_window_is_marked_unmatched_not_forced():
    p = cp.BoundedPairer(slop_s=0.020)
    p.add_d435i(D(0, 100.000)); p.add_d435i(D(1, 100.100))
    p.add_ego(E(0, 100.050))                 # 50 ms from both
    p.add_d435i(D(2, 100.200))
    dec = p.poll(now=100.3)[0]
    assert not dec.matched and dec.reason == "outside_slop" and dec.pair_id is None
    assert dec.d435i is not None and abs(dec.delta_s) == pytest.approx(0.050, abs=1e-9)   # the rejected nearest candidate
    p.add_ego(E(1, 100.195))
    p.add_d435i(D(3, 100.300))
    nxt = p.poll(now=100.4)[0]
    assert nxt.matched and nxt.pair_id == "pair_000000"   # an unmatched frame consumes no pair id


def test_each_d435i_frame_is_used_at_most_once():
    p = cp.BoundedPairer(slop_s=0.050)
    p.add_d435i(D(0, 100.000)); p.add_d435i(D(1, 100.040))
    for i, t in enumerate((100.010, 100.015, 100.020)):
        p.add_ego(E(i, t))
    p.add_d435i(D(2, 100.500))
    a, b, c = p.poll(now=100.6)
    assert a.matched and a.d435i.frame_id == 0
    assert b.matched and b.d435i.frame_id == 1                   # its nearest (0) was taken: next one inside the window
    assert not c.matched and c.reason == "d435i_used"            # both in-window frames taken: no duplicate pair
    assert [x.pair_id for x in (a, b)] == ["pair_000000", "pair_000001"]


def test_ties_go_to_the_earlier_d435i_frame():
    p = cp.BoundedPairer(slop_s=0.375)               # binary-exact times: the two distances are exactly equal
    p.add_d435i(D(0, 64.0)); p.add_d435i(D(1, 64.5))
    p.add_ego(E(0, 64.25))
    p.add_d435i(D(2, 65.0))
    dec = p.poll(now=65.1)[0]
    assert abs(dec.delta_s) == 0.25 and dec.d435i.frame_id == 0


# ------------------------------------------------------- when to decide
def test_decision_waits_for_a_d435i_frame_past_the_window():
    p = cp.BoundedPairer(slop_s=0.020, max_wait_s=0.35)
    for k in range(4):
        p.add_d435i(D(k, 100.0 + k / 30))    # newest 100.100, received 100.151
    p.add_ego(E(0, 100.110, rx=100.113))
    assert p.poll(now=100.15) == []          # a closer frame may still come (100.100 + 33 ms = 100.133)
    p.add_d435i(D(4, 100.0 + 4 / 30))        # 100.133 >= 100.110 + 20 ms
    out = p.poll(now=100.19)
    assert len(out) == 1 and out[0].d435i.frame_id == 3


def test_max_wait_decides_when_the_d435i_stream_lags():
    p = cp.BoundedPairer(slop_s=0.020, max_wait_s=0.35, silence_s=1.0)
    for k in range(4):
        p.add_d435i(D(k, 100.0 + k / 30))
    p.add_ego(E(0, 100.110, rx=100.113))
    assert p.poll(now=100.40) == []
    out = p.poll(now=100.47)                 # 357 ms after the ego frame arrived
    assert len(out) == 1 and out[0].matched and out[0].d435i.frame_id == 3


def test_silent_d435i_stream_does_not_delay_the_ego_view():
    p = cp.BoundedPairer(slop_s=0.020, max_wait_s=0.35, silence_s=1.0)
    p.add_ego(E(0, 100.0, rx=100.003))
    out = p.poll(now=100.004)
    assert len(out) == 1 and out[0].reason == "no_d435i_frames" and out[0].d435i is None
    p.add_d435i(D(0, 98.0))                  # received at 98.051, then silence
    p.add_ego(E(1, 100.07, rx=100.073))
    out = p.poll(now=100.074)
    assert len(out) == 1 and out[0].reason == "outside_slop"


def test_ego_frames_without_a_time_model_are_unmatched():
    p = cp.BoundedPairer()
    p.add_d435i(D(0, 100.0))
    f = E(0, 100.0); f.common_time = None
    p.add_ego(f)
    out = p.poll(now=100.01)
    assert len(out) == 1 and out[0].reason == "no_time_model" and out[0].delta_s is None


def test_decisions_keep_the_ego_arrival_order():
    p = cp.BoundedPairer(slop_s=0.020)
    for k in range(4):
        p.add_d435i(D(k, 100.0 + k / 30))
    p.add_ego(E(0, 100.110, rx=100.113))     # not ready yet
    p.add_ego(E(1, 100.040, rx=100.114))     # would be ready, but must not overtake frame 0
    assert p.poll(now=100.12) == []
    p.add_d435i(D(4, 100.0 + 4 / 30))
    assert [d.ego.frame_id for d in p.poll(now=100.2)] == [0, 1]


def test_flush_decides_everything_pending():
    p = cp.BoundedPairer(slop_s=0.020)
    p.add_d435i(D(0, 100.0))
    p.add_ego(E(0, 100.005)); p.add_ego(E(1, 100.075))
    out = p.flush()
    assert [d.matched for d in out] == [True, False]
    assert p.pending_count == 0


def test_buffer_is_bounded_and_forgets_old_frames():
    p = cp.BoundedPairer(slop_s=0.020, keep_s=1.5, max_keep=90)
    for k in range(1000):
        p.add_d435i(D(k, 100.0 + k / 30))
    assert len(p.d435i_buffer) <= 46                # 1.5 s at 30 fps (+1)
    assert all(f.common_time >= 100.0 + 999 / 30 - 1.5 - 1e-9 for f in p.d435i_buffer)


# ------------------------------------------------- end to end on the rig
def rig(seconds=20.0, L=0.150, eD=0.020, d_drop=0.03, seed=0):
    rng = np.random.default_rng(seed)
    t0 = 1_791_317_440.0
    n_e = int(seconds * EGO_FPS)
    Te = t0 + np.arange(n_e) / EGO_FPS + rng.normal(0, 0.0005, n_e)
    tf = ego_counter(n_e)
    arr_e = Te + L + rng.exponential(0.003, n_e)
    Td = t0 - 0.3 + np.arange(int((seconds + 0.6) * 30)) / 30.0
    Td = Td[rng.random(Td.size) > d_drop]
    stamp = Td + eD
    arr_d = stamp + 0.051 + rng.exponential(0.002, Td.size)
    return Te, tf, arr_e, Td, stamp, arr_d


def run_pairer(Te, tf, arr_e, Td, stamp, arr_d, l_cfg, ed_cfg, slop=0.020):
    fit = ts.fit_tick_clock(tf, arr_e, iters=2, nominal_s_per_tick=1e-5)
    ce = fit.time_of(tf) - l_cfg
    cd = stamp - ed_cfg
    events = sorted([(a, 0, i) for i, a in enumerate(arr_e)] + [(a, 1, j) for j, a in enumerate(arr_d)])
    p = cp.BoundedPairer(slop_s=slop)
    out = []
    for a, kind, i in events:
        if kind == 0:
            p.add_ego(cp.Frame("ego", i, int(tf[i]), float(a), float(ce[i])))
        else:
            p.add_d435i(cp.Frame("d435i", i, int(round(stamp[i] * 1e9)), float(a), float(cd[i])))
        out += p.poll(float(a))
    out += p.flush()
    return out, p


def test_relative_calibration_pairs_frames_captured_at_the_same_instant():
    Te, tf, arr_e, Td, stamp, arr_d = rig(L=0.150, eD=0.020)
    out, p = run_pairer(Te, tf, arr_e, Td, stamp, arr_d, l_cfg=0.130, ed_cfg=0.0)   # L - eD from a common event
    m = [d for d in out if d.matched]
    skew = np.array([Td[d.d435i.frame_id] - Te[d.ego.frame_id] for d in m])
    assert len(out) == Te.size and len(m) >= 0.9 * Te.size
    assert np.abs(skew).max() <= 0.022 and np.median(np.abs(skew)) <= 0.010


def test_uncalibrated_pairs_are_off_by_the_measured_common_event_lag():
    L, eD = 0.150, 0.020
    Te, tf, arr_e, Td, stamp, arr_d = rig(L=L, eD=eD)
    out, p = run_pairer(Te, tf, arr_e, Td, stamp, arr_d, l_cfg=0.0, ed_cfg=0.0)
    m = [d for d in out if d.matched]
    true_skew = np.array([Td[d.d435i.frame_id] - Te[d.ego.frame_id] for d in m])
    assert np.median(true_skew) == pytest.approx(L - eD, abs=0.012)      # receive-time pairing is ~L - eD off
    est = np.array([cp.true_skew_s(d.delta_s, L - eD, 0.0, 0.0) for d in m])
    assert np.abs(est - true_skew).max() <= 0.004                        # delta + lag - (Lcfg - eDcfg)


def test_pairing_report_on_the_rig():
    Te, tf, arr_e, Td, stamp, arr_d = rig(L=0.150, eD=0.020, d_drop=0.03)
    out, p = run_pairer(Te, tf, arr_e, Td, stamp, arr_d, l_cfg=0.130, ed_cfg=0.0)
    r = p.stats.as_dict(n_d435i_frames=Td.size)
    assert r["decided"] == Te.size == r["matched"] + r["unmatched"]
    assert set(r["unmatched_by_reason"]) <= {"outside_slop", "d435i_used"}
    assert r["unmatched_ratio"] == pytest.approx(r["unmatched"] / r["decided"])
    assert r["duplicate_d435i_in_pairs"] == 0
    assert r["d435i_unpaired"] == Td.size - r["matched"]
    assert r["delta_ms"]["max_abs"] <= 20.0 and r["slop_ms"] == 20.0
    assert r["max_pending"] >= 1


# ------------------------------------------------------- time sources
def test_ego_time_source_report_identifies_a_frame_counter():
    Te, tf, arr_e, *_ = rig()
    r = cp.ego_time_source_report(tf, arr_e)
    assert r["verdict"] == "frame_counter" and r["single_frame_fraction"] == 1.0
    assert r["receive_rate_hz"] == pytest.approx(EGO_FPS, rel=0.01)
    assert r["receive_over_counter"] == pytest.approx(15 / EGO_FPS, rel=0.01)   # 1.0 would be a real 10 us clock
    assert r["backward_steps"] == 0 and r["lost_frames"] == 0


def test_ego_time_source_report_counts_lost_messages_and_resets():
    Te, tf, arr_e, *_ = rig()
    keep = np.ones(tf.size, bool); keep[[50, 120, 121]] = False             # 1 + 2 lost messages
    tf2, arr2 = tf[keep], arr_e[keep]
    r = cp.ego_time_source_report(tf2, arr2)
    assert r["verdict"] == "frame_counter" and r["multi_frame_steps"] == 2 and r["lost_frames"] == 3
    reset = np.concatenate((tf[:100], tf[100:] - tf[100] + 5))              # camera service restarted
    r = cp.ego_time_source_report(reset, arr_e)
    assert r["backward_steps"] == 1


def test_ego_time_source_report_flags_a_real_clock():
    Te, tf, arr_e, *_ = rig()
    clock = np.round((Te - Te[0]) * 1e5).astype(np.int64) + TF0             # a true 10 us capture clock
    r = cp.ego_time_source_report(clock, arr_e)
    assert r["verdict"] == "irregular"
    assert r["receive_over_counter"] == pytest.approx(1.0, abs=0.01)


def test_d435i_time_source_report_counts_drops_and_latency():
    s = 1_791_317_440.0 + np.arange(300) / 30.0
    s = np.delete(s, [100, 200])
    rx = s + 0.051
    r = cp.d435i_time_source_report(np.round(s * 1e9).astype(np.int64), rx)
    assert r["period_ms"] == pytest.approx(33.333, abs=0.01)
    assert r["gaps"] == 2 and r["lost_frames"] == 2 and r["backward_steps"] == 0 and r["zero_steps"] == 0
    assert r["recv_minus_stamp_ms"]["median"] == pytest.approx(51.0, abs=0.01)


# ------------------------------------------------ common-event validation
def test_motion_validation_reports_offset_and_suggestion():
    v = cp.motion_validation([42.3, 43.7, 49.6], l_cfg_ms=0.0, ed_cfg_ms=0.0, deltas_ms=[-10.0, 0.0, 12.0])
    assert v["n"] == 3 and v["lag_ms"]["median"] == pytest.approx(43.7)
    assert v["pairing_offset_ms"] == pytest.approx(43.7)                     # pairs: D435i captured ~44 ms after ego
    assert v["suggested_go2_latency_ms"] == pytest.approx(43.7)
    assert v["true_skew_ms"]["median"] == pytest.approx(43.7)
    v2 = cp.motion_validation([42.3, 43.7, 49.6], l_cfg_ms=43.7, ed_cfg_ms=0.0)
    assert v2["pairing_offset_ms"] == pytest.approx(0.0) and v2["true_skew_ms"] is None
    assert cp.motion_validation([], 0.0, 0.0)["n"] == 0


def test_aggregate_and_relative_calibration():
    agg = cp.aggregate_motion_lags([("a", [42.0, 44.0, 50.0]), ("b", [40.0, 45.0]), ("c", [])])
    assert agg["n_windows"] == 5 and agg["median_ms"] == pytest.approx(44.0)
    assert [s["median_ms"] for s in agg["sessions"]] == [44.0, 42.5, None]
    assert agg["session_median_spread_ms"] == pytest.approx(1.5)
    cal = cp.relative_calibration(agg, ed_cfg_ms=0.0)
    c = cal["model_input"]["calibration"]                                   # what ego_d435i_view.py --calib reads
    assert c["go2_ego_latency_ms"] == pytest.approx(44.0) and c["d435i_stamp_minus_exposure_ms"] == 0.0
    assert cal["kind"] == "relative_camera_motion"
    with pytest.raises(ValueError):
        cp.relative_calibration(cp.aggregate_motion_lags([("x", [])]))


# ---------------------------------------------------------- CSV rows
def test_frame_and_pair_rows():
    assert cp.FRAMES_HEADER[:3] == ["stream", "frame_id", "raw_time"]
    assert cp.frame_row(cp.Frame("ego", 7, 433012921, 1791317440.1234567, None)) == \
        ["ego", 7, 433012921, "go2_time_frame", "1791317440.123457"]
    p = cp.BoundedPairer(slop_s=0.020)
    p.add_d435i(D(0, 100.0)); p.add_d435i(D(1, 100.2))
    p.add_ego(E(0, 100.005)); p.add_ego(E(1, 100.100))
    m, u = p.flush()
    row = dict(zip(cp.PAIRS_HEADER, cp.pair_row(m, 12, cp.SYNC_RECEIVE, "none")))
    assert row["pair_id"] == "pair_000000" and row["status"] == "matched"
    assert row["pair_time"] == "100.005000" == row["ego_common_time"] and row["delta_ms"] == "-5.000"
    assert row["video_frame"] == 12 and row["sync_method"] == "receive_time_approx"
    row = dict(zip(cp.PAIRS_HEADER, cp.pair_row(u, None, cp.SYNC_CAPTURE, "calib:x@abc")))
    assert row["pair_id"] == "" and row["pair_time"] == "" and row["status"] == "unmatched:outside_slop"
    assert row["d435i_frame_id"] in (0, 1) and row["video_frame"] == ""   # the rejected nearest candidate, for diagnosis
