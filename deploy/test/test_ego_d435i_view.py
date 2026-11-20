"""ROS-free tests for scripts/x5/ego_d435i_view.py (laptop viewer: Go2 ego and D435i, two windows).

Covered: the c key's FLASH branch includes the D435i stamp offset eD (raw = eD - L, so
L = eD - median(raw)); pair / banner residuals are raw + (L - eD); the banner judges a calibration
only with flashes recorded AFTER it; flash events are stamped at the midpoint of the previous-frame /
first-bright-frame interval (also across a dropped frame); a writer-size guard resizes (never
silently drops) a frame of another size; the hand-wave motion lag waits for 10 s of common data;
bounded-window pairing and its per-session report; the 50 Hz grid, two windows and two videos; the
ego_stream topic as ego source.
No robot, no X5, no ROS: Streams() only starts its (idle) daemon decode thread.
"""
import json
import os
import sys
import time
import types

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("av")

X5_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "x5"))
if X5_DIR not in sys.path:
    sys.path.insert(0, X5_DIR)
import ego_d435i_view as view  # noqa: E402

L_TRUE = 0.150                    # Go2 capture -> arrival-model latency of the synthetic rig
SLOPE = (1 / 14.23) / 6666.5      # s per time_frame unit (14.23 fps, 6666.5 units per frame)
TF0 = 433_012_921


class _LinFit:
    """Exact stand-in for ts.TickFit: Go2 arrival model = slope * time_frame + offset."""

    def __init__(self, slope, offset_s):
        self.slope, self.offset_s = slope, offset_s
        self.n, self.resid_ms = 100, {"p99": 0.0}

    def time_of(self, tf):
        return self.slope * np.asarray(tf, dtype=np.float64) + self.offset_s


def _streams(latency_s=0.0, d_offset_s=0.0):
    st = view.Streams(720, 20.0, latency_s, d_offset_s)
    st.go2_fit = _LinFit(SLOPE, time.time() - SLOPE * TF0)
    return st


def _feed_flashes(st, n, latency_true, d_offset_true, order="go2_first"):
    """Exact (unquantised) flashes at true capture instants T near now. With capture T the Go2
    arrival model reads T + L and the D435i header.stamp T + eD. The Go2 event goes in the way
    _decode_loop passes it (calibrated model time, time_frame), the D435i one the way on_color
    does (header.stamp, None)."""
    now = time.time()
    for k in range(n):
        T = now - 3.0 + 0.3 * k
        tf = int(round((T + latency_true - st.go2_fit.offset_s) / st.go2_fit.slope))
        go2 = ("go2", st.go2_time(tf), tf)
        d = ("d435i", T + d_offset_true, None)
        for ev in ((go2, d) if order == "go2_first" else (d, go2)):
            st._flash(*ev)


# ---------------------------------------------------------------- c key / flash
@pytest.mark.parametrize("start_latency_s", [0.0, 0.100])
@pytest.mark.parametrize("d_offset_s", [0.020, 0.0, -0.015])
def test_flash_calibration_installs_true_latency(d_offset_s, start_latency_s):
    st = _streams(start_latency_s, d_offset_s)
    _feed_flashes(st, 7, L_TRUE, d_offset_s, order="d435i_first" if start_latency_s else "go2_first")
    assert len(st.flash_pairs) == 7
    assert st.flash_median_raw() == pytest.approx((d_offset_s - L_TRUE) * 1e3, abs=0.01)   # raw = eD - L
    msg = st.apply_calibration()
    assert msg and "flash" in msg
    # -median(raw) alone would be L - eD (130 ms at eD = 20 ms)
    assert st.go2_latency_s == pytest.approx(L_TRUE, abs=1e-5)


def test_pair_residual_is_zero_with_correct_constants():
    ed = 0.020
    st = _streams(L_TRUE, ed)                   # both constants already right (e.g. from --calib)
    _feed_flashes(st, 4, L_TRUE, ed)
    assert [p["residual_ms"] for p in st.flash_pairs] == pytest.approx([0.0] * 4, abs=0.01)
    r, n = st.flash_residual_ms()
    assert n == 4 and r == pytest.approx(0.0, abs=0.01)


def test_flash_residual_uses_only_flashes_after_the_last_calibration():
    ed = 0.020
    st = _streams(0.0, ed)
    _feed_flashes(st, 5, L_TRUE, ed)
    r, n = st.flash_residual_ms()               # uncalibrated: displayed Go2 capture = arrival model
    assert n == 5 and r == pytest.approx(-L_TRUE * 1e3, abs=0.01)
    st.apply_calibration()
    assert st.go2_latency_s == pytest.approx(L_TRUE, abs=1e-5)
    assert st.flash_residual_ms() == (None, 0)  # the calibrating flashes would read 0 by construction
    _feed_flashes(st, 3, L_TRUE, ed)
    r, n = st.flash_residual_ms()
    assert n == 3 and r == pytest.approx(0.0, abs=0.01)
    # the pipeline becomes 20 ms slower: the 8 post-calibration pairs say -20 ms; mixing in the 5
    # calibrating pairs (0 by construction) would have shown a median of 0
    _feed_flashes(st, 5, L_TRUE + 0.020, ed)
    r, n = st.flash_residual_ms()
    assert n == 8 and r == pytest.approx(-20.0, abs=0.01)


def test_residual_after_calibration_survives_a_laptop_clock_step_back(monkeypatch):
    """The post-calibration selection is by pair index, not wall time: if the laptop clock steps
    backwards after c, the newer flashes must still count (a time-based marker would hide them all)."""
    ed = 0.020
    st = _streams(0.0, ed)
    _feed_flashes(st, 5, L_TRUE, ed)
    st.apply_calibration()
    real = time.time
    monkeypatch.setattr(view.time, "time", lambda: real() - 100.0)   # the whole process sees the step
    _feed_flashes(st, 3, L_TRUE, ed)
    r, n = st.flash_residual_ms()
    assert n == 3 and r == pytest.approx(0.0, abs=0.01)


def test_clearing_flashes_after_calibration_keeps_new_flashes():
    """The r key empties the flash records; the post-calibration marker must follow, otherwise the
    first len(old pairs) new flashes are silently skipped by the banner residual."""
    ed = 0.020
    st = _streams(0.0, ed)
    _feed_flashes(st, 5, L_TRUE, ed)
    st.apply_calibration()
    st.clear_flashes()
    assert st.flash_pairs == [] and st.ev_go2 == [] and st.ev_d == []
    _feed_flashes(st, 3, L_TRUE, ed)
    r, n = st.flash_residual_ms()
    assert n == 3 and r == pytest.approx(0.0, abs=0.01)
    # without any calibration, clearing leaves every later pair counted
    st2 = _streams(0.0, ed)
    _feed_flashes(st2, 2, L_TRUE, ed)
    st2.clear_flashes()
    _feed_flashes(st2, 2, L_TRUE, ed)
    assert st2.flash_residual_ms()[1] == 2 and st2.calib_index is None


def test_motion_branch_unchanged_and_preferred_over_flash():
    ed = 0.020
    st = _streams(0.0, ed)
    _feed_flashes(st, 5, 0.300, ed)             # flashes alone would say 300 ms
    lags = [131.0, 129.5, 130.2]
    st.motion_hist = [{"time": 0.0, "lag_ms": x, "corr": 0.9} for x in lags]
    msg = st.apply_calibration()
    assert msg and "motion" in msg
    assert st.go2_latency_s == float(np.median(lags)) * 1e-3 + ed      # L = median(lag) + eD, bit for bit


def test_calibration_without_data_changes_nothing():
    st = _streams(0.050, 0.020)
    assert st.apply_calibration() is None
    assert st.go2_latency_s == 0.050
    _feed_flashes(st, 2, L_TRUE, 0.020)
    assert st.flash_residual_ms()[1] == 2       # no calibration happened: every pair still counts


# ---------------------------------------------------------- midpoint flash stamps
def _img(v):
    return np.full((48, 64, 3), v, np.uint8)


def _run_detector(det, times, keys, bright_from):
    events = []
    for i, t in enumerate(times):
        e = det.update(_img(200 if i >= bright_from else 30), t, key=None if keys is None else keys[i])
        if e is not None:
            events.append((i, e, det.mid_key))
    return events


def test_flash_event_at_midpoint_of_last_dark_and_first_bright_frame():
    P, K = 1 / 14.23, 6666.5
    t0 = 1790000000.0
    times = [t0 + i * P for i in range(12)]
    keys = [TF0 + int(round(i * K)) for i in range(12)]
    ev = _run_detector(view.FlashDetector(20.0, period_s=0.0703), times, keys, bright_from=8)
    assert len(ev) == 1 and ev[0][0] == 8
    assert ev[0][1] == pytest.approx(0.5 * (times[7] + times[8]), abs=1e-9)
    assert ev[0][2] == pytest.approx(0.5 * (keys[7] + keys[8]))
    # D435i: no keys, the event time is the stamp midpoint
    Pd = 1 / 30.0
    td = [t0 + i * Pd for i in range(12)]
    ev = _run_detector(view.FlashDetector(20.0, period_s=0.0334), td, None, bright_from=9)
    assert len(ev) == 1 and ev[0][1] == pytest.approx(0.5 * (td[8] + td[9]), abs=1e-9) and ev[0][2] is None


def test_flash_after_a_dropped_frame_stays_at_the_bracket_midpoint():
    # frames 8 and 9 lost (decode gap), first bright frame received is 10: the lost frames may have
    # been dark or bright, so the torch-on instant is uniform over (t7, t10] and the midpoint of the
    # received bracket is the unbiased stamp (a frame_time - P/2 fallback measured +6 ms of bias
    # at 10 % Go2 frame loss, deliberately not used)
    P, K = 1 / 14.23, 6666.5
    t0 = 1790000000.0
    idx = list(range(8)) + [10, 11]
    times = [t0 + i * P for i in idx]
    keys = [TF0 + int(round(i * K)) for i in idx]
    ev = _run_detector(view.FlashDetector(20.0, period_s=0.0703), times, keys, bright_from=8)
    assert len(ev) == 1
    assert ev[0][1] == pytest.approx(0.5 * (times[7] + times[8]), abs=1e-9)
    assert ev[0][2] == pytest.approx(0.5 * (keys[7] + keys[8]))


def test_flash_period_is_estimated_from_recent_frames():
    # the banner's +-P/2: nominal until frames are seen, then the stream's own period (D435i at
    # 15 fps with the 30 fps nominal)
    det = view.FlashDetector(20.0, period_s=0.0334)
    assert det.period() == 0.0334
    P = 1 / 15.0
    t0 = 1790000000.0
    times = [t0 + i * P for i in range(12)]
    ev = _run_detector(det, times, None, bright_from=8)
    assert len(ev) == 1 and ev[0][1] == pytest.approx(0.5 * (times[7] + times[8]), abs=1e-9)
    assert det.period() == pytest.approx(P, rel=1e-6)


# ----------------------------------------------------------------- writer size
@pytest.fixture(scope="module")
def renderer():
    return view.Renderer()


def test_fit_box_keeps_aspect_and_pads():
    img = np.full((720, 1280, 3), 200, np.uint8)
    out = view.fit_box(img, 640, 480)
    assert out.shape == (480, 640, 3)
    assert (out[:60] == 0).all() and (out[420:] == 0).all() and (out[60:420] == 200).all()
    same = np.full((480, 640, 3), 9, np.uint8)
    o = view.fit_box(same, 640, 480)
    assert o is not same and np.array_equal(o, same)
    assert view.fit_box(np.full((240, 320, 3), 5, np.uint8), 640, 480).shape == (480, 640, 3)


def test_frame_size_guard_resizes_and_warns_once(capsys):
    guard = view.FrameSizeGuard((1493, 695))
    same = np.zeros((695, 1493, 3), np.uint8)
    assert guard(same) is same
    wide = np.full((695, 1706, 3), 7, np.uint8)
    assert guard(wide).shape == (695, 1493, 3)
    assert guard(wide).shape == (695, 1493, 3)
    out = capsys.readouterr().out
    assert out.count("WARNING") == 1 and "1706x695" in out and "1493x695" in out


# ------------------------------------------------------- motion lag minimum span
def _fill_motion(st, g_span, d_span, g_end=0.0, d_end=0.0, seed=0):
    rng = np.random.default_rng(seed)
    now = time.time()
    tg = np.arange(now - g_end - g_span, now - g_end, 1 / 14.2)
    td = np.arange(now - d_end - d_span, now - d_end, 1 / 30.0)
    with st.lock:
        st.motion_go2.clear(); st.motion_d.clear()
        st.motion_go2.extend(zip(tg.tolist(), (rng.random(tg.size) * 5).tolist()))
        st.motion_d.extend(zip(td.tolist(), (rng.random(td.size) * 5).tolist()))
    st._last_motion = 0.0
    st.motion_lag = None


def test_motion_lag_waits_for_10_s_of_common_data():
    st = view.Streams(720, 20.0, 0.0, 0.0)
    _fill_motion(st, 8.0, 8.0)           # 113 Go2 + 240 D435i samples: the sample-count gate alone passes
    st.update_motion_lag()
    assert st.motion_lag is None
    _fill_motion(st, 11.0, 11.0, g_end=12.5)   # each spans 11 s but they share none of it
    st.update_motion_lag()
    assert st.motion_lag is None
    _fill_motion(st, 12.0, 12.0)
    st.update_motion_lag()
    assert st.motion_lag is not None     # computed (ok or not) once 10 s of both are in the window


# ---------------------------------------------------------------------- pairing
# Bounded-window pairing, unmatched frames marked, pair_id records, per-session report.
from wam_go2 import camera_pairing as cp  # noqa: E402

class _Msg:
    def __init__(self, stamp_s, img, encoding="bgr8"):
        sec = int(stamp_s)
        self.header = types.SimpleNamespace(stamp=types.SimpleNamespace(sec=sec, nanosec=int(round((stamp_s - sec) * 1e9))))
        self.height, self.width = img.shape[:2]
        self.data = img.tobytes()
        self.encoding = encoding


def _paired_session(n_ego=40, latency_s=0.0, calib=("none", False), before=None):
    """Streams fed in arrival order like the callbacks: D435i at 30 fps (stamp = capture), Go2 frames
    whose arrival model reads capture + 0.150 s (the stand-in fit), message record before decode, and
    the main loop's poll_pairs after every Go2 frame; before[i](st) runs just before Go2 frame i (e.g.
    the c key). Returns (streams, time_frames, decisions, frame_record_rows)."""
    st = view.Streams(360, 20.0, latency_s, 0.0, slop_s=0.020, calibration_version=calib[0], go2_calibrated=calib[1])
    t0 = time.time() - 10.0
    st.go2_fit = _LinFit(SLOPE, t0 + 0.150 - SLOPE * TF0)          # arrival model of time_frame TF0 = t0 + 150 ms
    img_e = np.zeros((360, 640, 3), np.uint8)
    img_d = np.full((480, 640, 3), 50, np.uint8)
    d_times = [t0 - 0.2 + k / 30.0 for k in range(int(n_ego / 14.23 * 30) + 30)]
    k, tfs, out, rows = 0, [], [], []
    for i in range(n_ego):
        if before and i in before:
            before[i](st)
        tf = TF0 + int(round(i * 6666.5))
        t_model = float(st.go2_fit.time_of(tf))
        while k < len(d_times) and d_times[k] <= t_model + 0.1:
            st.on_color(_Msg(d_times[k], img_d)); k += 1
        arr = t_model + 0.002
        st._note_ego_msg(tf, arr)
        st._on_go2_frame(tf, arr, img_e)
        dec, r = st.poll_pairs(arr)
        out += dec; rows += r
        tfs.append(tf)
    return st, tfs, out, rows


def test_streams_record_every_frame_and_feed_the_pairer():
    st, tfs, out, rows = _paired_session()
    ego_rows = [r for r in rows if r[0] == "ego"]; d_rows = [r for r in rows if r[0] == "d435i"]
    assert [r[2] for r in ego_rows] == tfs and [r[1] for r in ego_rows] == list(range(len(tfs)))
    assert d_rows and all(r[3] == "d435i_header_stamp_ns" for r in d_rows)
    assert len(out) == len(tfs)
    m = [d for d in out if d.matched]
    assert len(m) == len(tfs) and all(abs(d.delta_s) <= 0.0167 + 1e-6 for d in m)
    assert m[0].ego.frame_id == 0 and m[0].ego.raw_time == tfs[0]
    assert m[0].ego.payload["sync"] == (cp.SYNC_RECEIVE, "none")   # L uncalibrated: receive-time pairing


def test_calibration_labels_follow_the_constants(tmp_path):
    st = view.Streams(360, 20.0, 0.0, 0.0)
    assert (st.sync_method, st.calibration_version) == (cp.SYNC_RECEIVE, "none")
    calib = tmp_path / "relative.json"
    calib.write_text(json.dumps(cp.relative_calibration(cp.aggregate_motion_lags([("s", [44.0])]))))
    L, eD, version, calibrated = view.calibration_from_file(str(calib))
    assert (L, eD, calibrated) == (44.0, 0.0, True) and version.startswith("calib:relative.json@") and len(version) == len("calib:relative.json@") + 12
    st = view.Streams(360, 20.0, 0.044, 0.0, calibration_version=version, go2_calibrated=True)
    assert st.sync_method == cp.SYNC_CAPTURE
    st = view.Streams(360, 20.0, 0.0, 0.0)
    st.motion_hist.append({"time": 0, "lag_ms": 44.0, "corr": 0.9, "rms_d435i": 1, "rms_go2": 1})
    assert st.apply_calibration() is not None
    assert st.sync_method == cp.SYNC_CAPTURE and st.calibration_version.startswith("c-key@")


def test_sync_summary_reports_sources_pairs_and_validation():
    st, tfs, out, _ = _paired_session(n_ego=60)
    st.motion_hist += [{"time": 0, "lag_ms": v, "corr": 0.9, "rms_d435i": 1, "rms_go2": 1} for v in (42.3, 43.7, 49.6)]
    s = view.sync_summary(st)
    assert s["sync_method"] == cp.SYNC_RECEIVE and s["calibration_version"] == "none"
    assert s["time_sources"]["ego"]["verdict"] == "frame_counter"
    assert s["time_sources"]["d435i"]["period_ms"] == pytest.approx(33.333, abs=0.01)
    assert s["pairing"]["decided"] == 60 and s["pairing"]["duplicate_d435i_in_pairs"] == 0
    assert s["validation"]["motion"]["pairing_offset_ms"] == pytest.approx(43.7)
    line = view.pairing_line(s, "/x/demo_view_pairs.csv")
    assert line.startswith("pairing: ") and "60/60" in line and "receive_time_approx" in line and "+44 ms" in line
    assert view.pairing_line(view.sync_summary(view.Streams(360, 20.0, 0.0, 0.0))).startswith("pairing: ")


def _press_c(st, lag_ms=44.0):
    st.motion_hist.append({"time": 0, "lag_ms": lag_ms, "corr": 0.9, "rms_d435i": 1, "rms_go2": 1})
    assert st.apply_calibration() is not None


def test_mid_session_calibration_is_reported_per_calibration_not_for_the_whole_session():
    """A c key mid-session must not relabel the receive-time pairs made before it as capture-time
    pairs in the session summary / pairing line."""
    st, tfs, out, _ = _paired_session(n_ego=40, before={20: _press_c})
    s = view.sync_summary(st)
    assert s["sync_method"] == "mixed" and s["calibration_version"] == "mixed"
    cals = s["calibrations"]
    assert [c["sync_method"] for c in cals] == [cp.SYNC_RECEIVE, cp.SYNC_CAPTURE]
    assert cals[0]["calibration_version"] == "none" and cals[1]["calibration_version"].startswith("c-key@")
    assert [c["decided"] for c in cals] == [20, 20] and [c["matched"] for c in cals] == [20, 19]
    # the time base jumps back by L at the key: the first frame after it loses its nearest D435i frame to
    # the last frame before it - marked d435i_used, never paired twice
    assert [(d.ego.frame_id, d.reason) for d in out if not d.matched] == [(20, "d435i_used")]
    assert cals[0]["go2_latency_ms"] == 0.0 and cals[1]["go2_latency_ms"] == pytest.approx(44.0)
    assert cals[0]["motion"]["pairing_offset_ms"] == pytest.approx(44.0)       # the receive-time pairs: ~44 ms apart
    assert cals[1]["motion"]["pairing_offset_ms"] == pytest.approx(0.0)        # after the calibration: same instant
    v = s["validation"]["motion"]
    assert v["calibration_version"] == cals[1]["calibration_version"] and v["pairing_offset_ms"] == pytest.approx(0.0)
    line = view.pairing_line(s)
    assert "mixed" in line and "receive_time_approx 20/20" in line and "capture_time 19/20" in line
    pairs = [cp.pair_row(d, None, *view.sync_of(d, st)) for d in out]            # the rows the main loop writes
    methods = [r[cp.PAIRS_HEADER.index("sync_method")] for r in pairs]
    assert methods == [cp.SYNC_RECEIVE] * 20 + [cp.SYNC_CAPTURE] * 20           # every row keeps its own label


def test_uniform_session_keeps_one_calibration_entry():
    st, tfs, out, _ = _paired_session(n_ego=30)
    s = view.sync_summary(st)
    assert s["sync_method"] == cp.SYNC_RECEIVE and s["calibration_version"] == "none"
    assert len(s["calibrations"]) == 1 and s["calibrations"][0]["decided"] == 30
    assert s["validation"]["motion"]["calibration_version"] == "none"



# ------------------------------------------------ 640x360, 50 Hz grid, two windows + two videos
from wam_go2 import grid_sync as gs  # noqa: E402

WINDOW_SHAPE = (view.GRID_LABEL_H + view.GRID_PANEL_H, view.GRID_PANEL_W, 3)


def test_each_window_is_one_640x360_picture_pixel_exact_under_its_size_and_fps_bar(renderer):
    rng = np.random.default_rng(2)
    ego = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
    win = renderer.window(ego, "Ego 640x360 14.2 fps", "sync 50 Hz")
    assert win.shape == WINDOW_SHAPE == (34 + 360, 640, 3)
    top = view.GRID_LABEL_H
    assert np.array_equal(win[top:], ego), "shown 1:1, nothing drawn on it"
    other = renderer.window(ego, "Ego 640x360 13.9 fps", "sync 50 Hz")
    rows, _ = np.nonzero(np.any(win != other, axis=2))
    assert rows.size and rows.max() < top, "the rate changed: only the bar"
    assert renderer.panel(ego) is ego, "the video frame of a 640x360 picture is the picture itself"


@pytest.mark.parametrize("wh,fills", [((1280, 720), True), ((640, 480), False), ((320, 180), True), ((848, 480), False)])
def test_other_sizes_are_fitted_whole_never_cropped(renderer, wh, fills):
    import cv2
    rng = np.random.default_rng(3)
    img = rng.integers(1, 256, (wh[1], wh[0], 3), dtype=np.uint8)       # no zero pixels: padding is detectable
    panel = renderer.panel(img)
    assert panel.shape == (360, 640, 3) and np.array_equal(panel, view.fit_box(img, 640, 360))
    s = min(640 / wh[0], 360 / wh[1])
    nw, nh = min(640, int(round(wh[0] * s))), min(360, int(round(wh[1] * s)))
    x0, y0 = (640 - nw) // 2, (360 - nh) // 2
    whole = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    assert np.array_equal(panel[y0:y0 + nh, x0:x0 + nw], whole)        # the WHOLE picture, scaled
    assert (nw, nh) == (640, 360) if fills else (panel == 0).any()     # 16:9 fills; other aspects are padded
    assert renderer.window(img, "Ego").shape == WINDOW_SHAPE


def test_a_camera_without_a_frame_shows_the_waiting_panel(renderer):
    blank = renderer.panel(None)
    assert blank.shape == (360, 640, 3) and blank.any() and renderer.window(None, "D435i").shape == WINDOW_SHAPE


def test_two_videos_named_after_the_recording(tmp_path):
    stem, paths = view.video_paths_for(str(tmp_path / "view" / "camera.mp4"), str(tmp_path), "20261108_120000")
    assert stem == "camera"
    assert paths == {"ego": str(tmp_path / "view" / "camera_ego.mp4"), "d435i": str(tmp_path / "view" / "camera_d435i.mp4")}
    stem, paths = view.video_paths_for(None, str(tmp_path), "20261108_120000")
    assert stem == "ego_d435i_20261108_120000" and paths["d435i"] == str(tmp_path / "ego_d435i_20261108_120000_d435i.mp4")
    assert view.video_paths_for(None, str(tmp_path), "x", no_record=True)[1] == {}


def test_grid_labels_name_the_size_and_the_measured_rate():
    assert view.grid_label("Ego", (640, 360), 14.23) == "Ego 640x360 14.2 fps"
    assert view.grid_label("D435i", (640, 360), None) == "D435i 640x360 -- fps"
    assert view.grid_label("Ego", None, None) == "Ego (no frames)"


class _TopicMsg(_Msg):
    def __init__(self, stamp_s, img, frame_id, encoding="rgb8"):
        super().__init__(stamp_s, img, encoding)
        self.header.frame_id = frame_id
        self.step = img.shape[1] * 3


def _grid_session(ego_fps=14.23, seconds=2.0):
    """The ego_stream topic (tf:<time_frame> ids, stamps on the laptop clock) and the D435i at 60 fps,
    fed in arrival order onto the 50 Hz grid in the viewer's ego mode (repeat, 40 ms)."""
    repeat, tol = view.ego_grid_mode()
    sampler = gs.GridSampler(rate_hz=50.0, ego_tol_s=tol, ego_repeat=repeat)
    st = view.Streams(360, 20.0, 0.0, 0.0, grid=sampler, ego_source="topic")
    t0 = time.time() - 5.0
    ev = [(t0 + i / ego_fps + 0.004, "ego", t0 + i / ego_fps, TF0 + int(round(i * 6666.5)))
          for i in range(int(seconds * ego_fps))]
    ev += [(t0 + 0.011 + k / 60.0 + 0.05, "d", t0 + 0.011 + k / 60.0, None) for k in range(int(seconds * 60))]
    ev.sort()
    ticks, rows = [], []
    img_e = np.zeros((360, 640, 3), np.uint8); img_d = np.full((360, 640, 3), 50, np.uint8)
    for rx, kind, t, tf in ev:
        if kind == "ego":
            st.on_ego_image(_TopicMsg(t, img_e, f"tf:{tf}"))
        else:
            st.on_color(_Msg(t, img_d, "rgb8"))
        got, r = st.poll_grid(rx)
        ticks += got; rows += r
    return st, ticks, rows


def test_topic_ego_frames_are_recorded_paired_and_put_on_the_grid():
    st, ticks, rows = _grid_session()
    ego_rows = [r for r in rows if r[0] == "ego"]
    assert ego_rows and all(r[3] == view.EGO_TOPIC_TIME_SOURCE for r in ego_rows)
    assert st.sync_method == cp.SYNC_RECEIVE                  # ego_stream's stamps, no c key yet
    assert len(ticks) > 80
    inner = ticks[1:-1]
    assert all(t.ego_status in ("new", "repeat") and t.d435i_status == "new" for t in inner)
    assert st.shapes == {"ego": (640, 360), "d435i": (640, 360)}
    decisions, _ = st.poll_pairs(time.time(), flush=True)         # every topic frame reached the pairer
    assert len(decisions) == len(ego_rows) and any(d.matched for d in decisions)
    assert {d.ego.payload["sync"] for d in decisions} == {(cp.SYNC_RECEIVE, "none")}


def test_topic_stamps_are_used_as_published_and_a_c_key_latency_is_subtracted():
    st = view.Streams(360, 20.0, 0.0, 0.0, ego_source="topic")
    img = np.zeros((360, 640, 3), np.uint8)
    t0 = time.time() - 5.0
    st.on_ego_image(_TopicMsg(t0, img, f"tf:{TF0}"))
    assert st.go2_time(TF0) == pytest.approx(t0, abs=1e-6), "ego_stream's time, nothing subtracted"
    _press_c(st, lag_ms=20.0)                                  # L = lag + eD = 20 ms
    assert st.go2_time(TF0) == pytest.approx(t0 - 0.020, abs=1e-6)
    assert st.go2_time(TF0, calibrated=False) == pytest.approx(t0, abs=1e-6)
    st.on_ego_image(_TopicMsg(t0 + 0.07, img, "camera"))     # no tf: id: the frame is keyed by its stamp
    key = int(round((t0 + 0.07) * 1e9 / 1000)) * 1000
    assert any(abs(k - key) < 1000 for k in st._topic_t)


def test_grid_sink_writes_one_frame_to_each_video_and_one_row_per_tick(renderer):
    st, ticks, _ = _grid_session()
    ego_video, d_video, rows = [], [], []

    def write(ego, d):
        ego_video.append(ego); d_video.append(d)
        return len(ego_video) - 1
    sink = view.GridSink(renderer, write_video=write, write_row=rows.append, grid_hz=50.0)
    shown = sink.handle(st, ticks, time.time())
    assert len(ego_video) == len(d_video) == len(ticks) == len(rows), "frame i of both videos is tick i"
    assert all(v.shape == (360, 640, 3) for v in ego_video + d_video), "the videos hold the bare 640x360 pictures"
    assert all((v == 0).all() for v in ego_video[1:-1]) and all((v == 50).all() for v in d_video[1:-1])
    assert [r[-3] for r in rows] == list(range(len(ticks))), "the CSV's video_frame is the index in both videos"
    assert rows[0][2] == "new" and rows[0][7] == "new"
    assert sink.video_frames == len(ticks)
    assert len(shown) == 2 and all(w.shape == WINDOW_SHAPE for w in shown), "two windows: Ego, D435i"
    assert (shown[0][view.GRID_LABEL_H:] == 0).all() and (shown[1][view.GRID_LABEL_H:] == 50).all()
    assert sink.handle(st, [], time.time()) is shown, "no tick for < 1 s: the windows keep the last tick"


def test_grid_sink_marks_repeated_ego_frames_of_the_14_fps_stream(renderer):
    st, ticks, _ = _grid_session(ego_fps=14.23, seconds=2.0)
    statuses = [t.ego_status for t in ticks[1:-1]]
    assert "repeat" in statuses and "new" in statuses


def test_grid_rows_keep_the_calibration_their_ego_frame_was_timed_with(renderer):
    # a c key mid-session must not relabel grid rows of ego frames timed before it (the same rule as the pairs)
    sampler = gs.GridSampler(rate_hz=50.0, ego_tol_s=0.040, ego_repeat=True)
    st = view.Streams(360, 20.0, 0.0, 0.0, grid=sampler, ego_source="topic", calibration_version="none")
    t0 = time.time() - 5.0
    img_e = np.zeros((360, 640, 3), np.uint8); img_d = np.full((360, 640, 3), 50, np.uint8)
    for k in range(40):                                        # D435i 60 fps over ~0.67 s
        st.on_color(_Msg(t0 + k / 60.0, img_d, "rgb8"))
    st.on_ego_image(_TopicMsg(t0 + 0.10, img_e, "tf:1"))        # timed under "none"
    st.motion_hist.append({"time": time.time(), "lag_ms": 5.0, "corr": 0.9, "rms_d435i": 1, "rms_go2": 1})
    assert st.apply_calibration()                               # the c key
    st.on_ego_image(_TopicMsg(t0 + 0.30, img_e, "tf:2"))        # timed under the new calibration
    ticks, _ = st.poll_grid(time.time())
    ticks += st.poll_grid(time.time(), flush=True)[0]
    rows = []
    view.GridSink(renderer, write_row=rows.append).handle(st, ticks, time.time())
    by_frame = {r[3]: (r[-2], r[-1]) for r in rows if r[2] in ("new", "repeat")}
    assert by_frame[0] == (cp.SYNC_RECEIVE, "none")
    assert by_frame[1][0] == cp.SYNC_CAPTURE and by_frame[1][1].startswith("c-key@")


def test_frontvideo_go2_latency_is_not_applied_again_to_the_ego_stream_topic():
    # --calib / --go2-latency-ms calibrate the /frontvideostream arrival model; the ego_stream node has
    # already subtracted its own latency from the topic stamps
    used = view.ego_calibration_for_source("topic", 44.0, 3.0, "calib:x.json@abc", True)
    assert used["go2_latency_ms"] == 0.0 and used["go2_calibrated"] is False
    assert used["d435i_stamp_offset_ms"] == 3.0 and "frontvideo" in used["note"]
    same = view.ego_calibration_for_source("frontvideo", 44.0, 3.0, "calib:x.json@abc", True)
    assert same == {"go2_latency_ms": 44.0, "d435i_stamp_offset_ms": 3.0, "calibration_version": "calib:x.json@abc",
                    "go2_calibrated": True, "note": ""}


# --- ego sampled with repeat --------------------------------------------------------------------
# The ~14 fps ego stream (frame intervals ~66-75 ms) on the 50 Hz grid: with repeat every tick shows the
# nearest frame; a frame at most once (no repeat, half a period) would leave most ticks black.

def _ego_ticks(mode, intervals_ms, seconds=3.0):
    repeat, tol = mode
    s = gs.GridSampler(rate_hz=50.0, ego_tol_s=tol, ego_repeat=repeat)
    t, fid, frames = T0_REPEAT, 0, []
    while t < T0_REPEAT + seconds:
        frames.append(cp.Frame("ego", fid, int(t * 1e9), t + 0.02, t, {"img": fid}))
        t += intervals_ms[fid % len(intervals_ms)] * 1e-3
        fid += 1
    d = [cp.Frame("d435i", i, int((T0_REPEAT + i / 60) * 1e9), T0_REPEAT + i / 60 + 0.03, T0_REPEAT + i / 60, i)
         for i in range(int(seconds * 60))]
    events = sorted([(f.receive_time, "e", f) for f in frames] + [(f.receive_time, "d", f) for f in d],
                    key=lambda e: e[0])
    ticks = []
    for rx, kind, f in events:
        (s.add_ego if kind == "e" else s.add_d435i)(f)
        ticks.extend(s.poll(rx))
    ticks += s.flush()
    return ticks[2:-2], len(frames)


T0_REPEAT = 1_800_000_000.0
EGO_STREAM_MS = (70, 72, 68, 75, 66, 71)            # ~14 fps with jitter


def test_ego_is_always_sampled_with_repeat_and_40_ms():
    assert view.ego_grid_mode() == (True, pytest.approx(0.040))
    assert view.ego_grid_mode(25) == (True, pytest.approx(0.025))


def test_the_14_fps_ego_stream_leaves_no_tick_black_with_repeat():
    inner, n = _ego_ticks(view.ego_grid_mode(), EGO_STREAM_MS)
    assert all(t.ego_status in ("new", "repeat") for t in inner), "no black (missing) tick"
    new = [t.ego.frame_id for t in inner if t.ego_status == "new"]
    assert len(new) == len(set(new)), "a frame is new once, then marked repeat"
    assert sum(t.ego_status == "repeat" for t in inner) > 0
    no_repeat, _ = _ego_ticks((False, 0.010), EGO_STREAM_MS)
    assert sum(t.ego_status == "missing" for t in no_repeat) / len(no_repeat) > 0.5


def test_a_stream_as_fast_as_the_grid_still_gets_a_new_frame_every_tick():
    inner, _ = _ego_ticks(view.ego_grid_mode(), (1000 / 60,))
    assert all(t.ego_status == "new" for t in inner)
    ids = [t.ego.frame_id for t in inner]
    assert len(ids) == len(set(ids))
