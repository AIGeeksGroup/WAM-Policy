"""scripts/x5/pair_report.py: per-session time-sync lines and the cross-session common-event re-test.
Inputs are ego_d435i_view.py summary JSONs; no ROS."""
import json
import os
import sys

import pytest

X5_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "x5"))
if X5_DIR not in sys.path:
    sys.path.insert(0, X5_DIR)
import pair_report  # noqa: E402


def _hist(*lags):
    return [{"time": 0, "lag_ms": v, "corr": 0.9, "rms_d435i": 1.0, "rms_go2": 2.0} for v in lags]


@pytest.fixture
def sessions(tmp_path):
    plain = {"time": "2026-10-07 07:11:01 +1100", "wall_s": 26.4, "msgs": {"go2": 378, "d435i": 762},
           "motion_lag_hist": _hist(49.6, 43.7, 42.3)}                  # a summary without a time_sync block
    full = {"time": "2026-10-08 10:00:00 +1100", "wall_s": 60.0, "msgs": {"go2": 850, "d435i": 1790},
           "motion_lag_hist": _hist(45.0, 46.0),
           "time_sync": {"sync_method": "receive_time_approx", "calibration_version": "none",
                         "pairing": {"decided": 840, "matched": 820, "unmatched": 20, "unmatched_ratio": 20 / 840,
                                     "delta_ms": {"n": 820, "median": 0.3, "p95_abs": 15.1, "max_abs": 19.8}},
                         "time_sources": {"ego": {"verdict": "frame_counter", "receive_rate_hz": 14.27, "lost_frames": 0},
                                          "d435i": {"stamp_rate_hz": 29.9, "lost_frames": 3}}}}
    quiet = {"time": "2026-10-08 11:00:00 +1100", "wall_s": 30.0, "msgs": {"go2": 0, "d435i": 0}, "motion_lag_hist": []}
    paths = []
    for name, d in (("demo_plate_walk_20261007_071034_view", plain), ("demo_box_push_20261008_100000_view", full),
                    ("demo_stand_fl_up_20261008_110000_view", quiet)):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(d))
        paths.append(str(p))
    return paths


def test_report_lines_and_aggregate(sessions, capsys):
    assert pair_report.main(sessions) == 0
    out = capsys.readouterr().out
    assert "demo_plate_walk_20261007_071034_view" in out and "820/840" in out and "receive_time_approx" in out
    assert "no time_sync block" in out                                   # such summaries still report their motion lags
    assert "windows 5" in out and "median +45.0 ms" in out               # 42.3 43.7 45 46 49.6
    assert "session medians spread 1.8 ms" in out                        # 43.7 vs 45.5


def test_write_calib_is_readable_by_the_viewer(sessions, tmp_path, capsys):
    out = tmp_path / "calib_relative.json"
    assert pair_report.main(sessions + ["--write-calib", str(out)]) == 0
    cal = json.loads(out.read_text())
    assert cal["kind"] == "relative_camera_motion" and cal["n_windows"] == 5
    assert cal["model_input"]["calibration"]["go2_ego_latency_ms"] == pytest.approx(45.0)
    assert cal["inputs"] == sessions
    import ego_d435i_view as view
    L, eD, version, calibrated = view.calibration_from_file(str(out))
    assert (L, eD, calibrated) == (pytest.approx(45.0), 0.0, True) and version.startswith("calib:calib_relative.json@")


def test_write_calib_refuses_without_any_common_event(sessions, tmp_path, capsys):
    out = tmp_path / "calib.json"
    assert pair_report.main([sessions[2], "--write-calib", str(out)]) == 2
    assert not out.exists() and "no common-event lag" in capsys.readouterr().out


def test_mixed_session_line_shows_each_calibration(tmp_path, capsys):
    d = {"time": "2026-10-08 12:00:00 +1100", "wall_s": 50.0, "motion_lag_hist": _hist(44.0),
         "time_sync": {"sync_method": "mixed", "calibration_version": "mixed",
                       "calibrations": [{"calibration_version": "none", "sync_method": "receive_time_approx", "decided": 300, "matched": 295},
                                        {"calibration_version": "c-key@12:00:20:L=44.0ms,eD=0.0ms", "sync_method": "capture_time",
                                         "decided": 400, "matched": 398}],
                       "pairing": {"decided": 700, "matched": 693, "unmatched": 7, "unmatched_ratio": 0.01,
                                   "delta_ms": {"n": 693, "p95_abs": 15.0}},
                       "time_sources": {"ego": {}, "d435i": {}}}}
    p = tmp_path / "mixed_view.json"
    p.write_text(json.dumps(d))
    assert pair_report.main([str(p)]) == 0
    out = capsys.readouterr().out
    assert "mixed: receive_time_approx 295/300 [none] then capture_time 398/400 [c-key@12:00:20" in out
