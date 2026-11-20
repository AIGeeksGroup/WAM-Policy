"""ROS-free tests for the analysis side of scripts/x5/sync_check.py (rclpy is
imported only inside capture(), which needs the robot and the X5 and is not
exercised here). Reference numbers come from a recorded static reference session
(test/fixtures/static_reference_session.*, read only)."""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
X5 = os.path.join(PKG, "scripts", "x5")
REF = os.path.join(PKG, "test", "fixtures", "static_reference_session")
sys.path.insert(0, PKG)

from wam_go2 import timesync as ts  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(REF + ".npz"), reason="reference session not present")

spec = importlib.util.spec_from_file_location("_test_x5_sync_check", os.path.join(X5, "sync_check.py"))
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)


def m0_report():
    with open(REF + ".json") as f:
        return json.load(f)


def m0_arrays():
    with np.load(REF + ".npz") as z:
        return {k: z[k].copy() for k in z.files}


def items(assessment):
    return {i["name"]: i for i in assessment["items"]}


# ------------------------------------------------ combined clock estimator
def test_host_clock_assessment_on_real_m0():
    a = sc.host_clock_assessment(m0_report()["hosts"])
    assert a["ok"], a
    it = items(a)
    # X5 raw - laptop raw = SNTP offset (X5 served minus laptop raw) - chrony offset (X5 served minus X5 raw)
    assert it["before"]["x5_raw_minus_laptop_raw_ms"] == pytest.approx(-0.08750 + 0.00181, abs=1e-4)
    assert it["after"]["x5_raw_minus_laptop_raw_ms"] == pytest.approx(-0.09882 - 0.01243, abs=1e-4)
    assert "in_band" not in it                            # host numbers only without arrays


def test_raw_offset_combines_the_two_halves():
    h = m0_report()["hosts"]
    h["x5"]["offset_s"] = 0.004            # X5 raw 4 ms slow of its served time ...
    h["x5_sntp"]["offset_s"] = 0.004       # ... and the served time 4 ms ahead of the laptop: raw clocks agree
    assert items(sc.host_clock_assessment(h))["before"]["ok"]
    h["x5_sntp"]["offset_s"] = -0.002      # served time 2 ms behind: X5 raw is 6 ms behind the laptop raw
    it = items(sc.host_clock_assessment(h))["before"]
    assert not it["ok"] and it["x5_raw_minus_laptop_raw_ms"] == pytest.approx(-6.0, abs=1e-9)


@pytest.mark.parametrize("edit", [
    lambda h: h.pop("x5"),
    lambda h: h.pop("x5_sntp_after"),
    lambda h: h.__setitem__("x5", {"error": "ssh: connect to host 192.0.2.11 port 22: timed out"}),
    lambda h: h.__setitem__("x5_sntp", {"error": "no SNTP reply"}),
    lambda h: h["x5_after"].update(synced=False),
    lambda h: h["x5_sntp"].update(remote_synced=False),
    lambda h: h["x5_sntp_after"].update(rtt_s=0.005),
    lambda h: h["x5"].update(offset_s=0.250),
])
def test_host_clock_assessment_fails_closed(edit):
    h = m0_report()["hosts"]
    edit(h)
    assert not sc.host_clock_assessment(h)["ok"]


def test_in_band_imu_check_uses_gyro_then_imu():
    h, arr = m0_report()["hosts"], m0_arrays()
    it = items(sc.host_clock_assessment(h, arr))["in_band"]
    assert it["ok"] and it["stream"] == "gyro" and it["median_ms"] == pytest.approx(1.3366, abs=1e-3)
    no_gyro = dict(arr, gyro_recv_ns=np.empty(0, np.int64), gyro_stamp_ns=np.empty(0, np.int64))
    it = items(sc.host_clock_assessment(h, no_gyro))["in_band"]
    assert it["ok"] and it["stream"] == "imu" and it["median_ms"] == pytest.approx(1.9328, abs=1e-3)
    for d_ms in (+10.0, -15.0):                          # recv - stamp median shifted by d
        shifted = dict(arr, gyro_stamp_ns=arr["gyro_stamp_ns"] - int(d_ms * 1e6))
        assert not items(sc.host_clock_assessment(h, shifted))["in_band"]["ok"], d_ms
    empty = {k: np.empty(0, np.int64) for k in ("gyro_recv_ns", "gyro_stamp_ns", "imu_recv_ns", "imu_stamp_ns")}
    a = sc.host_clock_assessment(h, empty)
    assert not a["ok"] and not items(a)["in_band"]["ok"]
    assert not sc.host_clock_assessment(h, {})["ok"]     # no IMU keys at all: fail closed


def test_host_verdict_lines_on_real_m0():
    lines, a = sc.host_verdict_lines(m0_report()["hosts"])
    assert a["ok"]
    assert lines[0].startswith("--  laptop chrony")
    assert lines[1].startswith("OK  X5 chrony offset")
    assert lines[2].startswith("--  SNTP") and "X5 served (corrected) time minus laptop raw time" in lines[2]
    assert "combine with the X5 chrony offset (line above)" in lines[2] and "independent" not in lines[2]
    assert lines[3].startswith("OK ") and "X5 raw clock minus laptop raw clock" in lines[3]
    assert "before -0.086 ms" in lines[3] and "after -0.111 ms" in lines[3]


def test_host_verdict_laptop_slew_is_bad_and_sntp_is_info():
    h = m0_report()["hosts"]
    for k in ("x5_sntp", "x5_sntp_after"):
        h[k]["offset_s"] += 0.020
    lines, a = sc.host_verdict_lines(h)
    assert not a["ok"]
    assert lines[1].startswith("OK  X5 chrony")          # chrony alone cannot see the laptop side
    assert lines[2].startswith("--  SNTP")               # informational, never BAD on its own
    assert lines[3].startswith("BAD") and "+19.9" in lines[3]


def test_host_verdict_sntp_error_is_bad_on_the_combined_line():
    h = m0_report()["hosts"]
    h["x5_sntp"] = {"error": "no SNTP reply from 192.0.2.11:123"}
    lines, a = sc.host_verdict_lines(h)
    assert lines[2].startswith("--  SNTP to X5: no SNTP reply")
    assert lines[3].startswith("BAD") and not a["ok"]


# ------------------------------------------------------ ego resolutions seen
def _imu_verdict(arr):
    _, verdict = sc.analyse(30.0, arr, 0.2)
    lines = [v for v in verdict if "camera IMU recv-stamp" in v]
    assert len(lines) == 1
    return lines[0]


@pytest.mark.parametrize("shift_ms, ok", [(0.0, True), (+4.0, True), (+15.0, False), (-6.0, False)])
def test_in_band_verdict_uses_the_recv_stamp_window(shift_ms, ok):
    """The camera IMU verdict window is IMU_RECV_STAMP_MS +- IMU_RECV_STAMP_TOL_MS (1.5 +- 5 ms; reference
    gyro median +1.34 ms)."""
    arr = m0_arrays()
    arr["gyro_recv_ns"] = arr["gyro_recv_ns"] + int(shift_ms * 1e6)
    line = _imu_verdict(arr)
    assert line.startswith("OK ") is ok, line
    assert "1.5 +- 5.0 ms" in line


def test_res_seen_array():
    a = sc.res_seen_array({720: 5, 360: 3})
    assert a.dtype == np.int64 and a.tolist() == [[360, 3], [720, 5]]
    e = sc.res_seen_array({})
    assert e.dtype == np.int64 and e.shape == (0, 2)


# --------------------------------- capture-grid offset label + res_seen JSON
def test_grid_line_is_offline_capture_grid_and_names_online_age():
    rep, arr = m0_report(), m0_arrays()
    arr["ego_res_seen"] = sc.res_seen_array({360: 428, 720: 431})
    fit = ts.TickFit(**rep["lowstate"]["tick_fit"])
    model, lines = sc.analyse_model_input(rep["observed_s"], arr, fit, go2_latency_ms=150.0, d435i_offset_ms=20.0)
    grid = [l for l in lines if "80 ms grid" in l]
    assert len(grid) == 1, lines
    assert "capture grid (offline)" in grid[0] and "online pick" not in grid[0]
    assert "online consumer's ego age = L" in grid[0] and "decode" in grid[0]
    assert model["go2_ego"]["res_seen"] == {"360": 428, "720": 431}


# -------------------------------------------------- main() with probes stubbed
def test_main_writes_host_clock_and_verdict_lines(tmp_path, monkeypatch, capsys):
    rep, arr = m0_report(), m0_arrays()
    h = rep["hosts"]
    chrony = iter([h["laptop"], h["x5"], h["x5_after"]])
    sntp = iter([h["x5_sntp"], h["x5_sntp_after"]])
    monkeypatch.setattr(sc, "chronyc_tracking", lambda cmd_prefix=(): next(chrony))
    monkeypatch.setattr(sc, "sntp_offset", lambda host, **kw: next(sntp))
    monkeypatch.setattr(sc, "x5_metadata_sample", lambda *a, **kw: rep["x5_metadata"])
    monkeypatch.setattr(sc, "capture", lambda *a, **kw: (rep["observed_s"], dict(arr, ego_res_seen=sc.res_seen_array({360: 428}))))
    monkeypatch.setattr(sys, "argv", ["sync_check.py", "--out-dir", str(tmp_path), "--label", "t"])
    monkeypatch.setenv("X5_IP", "192.0.2.11")                        # config/deploy.env
    assert sc.main() == 0
    capsys.readouterr()
    (js,) = tmp_path.glob("t_*.json")
    out = json.loads(js.read_text())
    assert out["host_clock"]["ok"] and [i["name"] for i in out["host_clock"]["items"]] == ["before", "after"]
    assert out["hosts"] == h                                         # the raw probe results stay in the JSON
    v = out["verdict"]
    assert v[0].startswith("--  laptop chrony") and v[1].startswith("OK  X5 chrony")
    assert v[2].startswith("--  SNTP X5 served (corrected) time") and v[3].startswith("OK  X5 raw clock minus laptop raw clock")
    assert out["model_input"]["go2_ego"]["res_seen"] == {"360": 428}
    with np.load(str(js)[:-5] + ".npz") as z:
        assert z["ego_res_seen"].tolist() == [[360, 428]]
