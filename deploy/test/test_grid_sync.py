"""wam_go2/grid_sync.py: both cameras resampled onto one 50 Hz (20 ms) capture-time grid.

The D435i colour sensor offers 640x360 at 60/30/15/6 fps only (no 50); the Go2 front camera
streams ~14.2 fps (its 720p and 360p messages carry the same time_frame: one capture encoded twice).
So every 20 ms tick takes, per camera, the frame whose capture time is nearest the tick:
  * D435i (60 fps): a NEW frame every tick, each frame used at most once, |offset| <= 10 ms;
  * Go2 ego (14.2 fps): the nearest frame within 40 ms; a frame shown on several ticks is
    "new" on the first and "repeat" afterwards - never presented as a fresh capture;
  * no frame within the tolerance: "missing" (nothing is filled in).
Synthetic rig: capture instants T on the laptop clock, arrival = T + latency.
"""
import numpy as np
import pytest

from wam_go2 import camera_pairing as cp
from wam_go2 import grid_sync as gs

T0 = 1_800_000_000.0           # an epoch-scale laptop time, so float rounding is exercised


def ego(fid, t, lat=0.055):
    return cp.Frame("ego", fid, 433_000_000 + 6667 * fid, t + lat, t, {"img": f"ego{fid}"})


def d435i(fid, t, lat=0.080):
    return cp.Frame("d435i", fid, int(round(t * 1e9)), t + lat, t, f"d{fid}")


def feed(sampler, duration=2.0, ego_fps=14.23, d_fps=60.0, start=T0 + 0.003, ego_drop=(), d_drop=()):
    """All frames of `duration` s, added in arrival order; returns the newest arrival time."""
    events = []
    for i in range(int(duration * ego_fps)):
        if i not in ego_drop:
            f = ego(i, start + i / ego_fps)
            events.append((f.receive_time, "ego", f))
    for i in range(int(duration * d_fps)):
        if i not in d_drop:
            f = d435i(i, start + 0.011 + i / d_fps)
            events.append((f.receive_time, "d", f))
    events.sort(key=lambda e: e[0])
    out = []
    for rx, kind, f in events:
        (sampler.add_ego if kind == "ego" else sampler.add_d435i)(f)
        out.extend(sampler.poll(rx))
    return out, events[-1][0]


def test_ticks_are_on_the_20_ms_grid_and_d435i_is_new_every_tick():
    s = gs.GridSampler(rate_hz=50.0)
    ticks, last_rx = feed(s)
    ticks += s.flush()
    assert len(ticks) > 80
    idx = [t.index for t in ticks]
    assert idx == list(range(idx[0], idx[0] + len(idx))), "consecutive ticks, none skipped"
    for t in ticks:
        assert t.time == pytest.approx(t.index * 0.020, abs=1e-6)
    inner = ticks[1:-1]
    assert all(t.d435i_status == "new" for t in inner)
    assert max(abs(t.d435i_offset_s) for t in inner) <= 1 / 120 + 1e-6      # half a 60 fps period
    used = [t.d435i.frame_id for t in ticks if t.d435i is not None]
    assert len(used) == len(set(used)), "a D435i frame is never used twice"


def test_ego_repeats_are_marked_and_new_frames_counted_once():
    s = gs.GridSampler(rate_hz=50.0)
    ticks, _ = feed(s)
    ticks += s.flush()
    inner = ticks[2:-2]
    news = [t for t in inner if t.ego_status == "new"]
    reps = [t for t in inner if t.ego_status == "repeat"]
    assert news and reps and not [t for t in inner if t.ego_status == "missing"]
    assert len({t.ego.frame_id for t in news}) == len(news), "each ego frame is new exactly once"
    assert 0.24 < len(news) / len(inner) < 0.33, "14.2 fps on a 50 Hz grid: ~28 % of the ticks bring a new frame"
    for t in reps:
        prev = [p for p in ticks if p.index < t.index and p.ego is not None][-1]
        assert prev.ego.frame_id == t.ego.frame_id, "a repeat is the frame of the previous tick"
    assert max(abs(t.ego_offset_s) for t in inner) <= 0.5 / 14.23 + 1e-6


def test_gaps_are_missing_never_filled():
    s = gs.GridSampler(rate_hz=50.0)
    ticks, _ = feed(s, d_drop=range(30, 40), ego_drop=range(10, 14))
    ticks += s.flush()
    d_missing = [t for t in ticks if t.d435i_status == "missing"]
    e_missing = [t for t in ticks if t.ego_status == "missing"]
    assert 5 <= len(d_missing) <= 9, "10 lost D435i frames (167 ms) -> the ticks with nothing within 10 ms"
    assert all(t.d435i is None and t.d435i_offset_s is None for t in d_missing)
    assert e_missing and all(t.ego is None for t in e_missing)
    stats = s.stats.as_dict()
    assert stats["d435i"]["missing"] == len(d_missing) and stats["ego"]["missing"] == len(e_missing)


def test_a_tick_waits_for_its_frames_but_not_forever():
    s = gs.GridSampler(rate_hz=50.0, max_wait_s=0.40)
    s.add_ego(ego(0, T0 + 0.000))
    s.add_d435i(d435i(0, T0 + 0.000))
    assert s.poll(T0 + 0.100) == [], "no stream is past the first tick yet: nothing decided"
    s.add_d435i(d435i(1, T0 + 1 / 60))
    s.add_d435i(d435i(2, T0 + 2 / 60))
    assert s.poll(T0 + 0.120) == [], "the ego stream is not past the tick (+40 ms) yet"
    late = s.poll(T0 + 0.000 + 0.41)
    assert late and late[0].index == round(T0 / 0.02), "after max_wait the tick is decided with what is there"


def test_a_silent_stream_does_not_stall_the_other_and_both_silent_stops_the_grid():
    s = gs.GridSampler(rate_hz=50.0, silence_s=1.0)
    t = T0
    for i in range(120):                                   # D435i only, 2 s
        s.add_d435i(d435i(i, t + i / 60))
    ticks = s.poll(t + 2.0 + 1.1)                          # the ego stream never came: silent
    assert len(ticks) > 90 and all(k.ego_status == "missing" and k.d435i_status == "new" for k in ticks[1:-1])
    more = s.poll(t + 30.0)
    assert len(more) <= 2, "no frames at all any more: the grid does not run on into the void"


def test_without_a_time_model_ego_frames_are_counted_not_used():
    s = gs.GridSampler(rate_hz=50.0)
    s.add_ego(cp.Frame("ego", 0, 433_000_000, T0 + 0.05, None, {"img": "x"}))
    assert s.stats.as_dict()["ego"]["no_time_model"] == 1


def test_report_and_row_and_line():
    s = gs.GridSampler(rate_hz=50.0)
    ticks, _ = feed(s, duration=3.0)
    ticks += s.flush()
    rep = s.stats.as_dict()
    assert rep["rate_hz"] == 50.0 and rep["ticks"] == len(ticks)
    assert rep["d435i"]["new_ratio"] > 0.97 and rep["d435i"]["duplicates"] == 0
    assert 55 < rep["d435i"]["fps"] < 65 and 13.5 < rep["ego"]["fps"] < 15.0
    assert rep["ego"]["offset_ms"]["max_abs"] <= 36.0 and rep["d435i"]["offset_ms"]["max_abs"] <= 8.4
    row = gs.grid_row(ticks[5], 5, cp.SYNC_RECEIVE, "none")
    assert len(row) == len(gs.GRID_HEADER) and row[0] == ticks[5].index and row[-3] == 5
    line = gs.grid_line(rep)
    assert line.startswith("grid 50 Hz:") and "D435i new" in line and "ego new" in line


def test_rate_and_tolerances_are_validated():
    with pytest.raises(ValueError):
        gs.GridSampler(rate_hz=0.0)
    with pytest.raises(ValueError):
        gs.GridSampler(rate_hz=50.0, d435i_tol_s=0.011)     # over half a period: one frame could serve two ticks
    gs.GridSampler(rate_hz=50.0, d435i_tol_s=0.010, ego_tol_s=0.040)
    assert np.isclose(gs.GridSampler(rate_hz=50.0).period_s, 0.020)


def test_without_repeats_an_ego_stream_as_fast_as_the_grid_gets_a_new_frame_every_tick():
    # ego_repeat=False: an ego stream at 60 fps behaves like the D435i
    s = gs.GridSampler(rate_hz=50.0, ego_tol_s=0.010, ego_repeat=False)
    ticks, _ = feed(s, ego_fps=60.0)
    ticks += s.flush()
    inner = ticks[1:-1]
    assert all(t.ego_status == "new" and t.d435i_status == "new" for t in inner)
    used = [t.ego.frame_id for t in ticks if t.ego is not None]
    assert len(used) == len(set(used))
    assert max(abs(t.ego_offset_s) for t in inner) <= 1 / 120 + 1e-6
    assert s.stats.as_dict()["ego"]["repeat"] == 0
    with pytest.raises(ValueError):
        gs.GridSampler(rate_hz=50.0, ego_tol_s=0.040, ego_repeat=False)   # over half a period without repeats
