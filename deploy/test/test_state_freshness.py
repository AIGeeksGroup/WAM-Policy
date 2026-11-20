"""StateFreshnessMonitor: stale-LowState detection per control tick. With
/lowstate streamed over WiFi, 12-17 % of ticks ran on an unchanged frame in
bursts of up to 170 ms, below the 0.25 s communication watchdog."""

import pytest

from wam_go2.safety.state_freshness import StateFreshnessMonitor

DT = 0.01


def feed(mon, pattern, start=0):
    """pattern: 'F' = a new frame arrived before the tick, 'S' = no new frame.
    Returns the list of non-None warnings."""
    count = start
    out = []
    for ch in pattern:
        if ch == "F":
            count += 1
        w = mon.update(count)
        if w is not None:
            out.append(w)
    return out


def test_first_tick_is_fresh_and_all_fresh_is_silent():
    mon = StateFreshnessMonitor(DT)
    assert feed(mon, "F" * 500) == []
    assert mon.total_ticks == 500 and mon.total_stale == 0
    assert mon.longest_run == 0 and mon.gaps == 0
    assert mon.stale_fraction == 0.0
    assert not mon.last_stale
    assert mon.summary().startswith("500 ticks, 0 (0.0%)")


def test_frames_arriving_faster_than_ticks_count_as_fresh():
    mon = StateFreshnessMonitor(DT)
    count = 0
    for _ in range(100):
        count += 5                     # 500 Hz stream, 100 Hz ticks
        assert mon.update(count) is None
    assert mon.total_stale == 0


def test_one_long_gap_is_reported_with_its_length():
    mon = StateFreshnessMonitor(DT, window=2.0, gap_warn=0.05)
    # 17 ticks on the same frame = the 170 ms gaps seen on the robot
    warnings = feed(mon, "F" * 10 + "S" * 17 + "F" * 173)
    assert mon.longest_run == 17 and mon.gaps == 1 and mon.total_stale == 17
    assert len(warnings) == 1
    assert "170 ms" in warnings[0] and "1 gaps >= 50 ms" in warnings[0]
    assert "8.5%" in warnings[0]        # 17 / 200


def test_short_scattered_gaps_below_threshold_are_silent():
    mon = StateFreshnessMonitor(DT, window=2.0, gap_warn=0.05, fraction_warn=0.10)
    # 4 % stale, longest run 2 ticks (20 ms): below both thresholds
    assert feed(mon, ("F" * 24 + "S") * 8 * 2) == []
    assert mon.total_stale == 16 and mon.longest_run == 1 and mon.gaps == 0


def test_high_stale_fraction_without_long_gaps_is_reported_once_per_window():
    mon = StateFreshnessMonitor(DT, window=2.0, gap_warn=0.05, fraction_warn=0.10)
    warnings = feed(mon, "FFS" * 200)      # 33 % stale, runs of 1 tick, 6 s
    assert len(warnings) == 3
    # 66 or 67 stale ticks per 200-tick window; the longest run is one tick
    assert all(w.startswith("LowState stale on 33.") and "longest gap 10 ms" in w
               and "0 gaps" in w for w in warnings)


def test_stale_run_spanning_a_window_boundary_is_counted_once():
    mon = StateFreshnessMonitor(DT, window=1.0, gap_warn=0.05)
    feed(mon, "F" * 95 + "S" * 10 + "F" * 95)   # run crosses the 100-tick boundary
    assert mon.gaps == 1 and mon.longest_run == 10


def test_reset_clears_the_run_totals_but_keeps_the_frame_counter():
    mon = StateFreshnessMonitor(DT)
    feed(mon, "F" * 5 + "S" * 6)
    assert mon.total_stale == 6
    mon.reset()
    assert mon.total_ticks == 0 and mon.total_stale == 0 and mon.longest_run == 0
    assert mon.gaps_including_open == 0      # the pre-reset run is not carried over
    # the next tick with the same frame count is still stale (counter kept)
    assert mon.update(5) is None
    assert mon.last_stale and mon.total_stale == 1 and mon.longest_run == 1


def test_summary_counts_a_gap_still_open_at_the_end_of_the_run():
    """A stale run that ends the session (e.g. a fall) never sees the fresh
    tick that would close it; the POLICY-exit summary must still count it."""
    mon = StateFreshnessMonitor(DT)
    feed(mon, "F" * 90 + "S" * 10)
    assert mon.last_stale
    assert mon.gaps == 0 and mon.gaps_including_open == 1
    s = mon.summary()
    assert "100 ticks" in s and "10 (10.0%)" in s and "100 ms" in s and "1 gaps" in s
    feed(mon, "F")
    assert mon.gaps == 1 and mon.gaps_including_open == 1   # closed, not double counted


def test_bad_dt_rejected():
    with pytest.raises(ValueError):
        StateFreshnessMonitor(0.0)
