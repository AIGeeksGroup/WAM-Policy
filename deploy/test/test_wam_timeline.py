"""wam_go2/wam/timeline.py: the logical episode clock and the chunk schedule.

The protocol counts 2 ms physics ticks: a native9 command is 10 ticks (20 ms), a
feedback sample every 40 ticks (80 ms), a chunk 16 commands = 160 ticks, and a
controller step 5 ticks (10 ms). On the robot the ticks advance only while a
command is being held; wall time is a separate axis.
"""
import pytest

from wam_go2.wam import timeline as tl


def test_protocol_constants_match_qlm_timing_profile():
    assert tl.PHYSICS_STEP_NS == 2_000_000
    assert tl.COMMAND_TICKS == 10 and tl.STEP_TICKS == 5
    assert tl.SAMPLE_TICKS == 40 and tl.GROUP_COMMANDS == 16
    assert tl.GROUP_TICKS == 160
    assert tl.COMMAND_S == pytest.approx(0.020) and tl.STEP_S == pytest.approx(0.010)
    assert tl.SAMPLE_S == pytest.approx(0.080)


def test_logical_clock_starts_at_origin_and_only_advances_explicitly():
    clock = tl.LogicalClock()
    assert clock.tick == 0 and clock.origin_ns == 0
    assert clock.sim_time_ns() == 0
    clock.advance_steps(2)                  # one 20 ms command = two controller steps
    assert clock.tick == 10
    assert clock.sim_time_ns() == 20_000_000
    assert clock.sim_time_ns(40) == 80_000_000
    with pytest.raises(ValueError):
        clock.advance_steps(0)
    with pytest.raises(ValueError):
        clock.advance_steps(-1)


def test_chunk_schedule_commands_steps_and_samples():
    sched = tl.ChunkSchedule(start_tick=160, start_wall=100.0)
    assert sched.n_commands == 16
    assert sched.command_tick(0) == 160 and sched.command_tick(15) == 310
    assert sched.command_wall(0) == pytest.approx(100.0)
    assert sched.command_wall(15) == pytest.approx(100.0 + 15 * 0.02)
    assert sched.end_tick == 320
    assert sched.end_wall == pytest.approx(100.32)
    # 32 controller steps of 10 ms, each belonging to one command
    steps = list(sched.steps())
    assert len(steps) == 32
    assert steps[0] == tl.StepSlot(index=0, command_index=0, start_tick=160, end_tick=165,
                                   start_wall=pytest.approx(100.0), end_wall=pytest.approx(100.01))
    assert steps[31].command_index == 15 and steps[31].end_tick == 320
    # the four feedback samples: 80/160/240/320 ms after the boundary
    assert sched.sample_ticks == [200, 240, 280, 320]
    assert [round(t - 100.0, 3) for t in sched.sample_walls] == [0.08, 0.16, 0.24, 0.32]
    assert sched.sample_wall_of_tick(240) == pytest.approx(100.16)
    with pytest.raises(ValueError):
        sched.sample_wall_of_tick(241)


def test_chunk_schedule_rejects_misaligned_start():
    with pytest.raises(ValueError):
        tl.ChunkSchedule(start_tick=5, start_wall=0.0)
    with pytest.raises(ValueError):
        tl.ChunkSchedule(start_tick=-160, start_wall=0.0)


def test_nearest_picks_the_closest_entry_within_tolerance_and_earlier_on_ties():
    buf = [(1.00, "a"), (1.07, "b"), (1.14, "c"), (1.21, "d")]
    assert tl.nearest(buf, 1.08, 0.040) == (1.07, "b")
    assert tl.nearest(buf, 1.105, 0.040) == (1.07, "b")        # exact tie: the earlier one
    assert tl.nearest(buf, 1.30, 0.040) is None                  # 90 ms away
    assert tl.nearest([], 1.0, 0.1) is None
    assert tl.nearest(buf, 1.00, 0.0) == (1.00, "a")             # zero tolerance still matches exact


def test_nearest_rejects_bad_tolerance():
    with pytest.raises(ValueError):
        tl.nearest([(0.0, 1)], 0.0, -0.1)
