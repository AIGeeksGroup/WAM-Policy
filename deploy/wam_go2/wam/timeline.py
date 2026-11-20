"""Logical episode clock and chunk schedule of the qlm-live-v1 protocol.

The protocol counts 2 ms physics ticks (QLM timing profile
``sim-12p5rgb-50action-100control-500physics-v2``): a controller step is 5 ticks
(10 ms), a native9 command 10 ticks (20 ms, held), a feedback sample every
40 ticks (80 ms) and a chunk 16 commands = 160 ticks. In simulation the ticks
ARE the physics clock and freeze while the policy computes. On the robot they
are a logical clock that advances only while a command is being held
(``LogicalClock.advance_steps``); wall time is a separate axis that keeps
running, which ``ChunkSchedule`` maps tick by tick onto a start instant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Tuple

PHYSICS_STEP_NS = 2_000_000          # one tick
STEP_TICKS = 5                       # one 100 Hz controller step
COMMAND_TICKS = 10                   # one native9 command (held)
SAMPLE_TICKS = 40                    # one 12.5 Hz RGB/state sample
GROUP_COMMANDS = 16                  # commands per prediction (one chunk)
GROUP_TICKS = GROUP_COMMANDS * COMMAND_TICKS
STEP_S = STEP_TICKS * PHYSICS_STEP_NS / 1e9
COMMAND_S = COMMAND_TICKS * PHYSICS_STEP_NS / 1e9
SAMPLE_S = SAMPLE_TICKS * PHYSICS_STEP_NS / 1e9
_TIE_DIGITS = 9                      # nearest(): |dt| equal to a nanosecond is a tie


def ticks_to_s(ticks: int) -> float:
    return ticks * PHYSICS_STEP_NS / 1e9


class LogicalClock:
    """Episode-local tick counter; ``sim_time_ns`` is what the packets carry."""

    def __init__(self, origin_ns: int = 0):
        if type(origin_ns) is not int or origin_ns < 0:
            raise ValueError("origin_ns must be a nonnegative int")
        self.origin_ns = origin_ns
        self.tick = 0

    def sim_time_ns(self, tick: Optional[int] = None) -> int:
        tick = self.tick if tick is None else int(tick)
        return self.origin_ns + tick * PHYSICS_STEP_NS

    def advance_steps(self, steps: int) -> int:
        """One held controller step = STEP_TICKS; returns the new tick."""
        if type(steps) is not int or steps < 1:
            raise ValueError("advance_steps needs a positive number of 10 ms steps")
        self.tick += steps * STEP_TICKS
        return self.tick


@dataclass(frozen=True)
class StepSlot:
    """One 10 ms controller step of a chunk: ticks and the wall instants it covers."""
    index: int
    command_index: int
    start_tick: int
    end_tick: int
    start_wall: float
    end_wall: float


class ChunkSchedule:
    """Wall-clock schedule of one chunk starting at ``start_tick`` / ``start_wall``.

    Command i is held over [start_wall + i*20 ms, +20 ms); the feedback samples
    are at 80/160/240/320 ms after the boundary (ticks start+40k).
    """

    def __init__(self, start_tick: int, start_wall: float, n_commands: int = GROUP_COMMANDS):
        if type(start_tick) is not int or start_tick < 0 or start_tick % COMMAND_TICKS:
            raise ValueError("chunk start must be a nonnegative tick on a command boundary")
        if type(n_commands) is not int or n_commands < 1 or n_commands % GROUP_COMMANDS:
            raise ValueError("a chunk holds a positive multiple of 16 commands")
        self.start_tick = start_tick
        self.start_wall = float(start_wall)
        self.n_commands = n_commands

    # ticks <-> wall ------------------------------------------------------
    def wall_of_tick(self, tick: int) -> float:
        return self.start_wall + ticks_to_s(tick - self.start_tick)

    def command_tick(self, i: int) -> int:
        self._check_index(i)
        return self.start_tick + i * COMMAND_TICKS

    def command_wall(self, i: int) -> float:
        return self.wall_of_tick(self.command_tick(i))

    @property
    def end_tick(self) -> int:
        return self.start_tick + self.n_commands * COMMAND_TICKS

    @property
    def end_wall(self) -> float:
        return self.wall_of_tick(self.end_tick)

    def steps(self) -> Iterator[StepSlot]:
        per_command = COMMAND_TICKS // STEP_TICKS
        for j in range(self.n_commands * per_command):
            start = self.start_tick + j * STEP_TICKS
            yield StepSlot(index=j, command_index=j // per_command, start_tick=start,
                           end_tick=start + STEP_TICKS, start_wall=self.wall_of_tick(start),
                           end_wall=self.wall_of_tick(start + STEP_TICKS))

    @property
    def sample_ticks(self) -> list:
        return [self.start_tick + k * SAMPLE_TICKS
                for k in range(1, self.n_commands * COMMAND_TICKS // SAMPLE_TICKS + 1)]

    @property
    def sample_walls(self) -> list:
        return [self.wall_of_tick(t) for t in self.sample_ticks]

    def sample_wall_of_tick(self, tick: int) -> float:
        if tick not in self.sample_ticks:
            raise ValueError(f"tick {tick} is not a feedback sample of this chunk")
        return self.wall_of_tick(tick)

    def _check_index(self, i: int) -> None:
        if type(i) is not int or not 0 <= i < self.n_commands:
            raise ValueError(f"command index {i} outside 0..{self.n_commands - 1}")


def nearest(buffer: Sequence[Tuple[float, object]], t: float, tolerance: float):
    """The (time, payload) entry closest to ``t`` within ``tolerance`` seconds.

    Ties (to a nanosecond) go to the EARLIER entry; None when nothing qualifies.
    """
    if not (tolerance >= 0.0):
        raise ValueError("tolerance must be >= 0")
    best = None
    for entry in buffer:
        key = (round(abs(entry[0] - t), _TIE_DIGITS), entry[0])
        if best is None or key < best[0]:
            best = (key, entry)
    if best is None or abs(best[1][0] - t) > tolerance + 1e-12:
        return None
    return best[1]
