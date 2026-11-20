"""The controller's state for consumers (the WAM bridge) as one flat float array.

Published by the controller node on ``rambo/state`` (std_msgs/Float64MultiArray)
once per 100 Hz control tick after the first LowState, in every mode. It carries
the quantities the controller already computes for RAMBO's own observation
(estimator height and body-frame velocity, IMU quaternion and gyro) plus what a
consumer needs to judge them: mode, safety latch, LowState freshness, estimator
validity, the LowState age at the tick and the control gap. The last 18 values
are the policy action of that tick (zeros outside POLICY), kept for the
protocol's control-step records.

A flat array with a version word keeps the controller free of new message
packages; encoder and decoder live here so both sides agree by test.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence

from .wam.observation import StateRecord

VERSION = 1.0
MODES = ("DAMP", "STAND", "POLICY")
TOPIC = "rambo/state"
# index layout (see encode)
_I_VERSION, _I_WALL, _I_TICK, _I_MODE, _I_TRIPPED, _I_FRESH, _I_EST_VALID, _I_HEIGHT = range(8)
_I_QUAT, _I_LIN, _I_ANG = 8, 12, 15
_I_SOURCE_AGE, _I_CONTROL_GAP, _I_ACTION = 18, 19, 20
LENGTH = _I_ACTION + 18


def _finite(values: Sequence[float], n: int, name: str) -> List[float]:
    out = [float(v) for v in values]
    if len(out) != n or not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name}: {n} finite values required")
    return out


def encode(*, wall: float, tick_ms: int, mode: str, tripped: bool, fresh: bool, estimator_valid: bool,
           base_height: float, quat_wxyz: Sequence[float], lin_vel_b: Sequence[float], ang_vel_b: Sequence[float],
           source_age: float, control_gap: float, action18: Optional[Sequence[float]] = None) -> List[float]:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    values = [VERSION] + _finite([wall], 1, "wall") + [float(int(tick_ms)), float(MODES.index(mode)),
                                                        float(bool(tripped)), float(bool(fresh)),
                                                        float(bool(estimator_valid))]
    values += _finite([base_height], 1, "base_height")
    values += _finite(quat_wxyz, 4, "quat_wxyz") + _finite(lin_vel_b, 3, "lin_vel_b") + _finite(ang_vel_b, 3, "ang_vel_b")
    values += _finite([source_age, control_gap], 2, "source_age/control_gap")
    values += [0.0] * 18 if action18 is None else _finite(action18, 18, "action18")
    assert len(values) == LENGTH
    return values


@dataclass
class StateSample:
    wall: float
    tick_ms: int
    mode: str
    tripped: bool
    fresh: bool
    estimator_valid: bool
    base_height: float
    quat_wxyz: List[float]
    lin_vel_b: List[float]
    ang_vel_b: List[float]
    source_age: float
    control_gap: float
    action18: List[float]

    @property
    def state_time(self) -> float:
        """Host wall time of the LowState this state was computed from (tick time minus its age)."""
        return self.wall - self.source_age

    def state_record(self) -> StateRecord:
        mode = self.mode if not self.tripped else f"{self.mode} (safety)"
        return StateRecord(time=self.state_time, height=self.base_height, quat_wxyz=tuple(self.quat_wxyz),
                           lin_vel_b=tuple(self.lin_vel_b), ang_vel_b=tuple(self.ang_vel_b), mode=mode,
                           fresh=self.fresh, action18=list(self.action18))


def decode(values: Sequence[float]) -> StateSample:
    v = [float(x) for x in values]
    if len(v) != LENGTH:
        raise ValueError(f"rambo/state has {len(v)} values, expected {LENGTH}")
    if v[_I_VERSION] != VERSION:
        raise ValueError(f"rambo/state layout version {v[_I_VERSION]} is not {VERSION}")
    code = int(v[_I_MODE])
    if not 0 <= code < len(MODES):
        raise ValueError(f"rambo/state mode code {code} unknown")
    return StateSample(wall=v[_I_WALL], tick_ms=int(v[_I_TICK]), mode=MODES[code], tripped=bool(v[_I_TRIPPED]),
                       fresh=bool(v[_I_FRESH]), estimator_valid=bool(v[_I_EST_VALID]), base_height=v[_I_HEIGHT],
                       quat_wxyz=v[_I_QUAT:_I_QUAT + 4], lin_vel_b=v[_I_LIN:_I_LIN + 3], ang_vel_b=v[_I_ANG:_I_ANG + 3],
                       source_age=v[_I_SOURCE_AGE], control_gap=v[_I_CONTROL_GAP], action18=v[_I_ACTION:_I_ACTION + 18])
