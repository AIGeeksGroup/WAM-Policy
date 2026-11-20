"""Bounds on predicted native9 commands: refuse, never clamp.

The acknowledgement sent back to the model states what was executed; the
protocol's transform is identity apart from the zero force. A prediction
outside the configured envelope therefore ends the episode (the robot holds)
instead of being altered on the way to the controller.

QLM-Bench MuJoCo, which produced the training data, never clipped a command
(RAMBOCore.submit). WAM runs therefore launch the controller with
command_handling:=sim - velocity and FL target executed unclipped. The
defaults here are 1.5x RAMBO's training ranges and contain every training
value (964,151 rows; closest: fl_ee_z min -0.034 vs -0.05): they only catch
implausible predictions. Tighten them per task with the node parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping, Sequence, Tuple

import numpy as np

ACTION_NAMES = ("base_vx", "base_vy", "base_yaw_rate", "fl_ee_x", "fl_ee_y", "fl_ee_z",
                "fl_ee_fx", "fl_ee_fy", "fl_ee_fz")


@dataclass(frozen=True)
class Bounds:
    vx: float = 0.75                          # |base_vx| [m/s]
    vy: float = 0.75                          # |base_vy| [m/s]
    wz: float = 0.75                          # |base_yaw_rate| [rad/s]
    ee_x: Tuple[float, float] = (0.0, 0.65)   # fl_ee_x [m], projected-CoM frame
    ee_y: Tuple[float, float] = (-0.15, 0.35)
    ee_z: Tuple[float, float] = (-0.05, 0.50)

    def __post_init__(self):
        for name in ("vx", "vy", "wz"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0.0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        for name in ("ee_x", "ee_y", "ee_z"):
            lo, hi = getattr(self, name)
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError(f"{name} must be (low, high) with low < high")
            object.__setattr__(self, name, (float(lo), float(hi)))

    @classmethod
    def from_mapping(cls, values: Mapping) -> "Bounds":
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown bound(s): {sorted(unknown)}")
        kw = {}
        for key, value in values.items():
            kw[key] = tuple(value) if key.startswith("ee_") else float(value)
        return cls(**kw)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def violations(action: Sequence[float], bounds: Bounds) -> list:
    """Human-readable violations of one native9 command (empty = acceptable)."""
    a = np.asarray(action, dtype=np.float64)
    if a.shape != (9,):
        raise ValueError("a native9 command has exactly nine values")
    out = []
    for i, name in enumerate(ACTION_NAMES):
        if not np.isfinite(a[i]):
            out.append(f"{name} not finite ({a[i]})")
    if out:
        return out
    for i, (name, limit) in enumerate(zip(ACTION_NAMES[:3], (bounds.vx, bounds.vy, bounds.wz))):
        if abs(a[i]) > limit:
            out.append(f"{name} {a[i]:+.4f} outside +-{limit:g}")
    for i, (name, (lo, hi)) in enumerate(zip(ACTION_NAMES[3:6], (bounds.ee_x, bounds.ee_y, bounds.ee_z)), 3):
        if not lo <= a[i] <= hi:
            out.append(f"{name} {a[i]:+.4f} outside [{lo:g}, {hi:g}]")
    if np.any(a[6:] != 0.0):
        out.append(f"desired force not zero: {a[6:].tolist()}")
    return out


def first_violation(chunk: Sequence[Sequence[float]], bounds: Bounds):
    """(index, problems) of the first unacceptable command of a chunk, or (None, [])."""
    for index, action in enumerate(chunk):
        problems = violations(action, bounds)
        if problems:
            return index, problems
    return None, []
