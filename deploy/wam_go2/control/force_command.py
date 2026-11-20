"""Staleness watchdog for the FL end-effector force command, and the box-push phases.

The force part of the native9 command (fl_ee_fx, fl_ee_fy, fl_ee_fz; WAM-Policy
sends zeros) reaches the controller on rambo/ee_force_cmd. A publisher that dies
must not leave its last force applied (30 N on a 15 kg robot is 2 m/s^2 of
unbalanced thrust), so a nonzero request older than `command_timeout` is replaced
by zeros. After such a timeout the force stays zero until the publisher withdraws
it (sends zeros) or the robot stops (zero planar velocity command); a publisher
that merely comes back cannot restore a force on its own. 0 disables the watchdog.

These are the command-watchdog semantics of the development tree's push pipeline
with its contact gate off (the default). This release ships no foot contact-force
estimate and therefore no contact gate. The node and the MuJoCo harness both call
`ForceCommandWatchdog.step` at 100 Hz with the same inputs.
"""
from dataclasses import dataclass
import math
import time

import numpy as np


class ForceCommandWatchdog:
    def __init__(self, command_timeout=0.5):
        self.command_timeout = float(command_timeout)
        if not math.isfinite(self.command_timeout) or self.command_timeout < 0.0:
            raise ValueError("force command timeout must be finite and nonnegative")
        self.reset()

    def reset(self):
        self._withdraw_required = False
        self.command_expired = False

    def step(self, request, moving, now=None, command_time=None):
        """The force to apply this tick: `request` [N] (3,), or zeros when stale."""
        now = time.monotonic() if now is None else float(now)
        req = np.asarray(request, dtype=float)
        if req.shape != (3,) or not np.isfinite(req).all():
            raise ValueError("force request must contain three finite values")
        self.command_expired = bool(np.any(req) and self.command_timeout > 0.0 and (
            command_time is None or not math.isfinite(command_time)
            or now - command_time > self.command_timeout))
        # A stopped robot or a withdrawn request clears the requirement; an
        # expired request (re)sets it, so recovering from silence needs a withdrawal.
        if not (moving and np.any(req)):
            self._withdraw_required = False
        if self.command_expired:
            self._withdraw_required = True
        return np.zeros(3) if self._withdraw_required else req.copy()


@dataclass(frozen=True)
class PushMotionPhase:
    seconds: float
    velocity: tuple
    force: tuple


def push_motion_phases(vx, force, duration, vy=0.0, wz=0.0):
    """Moving push, then simultaneous zero velocity/force and 0.5 s hold.

    Duration <= 0 is open-ended and has no scheduled stop; both callers stop
    publishing force when the controller exits POLICY.
    """
    walk = PushMotionPhase(duration if duration > 0.0 else math.inf,
                           (float(vx), float(vy), float(wz)),
                           (float(force), 0.0, 0.0))
    if duration <= 0.0:
        return (walk,)
    return (walk, PushMotionPhase(0.5, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
