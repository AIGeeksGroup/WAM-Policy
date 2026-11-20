"""Basic runtime safety checks (deliberately minimal: essential protection
only).

Checks:
- LowState freshness (communication watchdog)
- tip-over via projected gravity z (official example: projected_gravity_z > 0
  means flipped; we trip earlier at a configurable threshold)
- non-finite sensor values
- optional bounds on the ESTIMATED base speed / height (KF divergence)

Trip -> the node switches to DAMPING and stays there until re-armed.
"""

import numpy as np


class SafetyManager:
    def __init__(self, lowstate_timeout=0.1, max_tilt_gravity_z=-0.3,
                 max_base_speed=1.5, base_height_range=(0.02, 0.6),
                 source_stale_timeout=0.05, control_gap_timeout=0.05,
                 estimator_timeout=0.05):
        """max_tilt_gravity_z: trip when projected_gravity_b[2] rises above
        this value (level is -1.0, flat on its side is 0.0). The default
        -0.3 corresponds to ~72 deg of tilt; the training termination used a
        gravity-error norm of 0.75 (~ 45 deg).

        max_base_speed / base_height_range bound the ESTIMATED base state:
        the KF free-integrates when contact is lost (measured on real data:
        10 m/s within 20 s) while staying finite, so an isfinite check alone
        never fires. Training commands top out at 0.5 m/s; 1.5 m/s estimated
        means the estimate is broken, not the robot sprinting."""
        for name, value in (("source_stale_timeout", source_stale_timeout),
                            ("control_gap_timeout", control_gap_timeout),
                            ("estimator_timeout", estimator_timeout)):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, float(value))
        self.lowstate_timeout = lowstate_timeout
        self.max_tilt_gravity_z = max_tilt_gravity_z
        self.max_base_speed = max_base_speed
        self.base_height_range = base_height_range
        self.tripped = False
        self.reason = ""

    def check(self, lowstate_age, projected_gravity_b, joint_pos, joint_vel,
              base_lin_vel_b=None, base_height=None,
              base_ang_vel_b=None, active=False, control_gap=0.0,
              wired_ok=True, source_fault=None, estimator_age=None):
        if self.tripped:
            return False
        if source_fault:
            return self._trip(f"LowState source fault: {source_fault}")
        if active:
            if not np.isfinite(control_gap) or control_gap >= self.control_gap_timeout:
                return self._trip(f"control loop stalled ({control_gap:.3f}s)")
            if wired_ok is not True:
                return self._trip("wired link lost or unverified")
            if not np.isfinite(lowstate_age) or lowstate_age >= self.source_stale_timeout:
                return self._trip(f"LowState source stale ({lowstate_age:.3f}s)")
        if estimator_age is not None and (not np.isfinite(estimator_age)
                                          or estimator_age >= self.estimator_timeout):
            return self._trip(f"estimator has no successful update for {estimator_age:.3f}s")
        if lowstate_age > self.lowstate_timeout:
            return self._trip(f"LowState timeout ({lowstate_age:.3f}s)")
        if not (np.all(np.isfinite(joint_pos)) and np.all(np.isfinite(joint_vel))
                and np.all(np.isfinite(projected_gravity_b))):
            return self._trip("non-finite sensor value")
        if base_ang_vel_b is not None and not np.all(np.isfinite(base_ang_vel_b)):
            return self._trip("non-finite angular velocity")
        g = np.asarray(projected_gravity_b, dtype=float)
        if g[2] > self.max_tilt_gravity_z:
            return self._trip(f"tilt limit (gravity_z {g[2]:.2f})")
        if base_lin_vel_b is not None:
            speed = float(np.linalg.norm(base_lin_vel_b))
            if not np.isfinite(speed) or speed > self.max_base_speed:
                return self._trip(f"estimated base speed {speed:.2f} m/s")
        if base_height is not None:
            lo, hi = self.base_height_range
            if not np.isfinite(base_height) or not lo <= base_height <= hi:
                return self._trip(f"estimated base height {base_height:.3f} m")
        return True

    def _trip(self, reason):
        self.tripped = True
        self.reason = reason
        return False

    def rearm(self):
        self.tripped = False
        self.reason = ""
