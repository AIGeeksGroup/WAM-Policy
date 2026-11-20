"""Heading hold for the quadruped walk: keep the yaw the robot had when the
straight walk started.

Why: with FL held up as the manipulation leg the three-leg gait yaws to the
right on its own - real robot -0.7 to -2.0 deg/s while walking at
0.1-0.15 m/s (MuJoCo: -1.5 to -2.4 deg/s), i.e. 25 deg off after 20 s. The
policy takes a yaw-rate command (vel_cmd[2], trained range +-0.5 rad/s), so a
PI loop on the IMU yaw closes the drift without touching the policy:
whenever the operator commands forward speed with (near) zero yaw rate, the
yaw at that moment is latched as the reference and
omega_z = Kp*err + Ki*integral(err) is sent instead of zero. Operator yaw
input (/cmd_vel angular.z) releases the hold at once; stopping releases it
too.

IN PLACE (in_place=True): the same turn happens with a ZERO command - the
three-leg gait keeps stepping and yaws right (real robot about -1.0 deg/s
over 22-28 s of a WAM run, MuJoCo -1.14 deg/s at the WAM entry target, FL at
0.1934, 0.142, 0.05). A WAM run spends most of its time like this, stepping
while the policy server predicts the next chunk (~3 s per 0.32 s of
actions), and in the simulation that produced its training data the physics
is frozen then. So with in_place the yaw is also latched while the whole
command is ~zero (|vx|, |vy| < vx_min, |wz| < operator_wz_min) and held with
the same PI: MuJoCo, 25 s at the WAM entry target, -28.4 -> +1.6 deg, |wz| <=
0.06 rad/s, posture unchanged. Any real command releases it, so commanded
motion is untouched. Only the yaw: the estimator's velocity is not good
enough to hold the position (MuJoCo: its integral is 0.16 m off the true
displacement after 25 s), so a position hold would chase a phantom error.
The walking hold and the in-place hold are switched separately (enabled /
in_place): WAM runs keep the walking hold OFF (training had none) and the
in-place hold ON. A switch between the two modes re-latches the reference.

Pure Python, no ROS. Yaw is taken from the IMU quaternion (w, x, y, z):
the Go2 fuses it on board and it is drift-free over the seconds a demo
lasts.
"""

import math


def yaw_from_quat(quat_wxyz):
    w, x, y, z = (float(v) for v in quat_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


class HeadingHold:
    def __init__(self, enabled=True, kp=1.0, ki=0.3, max_wz=0.3, deadband_deg=1.0,
                 vx_min=0.02, operator_wz_min=0.02, dt=0.01, in_place=False):
        if kp < 0.0 or ki < 0.0 or max_wz < 0.0:
            raise ValueError("heading hold gains and limit must be >= 0")
        self.enabled = bool(enabled)      # the walking hold (straight walk)
        self.in_place = bool(in_place)    # the in-place hold (zero command)
        self.kp = float(kp)
        self.ki = float(ki)
        self.max_wz = float(max_wz)
        self.deadband = math.radians(float(deadband_deg))
        self.vx_min = float(vx_min)
        self.operator_wz_min = float(operator_wz_min)
        self.dt = float(dt)
        self.reset()

    def reset(self):
        self.yaw_ref = None
        self.integral = 0.0
        self.last_wz = 0.0
        self.last_error = 0.0
        self.mode = None           # "walking" / "in_place" while active
        self.in_place_integral = 0.0   # carried from one in-place pause to the next (see update)
        self.event = None          # "latched" / "released" on the tick it happened, else None

    @property
    def active(self):
        return self.yaw_ref is not None

    def update(self, vel_cmd, yaw):
        """vel_cmd = [vx, vy, wz] as commanded by the operator; returns the
        command to give the controller (a new list)."""
        vx, vy, wz = (float(v) for v in vel_cmd)
        self.event = None
        walking = abs(vx) >= self.vx_min
        steering = abs(wz) >= self.operator_wz_min
        still = not walking and not steering and abs(vy) < self.vx_min
        mode = ("walking" if self.enabled and walking and not steering
                else "in_place" if self.in_place and still else None)
        if mode != self.mode:
            if self.yaw_ref is not None:
                self.event = "released"
            if self.mode == "in_place":
                # the gait's turn is a steady bias: keep the integral that cancels it for the next
                # pause instead of rebuilding it (with 3 s pauses that cost ~2 deg each, MuJoCo)
                self.in_place_integral = self.integral
            self.yaw_ref = None
            self.mode = None
            self.integral = 0.0
            self.last_wz = wz
            self.last_error = 0.0
        if mode is None:
            self.last_wz = wz
            return [vx, vy, wz]
        if self.yaw_ref is None:
            self.yaw_ref = float(yaw)
            self.mode = mode
            self.integral = self.in_place_integral if mode == "in_place" else 0.0
            self.event = "latched"
        err = wrap_angle(self.yaw_ref - float(yaw))
        self.last_error = err
        if abs(err) < self.deadband:
            err_p = 0.0
        else:
            err_p = err - math.copysign(self.deadband, err)
        # integral with anti-windup: only accumulate while not saturated
        cand = self.integral + self.ki * err * self.dt
        out = self.kp * err_p + cand
        if abs(out) <= self.max_wz:
            self.integral = cand
        else:
            out = self.kp * err_p + self.integral
        out = max(-self.max_wz, min(self.max_wz, out))
        self.last_wz = out
        return [vx, vy, wz + out if mode == "in_place" else out]
