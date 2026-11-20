"""Contact-aided Kalman filter for base velocity and height.

The estimator RAMBO describes is a simple Kalman filter on the binary foot
contacts and forward kinematics, assuming a contact point does not move. RAMBO
ships no reference implementation, so this is a standard contact-aided design:

State  x = [v_w (3), p_z (1)]  (world-frame base velocity + base height)
Predict (IMU):  v'  = v + dt * (R_wb @ acc_imu + g),  g = (0,0,-9.81)
                p_z'= p_z + dt * v_z
Update (per foot in contact, foot assumed stationary in the world):
  0 = v_foot_w = v_w + R_wb @ (omega_b x p_f_b + J_leg @ dq_leg)
  -> measurement  v_w = -R_wb @ (omega_b x p_f_b + J_leg @ dq_leg)
  height          p_z = -(R_wb @ p_f_b)_z + foot_radius

Outputs match the RAMBO observation definitions (qp_env.py:1178-1222):
base_lin_vel_b = R_bw @ v_w, base_height = p_z.

Noise defaults are tuned so the estimate comfortably beats the training-time
observation noise (lin vel +-0.1 m/s, height +-0.02 m, qp_env.py:1154-1163).
"""

import time

import numpy as np

from ..constants import FOOT_RADIUS
from ..control import kinematics
from ..control.math_utils import quat_rotate, quat_rotate_inverse


class StateEstimator:
    def __init__(self,
                 sigma_accel=0.6,        # process noise from IMU accel [m/s^2]
                 sigma_height_process=0.01,
                 sigma_vel_meas=0.05,    # foot-stationary velocity meas noise [m/s]
                 sigma_height_meas=0.02,
                 nominal_height=0.3):
        self._sigma_accel = sigma_accel
        self._sigma_hp = sigma_height_process
        self._sigma_vm = sigma_vel_meas
        self._sigma_hm = sigma_height_meas
        self._nominal_height = nominal_height
        self.reset()

    def reset(self):
        self.valid = False
        self.consecutive_failures = 0
        self.last_success_time = None
        self._reset_filter()

    def _reset_filter(self):
        self.x = np.zeros(4)
        self.x[3] = self._nominal_height
        self.P = np.diag([0.1, 0.1, 0.1, 0.05])
        self._initialized = False

    # ------------------------------------------------------------------
    def update(self, dt, base_quat, gyro_b, accel_b, joint_pos, joint_vel,
               contact_state, now=None):
        """One predict + update cycle.

        Args:
            dt: time since the previous call [s]
            base_quat: IMU quaternion (w,x,y,z)
            gyro_b: (3,) body angular velocity [rad/s]
            accel_b: (3,) IMU specific force in body frame [m/s^2]
                     (includes gravity reaction: ~+9.81 z when standing)
            joint_pos/joint_vel: (12,) RAMBO order
            contact_state: (4,) bool, feet [FL, FR, RL, RR]
        """
        now = time.monotonic() if now is None else float(now)
        self.valid = False
        self.consecutive_failures += 1
        if not np.isfinite(dt) or dt <= 0.0:
            return False
        dt = float(np.clip(dt, 1e-4, 0.05))

        # ---- input validation: one NaN would poison the filter state forever
        if not (np.all(np.isfinite(base_quat)) and np.all(np.isfinite(gyro_b))
                and np.all(np.isfinite(accel_b)) and np.all(np.isfinite(joint_pos))
                and np.all(np.isfinite(joint_vel))):
            return False  # retain diagnostics, but never refresh health
        if not (np.all(np.isfinite(self.x)) and np.all(np.isfinite(self.P))):
            self._reset_filter()
            return False

        # ---- predict
        a_w = quat_rotate(base_quat, accel_b) + np.array([0.0, 0.0, -9.81])
        F = np.eye(4)
        F[3, 2] = dt
        self.x[0:3] += a_w * dt
        self.x[3] += self.x[2] * dt
        Qv = (self._sigma_accel * dt) ** 2
        Qn = np.diag([Qv, Qv, Qv, (self._sigma_hp * dt) ** 2 + Qv * dt ** 2])
        self.P = F @ self.P @ F.T + Qn

        # ---- measurements from stationary contact feet
        feet_b = kinematics.foot_positions_b(joint_pos)
        jac = kinematics.foot_jacobian(joint_pos)
        contacts = np.flatnonzero(np.asarray(contact_state, dtype=bool))
        if not self._initialized and len(contacts) > 0:
            # initialize height directly from kinematics
            heights = [-(quat_rotate(base_quat, feet_b[i]))[2] + FOOT_RADIUS
                       for i in contacts]
            self.x[3] = float(np.mean(heights))
            self._initialized = True

        for i in contacts:
            q_dot_leg = joint_vel[[i, 4 + i, 8 + i]]
            J_leg = jac[3 * i:3 * i + 3, [i, 4 + i, 8 + i]]
            foot_vel_b = np.cross(gyro_b, feet_b[i]) + J_leg @ q_dot_leg
            v_meas = -quat_rotate(base_quat, foot_vel_b)
            z_meas = -(quat_rotate(base_quat, feet_b[i]))[2] + FOOT_RADIUS

            # velocity measurement: H = [I3, 0]
            H = np.zeros((3, 4))
            H[:, 0:3] = np.eye(3)
            R = np.eye(3) * self._sigma_vm ** 2
            self._kf_update(H, v_meas - self.x[0:3], R)

            # height measurement: H = [0,0,0,1]
            H = np.zeros((1, 4))
            H[0, 3] = 1.0
            R = np.array([[self._sigma_hm ** 2]])
            self._kf_update(H, np.array([z_meas - self.x[3]]), R)

        # with no contact feet the filter free-integrates the IMU (prediction only)

        # ---- divergence guard: recover instead of latching a broken state
        if not (np.all(np.isfinite(self.x)) and np.all(np.isfinite(self.P))):
            self._reset_filter()
            return False
        self.valid = True
        self.consecutive_failures = 0
        self.last_success_time = now
        return True

    def age(self, now=None):
        """Monotonic age of the last successful update; reset is not success."""
        if self.last_success_time is None:
            return float("inf")
        return max(0.0, (time.monotonic() if now is None else now) - self.last_success_time)

    def _kf_update(self, H, innovation, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ H) @ self.P

    # ------------------------------------------------------------------
    @property
    def base_height(self):
        return float(self.x[3])

    def base_lin_vel_b(self, base_quat):
        return quat_rotate_inverse(base_quat, self.x[0:3])
