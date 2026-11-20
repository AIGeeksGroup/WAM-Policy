"""Centroidal QP for ground reaction forces, solved with OSQP.

Real-time implementation of the ground-reaction-force QP the RAMBO policy was
trained with (training environment: rambo_quadruped/modules/qp_torque_optimizer.py,
solve_grf_qpth + compute_desired_acc + construct_mass_mat), solved with OSQP
instead of qpth. The formulation is numerically identical to the training-time
QP:

  min_f 0.5 ||M f + g_b - a_des||^2_Q + 0.5*Wfe sum_ee ||f_ee - f_ee_des||^2
        + 0.5*Wf ||f_rest||^2
  s.t.  normal-force bounds on the non-ee legs (BODY-frame z, not rotated -
        upstream quirk kept on purpose: the policy was trained with exactly
        that), friction pyramid (mu=0.6) on the non-ee legs in the
        gravity-aligned frame.
  The manipulation leg (task.ee_legs: FL) is unconstrained (force-tracking
  end effector).

Decision variable f: 12 GRFs in the body frame, feet [FL, FR, RL, RR].
Mapped to OSQP standard form: min 0.5 x'Px + q'x s.t. l <= Ax <= u.
The upstream 6 one-sided rows per constrained leg are re-encoded as 5 rows
with an identical feasible set: the per-leg normal-force pair (+fz <= u,
-fz <= -l) becomes one two-sided row l <= fz <= u. This matters for
convergence: the swing-leg |fz| <= 1e-3 pair is a near-equality that OSQP's
ADMM converges to slowly as two one-sided rows (~15% max-iter failures
measured on the walking gait) but handles quickly as a two-sided bound.
"""

import time

import numpy as np
import osqp
import scipy.sparse as sparse

from ..constants import (
    QP_ACC_CLIP_ANG,
    QP_ACC_CLIP_LIN,
    QP_BASE_ORIENTATION_KD,
    QP_BASE_ORIENTATION_KP,
    QP_BASE_POSITION_KD,
    QP_BASE_POSITION_KP,
    QP_BODY_INERTIA,
    QP_BODY_MASS,
    QP_EPS,
    QP_FRICTION_COEF,
    QP_FZ_MAX,
    QP_FZ_MIN,
    QP_GRAVITY,
    QP_WEIGHT_DDQ,
    QP_WEIGHT_EE_FORCE,
    QP_WEIGHT_GRF,
)
from ..tasks import QUADRUPED
from .math_utils import quat_error


def compute_desired_acc(base_pos, base_quat_rp, base_lin_vel, base_ang_vel,
                        desired_base_pos, desired_base_quat,
                        desired_lin_vel, desired_ang_vel,
                        desired_lin_acc, desired_ang_acc):
    """PD on base pose/velocity + policy feedforward (projected com frame)."""
    lin_pos_error = desired_base_pos - base_pos
    lin_vel_error = desired_lin_vel - base_lin_vel
    desired_lin = (QP_BASE_POSITION_KP * lin_pos_error
                   + QP_BASE_POSITION_KD * lin_vel_error
                   + desired_lin_acc)

    ang_pos_error = quat_error(desired_base_quat, base_quat_rp)
    ang_vel_error = desired_ang_vel - base_ang_vel
    desired_ang = (QP_BASE_ORIENTATION_KP * ang_pos_error
                   + QP_BASE_ORIENTATION_KD * ang_vel_error
                   + desired_ang_acc)
    return np.concatenate([desired_lin, desired_ang])


def construct_mass_mat(foot_positions_b):
    """Single-rigid-body inverse dynamics matrix M (6x12)."""
    inv_mass = np.eye(3) / QP_BODY_MASS
    inv_inertia = np.linalg.inv(QP_BODY_INERTIA)
    mass_mat = np.zeros((6, 12))
    for i in range(4):
        mass_mat[:3, 3 * i:3 * i + 3] = inv_mass
        x, y, z = foot_positions_b[i]
        skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        mass_mat[3:6, 3 * i:3 * i + 3] = inv_inertia @ skew
    return mass_mat


class QPSolveError(RuntimeError):
    """The current contact problem has no verified solution; stop POLICY."""


class QPTorqueOptimizer:
    """One bounded cold solve per cycle; failed solves never reuse old forces.

    There is no verified physical tolerance for a stale solution, especially
    across contact changes. The first failure raises QPSolveError so the node
    can latch damping. Health and the last valid force remain diagnostic only.
    """

    # Two different limits, on purpose. SOLVE_TIME_LIMIT is handed to OSQP and
    # bounds the solver's own work: a problem that needs more than 3 ms of ADMM
    # is not converging and its result is not used. WALL_TIME_LIMIT is the
    # wall-clock check on the whole solve call; it must sit at the control
    # period, not at the solver budget, because a wall-clock overrun with a
    # converged, verified solution is a HOST scheduling hiccup, and those are
    # the job of the node's control_gap_timeout (50 ms), not a reason to drop
    # a walking robot into damping. Calibration: five real-robot POLICY
    # traces replayed through this solver, 8320 ticks, cold solves: median 0.60 ms, p99 0.88, p99.9 1.12, max 1.51 ms,
    # zero failures. 8 ms is >5x the worst observed tick and still under the
    # 10 ms control period, so an overrun really does mean the tick is late.
    SOLVE_TIME_LIMIT = 0.003  # OSQP time_limit [s]: solver work budget
    WALL_TIME_LIMIT = 0.008   # whole-call wall clock [s]: late-tick guard
    CONSTRAINT_TOLERANCE = 0.01  # N; numerical feasibility tolerance only

    def __init__(self, task=None):
        self._task = task or QUADRUPED
        self._ee_legs = list(self._task.ee_legs)
        self._constrained_legs = list(self._task.constrained_legs)
        self._Wq = QP_WEIGHT_DDQ
        self.reset()

        # desired setpoints, projected com frame (set each control step)
        self.desired_base_position = np.zeros(3)
        self.desired_base_orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.desired_linear_velocity = np.zeros(3)
        self.desired_angular_velocity = np.zeros(3)
        self.desired_linear_acceleration = np.zeros(3)
        self.desired_angular_acceleration = np.zeros(3)

    def reset(self):
        """Discard the prior POLICY session's forces and solver health."""
        self._last_grf = np.zeros(12)
        self.fallback_count = 0  # kept for diagnostics compatibility; no fallback
        self.solve_count = 0
        self.consecutive_failures = 0
        self.last_success_time = None
        self.solve_duration = 0.0
        self.status = "not_run"
        self.failure_reason = ""

    def get_grf(self, robot_state, ee_force_commands, desired_contact_state):
        """Solve for GRFs; same result as the training QP's get_grf (:66-127).

        Args:
            robot_state: RobotState (finalized).
            ee_force_commands: (3,) for one manipulation leg or (n_ee, 3)
                in task.ee_legs order, projected com frame.
            desired_contact_state: (4,) bool, feet [FL, FR, RL, RR].
        Returns:
            grf (12,) body frame, qp_cost float, desired_acc_b (6,), solved_acc (6,)
        """
        rs = robot_state
        ee_cmds = np.atleast_2d(np.asarray(ee_force_commands, dtype=float))
        if ee_cmds.shape != (self._task.num_ee, 3):
            raise ValueError(
                f"ee_force_commands must be ({self._task.num_ee}, 3) for task "
                f"'{self._task.name}', got {ee_cmds.shape}")

        curr_base_pos = np.array([0.0, 0.0, rs.base_height])
        # body-frame velocities rotated to the projected frame
        curr_lin_vel = rs.base_rot_mat_rp @ rs.base_lin_vel_b
        curr_ang_vel = rs.base_rot_mat_rp @ rs.base_ang_vel_b

        desired_acc = compute_desired_acc(
            curr_base_pos, rs.base_quat_rp, curr_lin_vel, curr_ang_vel,
            self.desired_base_position, self.desired_base_orientation_quat,
            self.desired_linear_velocity, self.desired_angular_velocity,
            self.desired_linear_acceleration, self.desired_angular_acceleration)
        desired_acc = np.clip(
            desired_acc,
            [-QP_ACC_CLIP_LIN] * 3 + [-QP_ACC_CLIP_ANG] * 3,
            [QP_ACC_CLIP_LIN] * 3 + [QP_ACC_CLIP_ANG] * 3)

        # rotate to body frame (QP solved in the body frame)
        desired_acc_b = desired_acc.copy()
        desired_acc_b[:3] = rs.base_rot_mat_rp_t @ desired_acc[:3]
        desired_acc_b[3:] = rs.base_rot_mat_rp_t @ desired_acc[3:]

        desired_ee_force_b = (rs.base_rot_mat_rp_t @ ee_cmds.T).T  # (n_ee, 3)

        mass_mat = construct_mass_mat(rs.ee_pos_b)

        grf, solved_acc, qp_cost = self._solve(
            mass_mat, desired_acc_b, desired_ee_force_b,
            rs.base_rot_mat_rp, desired_contact_state)
        return grf, qp_cost, desired_acc_b, solved_acc

    # ------------------------------------------------------------------
    def _solve(self, mass_mat, desired_acc, desired_ee_force_b,
               base_rot_mat_rp, foot_contact_state):
        """The QP of solve_grf_qpth (:334-411), solved with OSQP."""
        desired_ee_reaction_force = -np.atleast_2d(desired_ee_force_b)

        g = np.zeros(6)
        g[2] = -QP_GRAVITY
        g[:3] = base_rot_mat_rp.T @ g[:3]

        Q = self._Wq
        # quad_term = M'QM + S_ee' R_ee S_ee + S_r' R_r S_r
        P = mass_mat.T @ Q @ mass_mat
        q = mass_mat.T @ Q @ (g - desired_acc)
        for slot, leg in enumerate(self._ee_legs):
            sl = slice(3 * leg, 3 * leg + 3)
            P[sl, sl] += np.eye(3) * QP_WEIGHT_EE_FORCE
            q[sl] -= QP_WEIGHT_EE_FORCE * desired_ee_reaction_force[slot]
        for leg in self._constrained_legs:
            sl = slice(3 * leg, 3 * leg + 3)
            P[sl, sl] += np.eye(3) * QP_WEIGHT_GRF
        P = P + 1e-6 * np.eye(12)  # PSD regularization (qpth used 1e-6*rand)

        # constraints on the non-ee legs only.
        # rows 0..n-1: two-sided normal-force bounds (upstream pairs)
        # rows n..:    friction pyramid, one-sided, 4 per leg
        n = len(self._constrained_legs)
        A = np.zeros((5 * n, 12))
        lo = np.full(5 * n, -np.inf)
        up = np.full(5 * n, QP_EPS)
        for k, leg in enumerate(self._constrained_legs):
            col = 3 * leg
            A[k, col + 2] = 1.0
            if foot_contact_state[leg]:
                lo[k] = QP_FZ_MIN
                up[k] = QP_FZ_MAX
            else:
                lo[k] = -QP_EPS
                up[k] = QP_EPS
            # friction pyramid in the gravity-aligned frame
            row = n + 4 * k
            C = np.array([
                [1.0, 0.0, -QP_FRICTION_COEF],
                [-1.0, 0.0, -QP_FRICTION_COEF],
                [0.0, 1.0, -QP_FRICTION_COEF],
                [0.0, -1.0, -QP_FRICTION_COEF],
            ])
            A[row:row + 4, col:col + 3] = C @ base_rot_mat_rp

        self.solve_count += 1
        self.failure_reason = ""
        started = time.monotonic()
        if not all(np.all(np.isfinite(v)) for v in (P, q, A)):
            grf = None
            self.failure_reason = "non-finite QP inputs"
        else:
            grf = self._osqp_solve(P, q, A, lo, up)
        self.solve_duration = time.monotonic() - started
        if self.solve_duration > self.WALL_TIME_LIMIT:
            grf = None
            self.failure_reason = "solve exceeded elapsed time budget"
        if grf is not None:
            grf = np.asarray(grf, dtype=float)
            if grf.shape != (12,) or not np.all(np.isfinite(grf)):
                grf = None
                self.failure_reason = "invalid solution dimensions or non-finite forces"
            else:
                constrained = A @ grf
                tol = self.CONSTRAINT_TOLERANCE
                if np.any(constrained < lo - tol) or np.any(constrained > up + tol):
                    grf = None
                    self.failure_reason = "solution violates current contact constraints"
        if grf is None:
            self.status = "failed"
            self.consecutive_failures += 1
            self.failure_reason = self.failure_reason or "no valid solution"
            raise QPSolveError(
                f"QP solve failed: {self.failure_reason}; "
                f"failures={self.consecutive_failures}, "
                f"solve_ms={1000.0 * self.solve_duration:.3f}")

        # NOTE: the training QP reports solved_acc = M@grf - g (qp_torque_optimizer
        # .py:406), 2g off from the physical M@grf + g; the same sign is used here
        # so qp_cost stays comparable with the training-time "Step Log/QP Cost"
        # metric. Diagnostics only - not used for control.
        solved_acc = mass_mat @ grf - g
        err = solved_acc - desired_acc
        qp_cost = float(err @ Q @ err)
        if not np.all(np.isfinite(solved_acc)) or not np.isfinite(qp_cost):
            self.status = "failed"
            self.consecutive_failures += 1
            self.failure_reason = "non-finite solved acceleration or cost"
            raise QPSolveError(f"QP solve failed: {self.failure_reason}")
        self.status = "solved"
        self.consecutive_failures = 0
        self.last_success_time = time.monotonic()
        self._last_grf = grf.copy()
        return grf, solved_acc, qp_cost

    def _osqp_solve(self, P, q, A, lo, up):
        # Contact changes can stall warm-start ADMM. A single cold solve avoids
        # both obsolete contact forces and an unbounded warm + cold retry.
        try:
            return self._solve_once(P, q, A, lo, up)
        except Exception as exc:
            self.failure_reason = f"{type(exc).__name__}: {exc}"
            return None

    def _solve_once(self, P, q, A, lo, up):
        prob = osqp.OSQP()
        prob.setup(P=sparse.csc_matrix(P), q=q,
                   A=sparse.csc_matrix(A), l=lo, u=up,
                   verbose=False, eps_abs=1e-5, eps_rel=1e-5,
                   max_iter=20000, time_limit=self.SOLVE_TIME_LIMIT)
        try:
            res = prob.solve(raise_error=False)      # osqp >= 1.0
        except TypeError:
            res = prob.solve()                       # osqp 0.6.x has no raise_error kwarg
        status = getattr(res.info, "status_val", None)
        # Even accepted statuses must pass shape, finite and constraint checks
        # in _solve before the result can be used for a motor command.
        if status in (1, 2) and res.x is not None:
            return np.asarray(res.x, dtype=float)
        self.failure_reason = str(getattr(res.info, "status", f"status {status}"))
        return None
