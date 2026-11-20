"""RAMBO controller output -> Go2 LowCmd values.

Final gate before the hardware: reorders RAMBO -> Go2, applies absolute
position/torque clamps and rejects NaN/Inf. The RAMBO controller already
clips internally; this is the independent last line of defense.
"""

import numpy as np

from ..constants import (
    JOINT_TORQUE_LIMIT,
    POLICY_KD,
    POLICY_KP,
)
from .hardware_limits import clip_joint_positions
from .joint_mapping import rambo_to_go2


class CommandError(Exception):
    pass


def _gain_vector(value, name):
    """Scalar or RAMBO-order (12,) gain -> Go2-order (12,) array."""
    g = np.asarray(value, dtype=float)
    if g.ndim == 0:
        g = np.full(12, float(g))
    elif g.shape == (12,):
        g = rambo_to_go2(g)
    else:
        raise CommandError(f"{name} must be a scalar or 12 values, got shape {g.shape}")
    if not np.all(np.isfinite(g)) or np.any(g < 0.0):
        raise CommandError(f"non-finite or negative {name}")
    return g


def policy_gains(ee_joint_indices, compliant, compliant_kp, compliant_kd,
                 kp=POLICY_KP, kd=POLICY_KD):
    """Per-joint POLICY gains in RAMBO order -> (kp, kd) arrays of shape (12,).

    Nominal ``kp`` / ``kd`` (training: 40 / 1) on every joint; when
    ``compliant`` the manipulation-leg joints (``TaskConfig.ee_joint_indices``:
    FL) get ``compliant_kp`` / ``compliant_kd`` instead. This is how the RAMBO paper made the arm
    compliant for the handshake demo (Sec. V-D: "lowering the PD gains at the
    joints associated with the manipulation end-effectors"). The QP
    feedforward (gravity compensation + stance torque) is not touched here,
    so the leg still holds its own weight.
    """
    kp_v = np.full(12, float(kp))
    kd_v = np.full(12, float(kd))
    if compliant:
        idx = np.asarray(ee_joint_indices, dtype=int)
        kp_v[idx] = float(compliant_kp)
        kd_v[idx] = float(compliant_kd)
    return kp_v, kd_v


def adapt_command(q_des, dq_des, tau_ff, kp=POLICY_KP, kd=POLICY_KD):
    """RAMBO-order command -> (q, dq, kp, kd, tau) arrays in Go2 order.

    ``kp`` / ``kd`` are scalars (same gain on all 12 motors) or RAMBO-order
    (12,) vectors (see :func:`policy_gains`); vectors are reordered like q.
    Raises CommandError if the command contains non-finite values.
    """
    q_des = np.asarray(q_des, dtype=float)
    dq_des = np.asarray(dq_des, dtype=float)
    tau_ff = np.asarray(tau_ff, dtype=float)
    if not (np.all(np.isfinite(q_des)) and np.all(np.isfinite(dq_des))
            and np.all(np.isfinite(tau_ff))):
        raise CommandError("non-finite value in joint command")

    q_des = clip_joint_positions(q_des)
    tau_ff = np.clip(tau_ff, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)

    return (
        rambo_to_go2(q_des),
        rambo_to_go2(dq_des),
        _gain_vector(kp, "kp"),
        _gain_vector(kd, "kd"),
        rambo_to_go2(tau_ff),
    )
