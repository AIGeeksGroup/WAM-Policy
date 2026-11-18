"""10D proprioceptive state from the measured root link, and its train-only normalizer."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCHEMA = "wam-policy-proprioception-10d-v1"
STATE_DIM = 10
# Canonical / QLM live root-link fields (positions and velocities in the world frame, xyzw quaternion).
FIELDS = (
    "observation.state.base.position", "observation.state.base.orientation",
    "observation.state.base.linear_velocity", "observation.state.base.angular_velocity",
    "observation.state.projected_gravity",
)
# What QLM-Bench live observations must declare; the ground plane is at world z = 0.
STATE_SEMANTICS = {"reference": "root_link", "quaternion": "xyzw", "velocity_frame": "world",
                   "projected_gravity_frame": "body", "ground_z": 0.0,
                   "ground_reference": "push-box-v2-plane-world-z0"}
GROUND_Z = 0.0
# Dimensions with quantile normalization; gravity (1..3) is already a unit vector.
SCALED = [0, 4, 5, 6, 7, 8, 9]


def rotate_inverse_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate world vectors into the root-link (body) frame."""
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    if q.shape[-1] != 4 or v.shape[-1] != 3 or not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError("Invalid quaternion/vector shape or values")
    if not np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-4, rtol=0):
        raise ValueError("Quaternion must be unit xyzw")
    xyz, w = q[..., :3], q[..., 3:4]
    twice = 2.0 * np.cross(xyz, v)
    return v - w * twice + np.cross(xyz, twice)


def select_state(columns: dict, *, ground_z: float = GROUND_Z) -> np.ndarray:
    """{field: [N, 3|4]} root-link columns -> [N, 10] (h, gravity_b, v_b, w_b)."""
    pos, q, lin, ang, observed = [np.asarray(columns[k], dtype=np.float64) for k in FIELDS]
    if pos.shape[-1] != 3 or q.shape[-1] != 4 or lin.shape != pos.shape or ang.shape != pos.shape \
            or observed.shape != pos.shape:
        raise ValueError("Inconsistent root-link state columns")
    gravity = rotate_inverse_xyzw(q, np.broadcast_to([0., 0., -1.], pos.shape))
    if not np.allclose(gravity, observed, atol=2e-4, rtol=0):
        raise ValueError("Stored projected gravity conflicts with the root orientation")
    result = np.concatenate([pos[..., 2:3] - ground_z, gravity,
                             rotate_inverse_xyzw(q, lin), rotate_inverse_xyzw(q, ang)], axis=-1)
    if not np.isfinite(result).all() or result.shape[-1] != STATE_DIM:
        raise ValueError("Invalid 10D proprioception")
    return result.astype(np.float32)


@dataclass(frozen=True)
class StateNormalizer:
    """Affine map of [q01, q99] to [-1, 1] on the scaled dimensions (no clipping)."""
    q01: tuple
    q99: tuple
    low_variance: tuple
    epsilon: float = 1e-6

    @classmethod
    def fit(cls, selected: np.ndarray) -> "StateNormalizer":
        samples = np.asarray(selected, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != STATE_DIM or not len(samples) or not np.isfinite(samples).all():
            raise ValueError("Finite [N,10] training states required")
        q = np.quantile(samples[:, SCALED], [0.01, 0.99], axis=0, method="linear")
        low = (q[1] - q[0]) < 1e-6
        return cls(tuple(q[0].tolist()), tuple(q[1].tolist()), tuple(low.tolist()))

    @classmethod
    def from_record(cls, record: dict) -> "StateNormalizer":
        value = cls(tuple(record["q01"]), tuple(record["q99"]), tuple(record["low_variance"]),
                    float(record.get("epsilon", 1e-6)))
        value._parameters()
        return value

    def _parameters(self):
        if len(self.q01) != len(SCALED) or len(self.q99) != len(SCALED) or len(self.low_variance) != len(SCALED):
            raise ValueError("State normalizer needs 7 quantile pairs")
        lo, hi = np.asarray(self.q01, np.float64), np.asarray(self.q99, np.float64)
        return lo, hi, np.asarray(self.low_variance, dtype=bool)

    def encode(self, physical: np.ndarray) -> np.ndarray:
        value = np.asarray(physical, dtype=np.float64)
        if value.shape[-1] != STATE_DIM or not np.isfinite(value).all():
            raise ValueError("Finite 10D state required")
        lo, hi, low = self._parameters()
        out = value.copy()
        out[..., SCALED] = np.where(low, 0., 2. * (value[..., SCALED] - lo) / np.where(low, 1., hi - lo) - 1.)
        return out.astype(np.float32)

    def decode(self, normalized: np.ndarray) -> np.ndarray:
        value = np.asarray(normalized, dtype=np.float64)
        if value.shape[-1] != STATE_DIM or not np.isfinite(value).all():
            raise ValueError("Finite normalized 10D state required")
        lo, hi, low = self._parameters()
        out = value.copy()
        out[..., SCALED] = np.where(low, (lo + hi) / 2., lo + (value[..., SCALED] + 1.) * (hi - lo) / 2.)
        return out.astype(np.float32)

    def record(self) -> dict:
        return {"schema": SCHEMA, "q01": list(self.q01), "q99": list(self.q99),
                "low_variance": list(self.low_variance), "epsilon": self.epsilon,
                "dimensions": ["h", "vx", "vy", "vz", "wx", "wy", "wz"]}
