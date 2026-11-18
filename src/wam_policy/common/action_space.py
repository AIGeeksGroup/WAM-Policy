"""Native 9D command space (base velocity, front-left foot target, zero force), its normalizer and the
training-batch tensor contract. Physical timestamp alignment lives in ``wam_policy.interface``."""

from dataclasses import dataclass
import numpy as np
import torch

ACTION_NAMES = (
    "base_vx",
    "base_vy",
    "base_yaw_rate",
    "fl_ee_x",
    "fl_ee_y",
    "fl_ee_z",
    "fl_ee_fx",
    "fl_ee_fy",
    "fl_ee_fz",
)
ACTION_SCHEMA = "rambo-native9-v2"
CAMERA_KEYS = ("go2_ego", "d435i_rgb_task")


@dataclass(frozen=True)
class ActionNormalizer:
    offset: tuple[float, ...]
    scale: tuple[float, ...]
    diagnostic_only: bool = False

    def __post_init__(self):
        off, scale = np.asarray(self.offset), np.asarray(self.scale)
        if (
            off.shape != (9,)
            or scale.shape != (9,)
            or not np.isfinite(off).all()
            or not np.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError(
                "Normalization requires nine finite offsets and positive scales"
            )
        if np.any(off[6:] != 0) or np.any(scale[6:] != 1):
            raise ValueError("Zero-force baseline requires force offset=0 and scale=1")

    def encode(self, action):
        value = np.asarray(action)
        if value.shape[-1:] != (9,) or not np.isfinite(value).all():
            raise ValueError("Expected finite native 9D actions")
        if np.any(value[..., 6:] != 0):
            raise ValueError("The initial baseline requires zero desired force")
        return ((value - np.asarray(self.offset)) / np.asarray(self.scale)).astype(
            np.float32
        )

    def decode(self, action):
        value = np.asarray(action)
        if value.shape[-1:] != (9,) or not np.isfinite(value).all():
            raise ValueError("Expected finite native 9D predictions")
        # Preserve predictions; execution-time zero-force enforcement is an explicit later contract.
        return (value * np.asarray(self.scale) + np.asarray(self.offset)).astype(
            np.float32
        )


def validate_batch(batch):
    if set(batch) != {"latents", "actions", "actions_mask", "text_emb"}:
        raise ValueError("Batch requires latents/actions/actions_mask/text_emb only")
    v, a, m, t = (batch[k] for k in ("latents", "actions", "actions_mask", "text_emb"))
    if not all(isinstance(x, torch.Tensor) for x in (v, a, m, t)):
        raise ValueError("Batch fields must be tensors")
    if (
        v.ndim != 5
        or v.shape[1] != 48
        or v.shape[-2:] != (24, 20)
        or v.shape[0] < 1
        or v.shape[2] < 1
    ):
        raise ValueError("Expected video [B,48,T,24,20]")
    if (
        a.shape != (v.shape[0], 9, v.shape[2], 16, 1)
        or m.shape != a.shape
        or m.dtype != torch.bool
    ):
        raise ValueError("Expected actions and bool mask [B,9,T,16,1]")
    if t.shape != (v.shape[0], 512, 4096):
        raise ValueError("Expected text [B,512,4096]")
    if not all(
        x.is_floating_point() and bool(torch.isfinite(x).all()) for x in (v, a, t)
    ):
        raise ValueError("Batch values must be finite floating point")
    if bool((a[:, 6:][m[:, 6:]] != 0).any()):
        raise ValueError("Valid force labels must be zero")
    if bool((a[~m] != 0).any()):
        raise ValueError("Invalid action placeholders must be zero")
    if not bool(m.any()):
        raise ValueError("Batch has no valid action targets")
    return v.shape
