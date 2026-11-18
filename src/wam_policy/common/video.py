"""Deterministic two-view RGB and causal Wan VAE interface."""

import hashlib
import json
from collections.abc import Mapping
import numpy as np
import torch
import torch.nn.functional as F
from wan_va.modules.utils import WanVAEStreamingWrapper
from wam_policy.common.action_space import CAMERA_KEYS

PREPROCESS = {
    "schema": "dual-rgb-latent-v2",
    "cameras": list(CAMERA_KEYS),
    "source_hw": [720, 1280],
    "resize_hw": [180, 320],
    "interpolation": "bilinear",
    "antialias": True,
    "align_corners": False,
    "padding_lrtb": [0, 0, 6, 6],
    "padding_rgb": 0,
    "normalization": "x/255*2-1",
    "posterior": "mean",
    "composition": "ego_above_task",
    "latent_cthw": [48, None, 24, 20],
}
PREPROCESS_SHA256 = hashlib.sha256(
    json.dumps(PREPROCESS, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


PREPROCESS_360 = {**PREPROCESS, "source_hw": [360, 640], "rgb_spatial_profile": "dual-rgb-360p-v1"}
PREPROCESS_360_SHA256 = hashlib.sha256(
    json.dumps(PREPROCESS_360, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def preprocessing(rgb_profile="dual-rgb-360p-v1"):
    if rgb_profile == "dual-rgb-720p-v1": return PREPROCESS, PREPROCESS_SHA256
    if rgb_profile == "dual-rgb-360p-v1": return PREPROCESS_360, PREPROCESS_360_SHA256
    raise ValueError("Unknown input RGB spatial profile")


def preprocess_rgb(rgb, *, rgb_profile="dual-rgb-360p-v1"):
    profile, _ = preprocessing(rgb_profile)
    height, width = profile["source_hw"]
    value = torch.as_tensor(rgb)
    if value.ndim == 4:
        value = value.unsqueeze(0)
    if (
        value.ndim != 5
        or value.shape[-3:] != (height, width, 3)
        or value.shape[0] < 1
        or value.shape[1] < 1
        or value.dtype != torch.uint8
    ):
        raise ValueError(f"Expected uint8 RGB [B,N,{height},{width},3] with nonempty B,N")
    b, n = value.shape[:2]
    value = value.cpu().reshape(b * n, height, width, 3).permute(0, 3, 1, 2).float()
    value = F.interpolate(
        value, size=(180, 320), mode="bilinear", align_corners=False, antialias=True
    )
    value = F.pad(value, (0, 0, 6, 6), value=0).div(255).mul(2).sub(1)
    return value.reshape(b, n, 3, 192, 320).permute(0, 2, 1, 3, 4).contiguous()


def validate_views(views):
    if not isinstance(views, Mapping) or set(views) != set(CAMERA_KEYS):
        raise ValueError("Exactly go2_ego and d435i_rgb_task are required")
    shapes = [np.shape(views[k]) for k in CAMERA_KEYS]
    if shapes[0] != shapes[1]:
        raise ValueError("Camera streams must have identical batch/frame dimensions")


def compose_latents(views):
    if set(views) != set(CAMERA_KEYS):
        raise ValueError("Unexpected latent camera keys")
    a, b = (views[k] for k in CAMERA_KEYS)
    if not all(isinstance(x, torch.Tensor) and x.is_floating_point() for x in (a, b)):
        raise ValueError("Expected floating-point latent tensors")
    if (
        a.ndim != 5
        or a.shape[1] != 48
        or a.shape[-2:] != (12, 20)
        or a.shape != b.shape
        or a.shape[2] < 1
    ):
        raise ValueError("Expected two [B,48,T,12,20] latent streams")
    if (
        a.device != b.device
        or a.dtype != b.dtype
        or not torch.isfinite(a).all()
        or not torch.isfinite(b).all()
    ):
        raise ValueError("Latents must be finite with matching device/dtype")
    return torch.cat((a, b), dim=-2)


def split_latents(latents):
    if not isinstance(latents, torch.Tensor) or not latents.is_floating_point():
        raise ValueError("Expected floating-point latent tensor")
    if (
        latents.ndim != 5
        or latents.shape[1] != 48
        or latents.shape[-2:] != (24, 20)
        or latents.shape[2] < 1
        or not torch.isfinite(latents).all()
    ):
        raise ValueError("Expected finite [B,48,T,24,20]")
    return dict(zip(CAMERA_KEYS, latents.split(12, dim=-2)))


def normalize_latents(value, vae, *, inverse=False):
    mean = torch.as_tensor(
        vae.config.latents_mean, device=value.device, dtype=torch.float32
    ).view(1, 48, 1, 1, 1)
    std = torch.as_tensor(
        vae.config.latents_std, device=value.device, dtype=torch.float32
    ).view(1, 48, 1, 1, 1)
    if (
        not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
        or not (std > 0).all()
    ):
        raise ValueError("Invalid source VAE statistics")
    result = value.float() * std + mean if inverse else (value.float() - mean) / std
    return result.to(value.dtype)


class DualViewVAE:
    """One frozen weight set, two independent temporal cache states."""

    def __init__(self, vae, *, rgb_profile="dual-rgb-360p-v1"):
        self.rgb_profile = rgb_profile
        self.preprocess, _ = preprocessing(rgb_profile)
        self.vae = vae.eval().requires_grad_(False)
        self.streams = {key: WanVAEStreamingWrapper(vae) for key in CAMERA_KEYS}
        self.reset()

    def reset(self):
        for stream in self.streams.values():
            stream.clear_cache()
        self.buffers = {k: None for k in CAMERA_KEYS}
        self.first = True
        self.batch_size = None

    @property
    def pending_frames(self):
        buf = self.buffers[CAMERA_KEYS[0]]
        return 0 if buf is None else buf.shape[2]

    @torch.no_grad()
    def push(self, views):
        validate_views(views)
        prepared = {k: preprocess_rgb(views[k], rgb_profile=self.rgb_profile) for k in CAMERA_KEYS}
        b = prepared[CAMERA_KEYS[0]].shape[0]
        if self.batch_size is not None and b != self.batch_size:
            raise ValueError("Batch size changed without reset")
        self.batch_size = b
        for k in CAMERA_KEYS:
            self.buffers[k] = (
                prepared[k]
                if self.buffers[k] is None
                else torch.cat((self.buffers[k], prepared[k]), dim=2)
            )
        outputs = {k: [] for k in CAMERA_KEYS}
        weight = next(self.vae.parameters())
        while self.pending_frames >= (1 if self.first else 4):
            count = 1 if self.first else 4
            for k in CAMERA_KEYS:
                value = self.buffers[k][:, :, :count].to(
                    device=weight.device, dtype=weight.dtype
                )
                self.buffers[k] = self.buffers[k][:, :, count:]
                posterior = self.streams[k].encode_chunk(value)
                mu, _ = posterior.chunk(2, dim=1)
                z = normalize_latents(mu, self.vae)
                if z.shape != (b, 48, 1, 12, 20):
                    raise ValueError(f"Unexpected VAE output {tuple(z.shape)}")
                outputs[k].append(z)
            self.first = False
        if not outputs[CAMERA_KEYS[0]]:
            return None
        return compose_latents({k: torch.cat(v, dim=2) for k, v in outputs.items()})

    def finish(self):
        if self.first or self.pending_frames:
            raise ValueError(
                "Incomplete causal clip; require N=1+4k and do not silently pad/drop frames"
            )

    def encode_clip(self, views):
        validate_views(views)
        shape = np.shape(views[CAMERA_KEYS[0]])
        n = shape[-4] if len(shape) in (4, 5) else 0
        if n < 1 or (n - 1) % 4:
            raise ValueError("Offline clip requires N=1+4k frames")
        self.reset()
        result = self.push(views)
        self.finish()
        return result

    @torch.no_grad()
    def decode(self, latents, *, remove_letterbox=False):
        result = {}
        weight = next(self.vae.parameters())
        for k, z in split_latents(latents).items():
            raw = normalize_latents(
                z.to(device=weight.device, dtype=weight.dtype), self.vae, inverse=True
            )
            rgb = self.vae.decode(raw, return_dict=False)[0]
            if rgb.shape[1] != 3 or rgb.shape[-2:] != (192, 320):
                raise ValueError("Unexpected decoded geometry")
            if remove_letterbox:
                rgb = rgb[..., 6:-6, :]
            result[k] = rgb.clamp(-1, 1)
        return result
