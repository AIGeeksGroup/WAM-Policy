"""Native online codec and independent causal observation encoder."""
import numpy as np
import torch
import torch.nn.functional as F
from arm_opsd.config import ArmContract


class ActionCodec:
    def __init__(self, contract=None):
        self.contract = contract or ArmContract()
        self.q01 = torch.tensor(self.contract.q01, dtype=torch.float32)
        self.q99 = torch.tensor(self.contract.q99, dtype=torch.float32)

    def decode(self, normalized, *, first=False):
        if tuple(normalized.shape) != self.contract.action_shape:
            raise ValueError("Expected native [1,30,4,4,1] action endpoint")
        # Native performs (+1)/2 in the endpoint dtype BEFORE quantile tensors
        # promote to FP32. Upcasting earlier changes BF16 physical commands.
        values = normalized.detach().cpu()[0, :7, :, :, 0]
        decoded = ((values + 1) / 2 * (self.q99 - self.q01 + 1e-6)[:, None, None]
                   + self.q01[:, None, None])
        if not torch.isfinite(decoded).all():
            raise FloatingPointError("Non-finite decoded command")
        if first:
            decoded = decoded[:, 1:]
        return decoded.permute(1, 2, 0).reshape(-1, 7).numpy().copy()

    def encode_executed(self, commands, *, first=False):
        a = torch.as_tensor(np.asarray(commands).copy(), dtype=torch.float32)
        expected = 12 if first else 16
        if tuple(a.shape) != (expected, 7) or not torch.isfinite(a).all():
            raise ValueError("History commit requires complete execution-confirmed actions")
        a = (a - self.q01) / (self.q99 - self.q01 + 1e-6) * 2 - 1
        a = a.reshape(3 if first else 4, 4, 7)
        if first:
            a = torch.cat([torch.zeros(1, 4, 7), a], dim=0)
        out = torch.zeros(self.contract.action_shape)
        out[0, :7, :, :, 0] = a.permute(2, 0, 1)
        return out


def free_mask(value, *, action=False, first=False, contract=None):
    contract = contract or ArmContract()
    mask = torch.ones_like(value, dtype=torch.bool)
    if action:
        mask[:, len(contract.active_channels):] = False
    if first:
        mask[:, :, 0] = False
    return mask


def project(value, *, action=False, first=False, initial_latent=None, contract=None):
    contract = contract or ArmContract()
    result = value
    if action:
        active = torch.arange(value.shape[1], device=value.device) < len(contract.active_channels)
        result = result * active[None, :, None, None, None].to(result.dtype)
    if first:
        if action:
            known = torch.zeros_like(result[:, :, :1])
        else:
            if initial_latent is None or initial_latent.shape != result[:, :, :1].shape:
                raise ValueError("Cold-start state requires the initial observation latent")
            known = initial_latent.detach().to(result)
        result = torch.cat([known, result[:, :, 1:]], dim=2)
    return result


def time_vector(value, timestep, *, first=False):
    times = torch.full((value.shape[2],), float(timestep), dtype=torch.float32, device=value.device)
    if first:
        times[0] = 0.
    return times


class ObservationEncoder:
    def __init__(self, vae, contract=None):
        from wan_va.modules.utils import WanVAEStreamingWrapper
        self.contract = contract or ArmContract()
        self.vae = vae.eval().requires_grad_(False)
        self.streaming = WanVAEStreamingWrapper(vae)
        self.frames = 0

    def reset(self):
        self.streaming.clear_cache()
        self.frames = 0

    @torch.no_grad()
    def encode(self, observations):
        if isinstance(observations, dict):
            observations = [observations]
        n = len(observations)
        if n != (1 if self.frames == 0 else 12 if self.frames == 1 else 16):
            raise ValueError("Causal VAE requires initial1, first feedback12, then feedback16")
        videos = []
        for key in self.contract.cameras:
            images = np.stack([np.asarray(obs[key]) for obs in observations])
            if images.dtype != np.uint8 or images.shape[1:] != (128, 128, 3):
                raise ValueError("Expected native flipped contiguous uint8 RGB")
            video = torch.from_numpy(images.copy()).float().permute(3, 0, 1, 2)
            video = F.interpolate(video, size=(128, 128), mode="bilinear", align_corners=False)[None]
            videos.append(video)
        weight = next(self.vae.parameters())
        x = (torch.cat(videos, dim=0) / 255 * 2 - 1).to(weight.device, weight.dtype)
        encoded = self.streaming.encode_chunk(x)
        mean, _ = encoded.chunk(2, dim=1)
        latent_mean = torch.tensor(self.vae.config.latents_mean, device=mean.device).view(1, -1, 1, 1, 1)
        latent_std = torch.tensor(self.vae.config.latents_std, device=mean.device).view(1, -1, 1, 1, 1)
        latent = ((mean.float() - latent_mean) * (1.0 / latent_std)).to(mean.dtype)
        latent = torch.cat(latent.split(1, dim=0), dim=-1)
        expected_groups = 1 if self.frames == 0 else n // 4
        if tuple(latent.shape) != (1, 48, expected_groups, 8, 16) or not torch.isfinite(latent).all():
            raise ValueError("Invalid native VAE timeline/latent")
        self.frames += n
        return latent.detach()
