"""Three-stream flow matching: Stage 1.1 video/action flow plus a 10D state stream.

States share the video timestep and scheduler (SNR shift 5); their conditioning copy is always
clean. Invalid (padding) state slots are zero and excluded from the loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

from wam_policy.transfer.flow import JointFlow

BATCH_KEYS = {"latents", "actions", "actions_mask", "text_emb", "proprioception",
              "proprioception_mask", "proprioception_timestamps_ns"}


def state_grid_ids(mask: torch.Tensor) -> torch.Tensor:
    """[B,1,T,M,1] validity -> RoPE grid [B,4,T*M] with group-relative temporal positions."""
    if mask.ndim != 5 or mask.dtype != torch.bool or mask.shape[1] != 1 or mask.shape[-1] != 1:
        raise ValueError("Expected bool proprioception mask [B,1,T,M,1]")
    B, _, T, M, _ = mask.shape
    temporal = torch.zeros((B, T, M), dtype=torch.float32, device=mask.device)
    for k in range(T):
        for b in range(B):
            n = int(mask[b, 0, k, :, 0].sum())
            if (k == 0 and n != 1) or (k > 0 and n != 4) or not bool(mask[b, 0, k, :n, 0].all()):
                raise ValueError("Group 0 needs one state and every later group four")
            if k:
                temporal[b, k, :n] = k - 1 + torch.arange(1, n + 1, device=mask.device) / n
    h = torch.full_like(temporal, -2.)
    w = torch.full_like(temporal, -2.)
    role = torch.full_like(temporal, 2.)
    return torch.stack([temporal, h, w, role], dim=1).flatten(2)


class ProprioceptionFlow(JointFlow):
    def prepare(self, batch, *, chunk_size=None, window_size=None, noisy_video_probability=None):
        if set(batch) != BATCH_KEYS:
            raise ValueError(f"Batch keys must be {sorted(BATCH_KEYS)}")
        prepared = super().prepare({k: batch[k] for k in ("latents", "actions", "actions_mask", "text_emb")},
                                   chunk_size=chunk_size, window_size=window_size,
                                   noisy_video_probability=noisy_video_probability)
        source = batch["proprioception"].to(self.device)
        mask = batch["proprioception_mask"].to(self.device)
        timestamps = batch["proprioception_timestamps_ns"].to(self.device)
        B, C, T, M, W = source.shape
        if C != 10 or W != 1 or mask.shape != (B, 1, T, M, 1) or mask.dtype != torch.bool or \
                timestamps.shape != (B, T, M) or T != batch["latents"].shape[2]:
            raise ValueError("Proprioception must be [B,10,T,M,1] with T video latent groups")
        noise = torch.randn_like(source)
        timestep = prepared["latent_dict"]["timesteps"]
        if not torch.equal(timestep, timestep[0:1].expand_as(timestep)):
            raise ValueError("Video timesteps unexpectedly differ across the batch")
        target = self.train_scheduler_latent.training_target(source, noise, timestep[0])
        noisy = self.train_scheduler_latent.add_noise(source, noise, timestep[0], t_dim=2)
        valid = mask.expand_as(source)
        prepared["proprioception_dict"] = dict(
            noisy_latents=noisy.masked_fill(~valid, 0), targets=target.masked_fill(~valid, 0),
            latent=source.masked_fill(~valid, 0), timesteps=timestep,
            cond_timesteps=torch.zeros_like(timestep), validity_mask=mask,
            grid_id=state_grid_ids(mask), timestamps_ns=timestamps)
        return prepared

    def compute_loss(self, prepared, prediction):
        """-> (video, proprioception, action) losses."""
        if len(prediction) != 3:
            raise ValueError("The proprioception model returns video, proprioception and action")
        video_loss, action_loss = super().compute_loss(prepared, (prediction[0], prediction[2]))
        prop = prepared["proprioception_dict"]
        target = prop["targets"]
        B, _, T, M, _ = target.shape
        output = rearrange(prediction[1], "b (t m) c -> b c t m 1", t=T, m=M)
        weights = self.train_scheduler_latent.training_weight(prop["timesteps"].flatten()).reshape(B, T)
        errors = F.mse_loss(output.float(), target.float().detach(), reduction="none").mean(dim=1)
        weighted = errors * weights[:, :, None, None]
        valid = prop["validity_mask"][:, 0].float()
        loss = (weighted * valid).sum() / valid.sum()
        return video_loss, loss / self.gradient_accumulation_steps, action_loss
