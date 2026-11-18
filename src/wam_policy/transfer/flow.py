# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Modified by the WAM-Policy authors, 2026: derived from LingBot-VA train.py (_add_noise / compute_loss);
# native 9D action shapes, explicit chunk/window sampling, and valid-frame action-loss reduction.
"""Joint video/action flow-matching loss (JointFlow), derived from the official LingBot-VA train.py."""

import torch
import torch.nn.functional as F
from einops import rearrange
from wan_va.utils import (
    FlowMatchScheduler,
    get_mesh_id,
    sample_timestep_id,
    data_seq_to_patch,
)
from wam_policy.common.action_space import validate_batch
from wam_policy.common.flow_config import FlowConfig


class JointFlow:
    def __init__(self, device="cpu", *, gradient_accumulation_steps=1):
        self.config = FlowConfig()
        self.device = torch.device(device)
        self.patch_size = (1, 2, 2)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if (
            type(gradient_accumulation_steps) is not int
            or gradient_accumulation_steps < 1
        ):
            raise ValueError("Invalid gradient accumulation count")
        self.train_scheduler_latent = FlowMatchScheduler(
            shift=self.config.video_snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.train_scheduler_action = FlowMatchScheduler(
            shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

    @torch.no_grad()
    def prepare(
        self, batch, *, chunk_size=None, window_size=None, noisy_video_probability=None
    ):
        noisy_video_probability = (
            self.config.video_noisy_conditioning
            if noisy_video_probability is None
            else noisy_video_probability
        )
        shape = validate_batch(batch)
        chunk_size = (
            torch.randint(self.config.chunk_min, self.config.chunk_max + 1, (1,)).item()
            if chunk_size is None
            else chunk_size
        )
        window_size = (
            torch.randint(
                self.config.window_min, self.config.window_max + 1, (1,)
            ).item()
            if window_size is None
            else window_size
        )
        if (
            type(chunk_size) is not int
            or not 1 <= chunk_size <= 4
            or shape[2] <= chunk_size
        ):
            raise ValueError("Require multiple logical chunks and chunk_size in [1,4]")
        if type(window_size) is not int or not 4 <= window_size <= 64:
            raise ValueError("window_size must be in official logical units [4,64]")
        if not 0 <= noisy_video_probability <= 1:
            raise ValueError("Invalid history augmentation probability")
        b = {k: v.to(self.device) for k, v in batch.items()}
        latent = self._add_noise(
            b["latents"],
            self.train_scheduler_latent,
            action_mask=None,
            noisy_cond_prob=noisy_video_probability,
        )
        action = self._add_noise(
            b["actions"],
            self.train_scheduler_action,
            action_mask=b["actions_mask"],
            action_mode=True,
            noisy_cond_prob=self.config.action_noisy_conditioning,
        )
        latent["text_emb"] = b["text_emb"]
        action["text_emb"] = b["text_emb"]
        action["actions_mask"] = b["actions_mask"]
        return {
            "latent_dict": latent,
            "action_dict": action,
            "chunk_size": chunk_size,
            "window_size": window_size,
        }

    @torch.no_grad()
    def _add_noise(
        self,
        latent,
        train_scheduler,
        action_mask=None,
        action_mode=False,
        noisy_cond_prob=0.0,
    ):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(
            batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps
        )
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1

        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode,
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                batch_size=F,
                min_timestep_bd=0.5,
                max_timestep_bd=1.0,
                num_train_timesteps=train_scheduler.num_train_timesteps,
            )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(
                device=self.device
            )
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    def compute_loss(self, input_dict, pred):
        latent_pred, action_pred = pred
        action_pred = rearrange(
            action_pred,
            "b (f n) c -> b c f n 1",
            f=input_dict["action_dict"]["targets"].shape[-3],
        )
        latent_pred = data_seq_to_patch(
            self.patch_size,
            latent_pred,
            input_dict["latent_dict"]["targets"].shape[-3],
            input_dict["latent_dict"]["targets"].shape[-2],
            input_dict["latent_dict"]["targets"].shape[-1],
            batch_size=latent_pred.shape[0],
        )
        Bn, Fn = input_dict["latent_dict"]["timesteps"].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(
            input_dict["latent_dict"]["timesteps"].flatten()
        ).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(
            input_dict["action_dict"]["timesteps"].flatten()
        ).reshape(Bn, Fn)

        # Frame-wise video loss calculation
        latent_loss = F.mse_loss(
            latent_pred.float(),
            input_dict["latent_dict"]["targets"].float().detach(),
            reduction="none",
        )
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        latent_loss = latent_loss.permute(
            0, 2, 3, 4, 1
        )  # (B, C, F, H, W) -> (B, F, H, W, C)
        latent_loss = latent_loss.flatten(0, 1).flatten(
            1
        )  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and compute mask per frame
        latent_loss_per_frame = latent_loss.sum(dim=1)  # (B*F,)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F,)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        # Frame-wise action loss calculation
        action_loss = F.mse_loss(
            action_pred.float(),
            input_dict["action_dict"]["targets"].float().detach(),
            reduction="none",
        )
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict["action_dict"]["actions_mask"].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(
            0, 2, 3, 4, 1
        )  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_mask = (
            input_dict["action_dict"]["actions_mask"].float().permute(0, 2, 3, 4, 1)
        )  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(
            1
        )  # (B, F, H, W, C) -> (B*F, H*W*C)
        action_mask = action_mask.flatten(0, 1).flatten(
            1
        )  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize by mask per frame
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
        action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
        valid_frames = action_mask_per_frame > 0
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6))[
            valid_frames
        ].mean()

        return (
            latent_loss / self.gradient_accumulation_steps,
            action_loss / self.gradient_accumulation_steps,
        )
