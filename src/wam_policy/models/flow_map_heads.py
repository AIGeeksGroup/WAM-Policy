from __future__ import annotations

from math import prod

import torch
from torch import nn


class FlowTimePairEmbedding(nn.Module):
    """Embed (s_src, s_tgt, delta_s) for video flow-map training."""

    def __init__(
        self,
        hidden_dim: int = 256,
        mlp_hidden_dim: int | None = None,
        timestep_proj_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if mlp_hidden_dim is not None and mlp_hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be positive when provided.")

        self.hidden_dim = hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim or hidden_dim
        self.timestep_proj_dim = timestep_proj_dim or hidden_dim * 6
        self._build_layers()

    def forward(self, s_src: torch.Tensor, s_tgt: torch.Tensor) -> torch.Tensor:
        temb_delta, _ = self._embed(s_src, s_tgt)
        return temb_delta

    def forward_conditioning(
        self,
        s_src: torch.Tensor,
        s_tgt: torch.Tensor,
        timestep_proj_shape: torch.Size | tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temb_delta, timestep_proj_delta = self._embed(s_src, s_tgt)
        target_shape = tuple(timestep_proj_shape)
        if len(target_shape) < 3:
            raise ValueError("timestep_proj_shape must include batch, token, and channel dims.")

        batch_size, token_count = target_shape[:2]
        if temb_delta.ndim == 2:
            temb_delta = temb_delta[:, None, :].expand(batch_size, token_count, self.hidden_dim)
            timestep_proj_delta = timestep_proj_delta[:, None, :].expand(
                batch_size,
                token_count,
                self.timestep_proj_dim,
            )
        if temb_delta.shape[:2] != (batch_size, token_count):
            raise ValueError(
                "s_src/s_tgt token shape must match timestep_proj_shape leading dims."
            )

        trailing_size = prod(target_shape[2:])
        if timestep_proj_delta.shape[-1] != trailing_size:
            raise ValueError(
                "timestep projection delta dim does not match requested timestep_proj_shape."
            )
        return temb_delta, timestep_proj_delta.reshape(target_shape)

    def configure_output_dims(self, hidden_dim: int, timestep_proj_dim: int) -> None:
        if hidden_dim <= 0 or timestep_proj_dim <= 0:
            raise ValueError("hidden_dim and timestep_proj_dim must be positive.")
        if hidden_dim == self.hidden_dim and timestep_proj_dim == self.timestep_proj_dim:
            return
        param = next(self.parameters())
        device = param.device
        dtype = param.dtype
        self.hidden_dim = hidden_dim
        self.timestep_proj_dim = timestep_proj_dim
        self.mlp_hidden_dim = max(self.mlp_hidden_dim, hidden_dim)
        self._build_layers()
        self.to(device=device, dtype=dtype)

    def _embed(self, s_src: torch.Tensor, s_tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_time_pair(s_src, s_tgt)

        param = next(self.parameters())
        if s_src.device != param.device:
            raise ValueError("s_src and s_tgt must be on the same device as the module.")

        delta_s = s_tgt - s_src
        features = torch.stack([s_src, s_tgt, delta_s], dim=-1).to(dtype=param.dtype)
        hidden = self.trunk(features)
        return self.temb_out(hidden), self.timestep_proj_out(hidden)

    def _build_layers(self) -> None:
        self.trunk = nn.Sequential(
            nn.Linear(3, self.mlp_hidden_dim),
            nn.SiLU(),
        )
        self.temb_out = nn.Linear(self.mlp_hidden_dim, self.hidden_dim)
        self.timestep_proj_out = nn.Linear(self.mlp_hidden_dim, self.timestep_proj_dim)
        self._zero_init_outputs()

    def _zero_init_outputs(self) -> None:
        for layer in (self.temb_out, self.timestep_proj_out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)


def _validate_time_pair(s_src: torch.Tensor, s_tgt: torch.Tensor) -> None:
    _validate_time_tensor("s_src", s_src)
    _validate_time_tensor("s_tgt", s_tgt)
    if s_src.shape != s_tgt.shape:
        raise ValueError(f"s_src shape must match s_tgt shape: {s_src.shape} != {s_tgt.shape}")
    if s_src.device != s_tgt.device:
        raise ValueError("s_src and s_tgt must be on the same device.")
    if not torch.all(s_src < s_tgt).item():
        raise ValueError("s_src must be strictly smaller than s_tgt for every sample.")


def _validate_time_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim not in {1, 2}:
        raise ValueError(f"{name} must have shape [B] or [B, L].")
