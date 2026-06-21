from __future__ import annotations

import torch
from torch import nn


class FlowTimePairEmbedding(nn.Module):
    """Embed (s_src, s_tgt, delta_s) for video flow-map training."""

    def __init__(self, hidden_dim: int = 256, mlp_hidden_dim: int | None = None) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if mlp_hidden_dim is not None and mlp_hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be positive when provided.")

        inner_dim = mlp_hidden_dim or hidden_dim
        self.hidden_dim = hidden_dim
        self.mlp_hidden_dim = inner_dim
        self.net = nn.Sequential(
            nn.Linear(3, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, hidden_dim),
        )
        self._zero_init_output()

    def forward(self, s_src: torch.Tensor, s_tgt: torch.Tensor) -> torch.Tensor:
        _validate_time_pair(s_src, s_tgt)

        param = next(self.parameters())
        if s_src.device != param.device:
            raise ValueError("s_src and s_tgt must be on the same device as the module.")

        delta_s = s_tgt - s_src
        features = torch.stack([s_src, s_tgt, delta_s], dim=-1).to(dtype=param.dtype)
        return self.net(features)

    def _zero_init_output(self) -> None:
        output_layer = self.net[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("FlowTimePairEmbedding output layer must be nn.Linear.")
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)


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
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [B].")
