"""Tiny model configuration and synthetic tensors for smoke runs (never demonstrations)."""

import numpy as np
import torch
from wam_policy.common.action_space import CAMERA_KEYS

TINY_CONFIG = dict(
    num_layers=2,
    num_attention_heads=2,
    attention_head_dim=64,
    ffn_dim=512,
    action_dim=9,
    in_channels=48,
    out_channels=48,
    text_dim=4096,
    patch_size=(1, 2, 2),
)


