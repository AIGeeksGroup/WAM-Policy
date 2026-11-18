# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Modified by the WAM-Policy authors, 2026: lazy optional FlashAttention import, reference (dense) attention
# backend, rotary dimensions rounded to even sizes, inference-only key/value cache with logical ids and
# eviction, strict loading, native 9D action interface, and an optional proprioception branch.
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from typing import Callable, ClassVar
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
    and_masks,
    or_masks
)
from functools import partial

def flash_attn_func(*args, **kwargs):
    """Load optional FlashAttention only when explicitly selected."""
    try:
        from flash_attn_interface import flash_attn_func as implementation
    except ImportError:
        from flash_attn import flash_attn_func as implementation
    return implementation(*args, **kwargs)

__all__ = ['WanTransformer3DModel']


def custom_sdpa(q, k, v):
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2))
    return out.transpose(1, 2)


def proprioception_attention_metadata(latent_shape, proprioception_mask, action_mask, action_shape, patch_size, chunk_size, device):
    """Six token roles with native causal block IDs and explicit physical validity."""
    B, _, T, H, W = latent_shape
    AB, _, AT, AH, AW = action_shape
    if AB != B or proprioception_mask.ndim != 5 or tuple(proprioception_mask.shape[:3]) != (B, 1, T):
        raise ValueError("Video/proprioception attention shapes disagree")
    M = proprioception_mask.shape[3]
    V = (H // patch_size[1]) * (W // patch_size[2])
    def segment(count, positions, offset, valid=None):
        seq = torch.arange(B, device=device)[:, None].expand(B, count).flatten()
        frame = torch.arange(positions, device=device)[None, :, None].expand(B, positions, count // positions).flatten()
        frame = (frame // chunk_size) * 2 + offset
        if valid is not None:
            seq = seq.masked_fill(~valid.flatten(), -1)
        return seq, frame
    video_seq, video_frame = segment(T * V, T, 0)
    prop_seq, prop_frame = segment(T * M, T, 0, proprioception_mask[:, 0, :, :, 0].to(device))
    if action_mask.shape != (B, action_shape[1], AT, AH, AW) or action_mask.dtype != torch.bool or \
            not bool((action_mask == action_mask[:, :1]).all()):
        raise ValueError("Action validity must agree across the native9 channels")
    action_seq, action_frame = segment(AT * AH * AW, AT, 1,
                                       action_mask[:, 0].to(device))
    seq = torch.cat([video_seq, video_seq, prop_seq, prop_seq, action_seq, action_seq])
    frame = torch.cat([video_frame, video_frame, prop_frame, prop_frame, action_frame, action_frame])
    noise = torch.cat([torch.zeros_like(video_frame), torch.ones_like(video_frame),
                       torch.zeros_like(prop_frame), torch.ones_like(prop_frame),
                       torch.zeros_like(action_frame), torch.ones_like(action_frame)])
    return seq, frame, noise

class FlexAttnFunc(nn.Module):
    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention, dynamic=True, 
    )
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    attention_mask: ClassVar[BlockMask] = None
    cross_attention_mask: ClassVar[BlockMask] = None

    def __init__(
        self, 
        is_cross=False,
    ) -> None:
        super().__init__()
        self.is_cross = is_cross
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        q_varlen = rearrange(query[0], "s n d -> 1 n s d")
        k_varlen = rearrange(key[0], "s n d -> 1 n s d")
        v_varlen = rearrange(value[0], "s n d -> 1 n s d")

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)
        
        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)

        block_mask = FlexAttnFunc.cross_attention_mask if self.is_cross else FlexAttnFunc.attention_mask

        x_out = FlexAttnFunc.flex_attn(q_varlen, k_varlen, v_varlen, block_mask=block_mask, kernel_options = {
                                                    "BLOCK_M": 64,
                                                    "BLOCK_N": 64,
                                                    "BLOCK_M1": 32,
                                                    "BLOCK_N1": 64,
                                                    "BLOCK_M2": 64,
                                                    "BLOCK_N2": 32,
                                                })

        x_out = rearrange(x_out, "b n s d -> b s n d")
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_mask(
        latent_shape, 
        action_shape, 
        padded_length, 
        chunk_size,
        window_size,
        patch_size,
        device,
    ):
        torch._inductor.config.realize_opcount_threshold = 100
        B, _, L_F, L_H, L_W = latent_shape
        _, _, A_F, A_H, A_W = action_shape

        latent_seq_id = torch.arange(B)[:, None, None, None].\
            expand(-1, L_F // patch_size[0], L_H // patch_size[1], L_W // patch_size[2]).flatten()
        action_seq_id = torch.arange(B)[:, None, None, None].expand(-1, A_F, A_H, A_W).flatten()
        seq_ids = torch.cat([latent_seq_id] * 2 + [action_seq_id] * 2)

        latent_frame_id = torch.arange(L_F)[None, :, None, None].expand(B, -1, L_H // patch_size[1], L_W // patch_size[2])[None].flatten()
        action_frame_id = torch.arange(A_F)[None, :, None, None].expand(B, -1, A_H, A_W)[None].flatten()
        frame_ids = torch.cat([latent_frame_id // chunk_size * 2] * 2 + [action_frame_id // chunk_size * 2 + 1] * 2)

        noise_ids = torch.cat(
            [
                torch.zeros_like(latent_frame_id),
                torch.ones_like(latent_frame_id),
                torch.zeros_like(action_frame_id),
                torch.ones_like(action_frame_id),
            ]
        )

        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)
        frame_ids = F.pad(frame_ids, (0, padded_length), value=-1)
        noise_ids = F.pad(noise_ids, (0, padded_length), value=-1)

        mask_mod = FlexAttnFunc._get_mask_mod(seq_ids.long().to(device), frame_ids.long().to(device), noise_ids.long().to(device), window_size)
        block_mask = FlexAttnFunc.compiled_create_block_mask(
                mask_mod, 1, 1, len(seq_ids), len(seq_ids), device=device, _compile=True
            )
        FlexAttnFunc.attention_mask = block_mask

        text_seq_ids = torch.arange(B)[:, None].expand(-1, 512).flatten()
        mask_mod_cross = FlexAttnFunc._get_cross_mask_mod(seq_ids.long().to(device), text_seq_ids.long().to(device))
        block_mask_cross = FlexAttnFunc.compiled_create_block_mask(
                mask_mod_cross, 1, 1, len(seq_ids), len(text_seq_ids), device=device, _compile=True
            )
        FlexAttnFunc.cross_attention_mask = block_mask_cross
    
    @staticmethod
    @torch.no_grad()
    def init_mask_proprioception(latent_shape, proprioception_mask, action_mask, action_shape, padded_length,
                     chunk_size, window_size, patch_size, device):
        seq, frame, noise = proprioception_attention_metadata(latent_shape, proprioception_mask, action_mask, action_shape,
                                                    patch_size, chunk_size, device)
        B = latent_shape[0]
        seq = F.pad(seq, (0, padded_length), value=-1)
        frame = F.pad(frame, (0, padded_length), value=-1)
        noise = F.pad(noise, (0, padded_length), value=-1)
        predicate = FlexAttnFunc._get_mask_mod(seq, frame, noise, window_size)
        FlexAttnFunc.attention_mask = FlexAttnFunc.compiled_create_block_mask(
            predicate, 1, 1, len(seq), len(seq), device=device, _compile=True)
        text_seq = torch.arange(B, device=device)[:, None].expand(-1, 512).flatten()
        cross = FlexAttnFunc._get_cross_mask_mod(seq, text_seq)
        FlexAttnFunc.cross_attention_mask = FlexAttnFunc.compiled_create_block_mask(
            cross, 1, 1, len(seq), len(text_seq), device=device, _compile=True)

    @staticmethod
    @torch.no_grad()
    def _get_cross_mask_mod(seq_ids, text_seq_ids):
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_ids[q_idx] == text_seq_ids[kv_idx]) & (seq_ids[q_idx] >=0 ) & (text_seq_ids[kv_idx] >= 0)
        return seq_mask
    
    @staticmethod
    @torch.no_grad()
    def _get_mask_mod(seq_ids, frame_ids, noise_ids, window_size):
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_ids[q_idx] == seq_ids[kv_idx]) & (seq_ids[q_idx] >=0 ) & (seq_ids[kv_idx] >= 0)
        
        def block_causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (frame_ids[kv_idx] <= frame_ids[q_idx])
        
        def block_causal_mask_exclude_self(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (frame_ids[kv_idx] < frame_ids[q_idx])
        
        def block_self_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (frame_ids[kv_idx] == frame_ids[q_idx])
        
        def clean2clean_mask(
                b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 1) & (noise_ids[kv_idx] == 1)
        
        def noise2clean_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 1)
        def noise2noise_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 0)
        
        def block_window_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor, window_size: int
        ):
            return ((frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= window_size)

        mask_list = []
        mask_list.append(and_masks(clean2clean_mask, block_causal_mask))
        mask_list.append(and_masks(noise2clean_mask, block_causal_mask_exclude_self))
        mask_list.append(and_masks(noise2noise_mask, block_self_mask))
        mask = or_masks(*mask_list)
        mask = and_masks(mask, seq_mask)
        mask = and_masks(mask, partial(block_window_mask, window_size=window_size))
        return mask
       
class WanTimeTextImageEmbedding(nn.Module):

    def __init__(
        self,
        dim,
        time_freq_dim,
        time_proj_dim,
        text_embed_dim,
        pos_embed_seq_len,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim,
                                        flip_sin_to_cos=True,
                                        downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim,
                                               time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim,
                                                       dim,
                                                       act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        dtype=None,
    ):
        B, L = timestep.shape
        timestep = timestep.reshape(-1)
        timestep = self.timesteps_proj(timestep)
        # time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).to(dtype=dtype)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb.reshape(B, L, -1), timestep_proj.reshape(B, L, -1)


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Rotary dimensions must be even. This is identical to upstream at
        # head_dim=128 (44/42/42), and supports the 64-dim diagnostic (24/20/20).
        if attention_head_dim < 6 or attention_head_dim % 2:
            raise ValueError("Rotary attention head dimension must be even and >= 6")
        self.h_dim = self.w_dim = 2 * (self.attention_head_dim // 6)
        self.f_dim = self.attention_head_dim - self.h_dim - self.w_dim

        # Precompute and register buffers
        f_freqs_base, h_freqs_base, w_freqs_base = self._precompute_freqs_base()
        self.f_freqs_base = f_freqs_base
        self.h_freqs_base = h_freqs_base
        self.w_freqs_base = w_freqs_base

    def _precompute_freqs_base(self):
        # freqs_base = 1.0 / (theta ** (2k / dim))
        f_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.f_dim, 2)[:(self.f_dim // 2)].double() / self.f_dim))
        h_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.h_dim, 2)[:(self.h_dim // 2)].double() / self.h_dim))
        w_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.w_dim, 2)[:(self.w_dim // 2)].double() / self.w_dim))
        return f_freqs_base, h_freqs_base, w_freqs_base

    def forward(self, grid_ids):
        with torch.no_grad():
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(grid_ids.device)
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(grid_ids.device)
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(grid_ids.device)
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis


class ReferenceAttnFunc(nn.Module):
    """Dense attention oracle for small diagnostic models, never production training."""
    attention_mask = None
    cross_attention_mask = None

    def __init__(self, is_cross=False):
        super().__init__()
        self.is_cross = is_cross

    @staticmethod
    def init_mask(latent_shape, action_shape, padded_length, chunk_size, window_size, patch_size, device):
        B, _, T, H, W = latent_shape
        _, _, AT, AH, AW = action_shape
        video_seq = torch.arange(B, device=device)[:, None].expand(B, T*(H//patch_size[1])*(W//patch_size[2])).flatten()
        action_seq = torch.arange(B, device=device)[:, None].expand(B, AT*AH*AW).flatten()
        video_frame = torch.arange(T, device=device)[None, :, None].expand(B, T, (H//patch_size[1])*(W//patch_size[2])).flatten()
        action_frame = torch.arange(AT, device=device)[None, :, None].expand(B, AT, AH*AW).flatten()
        seq = F.pad(torch.cat([video_seq]*2+[action_seq]*2), (0,padded_length), value=-1)
        frame = F.pad(torch.cat([video_frame//chunk_size*2]*2+[action_frame//chunk_size*2+1]*2), (0,padded_length), value=-1)
        noise = F.pad(torch.cat([torch.zeros_like(video_frame),torch.ones_like(video_frame),torch.zeros_like(action_frame),torch.ones_like(action_frame)]), (0,padded_length), value=-1)
        ids = torch.arange(len(seq), device=device)
        mod = FlexAttnFunc._get_mask_mod(seq, frame, noise, window_size)
        ReferenceAttnFunc.attention_mask = mod(torch.tensor(0,device=device), torch.tensor(0,device=device), ids[:,None], ids[None,:])
        text_seq = torch.arange(B,device=device)[:,None].expand(B,512).flatten()
        mod_cross = FlexAttnFunc._get_cross_mask_mod(seq,text_seq)
        ReferenceAttnFunc.cross_attention_mask = mod_cross(torch.tensor(0,device=device),torch.tensor(0,device=device),ids[:,None],torch.arange(len(text_seq),device=device)[None,:])

    @staticmethod
    def init_mask_proprioception(latent_shape, proprioception_mask, action_mask, action_shape, padded_length,
                     chunk_size, window_size, patch_size, device):
        seq, frame, noise = proprioception_attention_metadata(latent_shape, proprioception_mask, action_mask, action_shape,
                                                    patch_size, chunk_size, device)
        B = latent_shape[0]
        seq = F.pad(seq, (0, padded_length), value=-1)
        frame = F.pad(frame, (0, padded_length), value=-1)
        noise = F.pad(noise, (0, padded_length), value=-1)
        ids = torch.arange(len(seq), device=device)
        predicate = FlexAttnFunc._get_mask_mod(seq, frame, noise, window_size)
        ReferenceAttnFunc.attention_mask = predicate(torch.tensor(0, device=device),
            torch.tensor(0, device=device), ids[:, None], ids[None, :])
        text_seq = torch.arange(B, device=device)[:, None].expand(-1, 512).flatten()
        cross = FlexAttnFunc._get_cross_mask_mod(seq, text_seq)
        ReferenceAttnFunc.cross_attention_mask = cross(torch.tensor(0, device=device),
            torch.tensor(0, device=device), ids[:, None], torch.arange(len(text_seq), device=device)[None, :])

    def forward(self, query, key, value):
        mask = self.cross_attention_mask if self.is_cross else self.attention_mask
        return F.scaled_dot_product_attention(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2), attn_mask=mask).transpose(1,2)


class WanAttention(torch.nn.Module):

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        eps=1e-5,
        dropout=0.0,
        cross_attention_dim_head=None,
        attn_mode='torch',
    ):
        super().__init__()
        if attn_mode == 'torch':
            self.attn_op = custom_sdpa
        elif attn_mode == 'flashattn':
            self.attn_op = flash_attn_func
        elif attn_mode == 'reference':
            self.attn_op = ReferenceAttnFunc(cross_attention_dim_head is not None)
        elif attn_mode == 'flex':
            self.attn_op = FlexAttnFunc(cross_attention_dim_head is not None)
        else:
            raise ValueError(
                f"Unsupported attention mode: {attn_mode}, expected torch, flashattn, flex or diagnostic reference"
            )

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList([
            torch.nn.Linear(self.inner_dim, dim, bias=True),
            torch.nn.Dropout(dropout),
        ])
        self.norm_q = torch.nn.RMSNorm(dim_head * heads,
                                       eps=eps,
                                       elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads,
                                       eps=eps,
                                       elementwise_affine=True)
        self.attn_caches = {} if cross_attention_dim_head is None else None

    def clear_pred_cache(self, cache_name):
        if self.attn_caches is None:
            return
        cache = self.attn_caches[cache_name]
        is_pred = cache['is_pred']
        cache['mask'][is_pred] = False

    def clear_cache(self, cache_name):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = None

    def init_kv_cache(self, cache_name, total_tolen, num_head, head_dim,
                      device, dtype, batch_size):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = {
            'k':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            'v':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            'id':
            torch.full((total_tolen, ), -1, device=device),
            "mask":
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
            "is_pred":
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
        }

    def allocate_slots(self, cache_name, key_size):
        cache = self.attn_caches[cache_name]
        mask = cache["mask"]
        ids = cache["id"]
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            used = mask.nonzero(as_tuple=False).squeeze(-1)

            used_ids = ids[used]
            order = torch.argsort(used_ids)
            # Evict complete forward entries, never an arbitrary token prefix.
            need = key_size - free.numel()
            oldest = used_ids[order]
            cutoff = oldest[need - 1]
            to_free = used[used_ids <= cutoff]
            mask[to_free] = False
            ids[to_free] = -1
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        assert free.numel() >= key_size
        return free[:key_size]

    def _next_cache_id(self, cache_name):
        ids = self.attn_caches[cache_name]['id']
        mask = self.attn_caches[cache_name]['mask']

        if mask.any():
            return ids[mask].max() + 1
        else:
            return torch.tensor(0, device=ids.device, dtype=ids.dtype)

    def update_cache(self, cache_name, key, value, is_pred):
        cache = self.attn_caches[cache_name]

        key_size = key.shape[1]
        slots = self.allocate_slots(cache_name, key_size)

        new_id = self._next_cache_id(cache_name)

        cache['k'][:, slots] = key
        cache['v'][:, slots] = value
        cache['mask'][slots] = True
        cache['id'][slots] = new_id
        cache['is_pred'][slots] = is_pred
        if 'logical_id' in cache:
            cache['logical_id'][slots] = cache['query_logical_id']
        return slots

    def restore_cache(self, cache_name, slots):
        self.attn_caches[cache_name]['mask'][slots] = False

    def forward(
        self,
        q,
        k,
        v,
        rotary_emb,
        update_cache=0,
        cache_name='pos',
    ):
        kv_cache = self.attn_caches[
            cache_name] if (self.attn_caches is not None) and (cache_name in self.attn_caches) else None

        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query)
        query = query.unflatten(2, (self.heads, -1))
        key = self.norm_k(key)
        key = key.unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:

            def apply_rotary_emb(x, freqs):
                x_out = torch.view_as_complex(
                    x.to(torch.float64).reshape(x.shape[0], x.shape[1],
                                                x.shape[2], -1, 2))
                x_out = torch.view_as_real(x_out * freqs).flatten(3)
                return x_out.to(x.dtype)
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        if kv_cache is not None and kv_cache['k'] is not None:
            if update_cache not in (0, 1, 2):
                raise ValueError("Unknown inference cache lifecycle")
            if update_cache:
                self.update_cache(cache_name, key, value, is_pred=(update_cache == 1))
            valid_mask = kv_cache['mask']
            if 'logical_id' in kv_cache:
                logical = kv_cache['query_logical_id']
                valid_mask = valid_mask & (kv_cache['logical_id'] >= logical - kv_cache['window_units'])
                valid_mask = valid_mask & (kv_cache['logical_id'] <= logical)
            valid = valid_mask.nonzero(as_tuple=False).squeeze(-1)
            cached_key, cached_value = kv_cache['k'][:, valid], kv_cache['v'][:, valid]
            if update_cache == 0:
                # Denoising uses scratch K/V. It cannot allocate/evict committed
                # history, even when the persistent cache is completely full.
                key = torch.cat((cached_key, key), dim=1)
                value = torch.cat((cached_value, value), dim=1)
            else:
                key, value = cached_key, cached_value

        hidden_states = self.attn_op(query, key, value)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class WanTransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode: str = "flashattn",
    ):
        super().__init__()
        self.attn_mode = attn_mode

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode=attn_mode,
        )
        self.norm2 = FP32LayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim,
                               inner_dim=ffn_dim,
                               activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        update_cache=0,
        cache_name='pos',
    ) -> torch.Tensor:
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = \
            rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(6, dim=1)
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        c_shift_msa = c_shift_msa.squeeze(1)
        c_scale_msa = c_scale_msa.squeeze(1)
        c_gate_msa = c_gate_msa.squeeze(1)
        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) *
                              (1. + scale_msa) +
                              shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states,
                                 norm_hidden_states,
                                 norm_hidden_states,
                                 rotary_emb,
                                 update_cache=update_cache,
                                 cache_name=cache_name)
        hidden_states = (hidden_states.float() +
                         attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(
            hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states,
                                 encoder_hidden_states,
                                 encoder_hidden_states,
                                 None,
                                 update_cache=0,
                                 cache_name=cache_name)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) *
                              (1. + c_scale_msa) +
                              c_shift_msa).type_as(hidden_states)

        ff_output = self.ffn(norm_hidden_states)

        hidden_states = (hidden_states.float() +
                         ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states


class WanTransformer3DModel(ModelMixin, ConfigMixin):
    r"""
    TODO
    """
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
                                        # "patch_embedding", 
                                        "patch_embedding_mlp",
                                        "condition_embedder", 
                                        'condition_embedder_action',
                                        "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", 
                             "scale_shift_table", 
                             "scale_shift_table_action",
                             "norm1", 
                             'action_norm1',
                             'text_norm1',
                             "norm2", 
                             'action_norm2',
                             'text_norm2',
                             "norm3",
                             'action_norm3',
                             'text_norm3'
                             ]
    _keys_to_ignore_on_load_unexpected = []
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(self,
                 patch_size=[1, 2, 2],
                 num_attention_heads=24,
                 attention_head_dim=128,
                 in_channels=48,
                 out_channels=48,
                 action_dim=30,
                 text_dim=4096,
                 freq_dim=256,
                 ffn_dim=14336,
                 num_layers=30,
                 cross_attn_norm=True,
                 eps=1e-06,
                 rope_max_seq_len=1024,
                 pos_embed_seq_len=None,
                 attn_mode="torch",
                 proprioception_dim=0):
        r"""
        TODO
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        if type(action_dim) is not int or action_dim <= 0:
            raise ValueError("action_dim must be a positive integer")
        inner_dim = num_attention_heads * attention_head_dim
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size,
                                      rope_max_seq_len)
        # Serialized by source 68b7bc1, but not used by upstream forward.
        # Keep canonical keys for strict lossless conversion; freeze in the policy.
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding_mlp = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2],
            inner_dim)
        self.action_embedder = nn.Linear(action_dim, inner_dim)
        if proprioception_dim not in (0, 10):
            raise ValueError("proprioception_dim must be 0 or 10")
        self.proprioception_dim = proprioception_dim
        if proprioception_dim:
            self.proprioception_embedder = nn.Linear(10, inner_dim)
            self.proprioception_proj_out = nn.Linear(inner_dim, 10)
            if self.proprioception_embedder.weight.device.type != "meta":
                generator = torch.Generator(device="cpu").manual_seed(0)
                with torch.no_grad():
                    for layer in (self.proprioception_embedder, self.proprioception_proj_out):
                        initialized = torch.empty(tuple(layer.weight.shape), dtype=torch.float32, device="cpu")
                        nn.init.xavier_uniform_(initialized, gain=1.0, generator=generator)
                        layer.weight.copy_(initialized.to(device=layer.weight.device, dtype=layer.weight.dtype))
                        layer.bias.zero_()
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        self.condition_embedder_action = deepcopy(self.condition_embedder)

        self.blocks = nn.ModuleList([
            WanTransformerBlock(inner_dim,
                                ffn_dim,
                                num_attention_heads,
                                cross_attn_norm,
                                eps,
                                attn_mode=attn_mode) for _ in range(num_layers)
        ])

        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim,
                                  out_channels * math.prod(patch_size))
        self.action_proj_out = nn.Linear(inner_dim, action_dim)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)

    def clear_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_cache(cache_name)

    def clear_pred_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_pred_cache(cache_name)

    def create_empty_cache(self, cache_name, attn_window,
                           latent_token_per_chunk, action_token_per_chunk,
                           device, dtype, batch_size):
        total_tolen = (attn_window // 2) * latent_token_per_chunk + (
            attn_window // 2) * action_token_per_chunk
        for block in self.blocks:
            block.attn1.init_kv_cache(cache_name, total_tolen,
                                      self.num_attention_heads,
                                      self.attention_head_dim, device, dtype, batch_size)
    
    def _input_embed(self, latents, input_type='latent'):
        if input_type == 'latent':
            hidden_states = rearrange(
                latents,
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2])
            hidden_states = self.patch_embedding_mlp(hidden_states)
        elif input_type == 'action':
            hidden_states = rearrange(latents, 'b c f h w -> b (f h w) c')
            hidden_states = self.action_embedder(hidden_states)
        elif input_type == 'proprioception':
            if self.proprioception_dim != 10:
                raise ValueError("This model has no proprioception pathway (proprioception_dim=0)")
            hidden_states = rearrange(latents, 'b c f h w -> b (f h w) c')
            hidden_states = self.proprioception_embedder(hidden_states)
        elif input_type == 'text':
            hidden_states = self.condition_embedder.text_embedder(latents)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        return hidden_states

    def _time_embed(self, timesteps, H, W, dtype, action_mode=False, proprioception_mode=False):
        pach_scale_h, pach_scale_w = (1, 1) if action_mode or proprioception_mode else (
            self.patch_size[1], self.patch_size[2])
        latent_time_steps = torch.repeat_interleave(
            timesteps,
            (H // pach_scale_h) *
            (W // pach_scale_w), dim=1)  # L
        current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C
        return temb, timestep_proj

    def forward_train(self, input_dict):
        if self.proprioception_dim:
            return self.forward_train_proprioception(input_dict)
        if "proprioception_dict" in input_dict:
            raise ValueError("Proprioception inputs require proprioception_dim=10")
        if self.config.attn_mode not in ("flex", "reference"):
            raise ValueError("Joint training requires masked flex attention; reference is diagnostic-only")
        input_dtype = self.patch_embedding_mlp.weight.dtype
        input_dict['latent_dict']['text_emb'] = input_dict['latent_dict']['text_emb'].to(input_dtype)
        input_dict['latent_dict']['noisy_latents'] = input_dict['latent_dict']['noisy_latents'].to(input_dtype)
        input_dict['latent_dict']['latent'] = input_dict['latent_dict']['latent'].to(input_dtype)
        input_dict['action_dict']['noisy_latents'] = input_dict['action_dict']['noisy_latents'].to(input_dtype)
        input_dict['action_dict']['latent'] = input_dict['action_dict']['latent'].to(input_dtype)

        latent_dict = input_dict['latent_dict']
        action_dict = input_dict['action_dict']
        batch_size = latent_dict['noisy_latents'].shape[0]

        latent_hidden_states = self._input_embed(latent_dict['noisy_latents'], input_type='latent').flatten(0, 1)[None]
        action_hidden_states = self._input_embed(action_dict['noisy_latents'], input_type='action').flatten(0, 1)[None]
        text_hidden_states = self._input_embed(latent_dict["text_emb"], input_type='text')

        text_hidden_states = text_hidden_states.flatten(0, 1)[None]

        condition_latent_hidden_states = self._input_embed(latent_dict['latent'], input_type='latent').flatten(0, 1)[None]
        condition_action_hidden_states = self._input_embed(action_dict['latent'], input_type='action').flatten(0, 1)[None]

        hidden_states = torch.cat([latent_hidden_states, 
                                   condition_latent_hidden_states,
                                   action_hidden_states, 
                                   condition_action_hidden_states], dim=1)


        latent_grid_id = latent_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        action_grid_id = action_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        full_grid_id = torch.cat([latent_grid_id] * 2 + [action_grid_id] * 2, dim=2)

        rotary_emb = self.rope(full_grid_id)[:, :, None] 

        latent_time_steps = torch.cat(
            [latent_dict['timesteps'].flatten(0, 1), latent_dict['cond_timesteps'].flatten(0, 1)]
        )[None]
        action_time_steps = torch.cat(
            [action_dict['timesteps'].flatten(0, 1), action_dict['cond_timesteps'].flatten(0, 1)]
        )[None]
        latent_temb, latent_timestep_proj =self._time_embed(latent_time_steps, 
                        latent_dict['noisy_latents'].shape[-2], 
                        latent_dict['noisy_latents'].shape[-1], 
                        dtype=hidden_states.dtype, 
                        action_mode=False)
        action_temb, action_timestep_proj = self._time_embed(action_time_steps,
                        action_dict['noisy_latents'].shape[-2], 
                        action_dict['noisy_latents'].shape[-1], 
                        dtype=hidden_states.dtype, 
                        action_mode=True)
        temb = torch.cat([latent_temb, action_temb], dim=1)
        timestep_proj = torch.cat([latent_timestep_proj, action_timestep_proj], dim=1)

        total_length = hidden_states.shape[1]
        padded_length = (128 - total_length % 128) % 128
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

        split_list = [latent_hidden_states.shape[1], 
                      condition_latent_hidden_states.shape[1], 
                      action_hidden_states.shape[1], 
                      condition_action_hidden_states.shape[1],
                      padded_length]

        mask_type = ReferenceAttnFunc if self.config.attn_mode == 'reference' else FlexAttnFunc
        mask_type.init_mask(latent_dict['noisy_latents'].shape, 
                               action_dict['noisy_latents'].shape, 
                               padded_length, 
                               input_dict["chunk_size"],
                               window_size=input_dict['window_size'],
                               patch_size=self.patch_size,
                               device=hidden_states.device
                               )

        for block in self.blocks:
            hidden_states = block(hidden_states,
                                         text_hidden_states,
                                         timestep_proj,
                                         rotary_emb,
                                         update_cache=False)
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table,
                                 'b l n c -> b n l c').chunk(2, dim=1)
        shift = shift.to(hidden_states.device).squeeze(1)
        scale = scale.to(hidden_states.device).squeeze(1)
        hidden_states = (self.norm_out(hidden_states.float()) *
                                (1. + scale) +
                                shift).type_as(hidden_states)
        latent_hidden_states, _, action_hidden_states, _, _ = torch.split(hidden_states, split_list, dim=1)
        latent_hidden_states = self.proj_out(latent_hidden_states)
        latent_hidden_states = rearrange(latent_hidden_states,
                                             '1 (b l) (n c) -> b (l n) c',
                                             n=math.prod(self.patch_size), b=batch_size)  #
        action_hidden_states = self.action_proj_out(action_hidden_states)
        action_hidden_states = rearrange(action_hidden_states,
                                             '1 (b l) c -> b l c',
                                             b=batch_size)  #

        return latent_hidden_states, action_hidden_states

    def forward_train_proprioception(self, input_dict):
        """Teacher-forced six-segment (video, proprioception, action) shared-DiT training call."""
        if self.config.attn_mode not in ("flex", "reference"):
            raise ValueError("Proprioception training requires causal masked attention")
        if self.proprioception_dim != 10:
            raise ValueError("Proprioception training requires the 10D pathway")
        video = input_dict['latent_dict']
        prop = input_dict['proprioception_dict']
        action = input_dict['action_dict']
        dtype = self.patch_embedding_mlp.weight.dtype
        B = video['noisy_latents'].shape[0]
        if prop['noisy_latents'].shape[:3] != (B, 10, video['noisy_latents'].shape[2]) or                 prop['noisy_latents'].shape[-1] != 1 or                 prop['validity_mask'].shape != (B, 1, prop['noisy_latents'].shape[2], prop['noisy_latents'].shape[3], 1):
            raise ValueError("Video/proprioception temporal shape mismatch")
        text = self._input_embed(video['text_emb'].to(dtype), input_type='text').flatten(0, 1)[None]
        roles = [
            self._input_embed(video['noisy_latents'].to(dtype), 'latent').flatten(0, 1)[None],
            self._input_embed(video['latent'].to(dtype), 'latent').flatten(0, 1)[None],
            self._input_embed(prop['noisy_latents'].to(dtype), 'proprioception').flatten(0, 1)[None],
            self._input_embed(prop['latent'].to(dtype), 'proprioception').flatten(0, 1)[None],
            self._input_embed(action['noisy_latents'].to(dtype), 'action').flatten(0, 1)[None],
            self._input_embed(action['latent'].to(dtype), 'action').flatten(0, 1)[None],
        ]
        hidden = torch.cat(roles, dim=1)
        video_grid = video['grid_id'].permute(1, 0, 2).flatten(1)[None]
        prop_grid = prop['grid_id'].permute(1, 0, 2).flatten(1)[None]
        action_grid = action['grid_id'].permute(1, 0, 2).flatten(1)[None]
        grid = torch.cat([video_grid, video_grid, prop_grid, prop_grid,
                          action_grid, action_grid], dim=2)
        rotary = self.rope(grid)[:, :, None]
        video_t = torch.cat([video['timesteps'].flatten(), video['cond_timesteps'].flatten()])[None]
        prop_t = torch.cat([prop['timesteps'].flatten(), prop['cond_timesteps'].flatten()])[None]
        action_t = torch.cat([action['timesteps'].flatten(), action['cond_timesteps'].flatten()])[None]
        ve, vp = self._time_embed(video_t, *video['noisy_latents'].shape[-2:], dtype=hidden.dtype)
        pe, pp = self._time_embed(prop_t, *prop['noisy_latents'].shape[-2:],
                                  dtype=hidden.dtype, proprioception_mode=True)
        ae, ap = self._time_embed(action_t, *action['noisy_latents'].shape[-2:],
                                  dtype=hidden.dtype, action_mode=True)
        temb = torch.cat([ve, pe, ae], dim=1)
        timestep_proj = torch.cat([vp, pp, ap], dim=1)
        padding = (-hidden.shape[1]) % 128
        hidden = F.pad(hidden, (0, 0, 0, padding))
        rotary = F.pad(rotary, (0, 0, 0, 0, 0, padding))
        temb = F.pad(temb, (0, 0, 0, padding))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padding))
        backend = ReferenceAttnFunc if self.config.attn_mode == 'reference' else FlexAttnFunc
        backend.init_mask_proprioception(video['noisy_latents'].shape, prop['validity_mask'],
                             action['actions_mask'], action['noisy_latents'].shape, padding, input_dict['chunk_size'],
                             input_dict['window_size'], self.patch_size, hidden.device)
        for block in self.blocks:
            hidden = block(hidden, text, timestep_proj, rotary, update_cache=False)
        shift, scale = rearrange(self.scale_shift_table[None] + temb[:, :, None],
                                 'b l n c -> b n l c').chunk(2, dim=1)
        hidden = (self.norm_out(hidden.float()) * (1. + scale.squeeze(1)) +
                  shift.squeeze(1)).type_as(hidden)
        vh, _, ph, _, ah, _, _ = torch.split(hidden, [x.shape[1] for x in roles] + [padding], dim=1)
        video_out = self.proj_out(vh)
        video_out = rearrange(video_out, '1 (b l) (n c) -> b (l n) c',
                               b=B, n=math.prod(self.patch_size))
        prop_out = rearrange(self.proprioception_proj_out(ph), '1 (b l) c -> b l c', b=B)
        action_out = rearrange(self.action_proj_out(ah), '1 (b l) c -> b l c', b=B)
        return video_out, prop_out, action_out

    def forward_embodied(self, video, proprioception, text_emb, video_grid_id,
                         proprioception_grid_id, timesteps, *, update_cache=0,
                         cache_name='pos'):
        """One no-grad-compatible joint state sampling/cache call; action uses native forward."""
        if self.proprioception_dim != 10 or self.config.attn_mode == 'flex':
            if self.proprioception_dim != 10:
                raise ValueError("Proprioception pathway unavailable")
            # Flex mask is a training backend; streaming uses torch/flashattn.
            raise ValueError("Streaming inference requires torch or flashattn attention")
        if video.shape[0] != proprioception.shape[0] or video.shape[2] != proprioception.shape[2]:
            raise ValueError("Embodied state batch/group mismatch")
        dtype = self.patch_embedding_mlp.weight.dtype
        B, _, T, _, _ = video.shape
        V = self._input_embed(video.to(dtype), 'latent')
        P = self._input_embed(proprioception.to(dtype), 'proprioception')
        hidden = torch.cat([V, P], dim=1)
        grid = torch.cat([video_grid_id, proprioception_grid_id], dim=2)
        rotary = self.rope(grid)[:, :, None]
        vt, vp = self._time_embed(timesteps, *video.shape[-2:], dtype=hidden.dtype)
        pt, pp = self._time_embed(timesteps, *proprioception.shape[-2:],
                                  dtype=hidden.dtype, proprioception_mode=True)
        temb, timestep_proj = torch.cat([vt, pt], dim=1), torch.cat([vp, pp], dim=1)
        text = self._input_embed(text_emb.to(dtype), 'text')
        for block in self.blocks:
            hidden = block(hidden, text, timestep_proj, rotary,
                           update_cache=update_cache, cache_name=cache_name)
        shift, scale = rearrange(self.scale_shift_table[None] + temb[:, :, None],
                                 'b l n c -> b n l c').chunk(2, dim=1)
        hidden = (self.norm_out(hidden.float()) * (1. + scale.squeeze(1)) +
                  shift.squeeze(1)).type_as(hidden)
        vh, ph = hidden.split([V.shape[1], P.shape[1]], dim=1)
        video_out = self.proj_out(vh)
        video_out = rearrange(video_out, 'b l (n c) -> b (l n) c', n=math.prod(self.patch_size))
        prop_out = self.proprioception_proj_out(ph)
        return video_out, prop_out

    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        train_mode=False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if train_mode:
            return self.forward_train(input_dict)
        if action_mode:  # action input emb
            latent_hidden_states = rearrange(input_dict['noisy_latents'],
                                             'b c f h w -> b (f h w) c')
            latent_hidden_states = self.action_embedder(
                latent_hidden_states)  # B L1 C
        else:  # latent input emb
            latent_hidden_states = rearrange(
                input_dict['noisy_latents'],
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2])
            latent_hidden_states = self.patch_embedding_mlp(
                latent_hidden_states)
        text_hidden_states = self.condition_embedder.text_embedder(
            input_dict["text_emb"])  # B L2 C

        latent_grid_id = input_dict['grid_id']
        rotary_emb = self.rope(latent_grid_id)[:, :, None]  # 1 L 1 C
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            self.patch_size[1], self.patch_size[2])

        latent_time_steps = torch.repeat_interleave(
            input_dict['timesteps'],
            (input_dict['noisy_latents'].shape[-2] // pach_scale_h) *
            (input_dict['noisy_latents'].shape[-1] // pach_scale_w), dim=1)  # L
        current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=latent_hidden_states.dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C

        for block in self.blocks:
            latent_hidden_states = block(latent_hidden_states,
                                         text_hidden_states,
                                         timestep_proj,
                                         rotary_emb,
                                         update_cache=update_cache,
                                         cache_name=cache_name)
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table,
                                 'b l n c -> b n l c').chunk(2, dim=1)
        shift = shift.to(latent_hidden_states.device).squeeze(1)
        scale = scale.to(latent_hidden_states.device).squeeze(1)
        latent_hidden_states = (self.norm_out(latent_hidden_states.float()) *
                                (1. + scale) +
                                shift).type_as(latent_hidden_states)

        if action_mode:
            latent_hidden_states = self.action_proj_out(latent_hidden_states)
        else:
            latent_hidden_states = self.proj_out(latent_hidden_states)
            latent_hidden_states = rearrange(latent_hidden_states,
                                             'b l (n c) -> b (l n) c',
                                             n=math.prod(self.patch_size))  #

        return latent_hidden_states


if __name__ == '__main__':
    model = WanTransformer3DModel(patch_size=[1, 2, 2],
                                  num_attention_heads=24,
                                  attention_head_dim=128,
                                  in_channels=48,
                                  out_channels=48,
                                  action_dim=30,
                                  text_dim=4096,
                                  freq_dim=256,
                                  ffn_dim=14336,
                                  num_layers=30,
                                  cross_attn_norm=True,
                                  eps=1e-6,
                                  rope_max_seq_len=1024,
                                  pos_embed_seq_len=None,
                                  attn_mode="torch")
    print(model)
