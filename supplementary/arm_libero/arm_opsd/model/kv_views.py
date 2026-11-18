"""Immutable role-specific KV views and differentiable native forwards.

No call in this module reads or writes ``attn_caches``. History tensors are
detached; current query K/V stay in the autograd graph until its endpoint loss.
"""
from dataclasses import dataclass, replace
from typing import Any

import torch
from einops import rearrange

from arm_opsd.model.kv_plan import CacheMetadata


@dataclass(frozen=True)
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor

    def __post_init__(self):
        if self.key.shape != self.value.shape or self.key.ndim != 4:
            raise ValueError("KV tensors must have equal [batch,tokens,heads,head_dim] shape")
        if self.key.requires_grad or self.value.requires_grad:
            raise ValueError("persistent history KV must be detached")


@dataclass(frozen=True)
class KVView:
    metadata: CacheMetadata
    layers: tuple[LayerKV, ...]
    frame_start: int = 0
    text_emb: torch.Tensor | None = None
    negative_text_emb: torch.Tensor | None = None
    view_key: Any = None
    role: str = ""
    parameter_version: int = -1

    def __post_init__(self):
        if self.frame_start < 0:
            raise ValueError("absolute frame_start cannot be negative")
        for layer in self.layers:
            if layer.key.shape[1] != self.metadata.size:
                raise ValueError("active-only layer KV must match metadata.active_slots")


def with_metadata(view: KVView, metadata: CacheMetadata) -> KVView:
    """Filter an existing value view; metadata alone cannot create new KV."""
    if metadata.capacity != view.metadata.capacity:
        raise ValueError("cache capacities differ")
    old = {slot: i for i, slot in enumerate(view.metadata.active_slots)}
    for slot in metadata.active_slots:
        if slot not in old or metadata.ids[slot] != view.metadata.ids[slot]:
            raise ValueError("new active cache entries require captured model K/V")
    layers = tuple(LayerKV(
        layer.key.index_select(1, torch.tensor([old[s] for s in metadata.active_slots],
                                             dtype=torch.long, device=layer.key.device)),
        layer.value.index_select(1, torch.tensor([old[s] for s in metadata.active_slots],
                                               dtype=torch.long, device=layer.value.device)),
    ) for layer in view.layers)
    return replace(view, metadata=metadata, layers=layers)


def reserve(view: KVView, tokens: int, update_cache: int = 0) -> KVView:
    """Return the post-query view: eviction persists, temporary masks do not."""
    if update_cache != 0:
        raise ValueError("value-free reserve only supports update_cache=0; use encode_clean to commit")
    return with_metadata(view, view.metadata.reserve(tokens, 0).after)


def clear_predictions(view: KVView) -> KVView:
    return with_metadata(view, view.metadata.clear_pred())


def _mesh(value: torch.Tensor, patch_size, frame_start: int, action: bool):
    _, _, frames, height, width = value.shape
    pf, ph, pw = (1, 1, 1) if action else tuple(patch_size)
    if frames % pf or height % ph or width % pw:
        raise ValueError("value shape is not divisible by native patch_size")
    ff, hh, ww = torch.meshgrid(
        torch.arange(frame_start, frame_start + frames // pf, device=value.device),
        torch.arange(height // ph, device=value.device),
        torch.arange(width // pw, device=value.device), indexing="ij")
    if action:
        offset = torch.arange(1, height + 1, device=value.device) / (height + 1)
        ff = ff + offset.view(1, -1, 1)
        hh = torch.full_like(hh, -1)
        ww = torch.full_like(ww, -1)
    grid = torch.stack((ff, hh, ww)).flatten(1)
    return torch.cat((grid, torch.full_like(grid[:1], int(action))), dim=0)


class NativeRole:
    """Explicit native role API used by actor, learner, and history replay.

    ``velocity`` preserves current-query autograd and does not mutate its view.
    ``encode_clean`` captures a new detached value view under no_grad. The model
    is deliberately unwrapped: DDP belongs around a loss unit, not each query.
    """
    def __init__(self, model, role: str = "student", version: int = 0, patch_size=None):
        if role not in ("teacher", "student"):
            raise ValueError("role must be teacher or student")
        self.model, self.role, self.version = model, role, int(version)
        self.patch_size = tuple(patch_size or model.patch_size)

    @property
    def device(self):
        return self.model.patch_embedding_mlp.weight.device

    @property
    def dtype(self):
        return self.model.patch_embedding_mlp.weight.dtype

    def empty_view(self, text_emb, negative_text_emb=None, frame_start=0,
                   capacity=2160, view_key=None) -> KVView:
        if text_emb.ndim != 3 or text_emb.shape[0] != 1:
            raise ValueError("one-stream text_emb must be [1,text_tokens,text_dim]")
        if negative_text_emb is not None and negative_text_emb.shape != text_emb.shape:
            raise ValueError("positive and negative prompt embedding shapes differ")
        batch = 2 if negative_text_emb is not None else 1
        shape = (batch, 0, self.model.num_attention_heads, self.model.attention_head_dim)
        layers = tuple(LayerKV(torch.empty(shape, device=self.device, dtype=self.dtype),
                               torch.empty(shape, device=self.device, dtype=self.dtype))
                       for _ in self.model.blocks)
        return KVView(CacheMetadata(capacity), layers, frame_start,
                      text_emb.detach().clone(),
                      None if negative_text_emb is None else negative_text_emb.detach().clone(),
                      view_key, self.role, self.version)

    def _validate(self, view):
        if view.role and (view.role != self.role or view.parameter_version != self.version):
            raise ValueError("KV role/version mismatch: rebuild from facts after parameter updates")
        if len(view.layers) != len(self.model.blocks):
            raise ValueError("KV layer count differs from model")
        if view.text_emb is None:
            raise ValueError("view requires prompt embeddings")

    def _prepare(self, value, timesteps, view, action):
        self._validate(view)
        if value.ndim != 5 or value.shape[0] != 1:
            raise ValueError("native streaming value must be [1,channels,frames,height,width]")
        value = value.to(device=self.device, dtype=self.dtype)
        timesteps = torch.as_tensor(timesteps, device=self.device, dtype=torch.float32)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(value.shape[2])
        if timesteps.shape == (1, value.shape[2]):
            timesteps = timesteps[0]
        if timesteps.shape != (value.shape[2],):
            raise ValueError("timesteps must be scalar, [frames], or [1,frames]")
        batch = 2 if view.negative_text_emb is not None else 1
        prompt = view.text_emb.to(device=self.device, dtype=self.dtype)
        if batch == 2:
            prompt = torch.cat((prompt, view.negative_text_emb.to(prompt)), dim=0)
        grid = _mesh(value, self.patch_size, view.frame_start, action)
        tokens = grid.shape[-1]
        reservation = view.metadata.reserve(tokens, 0)
        history = with_metadata(view, reservation.after_allocation)
        order = torch.tensor(reservation.gather_order, device=self.device, dtype=torch.long)
        functional_kv = tuple((layer.key, layer.value, order) for layer in history.layers)
        input_dict = {"noisy_latents": value.expand(batch, -1, -1, -1, -1),
                      "timesteps": timesteps[None].expand(batch, -1),
                      "grid_id": grid[None].expand(batch, -1, -1),
                      "text_emb": prompt}
        return value, input_dict, functional_kv, reservation

    def velocity(self, value, timesteps, view: KVView, action=False, guidance=1.0):
        if guidance != 1.0 and view.negative_text_emb is None:
            raise ValueError("classifier-free guidance requires a negative prompt branch")
        value, inputs, kv, _ = self._prepare(value, timesteps, view, action)
        packed = self.model(inputs, action_mode=action, functional_kv=kv)
        if action:
            output = rearrange(packed, "b (f h w) c -> b c f h w",
                               f=value.shape[2], h=value.shape[3], w=value.shape[4])
        else:
            pf, ph, pw = self.patch_size
            output = rearrange(packed,
                "b (f h w pt ph pw) c -> b c (f pt) (h ph) (w pw)",
                f=value.shape[2] // pf, h=value.shape[3] // ph,
                w=value.shape[4] // pw, pt=pf, ph=ph, pw=pw)
        if output.shape[0] == 2 and guidance != 1.0:
            return output[1:] + guidance * (output[:1] - output[1:])
        return output[:1]

    @torch.no_grad()
    def encode_clean(self, value, view: KVView, action=False, predicted=False) -> KVView:
        value, inputs, kv, _ = self._prepare(value, 0.0, view, action)
        _, captured = self.model(inputs, action_mode=action, functional_kv=kv, capture_kv=True)
        tokens = inputs["grid_id"].shape[-1]
        transition = view.metadata.reserve(tokens, 1 if predicted else 2)
        survivor = with_metadata(view, transition.after_allocation)
        order = torch.tensor(transition.gather_order, device=self.device, dtype=torch.long)
        layers = tuple(LayerKV(
            torch.cat((old.key, key.detach()), dim=1).index_select(1, order).detach().clone(),
            torch.cat((old.value, val.detach()), dim=1).index_select(1, order).detach().clone(),
        ) for old, (key, val) in zip(survivor.layers, captured))
        return replace(view, metadata=transition.after, layers=layers)

    def reserve(self, view: KVView, tokens: int, update_cache: int = 0) -> KVView:
        self._validate(view)
        return reserve(view, tokens, update_cache)

