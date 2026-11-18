"""Value-free simulation of the native physical-slot KV cache.

An update_cache=0 forward is NOT a read-only native operation: it reserves
slots, evicts old entries, writes the query, and only restores the new masks.
This module makes that state transition explicit without storing model tensors.
"""
from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class CacheMetadata:
    capacity: int
    ids: tuple[int, ...] = ()
    mask: tuple[bool, ...] = ()
    is_pred: tuple[bool, ...] = ()

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError("cache capacity must be positive")
        for name, default in (("ids", -1), ("mask", False), ("is_pred", False)):
            value = getattr(self, name)
            if not value:
                object.__setattr__(self, name, (default,) * self.capacity)
            elif len(value) != self.capacity:
                raise ValueError(f"{name} must contain capacity entries")

    @property
    def active_slots(self) -> tuple[int, ...]:
        return tuple(i for i, active in enumerate(self.mask) if active)

    @property
    def size(self) -> int:
        return sum(self.mask)

    def clear_pred(self) -> "CacheMetadata":
        # Native clear_pred_cache changes only mask, retaining stale IDs/flags.
        return replace(self, mask=tuple(m and not p for m, p in zip(self.mask, self.is_pred)))

    def reserve(self, key_size: int, update_cache: int = 0) -> "CacheReservation":
        if update_cache not in (0, 1, 2):
            raise ValueError("native cache update must be 0, 1 (predicted), or 2 (observed)")
        if not 0 < key_size <= self.capacity:
            raise ValueError("query tokens must fit the cache")
        ids, mask, predicted = list(self.ids), list(self.mask), list(self.is_pred)
        free = [i for i, active in enumerate(mask) if not active]
        evicted = ()
        if len(free) < key_size:
            used = self.active_slots
            # Match native torch.argsort, including equal-ID behavior on CPU.
            # Native LIBERO evicts whole write groups (128 video / 16 action).
            order = torch.argsort(torch.tensor([ids[i] for i in used])).tolist()
            evicted = tuple(used[j] for j in order[:key_size - len(free)])
            for slot in evicted:
                mask[slot], ids[slot] = False, -1
            free = [i for i, active in enumerate(mask) if not active]
        slots = tuple(free[:key_size])
        allocated = CacheMetadata(self.capacity, tuple(ids), tuple(mask), tuple(predicted))
        new_id = max((ids[i] for i in allocated.active_slots), default=-1) + 1
        for slot in slots:
            mask[slot], ids[slot], predicted[slot] = True, new_id, update_cache == 1
        written = CacheMetadata(self.capacity, tuple(ids), tuple(mask), tuple(predicted))
        if update_cache == 0:
            for slot in slots:
                mask[slot] = False
        after = CacheMetadata(self.capacity, tuple(ids), tuple(mask), tuple(predicted))
        return CacheReservation(self, allocated, written, after, slots, evicted, new_id)


@dataclass(frozen=True)
class CacheReservation:
    before: CacheMetadata
    after_allocation: CacheMetadata
    after_write: CacheMetadata
    after: CacheMetadata
    slots: tuple[int, ...]
    evicted_slots: tuple[int, ...]
    new_id: int

    @property
    def gather_order(self) -> tuple[int, ...]:
        """Physical gather for concat(surviving_history, current_query)."""
        joined = self.after_allocation.active_slots + self.slots
        index = {slot: i for i, slot in enumerate(joined)}
        return tuple(index[slot] for slot in self.after_write.active_slots)


@dataclass(frozen=True)
class JointCachePlan:
    video: CacheReservation
    video_clean: CacheReservation
    action: CacheReservation
    action_clean: CacheReservation
    post_generation: CacheMetadata


@dataclass(frozen=True)
class CachePlan:
    metadata: CacheMetadata

    def joint(self, video_tokens: int = 128, action_tokens: int = 16,
              video_queries: int = 1, action_queries: int = 2) -> JointCachePlan:
        if video_queries < 1 or action_queries < 1:
            raise ValueError("each modality needs a positive number of denoising queries")
        state = self.metadata
        video = state.reserve(video_tokens, 0)
        state = video.after
        for _ in range(video_queries - 1):
            state = state.reserve(video_tokens, 0).after
        video_clean = state.reserve(video_tokens, 1)
        state = video_clean.after
        action = state.reserve(action_tokens, 0)
        state = action.after
        for _ in range(action_queries - 1):
            state = state.reserve(action_tokens, 0).after
        action_clean = state.reserve(action_tokens, 1)
        return JointCachePlan(video, video_clean, action, action_clean,
                              action_clean.after.clear_pred())
