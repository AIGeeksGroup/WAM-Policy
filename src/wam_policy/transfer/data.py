"""Whole-episode training data: the frozen cache and a resumable per-rank episode sampler."""
from __future__ import annotations

import torch

from wam_policy.data.frozen_cache import FrozenEpisodes

__all__ = ["EpisodeSampler", "FrozenEpisodes"]


class EpisodeSampler:
    """Seeded per-epoch permutation of the train episodes, dealt round-robin to the ranks.

    Each rank gets ``len(uids) // world_size`` episodes per epoch (the remainder of the
    permutation is skipped), so all ranks take the same number of steps.
    """

    def __init__(self, uids, *, seed=42, rank=0, world_size=1):
        self.uids = list(uids)
        if len(self.uids) < world_size or not 0 <= rank < world_size:
            raise ValueError("Need at least one train episode per rank")
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.per_rank = len(self.uids) // world_size
        self.epoch, self.cursor = 0, 0

    def order(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(len(self.uids), generator=generator).tolist()
        return permutation[:self.per_rank * self.world_size][self.rank::self.world_size]

    def next(self):
        uid = self.uids[self.order()[self.cursor]]
        self.cursor += 1
        if self.cursor == self.per_rank:
            self.epoch, self.cursor = self.epoch + 1, 0
        return uid

    def state_dict(self):
        return {"seed": self.seed, "rank": self.rank, "world_size": self.world_size,
                "epoch": self.epoch, "cursor": self.cursor, "episodes": len(self.uids)}

    def load_state_dict(self, state):
        for key in ("seed", "rank", "world_size", "episodes"):
            if state[key] != self.state_dict()[key]:
                raise ValueError(f"Sampler state does not match this run ({key}: {state[key]} != "
                                 f"{self.state_dict()[key]})")
        self.epoch, self.cursor = state["epoch"], state["cursor"]
