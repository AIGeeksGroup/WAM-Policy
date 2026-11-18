"""Role/version-specific native history replay, scratch queries and detached plans.

Every context replays its observed history into a fresh KV cache before its queries. With
``config.history_reuse`` (default off) a HistoryStore keeps the pure-history KV of recently replayed
prefixes and a new context restores the longest stored prefix of its own history, then commits only
the remaining groups: the same operations on the same values as a full replay, so results are equal.
"""
from collections import Counter, OrderedDict
from contextlib import nullcontext
import itertools
import torch
from wan_va.utils import get_mesh_id
from wam_policy.opsd.history import ACTION_SHAPE

_NAMES = itertools.count()
HISTORY_CAPACITY = 8


def _copy(caches):
    return [{k: v.clone() if torch.is_tensor(v) else v for k, v in c.items()} for c in caches]


class HistoryStore:
    """Detached pure-history KV states, keyed by (role version, verified prefix digest).

    A state is a cache after replaying a prefix and nothing else: no plan or query writes.
    Restoring it and committing the remaining groups runs the operations of a complete
    replay, so attention reads the same masked slots in the same order. Entries are not
    cache leases; a version advance or a parameter restore clears them."""
    def __init__(self, capacity=HISTORY_CAPACITY):
        self.capacity, self.entries, self.stats = capacity, OrderedDict(), Counter()

    def clear(self):
        self.entries.clear()

    def get(self, key):
        state = self.entries.get(key)
        if state is not None:
            self.entries.move_to_end(key)
        return state

    def put(self, key, caches):
        if key in self.entries:
            self.entries.move_to_end(key)
            return False
        with torch.no_grad():
            self.entries[key] = _copy(caches)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return True


class RoleRuntime:
    def __init__(self, model, role, config, version=0, *, history=False, experts=None):
        self.model, self.role, self.config, self.version = model, role, config, version
        # M-OPD specialist teachers on this model (ExpertBank, teacher only) or None. A context's task
        # fixes its adapter, and a journal's digests bind its task, so stored history never mixes experts.
        self.experts = experts
        self.device = model.patch_embedding_mlp.weight.device
        self.dtype = model.patch_embedding_mlp.weight.dtype
        self.counts = Counter()
        self.active = set()
        self.history = HistoryStore() if history and config.history_reuse else None
        if model.config.attn_mode != "torch" or model.config.action_dim != 9:
            raise ValueError("Cached queries require a native9 model with torch SDPA attention")

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()

    def prepare(self, record):
        return QueryContext(self, record)

    def advance(self):
        if self.active:
            raise RuntimeError("Cannot change parameters/version with live cache leases")
        self.forget()
        self.version += 1

    def forget(self):
        """Drop stored history; required whenever parameters change."""
        if self.history is not None:
            self.history.clear()

    def remember(self, context, prefix_digest):
        """Share a pure-history state of this version, e.g. the actor's cache at predict entry."""
        if self.history is not None and self.history.put((self.version, prefix_digest), context.caches()):
            self.history.stats["shared"] += 1


class QueryContext:
    def __init__(self, runtime, record):
        self.runtime, self.model, self.version = runtime, runtime.model, runtime.version
        self.name = f"opsd:{runtime.role}:{runtime.version}:{next(_NAMES)}"
        self.group = record.snapshot.group
        self.task = record.journal.contract.get("task")
        self.closed, self.sealed, self.plan_set = False, False, False
        self.plan_digest = None
        z, text, entries, chain = record.journal.read(record.snapshot, digests=True)
        self.text = text.to(runtime.device, runtime.dtype)
        runtime.active.add(self.name)
        try:
            with torch.no_grad():
                start = self._restore(chain)
                if start is None:
                    self.model.create_empty_cache(self.name, runtime.config.window_units + 4, 120, 16,
                                                  runtime.device, runtime.dtype, 1)
                    for cache in self.caches():
                        cache["logical_id"] = torch.full_like(cache["id"], -1)
                    self.forward(z, 0., group=0, update=2, count="history")
                    self.forward(torch.zeros(ACTION_SHAPE), 0., group=0, action=True, update=2, count="history")
                    start = 0
                for group, (latent, action) in enumerate(entries[start:], start + 1):
                    self.commit(latent, action, group)
                if runtime.history is not None:
                    runtime.history.stats["replayed_groups"] += len(entries) - start
                    runtime.history.put((self.version, chain[-1]), self.caches())
        except BaseException:
            self.close()
            raise

    def _restore(self, chain):
        """Longest stored prefix of this exact history and version; None for a full replay."""
        store = self.runtime.history
        if store is None:
            return None
        store.stats["contexts"] += 1
        for prefix in range(len(chain) - 1, -1, -1):
            state = store.get((self.version, chain[prefix]))
            if state is not None:
                for block, cache in zip(self.model.blocks, _copy(state)):
                    block.attn1.attn_caches[self.name] = cache
                store.stats["restored"] += 1
                store.stats["reused_groups"] += prefix
                return prefix
        return None

    def caches(self):
        if self.closed or self.version != self.runtime.version:
            raise RuntimeError("Closed or stale-role cache")
        return [b.attn1.attn_caches[self.name] for b in self.model.blocks]

    def forward(self, value, timestep, *, group=None, action=False, update=0, count="query"):
        group = self.group if group is None else group
        if update and (self.sealed or torch.is_grad_enabled()):
            raise RuntimeError("Conditioning cache writes must precede graph construction and run in no_grad")
        if not update and torch.is_grad_enabled():
            self.sealed = True
        runtime = self.runtime
        for cache in self.caches():
            cache["query_logical_id"] = 2 * group + int(action)
            cache["window_units"] = runtime.config.window_units
        grid = get_mesh_id(1, 16 if action else 12, 1 if action else 10,
                           int(action), 1, group, action=action).to(runtime.device)
        payload = {"noisy_latents": value.to(runtime.device, runtime.dtype), "text_emb": self.text,
                   "timesteps": torch.full((1, 1), float(timestep), device=runtime.device, dtype=torch.float32),
                   "grid_id": grid[None]}
        runtime.counts[count + ("_action" if action else "_state")] += 1
        if runtime.experts is not None:
            runtime.experts.select(self.task)
        with runtime.autocast():
            return self.model(payload, update_cache=update, cache_name=self.name,
                              action_mode=action, train_mode=False)

    @torch.no_grad()
    def commit(self, latent, action, group):
        self.model.clear_pred_cache(self.name)
        self.forward(latent, 0., group=group, update=2, count="history")
        self.forward(action, 0., group=group, action=True, update=2, count="history")
        minimum = 2 * max(0, group + 1 - self.runtime.config.window_units // 2)
        for cache in self.caches():
            expired = cache["mask"] & (cache["logical_id"] < minimum)
            cache["mask"][expired] = False
            cache["id"][expired] = -1

    @torch.no_grad()
    def condition(self, plan):
        if self.plan_set:
            raise RuntimeError("Exactly one completed student state plan per action context")
        from wam_policy.opsd.history import tensor_hash
        self.plan_digest = tensor_hash(plan)
        self.forward(plan.detach(), 0., update=1, count="plan_cache")
        self.plan_set = True
        for cache in self.caches():
            if cache["k"].requires_grad or cache["v"].requires_grad or cache["k"].is_inference():
                raise RuntimeError("Conditioning KV must be ordinary detached tensors")

    def summary(self):
        c = self.caches()[0]
        return {name: sorted(set(c["logical_id"][c["mask"] & pred].cpu().tolist()))
                for name, pred in (("observed_units", ~c["is_pred"]), ("predicted_units", c["is_pred"]))}

    def close(self):
        if not self.closed:
            for block in self.model.blocks:
                block.attn1.attn_caches.pop(self.name, None)
            self.runtime.active.discard(self.name)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
