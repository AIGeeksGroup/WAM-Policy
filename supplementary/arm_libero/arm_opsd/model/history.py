"""Facts are durable; role/version KV representations are derived, immutable views."""
from dataclasses import replace
from uuid import uuid4
import time
import torch
from arm_opsd.config import ArmContract
from arm_opsd.records import ContextSnapshot, FeedbackRecord, owned_cpu, plain, tensor_hash


class EpisodeJournal:
    def __init__(self, *, stream_id, session_id, episode_id, task_id, instruction,
                 initial_latent, resource_hash, reset_receipt=None, contract=None):
        self.contract = contract or ArmContract()
        if tuple(initial_latent.shape) != (1, 48, 1, 8, 16):
            raise ValueError("Invalid initial latent")
        self.journal_id = uuid4().hex
        self.stream_id, self.session_id, self.episode_id = stream_id, session_id, episode_id
        self.task_id, self.instruction = int(task_id), instruction
        self.resource_hash = resource_hash
        self.initial_latent = owned_cpu(initial_latent)
        self.reset_receipt = reset_receipt
        self._records = ()
        self._receipt_ids = set()
        self.terminal_receipt = None

    @property
    def records(self):
        return self._records

    def state(self):
        """CPU-only, weights_only-loadable copy of every fact (target cache / replay / held-out)."""
        from dataclasses import asdict
        return {"journal_id": self.journal_id, "stream_id": self.stream_id, "session_id": self.session_id,
                "episode_id": self.episode_id, "task_id": self.task_id, "instruction": self.instruction,
                "resource_hash": self.resource_hash, "initial_latent": self.initial_latent.clone(),
                "reset_receipt": plain(self.reset_receipt), "contract": asdict(self.contract),
                "records": [{"receipt_id": r.receipt_id, "frame_start": r.frame_start,
                             "observed_latents": r.observed_latents.clone(),
                             "normalized_actions": r.normalized_actions.clone(),
                             "applied_actions": r.applied_actions.clone(), "actor_version": r.actor_version}
                            for r in self._records],
                "terminal_receipt": plain(self.terminal_receipt)}

    @classmethod
    def from_state(cls, state):
        contract = ArmContract(**{k: tuple(v) if isinstance(v, list) else v for k, v in state["contract"].items()})
        journal = cls(stream_id=state["stream_id"], session_id=state["session_id"], episode_id=state["episode_id"],
                      task_id=state["task_id"], instruction=state["instruction"],
                      initial_latent=state["initial_latent"], resource_hash=state["resource_hash"],
                      reset_receipt=state["reset_receipt"], contract=contract)
        journal.journal_id = state["journal_id"]
        records = []
        for index, r in enumerate(state["records"]):
            if r["frame_start"] != index * 4:
                raise ValueError("Journal state records are out of order")
            records.append(FeedbackRecord(r["receipt_id"], r["frame_start"], owned_cpu(r["observed_latents"]),
                                          owned_cpu(r["normalized_actions"]), owned_cpu(r["applied_actions"]),
                                          r["actor_version"]))
        journal._records = tuple(records)
        journal._receipt_ids = {r.receipt_id for r in records}
        journal.terminal_receipt = state["terminal_receipt"]
        return journal

    def snapshot(self, *, actor_version, round_id, slot_id):
        return ContextSnapshot(self.stream_id, self.session_id, self.episode_id,
                               self.task_id, self.instruction, self.journal_id,
                               len(self._records), len(self._records) * 4,
                               actor_version, round_id, slot_id, self.resource_hash,
                               self.last_receipt(len(self._records)))

    def append(self, *, receipt_id, latents, normalized_actions, applied_actions, actor_version):
        if receipt_id in self._receipt_ids:
            raise ValueError("Receipt already committed")
        if self.terminal_receipt is not None:
            raise ValueError("Cannot append after terminal")
        first = not self._records
        if tuple(latents.shape) != (1, 48, 3 if first else 4, 8, 16):
            raise ValueError("Only complete causal feedback groups may commit")
        if tuple(normalized_actions.shape) != self.contract.action_shape:
            raise ValueError("Expected native normalized history action")
        if tuple(torch.as_tensor(applied_actions).shape) != (12 if first else 16, 7):
            raise ValueError("Incomplete actual execution")
        state = torch.cat([self.initial_latent.to(latents), latents], dim=2) if first else latents
        for t in (state, normalized_actions, torch.as_tensor(applied_actions)):
            if not torch.isfinite(t).all():
                raise ValueError("Non-finite history")
        record = FeedbackRecord(receipt_id, len(self._records) * 4, owned_cpu(state),
                                owned_cpu(normalized_actions), owned_cpu(torch.as_tensor(applied_actions)),
                                actor_version)
        self._records = self._records + (record,)
        self._receipt_ids.add(receipt_id)
        return record

    def finish(self, receipt):
        if self.terminal_receipt is not None:
            raise ValueError("Episode already terminal")
        self.terminal_receipt = dict(receipt)

    def validate_snapshot(self, context):
        if (context.journal_id != self.journal_id or context.episode_id != self.episode_id
                or context.session_id != self.session_id or context.resource_hash != self.resource_hash
                or context.stream_id != self.stream_id or context.task_id != self.task_id
                or context.instruction != self.instruction or type(context.prefix) is not int
                or not 0 <= context.prefix <= len(self._records) or context.frame_start != context.prefix * 4):
            raise ValueError("Context does not belong to this journal or prefix")
        if context.last_receipt_id != self.last_receipt(context.prefix):
            raise ValueError("Context prefix ends at another receipt than this journal")

    def last_receipt(self, prefix):
        return self._records[prefix - 1].receipt_id if prefix else ""


class HistoryStore:
    def __init__(self, contract=None):
        self.contract = contract or ArmContract()
        self.journals = {}
        self._views = {}
        self.stats = {"requests": 0, "cache_hits": 0, "replayed_chunks": 0,
                      "evicted_tokens": 0, "replay_seconds": 0.0}

    def add(self, journal):
        if journal.journal_id in self.journals:
            raise ValueError("Duplicate journal")
        self.journals[journal.journal_id] = journal

    def merge(self, journal):
        """Register a journal copy (cache/pool/held-out). Copies of one journal differ only in how
        many committed facts they carry, so the longer prefix-consistent copy is kept."""
        existing = self.journals.get(journal.journal_id)
        if existing is None:
            self.add(journal)
            return journal
        if existing is journal:
            return existing
        common = min(len(existing.records), len(journal.records))
        if [r.receipt_id for r in existing.records[:common]] != [r.receipt_id for r in journal.records[:common]]:
            raise ValueError("Journal copies disagree on their common prefix")
        if len(journal.records) <= len(existing.records):
            return existing
        self.journals[journal.journal_id] = journal
        return journal

    def invalidate(self, role):
        name = role if isinstance(role, str) else role.role
        self._views = {k: v for k, v in self._views.items() if k[0] != name}

    def release(self, keep_journals):
        keep = set(keep_journals)
        self.journals = {k: v for k, v in self.journals.items() if k in keep}
        self._views = {k: v for k, v in self._views.items() if k[2] in keep}

    def drop_views(self, journal_ids):
        """Forget the cached views of given journals only (facts stay)."""
        drop = set(journal_ids)
        self._views = {k: v for k, v in self._views.items() if k[2] not in drop}

    def prune_views(self, keep_prefixes):
        """Retain only requested/live prefixes, never every replay intermediate."""
        keep = set(keep_prefixes)
        selected = {}
        for journal, prefix in keep:
            for key in self._views:
                if key[2] == journal and key[3] <= prefix:
                    group = (key[0], key[1], key[2], key[4])
                    if group not in selected or key[3] > selected[group][3]:
                        selected[group] = key
        retained = set(selected.values())
        self._views = {k: v for k, v in self._views.items() if k in retained}

    @torch.no_grad()
    def view(self, role, context, *, text_emb, negative_text_emb=None):
        from arm_opsd.model.kv_plan import CachePlan
        from arm_opsd.model.kv_views import with_metadata
        journal = self.journals[context.journal_id]
        journal.validate_snapshot(context)
        text_key = (tensor_hash(text_emb), tensor_hash(negative_text_emb) if negative_text_emb is not None else None)
        key = (role.role, role.version, journal.journal_id, context.prefix, text_key)
        self.stats["requests"] += 1
        if key in self._views:
            self.stats["cache_hits"] += 1
            return self._views[key]
        if role.device.type == "cuda":
            torch.cuda.synchronize(role.device)
        started = time.perf_counter()
        ancestors = [(k[3], v) for k, v in self._views.items()
                     if k[:3] == key[:3] and k[4] == text_key and k[3] < context.prefix]
        if ancestors:
            start, view = max(ancestors, key=lambda x: x[0])
        else:
            start = 0
            view = role.empty_view(text_emb, negative_text_emb, capacity=self.contract.capacity,
                                   view_key=(role.role, role.version, journal.episode_id, 0))
            self._views[(role.role, role.version, journal.journal_id, 0, text_key)] = view
        for index in range(start, context.prefix):
            record = journal.records[index]
            plan = CachePlan(view.metadata).joint(self.contract.video_tokens, self.contract.action_tokens)
            self.stats["replayed_chunks"] += 1
            self.stats["evicted_tokens"] += len(plan.video.evicted_slots) + len(plan.action.evicted_slots)
            view = with_metadata(view, plan.post_generation)
            view = replace(view, frame_start=record.frame_start)
            view = role.encode_clean(record.observed_latents.to(role.device, role.dtype), view, action=False)
            view = role.encode_clean(record.normalized_actions.to(role.device, role.dtype), view, action=True)
            view = replace(view, frame_start=(index + 1) * 4,
                           view_key=(role.role, role.version, journal.episode_id, index + 1))
        self._views[key] = view
        if role.device.type == "cuda":
            torch.cuda.synchronize(role.device)
        self.stats["replay_seconds"] += time.perf_counter() - started
        return view
