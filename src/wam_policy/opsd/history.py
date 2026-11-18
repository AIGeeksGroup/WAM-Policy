"""Immutable generation prefixes over an append-only, real-feedback CPU journal."""
from dataclasses import dataclass
import hashlib
import copy
import torch
from wam_policy.opsd.config import digest

STATE_SHAPE = (1, 48, 1, 24, 20)
ACTION_SHAPE = (1, 9, 1, 16, 1)


def cpu_copy(value):
    # Call outside inference_mode: autograd may save these constants later.
    with torch.inference_mode(False), torch.no_grad():
        return value.detach().to("cpu").clone().contiguous()


def tensor_hash(value):
    v = value.detach().cpu().contiguous()
    return hashlib.sha256(str((str(v.dtype), tuple(v.shape))).encode() +
                          v.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class HistorySnapshot:
    episode: tuple
    prefix_end: int
    group: int
    prefix_digest: str
    actor_version: int
    round_id: int
    observation_id: str
    tick: int
    contract_digest: str

    @property
    def key(self):
        return digest(self.__dict__)

    def to_dict(self):
        return {**self.__dict__, "episode": list(self.episode)}

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "episode": tuple(value["episode"])})


class EpisodeJournal:
    def __init__(self, episode, initial, text, *, contract=None):
        if tuple(initial.shape) != STATE_SHAPE or not torch.isfinite(initial).all():
            raise ValueError("Real initial dual-view latent required")
        self.episode = tuple(episode)
        self._initial, self._text = cpu_copy(initial), cpu_copy(text)
        self.contract = copy.deepcopy(contract or {})
        self.contract_digest = digest(self.contract)
        self._entries = []
        self._digests = [digest([self.episode, tensor_hash(self._initial), tensor_hash(self._text),
                                 self.contract_digest])]
        self.terminal = False

    @property
    def length(self):
        return len(self._entries)

    def append(self, latent, normalized_executed, *, tick, receipt):
        if self.terminal or tuple(latent.shape) != STATE_SHAPE or tuple(normalized_executed.shape) != ACTION_SHAPE:
            raise ValueError("Complete actual feedback group required")
        if tick != (self.length + 1) * 160:
            raise ValueError("Journal must retain actual 500Hz simulation chronology")
        if not torch.isfinite(latent).all() or not torch.isfinite(normalized_executed).all():
            raise FloatingPointError("Nonfinite observed history")
        if torch.any(normalized_executed[:,6:]!=0):
            raise ValueError('Executed force-zero history required; raw generation targets belong outside the journal')
        z, a = cpu_copy(latent), cpu_copy(normalized_executed)
        entry_digest = digest([tensor_hash(z), tensor_hash(a), tick, receipt])
        self._entries.append((z, a, tick, str(receipt), entry_digest))
        self._digests.append(digest([self._digests[-1], entry_digest]))

    def snapshot(self, *, actor_version, round_id, observation_id, tick):
        if self.terminal or tick != self.length * 160:
            raise ValueError("Generation must begin at a real complete group boundary")
        return HistorySnapshot(self.episode, self.length, self.length + 1, self._digests[-1],
                               actor_version, round_id, observation_id, tick, self.contract_digest)

    def read(self, snapshot, *, digests=False):
        """Verified prefix; with digests, also the content digest after each of its groups."""
        if (snapshot.episode != self.episode or snapshot.group != snapshot.prefix_end + 1 or
                not 0 <= snapshot.prefix_end <= self.length or snapshot.contract_digest != self.contract_digest or
                snapshot.tick != snapshot.prefix_end * 160):
            raise ValueError("History prefix does not belong to this journal")
        h = digest([self.episode, tensor_hash(self._initial), tensor_hash(self._text), self.contract_digest])
        entries, chain = [], [h]
        for z, a, tick, receipt, expected in self._entries[:snapshot.prefix_end]:
            actual = digest([tensor_hash(z), tensor_hash(a), tick, receipt])
            if actual != expected:
                raise ValueError("Immutable journal tensor was mutated")
            h = digest([h, actual])
            chain.append(h)
            entries.append((cpu_copy(z), cpu_copy(a)))
        if h != snapshot.prefix_digest:
            raise ValueError("Immutable history prefix changed")
        result = cpu_copy(self._initial), cpu_copy(self._text), tuple(entries)
        return (*result, tuple(chain)) if digests else result

    def prefix_state(self, prefix_end):
        """CPU tensors and contract of the first prefix_end groups, enough to rebuild this prefix
        (target cache entries, held-out shards). The future beyond the prefix is never stored."""
        if not 0 <= prefix_end <= self.length:
            raise ValueError("Prefix beyond the journal")
        return {"episode": list(self.episode), "initial": cpu_copy(self._initial), "text": cpu_copy(self._text),
                "contract": copy.deepcopy(self.contract),
                "entries": [(cpu_copy(z), cpu_copy(a), tick, receipt) for z, a, tick, receipt, _ in self._entries[:prefix_end]]}

    @classmethod
    def from_prefix_state(cls, value):
        journal = cls(tuple(value["episode"]), value["initial"], value["text"], contract=value["contract"])
        for z, a, tick, receipt in value["entries"]:
            journal.append(z, a, tick=tick, receipt=receipt)
        return journal


@dataclass(frozen=True)
class ContextRecord:
    snapshot: HistorySnapshot
    journal: EpisodeJournal
    requested: object = None
    executed: object = None
    terminal: object = None
    rng_before: object = None
    model_normalized: object = None
    # Actor-side initial noise (video, action) and the sampled video plan of this chunk; a
    # teacher actor's own sample is the regression target of the cache it builds.
    noise: object = None
    plan: object = None

    def validate(self):
        self.journal.read(self.snapshot)
        return self
