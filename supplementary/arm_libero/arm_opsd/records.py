"""Context and feedback records shared by collection, learner and the teacher cache."""
from dataclasses import dataclass
import hashlib
import torch


def tensor_hash(value):
    t = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((str(t.dtype), tuple(t.shape))).encode())
    digest.update(t.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def owned_cpu(value):
    return value.detach().to("cpu").contiguous().clone()


def plain(value):
    """JSON-like copy (tensors/arrays -> lists) for records that must survive weights_only loads."""
    import numpy as np
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


@dataclass(frozen=True)
class ContextSnapshot:
    stream_id: str
    session_id: str
    episode_id: str
    task_id: int
    instruction: str
    journal_id: str
    prefix: int
    frame_start: int
    actor_version: int
    round_id: int
    slot_id: int
    resource_hash: str
    # Receipt id of the last committed record of the prefix ("" for the empty prefix).
    last_receipt_id: str = ""

    @property
    def first(self):
        return self.prefix == 0

    @property
    def key(self):
        return (self.session_id, self.episode_id, self.prefix, self.frame_start,
                self.actor_version, self.round_id, self.slot_id, self.resource_hash)


@dataclass(frozen=True)
class FeedbackRecord:
    receipt_id: str
    frame_start: int
    observed_latents: torch.Tensor
    normalized_actions: torch.Tensor
    applied_actions: torch.Tensor
    actor_version: int


@dataclass(frozen=True)
class CollectedContext:
    snapshot: ContextSnapshot
    receipt_id: str
    executed_actions: int
    terminal: bool
