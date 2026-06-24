from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OnPolicyChunkRecord:
    sample_id: str
    task: str
    chunk_index: int
    context_path: str | None
    z_student_path: str | None
    a_student_path: str | None
    z_condition_path: str | None
    teacher_video_label_path: str | None
    teacher_action_label_path: str | None
    env_meta: dict[str, Any]
