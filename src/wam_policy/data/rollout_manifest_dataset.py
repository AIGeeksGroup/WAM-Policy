from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from wam_policy.opd.buffers import OnPolicyChunkRecord
from wam_policy.utils.paths import resolve_repo_root


class RolloutManifestDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest_path: str | Path, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.manifest_path = _resolve_path(manifest_path, self.repo_root)
        self.records = _load_rollout_manifest(self.manifest_path, self.repo_root)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "RolloutManifestDataset":
        opd_cfg = cfg.get("opd")
        if not isinstance(opd_cfg, Mapping):
            raise ValueError("Config must contain an 'opd' mapping.")
        manifest = opd_cfg.get("rollout_manifest")
        if not isinstance(manifest, str) or not manifest:
            raise ValueError("Config opd.rollout_manifest must be a non-empty string.")
        return cls(manifest_path=manifest, repo_root=resolve_repo_root(cfg))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "context": _load_optional(record.context_path),
            "z_student": _load_optional(record.z_student_path),
            "a_student": _load_optional(record.a_student_path),
            "z_condition": _load_optional(record.z_condition_path),
            "teacher_video_label": _load_optional(record.teacher_video_label_path),
            "teacher_action_label": _load_optional(record.teacher_action_label_path),
            "record": record,
            "meta": _record_to_meta(record),
        }


def rollout_manifest_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("batch must not be empty.")
    return {
        "context": [item["context"] for item in batch],
        "z_student": _collate_optional_tensor(batch, "z_student"),
        "a_student": _collate_optional_tensor(batch, "a_student"),
        "z_condition": _collate_optional_tensor(batch, "z_condition"),
        "teacher_video_label": _collate_optional_tensor(batch, "teacher_video_label"),
        "teacher_action_label": _collate_optional_tensor(batch, "teacher_action_label"),
        "record": [item["record"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }


def _load_rollout_manifest(path: Path, repo_root: Path) -> list[OnPolicyChunkRecord]:
    records: list[OnPolicyChunkRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                loaded = json.loads(line)
            except JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON: {exc.msg}") from exc
            records.append(_record_from_item(loaded, line_number, repo_root))
    return records


def _record_from_item(item: Any, line_number: int, repo_root: Path) -> OnPolicyChunkRecord:
    if not isinstance(item, dict):
        raise ValueError(f"Line {line_number}: rollout item must be a JSON object.")

    sample_id = _required_str(item, "sample_id", line_number)
    task = _required_str(item, "task", line_number)
    chunk_index = _required_int(item, "chunk_index", line_number)
    env_meta = item.get("env_meta", {})
    if not isinstance(env_meta, dict):
        raise ValueError(f"Line {line_number}: field 'env_meta' must be object.")

    return OnPolicyChunkRecord(
        sample_id=sample_id,
        task=task,
        chunk_index=chunk_index,
        context_path=_optional_resolved_str(item, "context_path", line_number, repo_root),
        z_student_path=_optional_resolved_str(item, "z_student_path", line_number, repo_root),
        a_student_path=_optional_resolved_str(item, "a_student_path", line_number, repo_root),
        z_condition_path=_optional_resolved_str(item, "z_condition_path", line_number, repo_root),
        teacher_video_label_path=_optional_resolved_str(
            item,
            "teacher_video_label_path",
            line_number,
            repo_root,
        ),
        teacher_action_label_path=_optional_resolved_str(
            item,
            "teacher_action_label_path",
            line_number,
            repo_root,
        ),
        env_meta=dict(env_meta),
    )


def _required_str(item: Mapping[str, Any], key: str, line_number: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Line {line_number}: field '{key}' must be a non-empty str.")
    return value


def _required_int(item: Mapping[str, Any], key: str, line_number: int) -> int:
    value = item.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Line {line_number}: field '{key}' must be int.")
    return value


def _optional_resolved_str(
    item: Mapping[str, Any],
    key: str,
    line_number: int,
    repo_root: Path,
) -> str | None:
    value = item.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Line {line_number}: field '{key}' must be str or null.")
    return str(_resolve_path(value, repo_root))


def _load_optional(path: str | None) -> Any:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Rollout manifest referenced file does not exist: {resolved}")
    return torch.load(resolved, map_location="cpu", weights_only=False)


def _collate_optional_tensor(batch: list[dict[str, Any]], key: str) -> torch.Tensor | list[Any] | None:
    values = [item[key] for item in batch]
    if all(value is None for value in values):
        return None
    if all(isinstance(value, torch.Tensor) for value in values):
        return torch.stack(values, dim=0)
    return values


def _record_to_meta(record: OnPolicyChunkRecord) -> dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "task": record.task,
        "chunk_index": record.chunk_index,
        "context_path": record.context_path,
        "z_student_path": record.z_student_path,
        "a_student_path": record.a_student_path,
        "z_condition_path": record.z_condition_path,
        "teacher_video_label_path": record.teacher_video_label_path,
        "teacher_action_label_path": record.teacher_action_label_path,
        "env_meta": dict(record.env_meta),
    }


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved.resolve()
