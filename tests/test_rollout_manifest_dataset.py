from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from wam_policy.data.rollout_manifest_dataset import (
    RolloutManifestDataset,
    rollout_manifest_collate_fn,
)


def _write_manifest(path: Path, items: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(item) for item in items), encoding="utf-8")


def test_rollout_manifest_dataset_loads_optional_tensors(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    context_path = rollout_dir / "context.pt"
    z_student_path = rollout_dir / "z_student.pt"
    a_student_path = rollout_dir / "a_student.pt"
    z_condition_path = rollout_dir / "z_condition.pt"
    torch.save({"text_emb": torch.ones(3, 4)}, context_path)
    torch.save(torch.zeros(1, 2, 2, 2), z_student_path)
    torch.save(torch.ones(4, 14), a_student_path)
    torch.save(torch.full((1, 2), 2.0), z_condition_path)

    manifest = tmp_path / "rollouts.jsonl"
    _write_manifest(
        manifest,
        [
            {
                "sample_id": "rollout_0",
                "task": "adjust_bottle",
                "chunk_index": 3,
                "context_path": "rollouts/context.pt",
                "z_student_path": "rollouts/z_student.pt",
                "a_student_path": "rollouts/a_student.pt",
                "z_condition_path": "rollouts/z_condition.pt",
                "teacher_video_label_path": None,
                "teacher_action_label_path": None,
                "env_meta": {"success": False, "seed": 7},
            }
        ],
    )

    sample = RolloutManifestDataset(manifest, repo_root=tmp_path)[0]

    assert sample["record"].sample_id == "rollout_0"
    assert sample["record"].chunk_index == 3
    assert torch.equal(sample["context"]["text_emb"], torch.ones(3, 4))
    assert sample["z_student"].shape == (1, 2, 2, 2)
    assert sample["a_student"].shape == (4, 14)
    assert sample["z_condition"].shape == (1, 2)
    assert sample["teacher_video_label"] is None
    assert sample["meta"]["env_meta"] == {"success": False, "seed": 7}


def test_rollout_manifest_dataset_rejects_missing_required_field(tmp_path: Path) -> None:
    manifest = tmp_path / "rollouts.jsonl"
    _write_manifest(manifest, [{"sample_id": "rollout_0", "task": "task"}])

    with pytest.raises(ValueError, match="chunk_index"):
        RolloutManifestDataset(manifest, repo_root=tmp_path)


def test_rollout_manifest_dataset_rejects_missing_tensor_file(tmp_path: Path) -> None:
    manifest = tmp_path / "rollouts.jsonl"
    _write_manifest(
        manifest,
        [
            {
                "sample_id": "rollout_0",
                "task": "task",
                "chunk_index": 0,
                "z_student_path": "missing.pt",
            }
        ],
    )
    dataset = RolloutManifestDataset(manifest, repo_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        dataset[0]


def test_rollout_manifest_collate_fn_stacks_optional_tensors() -> None:
    batch = [
        {
            "context": {"idx": 0},
            "z_student": torch.zeros(1, 2),
            "a_student": torch.ones(4, 2),
            "z_condition": None,
            "teacher_video_label": None,
            "teacher_action_label": torch.ones(4, 2),
            "record": "record_0",
            "meta": {"sample_id": "rollout_0"},
        },
        {
            "context": None,
            "z_student": torch.ones(1, 2),
            "a_student": torch.zeros(4, 2),
            "z_condition": None,
            "teacher_video_label": None,
            "teacher_action_label": torch.zeros(4, 2),
            "record": "record_1",
            "meta": {"sample_id": "rollout_1"},
        },
    ]

    collated = rollout_manifest_collate_fn(batch)

    assert collated["context"] == [{"idx": 0}, None]
    assert torch.equal(collated["z_student"], torch.stack([torch.zeros(1, 2), torch.ones(1, 2)]))
    assert torch.equal(collated["a_student"], torch.stack([torch.ones(4, 2), torch.zeros(4, 2)]))
    assert collated["z_condition"] is None
    assert collated["teacher_video_label"] is None
    assert torch.equal(
        collated["teacher_action_label"],
        torch.stack([torch.ones(4, 2), torch.zeros(4, 2)]),
    )


def test_rollout_manifest_collate_fn_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        rollout_manifest_collate_fn([])
