from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from wam_policy.data.action_dataset import (
    ActionChunkManifestDataset,
    action_chunk_collate_fn,
)


def _write_manifest(path: Path, items: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(item) for item in items), encoding="utf-8")


def test_action_chunk_manifest_dataset_loads_action_context_and_condition(
    tmp_path: Path,
) -> None:
    action_path = tmp_path / "actions/a.pt"
    context_path = tmp_path / "actions/context.pt"
    z_condition_path = tmp_path / "actions/z_condition.pt"
    action_path.parent.mkdir(parents=True)
    a_clean = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    context = {"text_emb": torch.ones(3, 5)}
    z_condition = torch.zeros(1, 2, 2, 2)
    torch.save(a_clean, action_path)
    torch.save(context, context_path)
    torch.save(z_condition, z_condition_path)

    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {
                "sample_id": "action_0",
                "a_clean_path": "actions/a.pt",
                "context_path": "actions/context.pt",
                "z_condition_path": "actions/z_condition.pt",
                "shape": [4, 2],
                "dtype": "float32",
                "prompt": "adjust the bottle",
            }
        ],
    )

    sample = ActionChunkManifestDataset(manifest, repo_root=tmp_path)[0]

    assert torch.equal(sample["a_clean"], a_clean)
    assert torch.equal(sample["context"]["text_emb"], context["text_emb"])
    assert torch.equal(sample["z_condition"], z_condition)
    assert sample["prompt"] == "adjust the bottle"
    assert sample["meta"]["sample_id"] == "action_0"
    assert sample["meta"]["a_clean_path"] == action_path.resolve()


def test_action_chunk_manifest_dataset_uses_none_for_optional_files(tmp_path: Path) -> None:
    action_path = tmp_path / "a.pt"
    torch.save(torch.zeros(4, 2), action_path)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [{"sample_id": "action_0", "a_clean_path": "a.pt"}])

    sample = ActionChunkManifestDataset(manifest, repo_root=tmp_path)[0]

    assert sample["context"] is None
    assert sample["z_condition"] is None
    assert sample["prompt"] == ""


def test_action_chunk_manifest_dataset_rejects_shape_mismatch(tmp_path: Path) -> None:
    action_path = tmp_path / "a.pt"
    torch.save(torch.zeros(4, 2), action_path)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [{"sample_id": "action_0", "a_clean_path": "a.pt", "shape": [2, 2]}],
    )
    dataset = ActionChunkManifestDataset(manifest, repo_root=tmp_path)

    with pytest.raises(ValueError, match="shape mismatch"):
        dataset[0]


def test_action_chunk_manifest_dataset_rejects_missing_required_field(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [{"sample_id": "action_0"}])

    with pytest.raises(ValueError, match="a_clean_path"):
        ActionChunkManifestDataset(manifest, repo_root=tmp_path)


def test_action_chunk_collate_fn_stacks_actions_and_optional_condition() -> None:
    batch = [
        {
            "a_clean": torch.zeros(4, 2),
            "context": {"idx": 0},
            "z_condition": torch.ones(1, 2),
            "prompt": "first",
            "meta": {"sample_id": "action_0"},
        },
        {
            "a_clean": torch.ones(4, 2),
            "context": None,
            "z_condition": torch.zeros(1, 2),
            "prompt": "second",
            "meta": {"sample_id": "action_1"},
        },
    ]

    collated = action_chunk_collate_fn(batch)

    assert torch.equal(collated["a_clean"], torch.stack([torch.zeros(4, 2), torch.ones(4, 2)]))
    assert collated["context"] == [{"idx": 0}, None]
    assert torch.equal(collated["z_condition"], torch.stack([torch.ones(1, 2), torch.zeros(1, 2)]))
    assert collated["prompt"] == ["first", "second"]
    assert collated["meta"] == [{"sample_id": "action_0"}, {"sample_id": "action_1"}]


def test_action_chunk_collate_fn_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        action_chunk_collate_fn([])
