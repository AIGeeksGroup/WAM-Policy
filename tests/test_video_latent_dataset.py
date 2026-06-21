from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from wam_policy.data.video_latent_dataset import (
    VideoLatentManifestDataset,
    video_latent_collate_fn,
)


def _write_manifest(path: Path, items: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(item) for item in items), encoding="utf-8")


def test_video_latent_manifest_dataset_loads_latent_and_context(tmp_path: Path) -> None:
    latent_path = tmp_path / "latents/z.pt"
    context_path = tmp_path / "latents/context.pt"
    latent_path.parent.mkdir(parents=True)
    z_clean = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    context = {"tokens": torch.tensor([1, 2, 3])}
    torch.save(z_clean, latent_path)
    torch.save(context, context_path)

    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {
                "sample_id": "sample_0",
                "z_clean": "latents/z.pt",
                "context": "latents/context.pt",
                "prompt": "adjust the bottle",
                "shape": [1, 1, 2, 2],
                "dtype": "float32",
            }
        ],
    )

    dataset = VideoLatentManifestDataset(manifest, repo_root=tmp_path)
    sample = dataset[0]

    assert len(dataset) == 1
    assert torch.equal(sample["z_clean"], z_clean)
    assert torch.equal(sample["context"]["tokens"], context["tokens"])
    assert sample["prompt"] == "adjust the bottle"
    assert sample["meta"]["sample_id"] == "sample_0"
    assert sample["meta"]["z_clean_path"] == latent_path.resolve()
    assert sample["meta"]["context_path"] == context_path.resolve()


def test_video_latent_manifest_dataset_uses_none_for_missing_context(tmp_path: Path) -> None:
    latent_path = tmp_path / "z.pt"
    torch.save(torch.zeros(1), latent_path)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [{"sample_id": "sample_0", "z_clean": "z.pt"}])

    sample = VideoLatentManifestDataset(manifest, repo_root=tmp_path)[0]

    assert sample["context"] is None
    assert sample["prompt"] == ""


def test_video_latent_manifest_dataset_rejects_missing_latent_file(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [{"sample_id": "sample_0", "z_clean": "missing.pt"}])
    dataset = VideoLatentManifestDataset(manifest, repo_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        dataset[0]


def test_video_latent_manifest_dataset_rejects_shape_mismatch(tmp_path: Path) -> None:
    latent_path = tmp_path / "z.pt"
    torch.save(torch.zeros(2, 2), latent_path)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [{"sample_id": "sample_0", "z_clean": "z.pt", "shape": [1, 2]}],
    )
    dataset = VideoLatentManifestDataset(manifest, repo_root=tmp_path)

    with pytest.raises(ValueError, match="shape mismatch"):
        dataset[0]


def test_video_latent_manifest_dataset_rejects_dtype_mismatch(tmp_path: Path) -> None:
    latent_path = tmp_path / "z.pt"
    torch.save(torch.zeros(2, dtype=torch.float32), latent_path)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [{"sample_id": "sample_0", "z_clean": "z.pt", "dtype": "bfloat16"}],
    )
    dataset = VideoLatentManifestDataset(manifest, repo_root=tmp_path)

    with pytest.raises(ValueError, match="dtype mismatch"):
        dataset[0]


def test_video_latent_collate_fn_stacks_latents_and_keeps_lists(tmp_path: Path) -> None:
    batch = [
        {
            "z_clean": torch.zeros(1, 2),
            "context": {"idx": 0},
            "prompt": "first",
            "meta": {"sample_id": "sample_0"},
        },
        {
            "z_clean": torch.ones(1, 2),
            "context": None,
            "prompt": "second",
            "meta": {"sample_id": "sample_1"},
        },
    ]

    collated = video_latent_collate_fn(batch)

    assert torch.equal(collated["z_clean"], torch.stack([torch.zeros(1, 2), torch.ones(1, 2)]))
    assert collated["context"] == [{"idx": 0}, None]
    assert collated["prompt"] == ["first", "second"]
    assert collated["meta"] == [{"sample_id": "sample_0"}, {"sample_id": "sample_1"}]


def test_video_latent_collate_fn_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        video_latent_collate_fn([])
