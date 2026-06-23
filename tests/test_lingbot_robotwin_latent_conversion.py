from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts.tools.convert_lingbot_robotwin_latents import (
    convert_lingbot_robotwin_latents,
)
from wam_policy.data.video_latent_dataset import VideoLatentManifestDataset


def _write_lingbot_latent(path: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    frames, height, width, channels = 2, 3, 4, 5
    item: dict[str, Any] = {
        "latent": torch.arange(frames * height * width * channels, dtype=torch.bfloat16).reshape(
            frames * height * width,
            channels,
        ),
        "latent_num_frames": frames,
        "latent_height": height,
        "latent_width": width,
        "text_emb": torch.ones(7, 8, dtype=torch.float32),
        "text": "adjust the bottle",
        "frame_ids": [0, 1],
        "start_frame": 0,
        "end_frame": 1,
        "fps": 10,
        "ori_fps": 30,
    }
    if overrides:
        item.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(item, path)
    return item


def test_convert_lingbot_robotwin_latent_writes_split_files_and_manifest(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "datasets/raw/robotwin"
    source_path = source_root / "latents/adjust_bottle/episode_000/chunk_000.pth"
    source_item = _write_lingbot_latent(source_path)
    output_root = tmp_path / "datasets/latents/robotwin_video_ffm_debug"
    manifest_path = tmp_path / "datasets/manifests/video_latents_debug.jsonl"

    samples = convert_lingbot_robotwin_latents(
        repo_root=tmp_path,
        source_root=source_root,
        output_root=output_root,
        manifest_path=manifest_path,
        max_samples=16,
        overwrite=True,
    )

    assert len(samples) == 1
    z_clean_path = output_root / "adjust_bottle/episode_000/chunk_000_z.pt"
    context_path = output_root / "adjust_bottle/episode_000/chunk_000_context.pt"
    assert z_clean_path.is_file()
    assert context_path.is_file()

    z_clean = torch.load(z_clean_path, map_location="cpu", weights_only=False)
    expected = source_item["latent"].reshape(2, 3, 4, 5).permute(3, 0, 1, 2).contiguous()
    assert torch.equal(z_clean, expected)
    assert tuple(z_clean.shape) == (5, 2, 3, 4)

    context = torch.load(context_path, map_location="cpu", weights_only=False)
    assert torch.equal(context["text_emb"], source_item["text_emb"])
    assert context["text"] == "adjust the bottle"
    assert context["source_latent"] == "latents/adjust_bottle/episode_000/chunk_000.pth"
    assert context["frame_ids"] == [0, 1]

    manifest_items = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_items == [
        {
            "context": "datasets/latents/robotwin_video_ffm_debug/adjust_bottle/episode_000/chunk_000_context.pt",
            "dtype": "bfloat16",
            "prompt": "adjust the bottle",
            "sample_id": "adjust_bottle__episode_000__chunk_000",
            "shape": [5, 2, 3, 4],
            "source": "datasets/raw/robotwin/latents/adjust_bottle/episode_000/chunk_000.pth",
            "z_clean": "datasets/latents/robotwin_video_ffm_debug/adjust_bottle/episode_000/chunk_000_z.pt",
        }
    ]

    dataset = VideoLatentManifestDataset(manifest_path, repo_root=tmp_path)
    sample = dataset[0]
    assert torch.equal(sample["z_clean"], expected)
    assert torch.equal(sample["context"]["text_emb"], source_item["text_emb"])


def test_convert_lingbot_robotwin_latent_respects_max_samples(tmp_path: Path) -> None:
    source_root = tmp_path / "raw"
    _write_lingbot_latent(source_root / "latents/task/sample_0.pth")
    _write_lingbot_latent(source_root / "latents/task/sample_1.pth")

    samples = convert_lingbot_robotwin_latents(
        repo_root=tmp_path,
        source_root=source_root,
        output_root=tmp_path / "latents_out",
        manifest_path=tmp_path / "manifest.jsonl",
        max_samples=1,
        overwrite=True,
    )

    assert len(samples) == 1
    assert samples[0].sample_id == "task__sample_0"


def test_convert_lingbot_robotwin_latent_dry_run_writes_nothing(tmp_path: Path) -> None:
    source_root = tmp_path / "raw"
    _write_lingbot_latent(source_root / "latents/task/sample_0.pth")
    output_root = tmp_path / "latents_out"
    manifest_path = tmp_path / "manifest.jsonl"

    samples = convert_lingbot_robotwin_latents(
        repo_root=tmp_path,
        source_root=source_root,
        output_root=output_root,
        manifest_path=manifest_path,
        max_samples=16,
        overwrite=False,
        dry_run=True,
    )

    assert len(samples) == 1
    assert not output_root.exists()
    assert not manifest_path.exists()


def test_convert_lingbot_robotwin_latent_rejects_missing_required_key(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "raw"
    source_path = source_root / "latents/task/sample_0.pth"
    source_item = _write_lingbot_latent(source_path)
    del source_item["text_emb"]
    torch.save(source_item, source_path)

    with pytest.raises(ValueError, match="missing required keys"):
        convert_lingbot_robotwin_latents(
            repo_root=tmp_path,
            source_root=source_root,
            output_root=tmp_path / "latents_out",
            manifest_path=tmp_path / "manifest.jsonl",
            max_samples=16,
            overwrite=True,
        )


def test_convert_lingbot_robotwin_latent_rejects_token_count_mismatch(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "raw"
    _write_lingbot_latent(
        source_root / "latents/task/sample_0.pth",
        overrides={"latent": torch.zeros(3, 4)},
    )

    with pytest.raises(ValueError, match="token count"):
        convert_lingbot_robotwin_latents(
            repo_root=tmp_path,
            source_root=source_root,
            output_root=tmp_path / "latents_out",
            manifest_path=tmp_path / "manifest.jsonl",
            max_samples=16,
            overwrite=True,
        )
