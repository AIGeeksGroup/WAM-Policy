from __future__ import annotations

import json
from pathlib import Path

import pytest

from wam_policy.data.manifests import load_jsonl_manifest, validate_manifest_item


def _write_jsonl(path: Path, items: list[object]) -> None:
    path.write_text("\n".join(json.dumps(item) for item in items), encoding="utf-8")


def test_load_jsonl_manifest_validates_and_resolves_paths(tmp_path: Path) -> None:
    absolute_latent = tmp_path / "absolute_z.pt"
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "sample_id": "sample_0",
                "z_clean": "latents/sample_0.pt",
                "context": "latents/sample_0_context.pt",
                "prompt": "adjust the bottle",
                "shape": [1, 16, 21, 32, 32],
                "dtype": "bf16",
            },
            {
                "sample_id": "sample_1",
                "z_clean": str(absolute_latent),
                "context": None,
            },
        ],
    )

    items = load_jsonl_manifest(manifest, repo_root=tmp_path)

    assert len(items) == 2
    assert items[0]["z_clean_path"] == (tmp_path / "latents/sample_0.pt").resolve()
    assert items[0]["context_path"] == (tmp_path / "latents/sample_0_context.pt").resolve()
    assert items[1]["z_clean_path"] == absolute_latent.resolve()
    assert items[1]["context_path"] is None


@pytest.mark.parametrize(
    "item",
    [
        {"z_clean": "z.pt"},
        {"sample_id": "sample_0"},
    ],
)
def test_validate_manifest_item_requires_sample_id_and_z_clean(item: object) -> None:
    with pytest.raises(ValueError, match="Line 4: missing required field"):
        validate_manifest_item(item, line_number=4)


@pytest.mark.parametrize(
    "item",
    [
        ["not", "an", "object"],
        {"sample_id": 3, "z_clean": "z.pt"},
        {"sample_id": "sample_0", "z_clean": 3},
        {"sample_id": "sample_0", "z_clean": "z.pt", "chunk_idx": True},
        {"sample_id": "sample_0", "z_clean": "z.pt", "shape": [1, False]},
    ],
)
def test_validate_manifest_item_rejects_invalid_types(item: object) -> None:
    with pytest.raises(ValueError, match="Line 2:"):
        validate_manifest_item(item, line_number=2)


def test_load_jsonl_manifest_rejects_invalid_json_with_line_number(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"sample_id": "ok", "z_clean": "z.pt"}\n{bad json}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Line 2: invalid JSON"):
        load_jsonl_manifest(manifest, repo_root=tmp_path)
