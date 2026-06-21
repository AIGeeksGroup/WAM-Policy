from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from wam_policy.data.manifests import load_jsonl_manifest
from wam_policy.utils.paths import resolve_repo_root


class VideoLatentManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        repo_root: str | Path,
        latent_root: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.manifest_path = Path(manifest_path)
        self.latent_root = (
            _resolve_path(latent_root, self.repo_root) if latent_root is not None else None
        )
        self.items = load_jsonl_manifest(self.manifest_path, self.repo_root)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "VideoLatentManifestDataset":
        data_cfg = cfg.get("data")
        if not isinstance(data_cfg, Mapping):
            raise ValueError("Config must contain a 'data' mapping.")

        manifest = data_cfg.get("manifest")
        if not isinstance(manifest, str) or not manifest:
            raise ValueError("Config data.manifest must be a non-empty string.")

        latent_root = data_cfg.get("latent_root")
        if latent_root is not None and not isinstance(latent_root, str):
            raise ValueError("Config data.latent_root must be a string when provided.")

        return cls(
            manifest_path=manifest,
            repo_root=resolve_repo_root(cfg),
            latent_root=latent_root,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        z_clean = _load_tensor(item["z_clean_path"], tensor_name="z_clean")
        _validate_loaded_tensor(z_clean, item, tensor_name="z_clean")

        context_path = item.get("context_path")
        context = _torch_load(context_path) if context_path is not None else None

        return {
            "z_clean": z_clean,
            "context": context,
            "prompt": item.get("prompt", ""),
            "meta": dict(item),
        }


def video_latent_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("batch must not be empty.")

    return {
        "z_clean": torch.stack([item["z_clean"] for item in batch], dim=0),
        "context": [item["context"] for item in batch],
        "prompt": [item["prompt"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }


def _load_tensor(path: Path, tensor_name: str) -> torch.Tensor:
    loaded = _torch_load(path)
    if not isinstance(loaded, torch.Tensor):
        raise TypeError(f"{tensor_name} file must contain a torch.Tensor: {path}")
    return loaded


def _torch_load(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest referenced file does not exist: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _validate_loaded_tensor(
    tensor: torch.Tensor,
    item: Mapping[str, Any],
    tensor_name: str,
) -> None:
    expected_shape = item.get("shape")
    if expected_shape is not None and tuple(tensor.shape) != tuple(expected_shape):
        raise ValueError(
            f"{tensor_name} shape mismatch for {item.get('sample_id')}: "
            f"{tuple(tensor.shape)} != {tuple(expected_shape)}"
        )

    expected_dtype = item.get("dtype")
    if expected_dtype is not None and _dtype_name(tensor.dtype) != expected_dtype:
        raise ValueError(
            f"{tensor_name} dtype mismatch for {item.get('sample_id')}: "
            f"{_dtype_name(tensor.dtype)} != {expected_dtype}"
        )


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype)
    return name.removeprefix("torch.")


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved.resolve()
