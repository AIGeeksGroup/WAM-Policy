from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from wam_policy.utils.paths import resolve_repo_root


class ActionChunkManifestDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest_path: str | Path, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.manifest_path = _resolve_path(manifest_path, self.repo_root)
        self.items = _load_action_manifest(self.manifest_path, self.repo_root)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ActionChunkManifestDataset":
        data_cfg = cfg.get("data")
        if not isinstance(data_cfg, Mapping):
            raise ValueError("Config must contain a 'data' mapping.")

        manifest = data_cfg.get("manifest")
        if not isinstance(manifest, str) or not manifest:
            raise ValueError("Config data.manifest must be a non-empty string.")

        return cls(manifest_path=manifest, repo_root=resolve_repo_root(cfg))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        a_clean = _load_tensor(item["a_clean_path"], tensor_name="a_clean")
        _validate_loaded_tensor(a_clean, item, tensor_name="a_clean")

        context_path = item.get("context_path")
        context = _torch_load(context_path) if context_path is not None else None

        z_condition_path = item.get("z_condition_path")
        z_condition = _torch_load(z_condition_path) if z_condition_path is not None else None

        return {
            "a_clean": a_clean,
            "context": context,
            "z_condition": z_condition,
            "prompt": item.get("prompt", ""),
            "meta": dict(item),
        }


def action_chunk_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("batch must not be empty.")

    z_conditions = [item["z_condition"] for item in batch]
    if all(item is None for item in z_conditions):
        z_condition: torch.Tensor | list[Any] | None = None
    elif all(isinstance(item, torch.Tensor) for item in z_conditions):
        z_condition = torch.stack(z_conditions, dim=0)
    else:
        z_condition = z_conditions

    return {
        "a_clean": torch.stack([item["a_clean"] for item in batch], dim=0),
        "context": [item["context"] for item in batch],
        "z_condition": z_condition,
        "prompt": [item["prompt"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }


def _load_action_manifest(path: Path, repo_root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                loaded = json.loads(line)
            except JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON: {exc.msg}") from exc

            item = _validate_action_item(loaded, line_number)
            item["a_clean_path"] = _resolve_path(item["a_clean_path"], repo_root)
            for key in ("context_path", "z_condition_path"):
                raw_path = item.get(key)
                item[key] = _resolve_path(raw_path, repo_root) if raw_path else None
            items.append(item)
    return items


def _validate_action_item(item: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"Line {line_number}: manifest item must be a JSON object.")

    validated = dict(item)
    _require_str(validated, "sample_id", line_number)
    _require_str(validated, "a_clean_path", line_number)
    for key in ("context_path", "z_condition_path", "prompt", "source", "dtype"):
        if key in validated and not isinstance(validated[key], str) and validated[key] is not None:
            raise ValueError(f"Line {line_number}: field '{key}' must be str or null.")
    if "shape" in validated:
        _validate_shape(validated["shape"], line_number)
    return validated


def _require_str(item: Mapping[str, Any], key: str, line_number: int) -> None:
    if key not in item:
        raise ValueError(f"Line {line_number}: missing required field '{key}'.")
    if not isinstance(item[key], str):
        raise ValueError(f"Line {line_number}: field '{key}' must be str.")


def _validate_shape(value: Any, line_number: int) -> None:
    if not isinstance(value, list):
        raise ValueError(f"Line {line_number}: field 'shape' must be list[int].")
    if not all(isinstance(dim, int) and not isinstance(dim, bool) for dim in value):
        raise ValueError(f"Line {line_number}: field 'shape' must be list[int].")


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
