from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any


REQUIRED_MANIFEST_FIELDS = {
    "sample_id": str,
    "z_clean": str,
}

OPTIONAL_MANIFEST_FIELDS = {
    "context": (str, type(None)),
    "prompt": str,
    "task": str,
    "chunk_idx": int,
    "source": str,
    "shape": list,
    "dtype": str,
}


def load_jsonl_manifest(path: str | Path, repo_root: str | Path) -> list[dict[str, Any]]:
    manifest_path = _resolve_path(path, Path(repo_root).expanduser().resolve())
    repo_root_path = Path(repo_root).expanduser().resolve()
    items: list[dict[str, Any]] = []

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                loaded = json.loads(line)
            except JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON: {exc.msg}") from exc

            item = validate_manifest_item(loaded, line_number)
            item["z_clean_path"] = _resolve_path(item["z_clean"], repo_root_path)
            context = item.get("context")
            item["context_path"] = _resolve_path(context, repo_root_path) if context else None
            items.append(item)

    return items


def validate_manifest_item(item: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"Line {line_number}: manifest item must be a JSON object.")

    validated = dict(item)
    for field, expected_type in REQUIRED_MANIFEST_FIELDS.items():
        if field not in validated:
            raise ValueError(f"Line {line_number}: missing required field '{field}'.")
        _validate_field_type(validated[field], expected_type, field, line_number)

    for field, expected_type in OPTIONAL_MANIFEST_FIELDS.items():
        if field in validated:
            value = validated[field]
            if field == "chunk_idx":
                _validate_int_field(value, field, line_number)
            elif field == "shape":
                _validate_shape(value, line_number)
            else:
                _validate_field_type(value, expected_type, field, line_number)

    return validated


def _resolve_path(path: str | Path | None, repo_root: Path) -> Path:
    if path is None:
        raise ValueError("Cannot resolve a missing path.")
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved.resolve()


def _validate_field_type(
    value: Any,
    expected_type: type | tuple[type, ...],
    field: str,
    line_number: int,
) -> None:
    if not isinstance(value, expected_type):
        expected = _format_expected_type(expected_type)
        raise ValueError(f"Line {line_number}: field '{field}' must be {expected}.")


def _validate_int_field(value: Any, field: str, line_number: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Line {line_number}: field '{field}' must be int.")


def _validate_shape(value: Any, line_number: int) -> None:
    if not isinstance(value, list):
        raise ValueError(f"Line {line_number}: field 'shape' must be list[int].")
    if not all(isinstance(dim, int) and not isinstance(dim, bool) for dim in value):
        raise ValueError(f"Line {line_number}: field 'shape' must be list[int].")


def _format_expected_type(expected_type: type | tuple[type, ...]) -> str:
    if isinstance(expected_type, tuple):
        return " or ".join(_type_name(item) for item in expected_type)
    return _type_name(expected_type)


def _type_name(value: type) -> str:
    if value is type(None):
        return "null"
    return value.__name__
