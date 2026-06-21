from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return loaded


def apply_overrides(
    cfg: Mapping[str, Any],
    overrides: Sequence[str] | None,
) -> dict[str, Any]:
    resolved: dict[str, Any] = deepcopy(dict(cfg))
    if not overrides:
        return resolved

    for override in overrides:
        key_path, value = _split_override(override)
        _set_dotted_value(resolved, key_path, _parse_override_value(value))

    return resolved


def _split_override(override: str) -> tuple[list[str], str]:
    if "=" not in override:
        raise ValueError(f"Override must use KEY=VALUE syntax: {override}")

    key, value = override.split("=", 1)
    key_path = key.split(".")
    if not key or any(part == "" for part in key_path):
        raise ValueError(f"Override key must be a non-empty dotted path: {override}")
    return key_path, value


def _set_dotted_value(cfg: MutableMapping[str, Any], key_path: list[str], value: Any) -> None:
    cursor: MutableMapping[str, Any] = cfg
    for part in key_path[:-1]:
        existing = cursor.get(part)
        if existing is None:
            next_cursor: dict[str, Any] = {}
            cursor[part] = next_cursor
            cursor = next_cursor
            continue
        if not isinstance(existing, MutableMapping):
            joined = ".".join(key_path)
            raise ValueError(f"Cannot apply override {joined}: {part} is not a mapping.")
        cursor = existing

    cursor[key_path[-1]] = value


def _parse_override_value(value: str) -> Any:
    if value == "":
        return ""

    parsed = yaml.safe_load(value)
    if parsed is None and value.lower() not in {"null", "none", "~"}:
        return value
    return parsed
