from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

from wam_policy.utils.paths import resolve_repo_root


EXTERNAL_REPO_KEYS = ("lingbot_repo", "anyflow_repo")


def bootstrap_external_repos(
    cfg: Mapping[str, Any],
    require_exists: bool = False,
) -> list[Path]:
    repo_root = resolve_repo_root(cfg)
    paths_cfg = cfg.get("paths")
    if not isinstance(paths_cfg, Mapping):
        raise ValueError("Config must contain a 'paths' mapping.")

    resolved_paths: list[Path] = []
    for key in EXTERNAL_REPO_KEYS:
        raw_path = paths_cfg.get(key)
        if not isinstance(raw_path, str) or raw_path == "":
            raise ValueError(f"Config paths.{key} must be a non-empty string.")

        path = _resolve_external_path(repo_root, raw_path)
        if require_exists and not path.is_dir():
            raise FileNotFoundError(f"Configured external repo does not exist: {path}")

        _append_sys_path_once(path)
        resolved_paths.append(path)

    return resolved_paths


def _resolve_external_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _append_sys_path_once(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)
