from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def resolve_repo_root(cfg: Mapping[str, Any] | None = None) -> Path:
    repo_root = "."
    if cfg is not None:
        paths_cfg = cfg.get("paths")
        if isinstance(paths_cfg, Mapping):
            configured_root = paths_cfg.get("repo_root")
            if isinstance(configured_root, str) and configured_root:
                repo_root = configured_root

    return Path(repo_root).expanduser().resolve()
