from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from wam_policy.utils.paths import resolve_repo_root


UNIMPLEMENTED_STAGES = {"action_ffm", "video_dmd", "action_dmd"}


def run(cfg: Mapping[str, Any], dry_run: bool = False) -> dict[str, Any] | None:
    experiment = cfg.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError("Config must contain an 'experiment' mapping.")

    stage = experiment.get("stage")
    if stage == "video_ffm":
        if dry_run:
            return _static_video_ffm_check(cfg)
        return train_video_forward_flowmap(cfg)

    if stage in UNIMPLEMENTED_STAGES:
        raise NotImplementedError(f"{stage} is not implemented yet. Current scope: video_ffm.")

    raise ValueError(f"Unknown stage: {stage}")


def train_video_forward_flowmap(cfg: Mapping[str, Any]) -> dict[str, Any]:
    from wam_policy.opd.stages import train_video_forward_flowmap as _train_video_ffm

    return _train_video_ffm(cfg)


def _static_video_ffm_check(cfg: Mapping[str, Any]) -> dict[str, Any]:
    paths = cfg.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("Config must contain a 'paths' mapping.")

    missing_path_keys = [
        key for key in ("lingbot_repo", "anyflow_repo") if not isinstance(paths.get(key), str)
    ]
    if missing_path_keys:
        joined = ", ".join(missing_path_keys)
        raise ValueError(f"Config paths missing required external repo fields: {joined}")

    repo_root = resolve_repo_root(cfg)
    return {
        "pipeline": "v2_anyflow_dual_head",
        "stage": "video_ffm",
        "repo_root": str(repo_root),
        "lingbot_repo": str(_resolve_under_repo(repo_root, paths["lingbot_repo"])),
        "anyflow_repo": str(_resolve_under_repo(repo_root, paths["anyflow_repo"])),
    }


def _resolve_under_repo(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()
