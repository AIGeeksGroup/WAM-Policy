from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from wam_policy.utils.paths import resolve_repo_root


STATIC_STAGE_REQUIREMENTS = {
    "video_ffm": {
        "data_keys": ("manifest",),
        "extra_sections": (),
    },
    "action_ffm": {
        "data_keys": ("manifest",),
        "extra_sections": (),
    },
    "video_dmd": {
        "data_keys": (),
        "extra_sections": ("opd",),
    },
    "action_dmd": {
        "data_keys": (),
        "extra_sections": ("opd",),
    },
}


def run(cfg: Mapping[str, Any], dry_run: bool = False) -> dict[str, Any] | None:
    experiment = cfg.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError("Config must contain an 'experiment' mapping.")

    stage = experiment.get("stage")
    if stage == "video_ffm":
        if dry_run:
            return _static_stage_check(cfg, stage="video_ffm")
        return train_video_forward_flowmap(cfg)

    if stage == "action_ffm":
        if dry_run:
            return _static_stage_check(cfg, stage="action_ffm")
        return train_action_forward_flowmap(cfg)

    if stage == "video_dmd":
        if dry_run:
            return _static_stage_check(cfg, stage="video_dmd")
        raise NotImplementedError("video_dmd non-dry-run trainer is not implemented yet.")

    if stage == "action_dmd":
        if dry_run:
            return _static_stage_check(cfg, stage="action_dmd")
        raise NotImplementedError("action_dmd non-dry-run trainer is not implemented yet.")

    raise ValueError(f"Unknown stage: {stage}")


def train_video_forward_flowmap(cfg: Mapping[str, Any]) -> dict[str, Any]:
    from wam_policy.opd.stages import train_video_forward_flowmap as _train_video_ffm

    return _train_video_ffm(cfg)


def train_action_forward_flowmap(cfg: Mapping[str, Any]) -> dict[str, Any]:
    from wam_policy.opd.stages import train_action_forward_flowmap as _train_action_ffm

    return _train_action_ffm(cfg)


def _static_stage_check(cfg: Mapping[str, Any], stage: str) -> dict[str, Any]:
    requirements = STATIC_STAGE_REQUIREMENTS.get(stage)
    if requirements is None:
        raise ValueError(f"Unknown stage for static check: {stage}")

    experiment = _required_mapping(cfg, "experiment")
    configured_stage = experiment.get("stage")
    if configured_stage != stage:
        raise ValueError(f"Config experiment.stage must be {stage!r}, got {configured_stage!r}.")

    paths = cfg.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("Config must contain a 'paths' mapping.")

    missing_path_keys = [
        key for key in ("lingbot_repo", "anyflow_repo") if not isinstance(paths.get(key), str)
    ]
    if missing_path_keys:
        joined = ", ".join(missing_path_keys)
        raise ValueError(f"Config paths missing required external repo fields: {joined}")

    _required_mapping(cfg, "teacher")
    _required_mapping(cfg, "model")
    data = _required_mapping(cfg, "data")
    _required_mapping(cfg, "flowmap")
    _required_mapping(cfg, "runtime")
    _required_mapping(cfg, "train")
    for section in requirements["extra_sections"]:
        _required_mapping(cfg, section)
    missing_data_keys = [
        key for key in requirements["data_keys"] if not isinstance(data.get(key), str)
    ]
    if missing_data_keys:
        joined = ", ".join(missing_data_keys)
        raise ValueError(f"Config data missing required fields for {stage}: {joined}")

    if stage in {"video_dmd", "action_dmd"}:
        opd = _required_mapping(cfg, "opd")
        if not isinstance(opd.get("rollout_manifest"), str):
            raise ValueError(f"Config opd.rollout_manifest is required for {stage}.")
        dmd = opd.get("dmd")
        if not isinstance(dmd, Mapping):
            raise ValueError(f"Config opd.dmd mapping is required for {stage}.")

    repo_root = resolve_repo_root(cfg)
    return {
        "pipeline": "v2_anyflow_dual_head",
        "stage": stage,
        "repo_root": str(repo_root),
        "lingbot_repo": str(_resolve_under_repo(repo_root, paths["lingbot_repo"])),
        "anyflow_repo": str(_resolve_under_repo(repo_root, paths["anyflow_repo"])),
    }


def _required_mapping(cfg: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    value = cfg.get(section)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config must contain a '{section}' mapping.")
    return value


def _resolve_under_repo(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()
