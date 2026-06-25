from __future__ import annotations

from typing import Any, Mapping


def static_visualization_check(cfg: Mapping[str, Any], mode: str) -> dict[str, str]:
    _required_mapping(cfg, "paths")
    _required_mapping(cfg, "teacher")
    _required_mapping(cfg, "model")
    _required_mapping(cfg, "flowmap")
    _required_mapping(cfg, "runtime")
    _required_mapping(cfg, "logging")
    visualization = _required_mapping(cfg, "visualization")

    paths = _required_mapping(cfg, "paths")
    for key in ("lingbot_repo", "anyflow_repo"):
        if not isinstance(paths.get(key), str) or not paths[key]:
            raise ValueError(f"Config paths.{key} must be a non-empty string.")

    if mode == "offline-video-stream":
        data = _required_mapping(cfg, "data")
        if not isinstance(data.get("manifest"), str) or not data["manifest"]:
            raise ValueError("Config data.manifest must be a non-empty string.")
        section = _required_nested_mapping(visualization, "offline_video_stream")
        _required_section_str(
            section,
            "student_checkpoint_path",
            "visualization.offline_video_stream",
        )
        return {
            "output_dir": _section_str(
                section,
                "output_dir",
                "results/v2/visualization/offline_video_stream",
            )
        }

    if mode == "robotwin-rollout":
        section = _required_nested_mapping(visualization, "robotwin_rollout")
        _required_section_str(section, "video_checkpoint_path", "visualization.robotwin_rollout")
        _required_section_str(section, "action_checkpoint_path", "visualization.robotwin_rollout")
        runner = section.get("runner", "not_implemented")
        if runner not in {"not_implemented", "external_command"}:
            raise ValueError(
                "visualization.robotwin_rollout.runner must be not_implemented "
                "or external_command."
            )
        if runner == "external_command":
            command = section.get("command")
            valid_command = isinstance(command, str) and bool(command)
            valid_command = valid_command or (
                isinstance(command, list)
                and all(isinstance(item, str) and item for item in command)
            )
            if not valid_command:
                raise ValueError(
                    "visualization.robotwin_rollout.command is required when "
                    "runner=external_command."
                )
        return {
            "output_dir": _section_str(
                section,
                "output_dir",
                "results/v2/visualization/robotwin_rollout",
            )
        }

    raise ValueError(f"Unknown visualization mode: {mode}")


def _required_mapping(cfg: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    value = cfg.get(section)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config must contain a '{section}' mapping.")
    return value


def _required_nested_mapping(parent: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    value = parent.get(section)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config visualization must contain a '{section}' mapping.")
    return value


def _required_section_str(section: Mapping[str, Any], key: str, prefix: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{prefix}.{key} must be a non-empty string.")
    return value


def _section_str(section: Mapping[str, Any], key: str, default: str) -> str:
    value = section.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"visualization.{key} must be a non-empty string.")
    return value
