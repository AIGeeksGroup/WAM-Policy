from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from wam_policy.adapters.checkpoint import load_student_checkpoint
from wam_policy.adapters.lingbot_va import LingBotVAAdapter
from wam_policy.adapters.robotwin import RoboTwinAdapter
from wam_policy.models.lingbot_student import (
    LingBotActionFlowMapStudent,
    LingBotVideoFlowMapStudent,
)
from wam_policy.utils.logging import TrainingLogger
from wam_policy.utils.paths import resolve_repo_root
from wam_policy.utils.runtime import resolve_device, resolve_torch_dtype
from wam_policy.utils.seed import set_seed
from wam_policy.visualization.video import (
    VideoEncodingResult,
    side_by_side,
    write_metadata,
    write_video_artifact,
)


def run_robotwin_rollout_visualization(cfg: Mapping[str, Any]) -> dict[str, Any]:
    vis_cfg = _robotwin_cfg(cfg)
    set_seed(_int_cfg(vis_cfg, "seed", default=0))

    output_dir = _resolve_path(
        cfg,
        _str_cfg(vis_cfg, "output_dir", default="results/v2/visualization/robotwin_rollout"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    robotwin_adapter = RoboTwinAdapter(cfg)
    if vis_cfg.get("runner") == "external_command":
        result = robotwin_adapter.rollout_student_episode(
            output_dir=output_dir,
            max_chunks=_int_cfg(vis_cfg, "max_chunks", default=32),
            fps=_int_cfg(vis_cfg, "fps", default=10),
            task_name=_optional_str_cfg(vis_cfg, "task_name"),
            seed=_int_cfg(vis_cfg, "seed", default=0),
            return_visualization=True,
        )
    else:
        device = resolve_device(cfg)
        dtype = resolve_torch_dtype(cfg)
        lingbot_adapter = LingBotVAAdapter(cfg)
        video_student = LingBotVideoFlowMapStudent(cfg, lingbot_adapter)
        action_student = LingBotActionFlowMapStudent(cfg, lingbot_adapter)
        video_student.build()
        action_student.build()
        video_student.to(device=device, dtype=dtype)
        action_student.to(device=device, dtype=dtype)
        strict = _bool_cfg(vis_cfg, "checkpoint_strict", default=False)
        load_student_checkpoint(
            video_student,
            _required_str(vis_cfg, "video_checkpoint_path"),
            strict=strict,
            map_location="cpu",
        )
        load_student_checkpoint(
            action_student,
            _required_str(vis_cfg, "action_checkpoint_path"),
            strict=strict,
            map_location="cpu",
        )
        video_student.eval()
        action_student.eval()

        with torch.no_grad():
            result = robotwin_adapter.rollout_student_episode(
                lingbot_adapter=lingbot_adapter,
                video_student=video_student,
                action_student=action_student,
                output_dir=output_dir,
                max_chunks=_int_cfg(vis_cfg, "max_chunks", default=32),
                fps=_int_cfg(vis_cfg, "fps", default=10),
                task_name=_optional_str_cfg(vis_cfg, "task_name"),
                seed=_int_cfg(vis_cfg, "seed", default=0),
                return_visualization=True,
            )

    videos, encoding = _coerce_robotwin_videos(
        result,
        output_dir=output_dir,
        fps=_int_cfg(vis_cfg, "fps", default=10),
    )
    metadata_payload: dict[str, Any] = {
        "videos": {key: str(path) for key, path in videos.items()},
        "encoding": {key: value.to_summary() for key, value in encoding.items()},
        "result_type": type(result).__name__,
    }

    run_url: str | None = None
    wandb_error: str | None = None
    wandb_status: dict[str, bool]
    try:
        logger = TrainingLogger(cfg)
        try:
            wandb_status = logger.log_videos(
                {
                    "robotwin_rollout/actual_observation": videos["actual_observation"],
                    "robotwin_rollout/student_imagined_video": videos["student_imagined_video"],
                    "robotwin_rollout/actual_vs_imagined": videos["actual_vs_imagined"],
                },
                fps=_int_cfg(vis_cfg, "fps", default=10),
            )
            run_url = logger.run_url
        finally:
            logger.finish()
    except Exception as exc:
        wandb_error = f"{type(exc).__name__}: {exc}"
        wandb_status = {
            "robotwin_rollout/actual_observation": False,
            "robotwin_rollout/student_imagined_video": False,
            "robotwin_rollout/actual_vs_imagined": False,
        }
        print(
            "WARNING: W&B video upload failed for RoboTwin visualization; "
            f"local artifacts remain under {output_dir}; error={wandb_error}"
        )

    metadata_payload["wandb_video_status"] = wandb_status
    metadata_payload["wandb_run_url"] = run_url
    if wandb_error is not None:
        metadata_payload["wandb_error"] = wandb_error
    metadata_path = write_metadata(
        output_dir / "robotwin_rollout_metadata.json",
        metadata_payload,
    )

    return {
        "output_dir": output_dir,
        "metadata_path": metadata_path,
        "wandb_run_url": run_url,
        "videos": videos,
        "encoding": encoding,
        "wandb_video_status": wandb_status,
    }


def _coerce_robotwin_videos(
    result: Any,
    *,
    output_dir: Path,
    fps: int,
) -> tuple[dict[str, Path], dict[str, VideoEncodingResult]]:
    if not isinstance(result, Mapping):
        raise TypeError(
            "RoboTwinAdapter.rollout_student_episode must return a mapping containing "
            "video paths or frame arrays for visualization."
        )

    videos: dict[str, Path] = {}
    encoding: dict[str, VideoEncodingResult] = {}
    frame_keys = {
        "actual_observation": "actual_observation_frames",
        "student_imagined_video": "student_imagined_video_frames",
    }
    path_keys = {
        "actual_observation": "actual_observation_video",
        "student_imagined_video": "student_imagined_video",
        "actual_vs_imagined": "actual_vs_imagined_video",
    }

    for output_key, result_key in path_keys.items():
        raw_path = result.get(result_key)
        if isinstance(raw_path, (str, Path)):
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"RoboTwin visualization video does not exist: {path}")
            videos[output_key] = path
            suffix = path.suffix.lower().lstrip(".") or "external"
            encoding[output_key] = VideoEncodingResult(
                path=path,
                backend="external_runner",
                codec=suffix,
                fps=fps,
                succeeded=True,
                fallback_level="external",
            )

    for output_key, result_key in frame_keys.items():
        if output_key in videos or result_key not in result:
            continue
        encoded = write_video_artifact(
            result[result_key],
            output_dir / f"{output_key}.mp4",
            fps=fps,
        )
        videos[output_key] = encoded.path
        encoding[output_key] = encoded

    if "actual_vs_imagined" not in videos:
        actual_frames = result.get("actual_observation_frames")
        imagined_frames = result.get("student_imagined_video_frames")
        if actual_frames is not None and imagined_frames is not None:
            encoded = write_video_artifact(
                side_by_side(actual_frames, imagined_frames),
                output_dir / "actual_vs_imagined.mp4",
                fps=fps,
            )
            videos["actual_vs_imagined"] = encoded.path
            encoding["actual_vs_imagined"] = encoded

    missing = {
        "actual_observation",
        "student_imagined_video",
        "actual_vs_imagined",
    } - set(videos)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(
            "RoboTwin rollout visualization result is missing required videos: "
            f"{joined}. Return MP4 paths or frame arrays from RoboTwinAdapter."
        )
    return videos, encoding


def _robotwin_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    visualization = cfg.get("visualization")
    if not isinstance(visualization, Mapping):
        return {}
    section = visualization.get("robotwin_rollout")
    return section if isinstance(section, Mapping) else {}


def _resolve_path(cfg: Mapping[str, Any], raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = resolve_repo_root(cfg) / path
    return path.resolve()


def _required_str(cfg: Mapping[str, Any], key: str) -> str:
    value = cfg.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"visualization.robotwin_rollout.{key} must be a non-empty string.")
    return value


def _optional_str_cfg(cfg: Mapping[str, Any], key: str) -> str | None:
    value = cfg.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"visualization.robotwin_rollout.{key} must be null or non-empty string.")
    return value


def _str_cfg(cfg: Mapping[str, Any], key: str, *, default: str) -> str:
    value = cfg.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"visualization.robotwin_rollout.{key} must be a non-empty string.")
    return value


def _int_cfg(cfg: Mapping[str, Any], key: str, *, default: int) -> int:
    value = cfg.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"visualization.robotwin_rollout.{key} must be an int.")
    return value


def _bool_cfg(cfg: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = cfg.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"visualization.robotwin_rollout.{key} must be bool.")
    return value
