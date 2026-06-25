from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Mapping

from wam_policy.utils.paths import resolve_repo_root


def log_step_metrics(step: int, metrics: Mapping[str, float]) -> None:
    metric_text = " ".join(f"{key}={value:.6g}" for key, value in sorted(metrics.items()))
    print(f"step={step} {metric_text}")


class TrainingLogger:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self._wandb: Any | None = None
        self._wandb_run: Any | None = None

        wandb_cfg = _wandb_config(cfg)
        if not _wandb_enabled(wandb_cfg):
            return

        self._init_wandb(wandb_cfg)

    def log_metrics(self, step: int, metrics: Mapping[str, float]) -> None:
        log_step_metrics(step, metrics)
        if self._wandb is None:
            return
        self._wandb.log(dict(metrics), step=step)

    def log_videos(
        self,
        videos: Mapping[str, str | Path],
        *,
        fps: int,
        step: int | None = None,
    ) -> dict[str, bool]:
        status: dict[str, bool] = {}
        for key, path in sorted(videos.items()):
            print(f"video/{key}: {Path(path)}")
        if self._wandb is None:
            return {key: False for key in videos}
        payload: dict[str, Any] = {}
        for key, raw_path in videos.items():
            path = Path(raw_path)
            suffix = path.suffix.lower().lstrip(".")
            if suffix not in {"mp4", "gif"}:
                print(f"wandb_video_skipped/{key}: unsupported_video_suffix={path.suffix}")
                status[key] = False
                continue
            payload[key] = self._wandb.Video(str(path), fps=fps, format=suffix)
            status[key] = True
        if not payload:
            return status
        if step is None:
            self._wandb.log(payload)
        else:
            self._wandb.log(payload, step=step)
        return status

    def finish(self) -> None:
        if self._wandb is None or self._wandb_run is None:
            return
        self._wandb.finish()
        self._wandb_run = None

    @property
    def run_url(self) -> str | None:
        if self._wandb_run is None:
            return None
        url = getattr(self._wandb_run, "url", None)
        return url if isinstance(url, str) and url else None

    def _init_wandb(self, wandb_cfg: Mapping[str, Any]) -> None:
        try:
            self._wandb = importlib.import_module("wandb")
        except ImportError as exc:
            raise RuntimeError(
                "Weights & Biases logging is enabled, but the 'wandb' package is not "
                "installed. Install wandb or set logging.wandb.enabled=false."
            ) from exc

        wandb_dir = _resolve_wandb_dir(self.cfg, _optional_str(wandb_cfg, "dir", "wandb"))
        wandb_dir.mkdir(parents=True, exist_ok=True)

        init_kwargs: dict[str, Any] = {
            "project": _required_str(wandb_cfg, "project"),
            "mode": _optional_str(wandb_cfg, "mode", "online"),
            "dir": str(wandb_dir),
            "config": _sanitize_for_wandb_config(self.cfg),
        }
        entity = _optional_str_or_none(wandb_cfg, "entity")
        if entity is not None:
            init_kwargs["entity"] = entity
        name = _optional_str_or_none(wandb_cfg, "name")
        if name is not None:
            init_kwargs["name"] = name
        run_id = _optional_str_or_none(wandb_cfg, "id")
        if run_id is not None:
            init_kwargs["id"] = run_id
        resume = _optional_str_or_none(wandb_cfg, "resume")
        if resume is not None:
            init_kwargs["resume"] = resume
        tags = _optional_str_list(wandb_cfg, "tags")
        if tags:
            init_kwargs["tags"] = tags

        self._wandb_run = self._wandb.init(**init_kwargs)


def _wandb_config(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    logging_cfg = cfg.get("logging")
    if not isinstance(logging_cfg, Mapping):
        return {}
    wandb_cfg = logging_cfg.get("wandb")
    return wandb_cfg if isinstance(wandb_cfg, Mapping) else {}


def _wandb_enabled(wandb_cfg: Mapping[str, Any]) -> bool:
    enabled = wandb_cfg.get("enabled", False)
    if enabled == "auto":
        return bool(os.environ.get("WANDB_API_KEY"))
    if not isinstance(enabled, bool):
        raise ValueError("logging.wandb.enabled must be a bool or 'auto'.")
    return enabled


def _resolve_wandb_dir(cfg: Mapping[str, Any], raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = resolve_repo_root(cfg) / path
    return path.resolve()


def _required_str(cfg: Mapping[str, Any], key: str) -> str:
    value = cfg.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"logging.wandb.{key} must be a non-empty string.")
    return value


def _optional_str(cfg: Mapping[str, Any], key: str, default: str) -> str:
    value = cfg.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"logging.wandb.{key} must be a non-empty string.")
    return value


def _optional_str_or_none(cfg: Mapping[str, Any], key: str) -> str | None:
    value = cfg.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"logging.wandb.{key} must be null or a non-empty string.")
    return value


def _optional_str_list(cfg: Mapping[str, Any], key: str) -> list[str]:
    value = cfg.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"logging.wandb.{key} must be a list of strings.")
    return value


def _sanitize_for_wandb_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_secret_key(key_text):
                sanitized[key_text] = "<redacted>"
            else:
                sanitized[key_text] = _sanitize_for_wandb_config(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_for_wandb_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_for_wandb_config(item) for item in value)
    return value


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in ("api_key", "token", "password", "secret"))
