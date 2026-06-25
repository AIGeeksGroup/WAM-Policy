from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from wam_policy.opd.buffers import OnPolicyChunkRecord
from wam_policy.utils.paths import resolve_repo_root


class RoboTwinAdapter:
    """Adapter boundary for server-side RoboTwin rollout integration."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.repo_root = resolve_repo_root(cfg)

    def make_env(self) -> Any:
        raise NotImplementedError(
            "RoboTwin rollout must run on the server/compute node with external/RoboTwin installed."
        )

    def rollout_student_episode(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        del args
        rollout_cfg = _robotwin_rollout_cfg(self.cfg)
        runner = rollout_cfg.get("runner", "not_implemented")
        if runner == "external_command":
            return self._rollout_with_external_command(rollout_cfg, kwargs)
        raise NotImplementedError(
            "RoboTwin rollout must run on the server/compute node with external/RoboTwin "
            "installed. Set visualization.robotwin_rollout.runner=external_command and "
            "visualization.robotwin_rollout.command to a server rollout command, or implement "
            "an in-process adapter binding. The command must write a JSON result containing "
            "actual_observation_video, student_imagined_video, and actual_vs_imagined_video."
        )

    def save_rollout_manifest(
        self,
        records: Iterable[OnPolicyChunkRecord],
        path: str | Path,
    ) -> Path:
        output_path = Path(path).expanduser()
        if not output_path.is_absolute():
            output_path = self.repo_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), sort_keys=True))
                handle.write("\n")
        return output_path.resolve()

    def _rollout_with_external_command(
        self,
        rollout_cfg: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        output_dir = Path(kwargs["output_dir"]).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = _resolve_output_path(
            rollout_cfg.get("result_json"),
            output_dir / "robotwin_rollout_result.json",
            output_dir,
        )
        request_path = output_dir / "robotwin_rollout_request.json"
        stdout_path = output_dir / "robotwin_rollout_stdout.log"
        stderr_path = output_dir / "robotwin_rollout_stderr.log"

        request = {
            "output_dir": str(output_dir),
            "result_json": str(result_path),
            "max_chunks": kwargs.get("max_chunks"),
            "fps": kwargs.get("fps"),
            "task_name": kwargs.get("task_name"),
            "seed": kwargs.get("seed"),
            "teacher_checkpoint_path": _nested_str(self.cfg, "teacher", "checkpoint_path"),
            "video_checkpoint_path": rollout_cfg.get("video_checkpoint_path"),
            "action_checkpoint_path": rollout_cfg.get("action_checkpoint_path"),
        }
        with request_path.open("w", encoding="utf-8") as handle:
            json.dump(request, handle, indent=2, sort_keys=True)
            handle.write("\n")

        command = _command_args(rollout_cfg)
        env = dict(os.environ)
        env.update(
            {
                "WAM_ROBOTWIN_REQUEST_JSON": str(request_path),
                "WAM_ROBOTWIN_RESULT_JSON": str(result_path),
                "WAM_ROBOTWIN_OUTPUT_DIR": str(output_dir),
            }
        )
        timeout = _optional_int(rollout_cfg.get("timeout_seconds"))
        cwd = _optional_cwd(rollout_cfg.get("workdir"), self.repo_root)
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w",
            encoding="utf-8",
        ) as stderr:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                check=False,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                "RoboTwin external rollout command failed. "
                f"exit_code={completed.returncode} stdout={stdout_path} stderr={stderr_path}"
            )
        if not result_path.is_file():
            raise FileNotFoundError(
                "RoboTwin external rollout command completed but did not write result JSON: "
                f"{result_path}"
            )
        with result_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"RoboTwin result JSON must contain an object: {result_path}")
        result = dict(loaded)
        result.setdefault("stdout_log", str(stdout_path))
        result.setdefault("stderr_log", str(stderr_path))
        result.setdefault("request_json", str(request_path))
        return result


def _robotwin_rollout_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    visualization = cfg.get("visualization")
    if not isinstance(visualization, Mapping):
        return {}
    rollout = visualization.get("robotwin_rollout")
    return rollout if isinstance(rollout, Mapping) else {}


def _command_args(rollout_cfg: Mapping[str, Any]) -> list[str]:
    command = rollout_cfg.get("command")
    if isinstance(command, str) and command:
        return shlex.split(command)
    if isinstance(command, list) and all(isinstance(item, str) and item for item in command):
        return list(command)
    raise ValueError(
        "visualization.robotwin_rollout.command must be a non-empty string or list[str] "
        "when runner=external_command."
    )


def _resolve_output_path(raw_path: Any, default: Path, output_dir: Path) -> Path:
    if raw_path is None:
        return default.resolve()
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("visualization.robotwin_rollout.result_json must be null or string.")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = output_dir / path
    return path.resolve()


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("visualization.robotwin_rollout.timeout_seconds must be positive int.")
    return value


def _optional_cwd(value: Any, repo_root: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("visualization.robotwin_rollout.workdir must be null or non-empty string.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _nested_str(cfg: Mapping[str, Any], section: str, key: str) -> str | None:
    value = cfg.get(section)
    if not isinstance(value, Mapping):
        return None
    raw = value.get(key)
    return raw if isinstance(raw, str) else None
