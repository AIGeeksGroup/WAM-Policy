"""Optional experiment tracking: a local metrics journal, mirrored to W&B when enabled.

W&B is off unless ``mode`` (or ``WANDB_MODE``) is ``online`` or ``offline``; entity and project
come from arguments or the site configuration (``WANDB_ENTITY``, ``WANDB_PROJECT``).
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from wam_policy.common import site


def redact(value):
    """Drop secrets (tokens, passwords, credentials in URLs) from values written to logs."""
    if isinstance(value, dict):
        return {k: "<redacted>" if re.search(r"password|secret|api[_-]?key|access[_-]?token|credential", k, re.I)
                else redact(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [redact(x) for x in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@", r"\1<redacted>@", value)
        value = re.sub(r"(?i)([?&](?:token|signature)=)[^&\s]+", r"\1<redacted>", value)
        value = re.sub(r"(?i)((?:WANDB_API_KEY|HF_TOKEN|PASSWORD|SECRET)\s*[=:]\s*)\S+", r"\1<redacted>", value)
    return value


class Tracker:
    """``metrics.jsonl`` in ``directory`` plus an optional W&B run.

    Tracker(directory, name="teacher-sft", group=None, job_type="train", config={...},
            mode=None, entity=None, project=None)
    """

    def __init__(self, directory, *, name, group=None, job_type=None, config=None,
                 mode=None, entity=None, project=None, step_metric="step"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mode, self.entity, self.project = site.wandb_settings(mode, entity, project)
        self.step_metric = step_metric
        self.run = self.sdk = None
        if self.mode != "disabled":
            import wandb
            self.sdk = wandb
            self.run = wandb.init(entity=self.entity, project=self.project, name=name, group=group,
                                  job_type=job_type, config=redact(config or {}), mode=self.mode,
                                  dir=str(self.directory),
                                  settings=wandb.Settings(disable_git=True, save_code=False))
            if hasattr(self.run, "define_metric"):
                self.run.define_metric(step_metric)
                self.run.define_metric("*", step_metric=step_metric)

    def log(self, values, step):
        values = redact(values)
        with (self.directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({self.step_metric: step, **values}, allow_nan=False, default=str) + "\n")
        if self.run:
            self.run.log({self.step_metric: step, **values})

    def log_videos(self, videos, step, *, fps, caption=""):
        """Log MP4 files ({key: path}); they always stay on disk."""
        with (self.directory / "media.jsonl").open("a") as stream:
            stream.write(json.dumps({self.step_metric: step, "fps": fps, "caption": caption,
                                     "videos": {k: str(v) for k, v in videos.items()}}) + "\n")
        if self.run:
            self.run.log({self.step_metric: step, **{k: self.sdk.Video(str(v), format="mp4", fps=fps, caption=caption)
                                                     for k, v in videos.items()}})

    def finish(self, status="completed", *, summary=None):
        if self.run:
            self.run.summary.update(redact({"status": status, **(summary or {})}))
            self.run.finish(exit_code=0 if status in ("completed", "stopped") else 1)
        return True
