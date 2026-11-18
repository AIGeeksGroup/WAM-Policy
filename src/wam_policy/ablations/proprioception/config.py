"""Training configuration of the proprioception ablation (JSON file -> TrainConfig)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path

from wam_policy.common import site

PATH_FIELDS = ("cache_root", "snapshot", "transformer", "state_root", "output_root")
# Fields that do not change the optimisation (allowed to differ when resuming).
RUNTIME_FIELDS = PATH_FIELDS + ("device", "wandb_mode", "wandb_entity", "wandb_project", "walltime_seconds")


@dataclass(frozen=True)
class TrainConfig:
    # Inputs: the Stage 1.1 frozen VAE/T5 cache (``python -m wam_policy.data cache``), the QLM-Bench
    # dataset snapshot it was built from (root-link state columns), the converted native9+proprioception
    # transformer and the 10D state directory written by the ``data`` command.
    cache_root: str = ""
    snapshot: str = ""
    transformer: str = ""
    state_root: str = ""
    output_root: str = ""
    seed: int = 42
    max_steps: int = 5000
    learning_rate: float = 1e-5
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.1
    warmup_steps: int = 10
    grad_clip_norm: float = 1.0
    accumulation: int = 1
    checkpoint_every: int = 100
    validation_every: int = 100
    keep_checkpoints: int = 2
    activation_checkpointing: bool = True
    attention: str = "flex"
    device: str = "cuda"
    walltime_seconds: int = 0  # 0 = no limit; otherwise stop and checkpoint before the limit
    wandb_mode: str | None = None  # None -> WANDB_MODE (default disabled)
    wandb_entity: str | None = None
    wandb_project: str | None = None

    def validate(self):
        for name in PATH_FIELDS:
            value = getattr(self, name)
            if not value or "$" in value:
                raise ValueError(f"{name} must be set (environment variables are expanded)")
        for name in ("max_steps", "warmup_steps", "accumulation", "checkpoint_every", "validation_every",
                     "keep_checkpoints"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.walltime_seconds) is not int or self.walltime_seconds < 0:
            raise ValueError("walltime_seconds must be a nonnegative integer")
        if self.device not in ("cpu", "cuda") or self.attention not in ("reference", "flex"):
            raise ValueError("device must be cpu|cuda and attention reference|flex")
        if not (0 < self.learning_rate < 1 and 0 <= self.weight_decay < 1 and self.grad_clip_norm > 0):
            raise ValueError("Invalid optimizer values")
        if len(self.betas) != 2 or not all(0 <= x < 1 for x in self.betas) or not math.isfinite(sum(self.betas)):
            raise ValueError("Invalid optimizer betas")
        site.wandb_settings(self.wandb_mode, self.wandb_entity, self.wandb_project)
        return self

    def training_values(self):
        """The fields a resumed run must share with its checkpoint."""
        value = asdict(self)
        for name in RUNTIME_FIELDS:
            value.pop(name)
        value["betas"] = list(value["betas"])
        return value


def load_config(path, **overrides):
    values = json.loads(Path(path).read_text())
    unknown = set(values) - set(TrainConfig.__dataclass_fields__)
    if unknown:
        raise ValueError("Unknown training keys: " + str(sorted(unknown)))
    values.update({k: v for k, v in overrides.items() if v is not None})
    for key in PATH_FIELDS:
        if key in values:
            values[key] = str(Path(os.path.expandvars(values[key])).expanduser())
    if "betas" in values:
        values["betas"] = tuple(values["betas"])
    return TrainConfig(**values).validate()
