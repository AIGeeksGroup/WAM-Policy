"""Stage 1.1 SFT configuration (JSON file + command-line overrides).

Paths may be given in the JSON (``$VAR`` is expanded) or on the command line. Keys starting with
``_`` are comments. The paper recipe is ``configs/stage1_1_transfer/teacher_sft.json``: 8 ranks x
accumulation 1 = global batch 8, 5000 steps, AdamW lr 1e-5, 10 warm-up steps. With fewer GPUs,
raise ``accumulation`` to keep the global batch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path

PATH_KEYS = ("cache_root", "transformer", "output_dir", "base_model")
# Fields that define the optimization; a resumed run must use the checkpoint's values.
RESUME_KEYS = ("seed", "learning_rate", "betas", "weight_decay", "warmup_steps", "grad_clip_norm",
               "accumulation", "activation_checkpointing", "attention", "device")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    partial.replace(path)


@dataclass(frozen=True)
class TrainConfig:
    cache_root: str = ""          # frozen cache (python -m wam_policy.data cache)
    transformer: str = ""         # native9-converted LingBot-VA transformer (initial weights)
    output_dir: str = ""          # run directory: checkpoints/, metrics.jsonl, teacher/
    base_model: str = ""          # LingBot-VA base snapshot; only needed for prediction videos
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
    attention: str = "flex"        # flex (GPU) | reference (CPU/debug)
    device: str = "cuda"           # cuda (CUDA or ROCm) | cpu
    walltime_seconds: int = 0      # 0 = no limit; otherwise checkpoint and stop before the limit
    visualize_every: int = 0       # optimizer steps between prediction videos; 0 = off
    visualize_episodes: tuple = field(default_factory=tuple)  # validation episode uids
    ffmpeg: str = "ffmpeg"
    run_name: str = ""             # W&B run name (default: output directory name)
    wandb_mode: str = ""           # "" = site default (WANDB_MODE, disabled unless set)
    wandb_entity: str = ""
    wandb_project: str = ""

    def validate(self):
        for key in ("cache_root", "transformer", "output_dir"):
            if not getattr(self, key):
                raise ValueError(f"{key} is required (config file or --{key.replace('_', '-')})")
        for key in ("max_steps", "warmup_steps", "accumulation", "checkpoint_every", "validation_every",
                    "keep_checkpoints"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.device not in ("cpu", "cuda") or self.attention not in ("flex", "reference"):
            raise ValueError("device must be cpu|cuda and attention flex|reference")
        if self.visualize_every and not (self.base_model and self.visualize_episodes):
            raise ValueError("Prediction videos need base_model and visualize_episodes")
        if len(self.betas) != 2:
            raise ValueError("betas must have two values")
        return self

    def resume_values(self):
        values = asdict(self)
        return {k: list(values[k]) if isinstance(values[k], tuple) else values[k] for k in RESUME_KEYS}


def load_config(path, **overrides):
    values = {k: v for k, v in json.loads(Path(path).read_text()).items() if not k.startswith("_")}
    unknown = set(values) - set(TrainConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown training keys: {sorted(unknown)}")
    values.update({k: v for k, v in overrides.items() if v is not None})
    for key in PATH_KEYS:
        if values.get(key):
            values[key] = os.path.expandvars(os.path.expanduser(values[key]))
    for key in ("betas", "visualize_episodes"):
        if key in values:
            values[key] = tuple(values[key])
    return TrainConfig(**values).validate()
