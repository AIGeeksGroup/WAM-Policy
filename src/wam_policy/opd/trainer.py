from __future__ import annotations

from itertools import count
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from wam_policy.adapters.anyflow import build_video_flowmap_pair, sample_s_pairs
from wam_policy.adapters.checkpoint import save_training_checkpoint
from wam_policy.adapters.lingbot_va import LingBotVAAdapter
from wam_policy.data.video_latent_dataset import (
    VideoLatentManifestDataset,
    video_latent_collate_fn,
)
from wam_policy.models.lingbot_student import LingBotVideoFlowMapStudent
from wam_policy.opd.losses import video_ffm_average_velocity_loss
from wam_policy.utils.logging import log_step_metrics
from wam_policy.utils.optim import build_adamw_optimizer
from wam_policy.utils.paths import resolve_repo_root
from wam_policy.utils.runtime import resolve_device, resolve_torch_dtype
from wam_policy.utils.seed import set_seed


class VideoFFMTrainer:
    def __init__(
        self,
        cfg: Mapping[str, Any],
        adapter: Any | None = None,
        student: nn.Module | None = None,
        dataset: Dataset[dict[str, Any]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg)
        self.dtype = resolve_torch_dtype(cfg)
        self.adapter = adapter if adapter is not None else LingBotVAAdapter(cfg)
        self.student = (
            student
            if student is not None
            else LingBotVideoFlowMapStudent(cfg, self.adapter)
        )
        self.dataset = dataset if dataset is not None else VideoLatentManifestDataset.from_config(cfg)
        self.optimizer: torch.optim.Optimizer | None = None
        self.loader_iter: Any | None = None

    def train(self) -> dict[str, Any]:
        set_seed(_train_int(self.cfg, "seed", default=42))
        self._build_student()
        self.student.to(device=self.device)
        dataloader = self._build_dataloader()
        self.optimizer = self._build_optimizer()

        max_steps = _train_int(self.cfg, "max_steps", default=1000)
        if max_steps <= 0:
            raise ValueError("train.max_steps must be positive.")

        last_metrics: dict[str, float] = {}
        for step in range(1, max_steps + 1):
            batch = self._next_batch(dataloader)
            last_metrics = self._train_step(batch)
            if step % _train_int(self.cfg, "log_every", default=10) == 0:
                log_step_metrics(step, last_metrics)
            if step % _train_int(self.cfg, "save_every", default=500) == 0:
                self._save_checkpoint(step, last_metrics)

        self._save_checkpoint(max_steps, last_metrics)
        return {"step": max_steps, "metrics": last_metrics}

    def _build_student(self) -> None:
        build = getattr(self.student, "build", None)
        if callable(build):
            build()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        trainable_params = [param for param in self.student.parameters() if param.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for video FFM training.")

        return build_adamw_optimizer(
            trainable_params,
            lr=_train_float(self.cfg, "lr", default=1.0e-5),
            weight_decay=_train_float(self.cfg, "weight_decay", default=0.0),
        )

    def _build_dataloader(self) -> DataLoader[dict[str, Any]]:
        try:
            dataset_len = len(self.dataset)
        except TypeError:
            dataset_len = None
        if dataset_len == 0:
            raise ValueError("Video FFM dataset must not be empty.")

        return DataLoader(
            self.dataset,
            batch_size=_data_int(self.cfg, "batch_size", default=1),
            shuffle=False,
            num_workers=_data_int(self.cfg, "num_workers", default=0),
            collate_fn=video_latent_collate_fn,
        )

    def _next_batch(self, dataloader: DataLoader[dict[str, Any]]) -> dict[str, Any]:
        for _ in count():
            if self.loader_iter is None:
                self.loader_iter = iter(dataloader)
            try:
                return next(self.loader_iter)
            except StopIteration:
                self.loader_iter = None

    def _train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been initialized.")

        self.student.train()
        z_clean = batch["z_clean"].to(device=self.device, dtype=self.dtype)
        context = self.adapter.build_video_context(batch)

        s_src, s_tgt = sample_s_pairs(
            batch_size=z_clean.shape[0],
            s_min=_flowmap_float(self.cfg, "s_min", default=0.0),
            s_max=_flowmap_float(self.cfg, "s_max", default=1.0),
            min_delta_s=_flowmap_float(self.cfg, "min_delta_s", default=0.02),
            device=self.device,
        )
        flowmap_pair = build_video_flowmap_pair(z_clean=z_clean, s_src=s_src, s_tgt=s_tgt)
        pred_u = self.student.forward_video_flowmap(
            flowmap_pair.z_src,
            flowmap_pair.s_src,
            flowmap_pair.s_tgt,
            context,
        )
        loss, metrics = video_ffm_average_velocity_loss(pred_u, flowmap_pair.target_u)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_clip_norm = _train_optional_float(self.cfg, "grad_clip_norm")
        if grad_clip_norm is not None and grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), grad_clip_norm)
        self.optimizer.step()
        return metrics

    def _save_checkpoint(self, step: int, metrics: Mapping[str, float]) -> None:
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been initialized.")
        save_training_checkpoint(
            student=self.student,
            optimizer=self.optimizer,
            cfg=self.cfg,
            step=step,
            metrics=metrics,
            checkpoint_dir=_checkpoint_dir(self.cfg),
        )


def _checkpoint_dir(cfg: Mapping[str, Any]) -> Path:
    paths_cfg = cfg.get("paths")
    raw_path = "experiments/checkpoints/v2/video_ffm"
    if isinstance(paths_cfg, Mapping):
        configured = paths_cfg.get("checkpoint_dir")
        if isinstance(configured, str) and configured:
            raw_path = configured

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = resolve_repo_root(cfg) / path
    return path.resolve()


def _section(cfg: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    value = cfg.get(section)
    return value if isinstance(value, Mapping) else {}


def _train_int(cfg: Mapping[str, Any], key: str, default: int) -> int:
    return _int_value(_section(cfg, "train").get(key, default), f"train.{key}")


def _data_int(cfg: Mapping[str, Any], key: str, default: int) -> int:
    return _int_value(_section(cfg, "data").get(key, default), f"data.{key}")


def _train_float(cfg: Mapping[str, Any], key: str, default: float) -> float:
    return _float_value(_section(cfg, "train").get(key, default), f"train.{key}")


def _train_optional_float(cfg: Mapping[str, Any], key: str) -> float | None:
    value = _section(cfg, "train").get(key)
    if value is None:
        return None
    return _float_value(value, f"train.{key}")


def _flowmap_float(cfg: Mapping[str, Any], key: str, default: float) -> float:
    return _float_value(_section(cfg, "flowmap").get(key, default), f"flowmap.{key}")


def _int_value(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int.")
    return value


def _float_value(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    return float(value)
