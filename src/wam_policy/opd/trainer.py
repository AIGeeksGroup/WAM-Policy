from __future__ import annotations

from itertools import count
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from wam_policy.adapters.anyflow import (
    build_action_flowmap_pair,
    build_video_flowmap_pair,
    sample_s_pairs,
)
from wam_policy.adapters.checkpoint import save_training_checkpoint
from wam_policy.adapters.lingbot_va import LingBotVAAdapter
from wam_policy.data.action_dataset import (
    ActionChunkManifestDataset,
    action_chunk_collate_fn,
)
from wam_policy.data.rollout_manifest_dataset import (
    RolloutManifestDataset,
    rollout_manifest_collate_fn,
)
from wam_policy.data.video_latent_dataset import (
    VideoLatentManifestDataset,
    video_latent_collate_fn,
)
from wam_policy.models.lingbot_student import (
    LingBotActionFlowMapStudent,
    LingBotVideoFlowMapStudent,
)
from wam_policy.opd.losses import (
    action_ffm_average_velocity_loss,
    dmd_score_surrogate_loss,
    video_ffm_average_velocity_loss,
)
from wam_policy.utils.logging import TrainingLogger
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

        logger = TrainingLogger(self.cfg)
        last_metrics: dict[str, float] = {}
        try:
            for step in range(1, max_steps + 1):
                batch = self._next_batch(dataloader)
                last_metrics = self._train_step(batch)
                if step % _train_int(self.cfg, "log_every", default=10) == 0:
                    logger.log_metrics(step, last_metrics)
                if step % _train_int(self.cfg, "save_every", default=500) == 0:
                    self._save_checkpoint(step, last_metrics)

            self._save_checkpoint(max_steps, last_metrics)
            return {"step": max_steps, "metrics": last_metrics}
        finally:
            logger.finish()

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
        z_clean = normalize_video_latent_batch(batch["z_clean"]).to(
            device=self.device,
            dtype=self.dtype,
        )
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


class ActionFFMTrainer:
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
            else LingBotActionFlowMapStudent(cfg, self.adapter)
        )
        self.dataset = dataset if dataset is not None else ActionChunkManifestDataset.from_config(cfg)
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

        logger = TrainingLogger(self.cfg)
        last_metrics: dict[str, float] = {}
        try:
            for step in range(1, max_steps + 1):
                batch = self._next_batch(dataloader)
                last_metrics = self._train_step(batch)
                if step % _train_int(self.cfg, "log_every", default=10) == 0:
                    logger.log_metrics(step, last_metrics)
                if step % _train_int(self.cfg, "save_every", default=500) == 0:
                    self._save_checkpoint(step, last_metrics)

            self._save_checkpoint(max_steps, last_metrics)
            return {"step": max_steps, "metrics": last_metrics}
        finally:
            logger.finish()

    def _build_student(self) -> None:
        build = getattr(self.student, "build", None)
        if callable(build):
            build()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        trainable_params = [param for param in self.student.parameters() if param.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for action FFM training.")

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
            raise ValueError("Action FFM dataset must not be empty.")

        return DataLoader(
            self.dataset,
            batch_size=_data_int(self.cfg, "batch_size", default=1),
            shuffle=False,
            num_workers=_data_int(self.cfg, "num_workers", default=0),
            collate_fn=action_chunk_collate_fn,
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
        a_clean = normalize_action_chunk_batch(batch["a_clean"]).to(
            device=self.device,
            dtype=self.dtype,
        )
        context = self.adapter.build_action_context(batch)
        z_condition = _move_optional_tensor(batch.get("z_condition"), self.device, self.dtype)

        s_src, s_tgt = sample_s_pairs(
            batch_size=a_clean.shape[0],
            s_min=_flowmap_float(self.cfg, "s_min", default=0.0),
            s_max=_flowmap_float(self.cfg, "s_max", default=1.0),
            min_delta_s=_flowmap_float(self.cfg, "min_delta_s", default=0.02),
            device=self.device,
        )
        flowmap_pair = build_action_flowmap_pair(a_clean=a_clean, s_src=s_src, s_tgt=s_tgt)
        pred_u = self.student.forward_action_flowmap(
            flowmap_pair.a_src,
            flowmap_pair.s_src,
            flowmap_pair.s_tgt,
            context,
            z_condition=z_condition,
        )
        loss, metrics = action_ffm_average_velocity_loss(pred_u, flowmap_pair.target_u)

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


class VideoDMDOPDTrainer:
    def __init__(
        self,
        cfg: Mapping[str, Any],
        adapter: Any | None = None,
        student: nn.Module | None = None,
        rollout_dataset: Dataset[dict[str, Any]] | None = None,
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
        self.rollout_dataset = (
            rollout_dataset
            if rollout_dataset is not None
            else RolloutManifestDataset.from_config(cfg)
        )
        self.optimizer: torch.optim.Optimizer | None = None
        self.loader_iter: Any | None = None

    def train(self) -> dict[str, Any]:
        set_seed(_train_int(self.cfg, "seed", default=42))
        _raise_if_teacher_score_not_implemented(self.cfg, stage="video_dmd")
        self._build_student()
        self.student.to(device=self.device)
        dataloader = self._build_dataloader()
        self.optimizer = self._build_optimizer()

        max_steps = _train_int(self.cfg, "max_steps", default=1000)
        if max_steps <= 0:
            raise ValueError("train.max_steps must be positive.")

        logger = TrainingLogger(self.cfg)
        last_metrics: dict[str, float] = {}
        try:
            for step in range(1, max_steps + 1):
                batch = self._next_batch(dataloader)
                last_metrics = self._train_step(batch)
                if step % _train_int(self.cfg, "log_every", default=10) == 0:
                    logger.log_metrics(step, last_metrics)
                if step % _train_int(self.cfg, "save_every", default=500) == 0:
                    self._save_checkpoint(step, last_metrics)

            self._save_checkpoint(max_steps, last_metrics)
            return {"step": max_steps, "metrics": last_metrics}
        finally:
            logger.finish()

    def _build_student(self) -> None:
        build = getattr(self.student, "build", None)
        if callable(build):
            build()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        trainable_params = [param for param in self.student.parameters() if param.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for video DMD OPD training.")
        return build_adamw_optimizer(
            trainable_params,
            lr=_train_float(self.cfg, "lr", default=1.0e-5),
            weight_decay=_train_float(self.cfg, "weight_decay", default=0.0),
        )

    def _build_dataloader(self) -> DataLoader[dict[str, Any]]:
        try:
            dataset_len = len(self.rollout_dataset)
        except TypeError:
            dataset_len = None
        if dataset_len == 0:
            raise ValueError("Video DMD rollout dataset must not be empty.")
        return DataLoader(
            self.rollout_dataset,
            batch_size=_data_int(self.cfg, "batch_size", default=1),
            shuffle=False,
            num_workers=_data_int(self.cfg, "num_workers", default=0),
            collate_fn=rollout_manifest_collate_fn,
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
        if not isinstance(batch.get("z_student"), torch.Tensor):
            raise ValueError("Video DMD rollout batch must contain tensor z_student.")
        z_student = normalize_video_latent_batch(batch["z_student"]).to(
            device=self.device,
            dtype=self.dtype,
        )
        context = self.adapter.build_video_context(batch)

        s_src, s_tgt = sample_s_pairs(
            batch_size=z_student.shape[0],
            s_min=_flowmap_float(self.cfg, "s_min", default=0.0),
            s_max=_flowmap_float(self.cfg, "s_max", default=1.0),
            min_delta_s=_flowmap_float(self.cfg, "min_delta_s", default=0.02),
            device=self.device,
        )
        flowmap_pair = build_video_flowmap_pair(z_clean=z_student, s_src=s_src, s_tgt=s_tgt)
        pred_u = self.student.forward_video_flowmap(
            flowmap_pair.z_src,
            flowmap_pair.s_src,
            flowmap_pair.s_tgt,
            context,
        )
        teacher_score = _resolve_video_teacher_score(
            cfg=self.cfg,
            adapter=self.adapter,
            batch=batch,
            reference=flowmap_pair.z_src,
            s_src=flowmap_pair.s_src,
            s_tgt=flowmap_pair.s_tgt,
            context=context,
        )
        student_score = _resolve_video_student_score(
            cfg=self.cfg,
            adapter=self.adapter,
            student=self.student,
            pred_u=pred_u,
            z_state=flowmap_pair.z_src,
            s_src=flowmap_pair.s_src,
            s_tgt=flowmap_pair.s_tgt,
            context=context,
        )
        student_state = _resolve_dmd_student_state(
            cfg=self.cfg,
            pred_u=pred_u,
            clean_sample=z_student,
            sample_name="z_student",
        )
        dmd_loss, metrics = dmd_score_surrogate_loss(
            student_state=student_state,
            teacher_score=teacher_score,
            student_score=student_score,
            weight=_opd_dmd_float(self.cfg, "weight", default=0.1),
            metric_prefix="video_dmd",
        )
        total_loss = dmd_loss

        if _opd_anchor_enabled(self.cfg):
            anchor_loss, anchor_metrics = video_ffm_average_velocity_loss(
                pred_u,
                flowmap_pair.target_u,
            )
            anchor_weight = _opd_anchor_float(self.cfg, "weight", default=1.0)
            total_loss = total_loss + anchor_weight * anchor_loss
            metrics.update({f"anchor/{key}": value for key, value in anchor_metrics.items()})

        metrics["loss/video_dmd_total"] = float(total_loss.detach().cpu())
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
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


class ActionDMDOPDTrainer:
    def __init__(
        self,
        cfg: Mapping[str, Any],
        adapter: Any | None = None,
        student: nn.Module | None = None,
        rollout_dataset: Dataset[dict[str, Any]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg)
        self.dtype = resolve_torch_dtype(cfg)
        self.adapter = adapter if adapter is not None else LingBotVAAdapter(cfg)
        self.student = (
            student
            if student is not None
            else LingBotActionFlowMapStudent(cfg, self.adapter)
        )
        self.rollout_dataset = (
            rollout_dataset
            if rollout_dataset is not None
            else RolloutManifestDataset.from_config(cfg)
        )
        self.optimizer: torch.optim.Optimizer | None = None
        self.loader_iter: Any | None = None

    def train(self) -> dict[str, Any]:
        set_seed(_train_int(self.cfg, "seed", default=42))
        _raise_if_teacher_score_not_implemented(self.cfg, stage="action_dmd")
        self._build_student()
        self.student.to(device=self.device)
        dataloader = self._build_dataloader()
        self.optimizer = self._build_optimizer()

        max_steps = _train_int(self.cfg, "max_steps", default=1000)
        if max_steps <= 0:
            raise ValueError("train.max_steps must be positive.")

        logger = TrainingLogger(self.cfg)
        last_metrics: dict[str, float] = {}
        try:
            for step in range(1, max_steps + 1):
                batch = self._next_batch(dataloader)
                last_metrics = self._train_step(batch)
                if step % _train_int(self.cfg, "log_every", default=10) == 0:
                    logger.log_metrics(step, last_metrics)
                if step % _train_int(self.cfg, "save_every", default=500) == 0:
                    self._save_checkpoint(step, last_metrics)

            self._save_checkpoint(max_steps, last_metrics)
            return {"step": max_steps, "metrics": last_metrics}
        finally:
            logger.finish()

    def _build_student(self) -> None:
        build = getattr(self.student, "build", None)
        if callable(build):
            build()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        trainable_params = [param for param in self.student.parameters() if param.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for action DMD OPD training.")
        return build_adamw_optimizer(
            trainable_params,
            lr=_train_float(self.cfg, "lr", default=1.0e-5),
            weight_decay=_train_float(self.cfg, "weight_decay", default=0.0),
        )

    def _build_dataloader(self) -> DataLoader[dict[str, Any]]:
        try:
            dataset_len = len(self.rollout_dataset)
        except TypeError:
            dataset_len = None
        if dataset_len == 0:
            raise ValueError("Action DMD rollout dataset must not be empty.")
        return DataLoader(
            self.rollout_dataset,
            batch_size=_data_int(self.cfg, "batch_size", default=1),
            shuffle=False,
            num_workers=_data_int(self.cfg, "num_workers", default=0),
            collate_fn=rollout_manifest_collate_fn,
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
        if not isinstance(batch.get("a_student"), torch.Tensor):
            raise ValueError("Action DMD rollout batch must contain tensor a_student.")
        if not isinstance(batch.get("z_condition"), torch.Tensor):
            raise ValueError(
                "Action DMD rollout batch must contain tensor z_condition from student video rollout."
            )
        a_student = normalize_action_chunk_batch(batch["a_student"]).to(
            device=self.device,
            dtype=self.dtype,
        )
        z_condition = batch["z_condition"].to(device=self.device, dtype=self.dtype)
        context = self.adapter.build_action_context(batch)

        s_src, s_tgt = sample_s_pairs(
            batch_size=a_student.shape[0],
            s_min=_flowmap_float(self.cfg, "s_min", default=0.0),
            s_max=_flowmap_float(self.cfg, "s_max", default=1.0),
            min_delta_s=_flowmap_float(self.cfg, "min_delta_s", default=0.02),
            device=self.device,
        )
        flowmap_pair = build_action_flowmap_pair(a_clean=a_student, s_src=s_src, s_tgt=s_tgt)
        pred_u = self.student.forward_action_flowmap(
            flowmap_pair.a_src,
            flowmap_pair.s_src,
            flowmap_pair.s_tgt,
            context,
            z_condition=z_condition,
        )
        teacher_score = _resolve_action_teacher_score(
            cfg=self.cfg,
            adapter=self.adapter,
            batch=batch,
            reference=flowmap_pair.a_src,
            s_src=flowmap_pair.s_src,
            s_tgt=flowmap_pair.s_tgt,
            context=context,
            z_condition=z_condition,
        )
        student_score = _resolve_action_student_score(
            cfg=self.cfg,
            adapter=self.adapter,
            student=self.student,
            pred_u=pred_u,
            a_state=flowmap_pair.a_src,
            s_src=flowmap_pair.s_src,
            s_tgt=flowmap_pair.s_tgt,
            context=context,
            z_condition=z_condition,
        )
        student_state = _resolve_dmd_student_state(
            cfg=self.cfg,
            pred_u=pred_u,
            clean_sample=a_student,
            sample_name="a_student",
        )
        dmd_loss, metrics = dmd_score_surrogate_loss(
            student_state=student_state,
            teacher_score=teacher_score,
            student_score=student_score,
            weight=_opd_dmd_float(self.cfg, "weight", default=0.1),
            metric_prefix="action_dmd",
        )
        total_loss = dmd_loss

        if _opd_anchor_enabled(self.cfg):
            anchor_loss, anchor_metrics = action_ffm_average_velocity_loss(
                pred_u,
                flowmap_pair.target_u,
            )
            anchor_weight = _opd_anchor_float(self.cfg, "weight", default=1.0)
            total_loss = total_loss + anchor_weight * anchor_loss
            metrics.update({f"anchor/{key}": value for key, value in anchor_metrics.items()})

        metrics["loss/action_dmd_total"] = float(total_loss.detach().cpu())
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
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


def normalize_video_latent_batch(z_clean: torch.Tensor) -> torch.Tensor:
    if not isinstance(z_clean, torch.Tensor):
        raise TypeError("z_clean must be a torch.Tensor.")
    if z_clean.ndim == 5:
        return z_clean
    if z_clean.ndim == 6 and z_clean.shape[1] == 1:
        return z_clean.squeeze(1)
    raise ValueError(
        "z_clean must have shape [B, C, F, H, W] or [B, 1, C, F, H, W]."
    )


def normalize_action_chunk_batch(a_clean: torch.Tensor) -> torch.Tensor:
    if not isinstance(a_clean, torch.Tensor):
        raise TypeError("a_clean must be a torch.Tensor.")
    if a_clean.ndim in {2, 3}:
        return a_clean
    if a_clean.ndim == 4 and a_clean.shape[1] == 1:
        return a_clean.squeeze(1)
    raise ValueError("a_clean must have shape [B, D], [B, T, D], or [B, 1, T, D].")


def _move_optional_tensor(value: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return value


def _resolve_video_teacher_score(
    *,
    cfg: Mapping[str, Any],
    adapter: Any,
    batch: Mapping[str, Any],
    reference: torch.Tensor,
    s_src: torch.Tensor,
    s_tgt: torch.Tensor,
    context: Any,
) -> torch.Tensor:
    source = _opd_dmd_choice(
        cfg,
        "teacher_score_source",
        default="not_implemented",
        allowed={"not_implemented", "lingbot_velocity_forward", "cached_label"},
    )
    if source == "not_implemented":
        raise RuntimeError(_dmd_teacher_not_implemented_message())
    if source == "cached_label":
        label = _cached_teacher_label(
            batch=batch,
            key="teacher_video_label",
            reference=reference,
            normalize=normalize_video_latent_batch,
        )
        return _validate_dmd_score("teacher_video_label", label, reference)
    score = adapter.teacher_video_score_or_velocity(
        z_state=reference,
        s_src=s_src,
        s_tgt=s_tgt,
        context=context,
    )
    return _validate_dmd_score("teacher_video_score", score, reference)


def _resolve_action_teacher_score(
    *,
    cfg: Mapping[str, Any],
    adapter: Any,
    batch: Mapping[str, Any],
    reference: torch.Tensor,
    s_src: torch.Tensor,
    s_tgt: torch.Tensor,
    context: Any,
    z_condition: torch.Tensor,
) -> torch.Tensor:
    source = _opd_dmd_choice(
        cfg,
        "teacher_score_source",
        default="not_implemented",
        allowed={"not_implemented", "lingbot_velocity_forward", "cached_label"},
    )
    if source == "not_implemented":
        raise RuntimeError(_dmd_teacher_not_implemented_message())
    if source == "cached_label":
        label = _cached_teacher_label(
            batch=batch,
            key="teacher_action_label",
            reference=reference,
            normalize=normalize_action_chunk_batch,
        )
        return _validate_dmd_score("teacher_action_label", label, reference)
    score = adapter.teacher_action_score_or_velocity(
        a_state=reference,
        s_src=s_src,
        s_tgt=s_tgt,
        context=context,
        z_condition=z_condition,
    )
    return _validate_dmd_score("teacher_action_score", score, reference)


def _resolve_video_student_score(
    *,
    cfg: Mapping[str, Any],
    adapter: Any,
    student: nn.Module,
    pred_u: torch.Tensor,
    z_state: torch.Tensor,
    s_src: torch.Tensor,
    s_tgt: torch.Tensor,
    context: Any,
) -> torch.Tensor:
    source = _opd_dmd_choice(
        cfg,
        "student_score_source",
        default="flowmap_small_interval",
        allowed={"flowmap_small_interval", "current_interval_average_velocity"},
    )
    if source == "current_interval_average_velocity":
        score = adapter.student_video_score_or_velocity_from_flowmap(
            pred_u=pred_u,
            z_state=z_state,
            s_src=s_src,
            s_tgt=s_tgt,
            context=context,
        )
        return _validate_dmd_score("student_video_score", score, z_state)

    s_small_tgt = _small_interval_tgt(cfg, s_src)
    pred_small = student.forward_video_flowmap(z_state, s_src, s_small_tgt, context)
    score = adapter.student_video_score_or_velocity_from_flowmap(
        pred_u=pred_small,
        z_state=z_state,
        s_src=s_src,
        s_tgt=s_small_tgt,
        context=context,
    )
    return _validate_dmd_score("student_video_score", score, z_state)


def _resolve_action_student_score(
    *,
    cfg: Mapping[str, Any],
    adapter: Any,
    student: nn.Module,
    pred_u: torch.Tensor,
    a_state: torch.Tensor,
    s_src: torch.Tensor,
    s_tgt: torch.Tensor,
    context: Any,
    z_condition: torch.Tensor,
) -> torch.Tensor:
    source = _opd_dmd_choice(
        cfg,
        "student_score_source",
        default="flowmap_small_interval",
        allowed={"flowmap_small_interval", "current_interval_average_velocity"},
    )
    if source == "current_interval_average_velocity":
        score = adapter.student_action_score_or_velocity_from_flowmap(
            pred_u=pred_u,
            a_state=a_state,
            s_src=s_src,
            s_tgt=s_tgt,
            context=context,
            z_condition=z_condition,
        )
        return _validate_dmd_score("student_action_score", score, a_state)

    s_small_tgt = _small_interval_tgt(cfg, s_src)
    pred_small = student.forward_action_flowmap(
        a_state,
        s_src,
        s_small_tgt,
        context,
        z_condition=z_condition,
    )
    score = adapter.student_action_score_or_velocity_from_flowmap(
        pred_u=pred_small,
        a_state=a_state,
        s_src=s_src,
        s_tgt=s_small_tgt,
        context=context,
        z_condition=z_condition,
    )
    return _validate_dmd_score("student_action_score", score, a_state)


def _resolve_dmd_student_state(
    *,
    cfg: Mapping[str, Any],
    pred_u: torch.Tensor,
    clean_sample: torch.Tensor,
    sample_name: str,
) -> torch.Tensor:
    source = _opd_dmd_choice(
        cfg,
        "surrogate_state",
        default="pred_u",
        allowed={"pred_u", "clean_sample"},
    )
    if source == "pred_u":
        return pred_u
    return _validate_dmd_score(sample_name, clean_sample, pred_u)


def _cached_teacher_label(
    *,
    batch: Mapping[str, Any],
    key: str,
    reference: torch.Tensor,
    normalize: Any,
) -> torch.Tensor:
    label = batch.get(key)
    if not isinstance(label, torch.Tensor):
        raise ValueError(
            f"opd.dmd.teacher_score_source=cached_label requires rollout batch tensor {key}."
        )
    label = normalize(label).to(device=reference.device, dtype=reference.dtype)
    return label


def _validate_dmd_score(name: str, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.shape != reference.shape:
        raise ValueError(f"{name} shape must match DMD reference: {value.shape} != {reference.shape}")
    if value.device != reference.device:
        raise ValueError(f"{name} device must match DMD reference: {value.device} != {reference.device}")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _small_interval_tgt(cfg: Mapping[str, Any], s_src: torch.Tensor) -> torch.Tensor:
    small_delta_s = _opd_dmd_float(cfg, "small_delta_s", default=0.02)
    s_min = _flowmap_float(cfg, "s_min", default=0.0)
    s_max = _flowmap_float(cfg, "s_max", default=1.0)
    if small_delta_s <= 0.0:
        raise ValueError("opd.dmd.small_delta_s must be positive.")
    if small_delta_s > s_max - s_min:
        raise ValueError("opd.dmd.small_delta_s must not exceed flowmap.s_max - flowmap.s_min.")
    s_max_tensor = torch.full_like(s_src, s_max)
    s_tgt = torch.minimum(s_src + small_delta_s, s_max_tensor)
    if not torch.all(s_tgt > s_src).item():
        raise ValueError("opd.dmd.small_delta_s must leave a positive small interval.")
    return s_tgt


def _raise_if_teacher_score_not_implemented(cfg: Mapping[str, Any], *, stage: str) -> None:
    source = _opd_dmd_choice(
        cfg,
        "teacher_score_source",
        default="not_implemented",
        allowed={"not_implemented", "lingbot_velocity_forward", "cached_label"},
    )
    if source == "not_implemented":
        raise RuntimeError(f"{stage}: {_dmd_teacher_not_implemented_message()}")


def _dmd_teacher_not_implemented_message() -> str:
    return (
        "DMD teacher score source is not implemented. "
        "Use teacher_score_source=lingbot_velocity_forward after server probe succeeds, "
        "or use teacher_score_source=cached_label with teacher labels in rollout manifest."
    )


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


def _opd_dmd_float(cfg: Mapping[str, Any], key: str, default: float) -> float:
    dmd_cfg = _nested_section(cfg, "opd", "dmd")
    return _float_value(dmd_cfg.get(key, default), f"opd.dmd.{key}")


def _opd_dmd_choice(
    cfg: Mapping[str, Any],
    key: str,
    *,
    default: str,
    allowed: set[str],
) -> str:
    dmd_cfg = _nested_section(cfg, "opd", "dmd")
    value = dmd_cfg.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"opd.dmd.{key} must be a non-empty string.")
    if value not in allowed:
        joined = ", ".join(sorted(allowed))
        raise ValueError(f"opd.dmd.{key} must be one of: {joined}.")
    return value


def _opd_anchor_float(cfg: Mapping[str, Any], key: str, default: float) -> float:
    anchor_cfg = _nested_section(cfg, "opd", "anchor_ffm")
    return _float_value(anchor_cfg.get(key, default), f"opd.anchor_ffm.{key}")


def _opd_anchor_enabled(cfg: Mapping[str, Any]) -> bool:
    anchor_cfg = _nested_section(cfg, "opd", "anchor_ffm")
    value = anchor_cfg.get("enabled", False)
    if not isinstance(value, bool):
        raise ValueError("opd.anchor_ffm.enabled must be a bool.")
    return value


def _nested_section(cfg: Mapping[str, Any], section: str, child: str) -> Mapping[str, Any]:
    parent = _section(cfg, section)
    value = parent.get(child)
    return value if isinstance(value, Mapping) else {}


def _int_value(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int.")
    return value


def _float_value(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    return float(value)
