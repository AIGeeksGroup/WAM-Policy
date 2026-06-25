from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from wam_policy.adapters.checkpoint import load_student_checkpoint
from wam_policy.adapters.lingbot_va import LingBotVAAdapter
from wam_policy.data.video_latent_dataset import VideoLatentManifestDataset
from wam_policy.models.lingbot_student import LingBotVideoFlowMapStudent
from wam_policy.utils.logging import TrainingLogger
from wam_policy.utils.paths import resolve_repo_root
from wam_policy.utils.runtime import resolve_device, resolve_torch_dtype
from wam_policy.utils.seed import set_seed
from wam_policy.visualization.video import (
    contact_sheet_video,
    side_by_side,
    stitch_chunks,
    write_metadata,
    write_video_artifact,
)


EPISODE_KEYS = (
    "episode_id",
    "episode_index",
    "episode_idx",
    "episode",
    "trajectory_id",
    "traj_id",
    "demo_id",
    "sequence_id",
    "episode_name",
)
CHUNK_KEYS = ("chunk_idx", "chunk_index", "segment_idx", "segment_index", "start_frame")


@dataclass(frozen=True)
class OfflineSelection:
    indices: list[int]
    composition_mode: str
    reason: str
    episode_key: str | None = None
    chunk_key: str | None = None


def run_offline_video_stream_visualization(cfg: Mapping[str, Any]) -> dict[str, Any]:
    vis_cfg = _offline_cfg(cfg)
    device = resolve_device(cfg)
    dtype = resolve_torch_dtype(cfg)
    set_seed(_int_cfg(vis_cfg, "seed", default=42))

    output_dir = _resolve_path(
        cfg,
        _str_cfg(vis_cfg, "output_dir", default="results/v2/visualization/offline_video_stream"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = VideoLatentManifestDataset.from_config(cfg)
    selection = select_offline_chunks(
        dataset.items,
        max_chunks=_int_cfg(vis_cfg, "max_chunks", default=16),
        selection_mode=_str_cfg(vis_cfg, "selection_mode", default="auto"),
    )

    adapter = LingBotVAAdapter(cfg)
    teacher = adapter.load_pretrained_video_action_model().to(device=device, dtype=dtype)
    student = LingBotVideoFlowMapStudent(cfg, adapter)
    student.build()
    student.to(device=device, dtype=dtype)
    load_student_checkpoint(
        student,
        _required_str(vis_cfg, "student_checkpoint_path"),
        strict=_bool_cfg(vis_cfg, "checkpoint_strict", default=False),
        map_location="cpu",
    )
    student.eval()
    vae = adapter.load_video_vae()

    teacher_chunks: list[torch.Tensor] = []
    student_chunks: list[torch.Tensor] = []
    teacher_latent_chunks: list[torch.Tensor] = []
    student_latent_chunks: list[torch.Tensor] = []
    selected_meta: list[dict[str, Any]] = []
    cumulative_frame_start = 0
    current_stage = "initialization"
    try:
        for index in selection.indices:
            sample = dataset[index]
            z_clean = sample["z_clean"]
            if z_clean.ndim == 5 and z_clean.shape[0] == 1:
                z_clean = z_clean.squeeze(0)
            if z_clean.ndim != 4:
                raise ValueError(
                    f"Expected video latent [C,F,H,W] sample after load, got {z_clean.shape}."
                )
            z_noise = torch.randn(
                (1, *z_clean.shape),
                device=device,
                dtype=dtype,
            )
            context = sample["context"]
            current_stage = "teacher_sampling"
            teacher_latents = adapter.sample_teacher_video(
                teacher,
                z_noise,
                context,
                num_steps=_int_cfg(vis_cfg, "teacher_steps", default=25),
                snr_shift=_float_cfg(vis_cfg, "teacher_snr_shift", default=5.0),
                guidance_scale=_float_cfg(vis_cfg, "teacher_guidance_scale", default=1.0),
                frame_start_id=_frame_start_id(sample["meta"], cumulative_frame_start),
            )
            current_stage = "student_sampling"
            student_latents = sample_student_video_flowmap(
                student,
                z_noise,
                context,
                num_steps=_int_cfg(vis_cfg, "student_steps", default=4),
            )
            teacher_latent_chunks.append(teacher_latents.detach().cpu())
            student_latent_chunks.append(student_latents.detach().cpu())

            current_stage = "vae_decode_teacher"
            teacher_frames = adapter.decode_video_latents(vae, teacher_latents)[0]
            current_stage = "vae_decode_student"
            student_frames = adapter.decode_video_latents(vae, student_latents)[0]
            teacher_chunks.append(teacher_frames)
            student_chunks.append(student_frames)
            selected_meta.append(dict(sample["meta"]))
            cumulative_frame_start += int(teacher_frames.shape[0])
    except Exception as exc:
        latent_paths = _save_latent_fallbacks(
            output_dir=output_dir,
            teacher_latent_chunks=teacher_latent_chunks,
            student_latent_chunks=student_latent_chunks,
        )
        failure_summary = write_metadata(
            output_dir / "offline_video_stream_failure_summary.json",
            {
                "failed_stage": current_stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "adapter_location": "src/wam_policy/adapters/lingbot_va.py",
                "latent_fallbacks": {
                    key: [str(path) for path in paths]
                    for key, paths in latent_paths.items()
                },
                "selected_meta": selected_meta,
            },
        )
        raise RuntimeError(
            f"Offline visualization failed during {current_stage}. "
            f"Latent fallbacks and summary were saved under {output_dir}; "
            f"summary={failure_summary}"
        ) from exc

    if selection.composition_mode == "stitch":
        teacher_video = stitch_chunks(teacher_chunks)
        student_video = stitch_chunks(student_chunks)
    elif selection.composition_mode == "grid":
        columns = _optional_int_cfg(vis_cfg, "grid_columns")
        teacher_video = contact_sheet_video(teacher_chunks, columns=columns)
        student_video = contact_sheet_video(student_chunks, columns=columns)
    else:
        raise ValueError(f"Unknown composition mode: {selection.composition_mode}")

    comparison_video = side_by_side(teacher_video, student_video)
    fps = _int_cfg(vis_cfg, "fps", default=10)
    teacher_encoding = write_video_artifact(
        teacher_video,
        output_dir / "teacher_25step.mp4",
        fps=fps,
    )
    student_encoding = write_video_artifact(
        student_video,
        output_dir / "student_4step.mp4",
        fps=fps,
    )
    comparison_encoding = write_video_artifact(
        comparison_video,
        output_dir / "teacher_vs_student.mp4",
        fps=fps,
    )
    metadata_payload: dict[str, Any] = {
        "composition_mode": selection.composition_mode,
        "selection_reason": selection.reason,
        "episode_key": selection.episode_key,
        "chunk_key": selection.chunk_key,
        "indices": selection.indices,
        "teacher_steps": _int_cfg(vis_cfg, "teacher_steps", default=25),
        "student_steps": _int_cfg(vis_cfg, "student_steps", default=4),
        "selected_meta": selected_meta,
        "encoding": {
            "offline_video_stream/teacher_25step": teacher_encoding.to_summary(),
            "offline_video_stream/student_4step": student_encoding.to_summary(),
            "offline_video_stream/teacher_vs_student": comparison_encoding.to_summary(),
        },
        "videos": {
            "offline_video_stream/teacher_25step": str(teacher_encoding.path),
            "offline_video_stream/student_4step": str(student_encoding.path),
            "offline_video_stream/teacher_vs_student": str(comparison_encoding.path),
        },
    }

    run_url: str | None = None
    wandb_error: str | None = None
    wandb_status: dict[str, bool]
    try:
        logger = TrainingLogger(cfg)
        try:
            wandb_status = logger.log_videos(
                {
                    "offline_video_stream/teacher_25step": teacher_encoding.path,
                    "offline_video_stream/student_4step": student_encoding.path,
                    "offline_video_stream/teacher_vs_student": comparison_encoding.path,
                },
                fps=fps,
            )
            run_url = logger.run_url
        finally:
            logger.finish()
    except Exception as exc:
        wandb_error = f"{type(exc).__name__}: {exc}"
        wandb_status = {
            "offline_video_stream/teacher_25step": False,
            "offline_video_stream/student_4step": False,
            "offline_video_stream/teacher_vs_student": False,
        }
        print(
            "WARNING: W&B video upload failed for offline visualization; "
            f"local artifacts remain under {output_dir}; error={wandb_error}"
        )

    metadata_payload["wandb_video_status"] = wandb_status
    metadata_payload["wandb_run_url"] = run_url
    if wandb_error is not None:
        metadata_payload["wandb_error"] = wandb_error
    metadata_path = write_metadata(
        output_dir / "offline_video_stream_metadata.json",
        metadata_payload,
    )

    return {
        "output_dir": output_dir,
        "metadata_path": metadata_path,
        "wandb_run_url": run_url,
        "videos": {
            "teacher_25step": teacher_encoding.path,
            "student_4step": student_encoding.path,
            "teacher_vs_student": comparison_encoding.path,
        },
        "encoding": {
            "teacher_25step": teacher_encoding,
            "student_4step": student_encoding,
            "teacher_vs_student": comparison_encoding,
        },
        "wandb_video_status": wandb_status,
        "selection": selection,
    }


@torch.no_grad()
def sample_student_video_flowmap(
    student: LingBotVideoFlowMapStudent,
    z_noise: torch.Tensor,
    context: Any,
    *,
    num_steps: int = 4,
) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    z_state = z_noise.clone()
    batch_size = z_state.shape[0]
    s_edges = torch.linspace(0.0, 1.0, num_steps + 1, device=z_state.device, dtype=torch.float32)
    for index in range(num_steps):
        s_src = torch.full((batch_size,), float(s_edges[index]), device=z_state.device)
        s_tgt = torch.full((batch_size,), float(s_edges[index + 1]), device=z_state.device)
        pred_u = student.forward_video_flowmap(z_state, s_src, s_tgt, context)
        z_state = z_state + (s_tgt - s_src).view(batch_size, 1, 1, 1, 1).to(z_state) * pred_u
    return z_state


def select_offline_chunks(
    items: list[Mapping[str, Any]],
    *,
    max_chunks: int,
    selection_mode: str = "auto",
) -> OfflineSelection:
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive.")
    if not items:
        raise ValueError("Video manifest must not be empty.")
    if selection_mode not in {"auto", "stitch", "grid"}:
        raise ValueError("selection_mode must be one of auto, stitch, or grid.")
    if selection_mode == "grid":
        return OfflineSelection(
            indices=list(range(min(max_chunks, len(items)))),
            composition_mode="grid",
            reason="selection_mode=grid",
        )

    continuous = _find_continuous_chunk_group(items, max_chunks=max_chunks)
    if continuous is not None:
        return continuous
    if selection_mode == "stitch":
        raise ValueError(
            "selection_mode=stitch requires reliable episode and chunk metadata, "
            "but none was found."
        )
    return OfflineSelection(
        indices=list(range(min(max_chunks, len(items)))),
        composition_mode="grid",
        reason="no reliable continuous episode/chunk metadata found",
    )


def _save_latent_fallbacks(
    *,
    output_dir: Path,
    teacher_latent_chunks: list[torch.Tensor],
    student_latent_chunks: list[torch.Tensor],
) -> dict[str, list[Path]]:
    latent_dir = output_dir / "latent_fallbacks"
    latent_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, list[Path]] = {"teacher": [], "student": []}
    for index, tensor in enumerate(teacher_latent_chunks):
        path = latent_dir / f"teacher_latents_{index:04d}.pt"
        torch.save(tensor, path)
        paths["teacher"].append(path)
    for index, tensor in enumerate(student_latent_chunks):
        path = latent_dir / f"student_latents_{index:04d}.pt"
        torch.save(tensor, path)
        paths["student"].append(path)
    return paths


def _find_continuous_chunk_group(
    items: list[Mapping[str, Any]],
    *,
    max_chunks: int,
) -> OfflineSelection | None:
    for episode_key in EPISODE_KEYS:
        if not any(episode_key in item for item in items):
            continue
        for chunk_key in CHUNK_KEYS:
            if not any(chunk_key in item for item in items):
                continue
            groups: dict[Any, list[tuple[int, Mapping[str, Any]]]] = {}
            for index, item in enumerate(items):
                if episode_key not in item or chunk_key not in item:
                    continue
                groups.setdefault(item[episode_key], []).append((index, item))
            for group in sorted(groups.values(), key=len, reverse=True):
                ordered = sorted(group, key=lambda pair: _numeric_or_text(pair[1][chunk_key]))
                if len(ordered) < 2:
                    continue
                if not _chunk_order_is_reliable([item for _, item in ordered], chunk_key):
                    continue
                selected = ordered[:max_chunks]
                return OfflineSelection(
                    indices=[index for index, _ in selected],
                    composition_mode="stitch",
                    reason="found continuous episode/chunk metadata",
                    episode_key=episode_key,
                    chunk_key=chunk_key,
                )
    return None


def _chunk_order_is_reliable(items: list[Mapping[str, Any]], chunk_key: str) -> bool:
    if chunk_key in {"chunk_idx", "chunk_index", "segment_idx", "segment_index"}:
        values = [_optional_int(item.get(chunk_key)) for item in items]
        if any(value is None for value in values):
            return False
        return all(b == a + 1 for a, b in zip(values, values[1:]))
    if chunk_key == "start_frame":
        starts = [_optional_int(item.get("start_frame")) for item in items]
        if any(value is None for value in starts):
            return False
        ends = [_optional_int(item.get("end_frame")) for item in items]
        if all(value is not None for value in ends):
            return all(end <= start for end, start in zip(ends, starts[1:]) if end is not None)
        return starts == sorted(starts)
    return False


def _frame_start_id(meta: Mapping[str, Any], fallback: int) -> int:
    for key in ("frame_start_id", "start_frame", "chunk_start_frame"):
        value = _optional_int(meta.get(key))
        if value is not None:
            return value
    return fallback


def _offline_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    visualization = cfg.get("visualization")
    if not isinstance(visualization, Mapping):
        return {}
    section = visualization.get("offline_video_stream")
    return section if isinstance(section, Mapping) else {}


def _resolve_path(cfg: Mapping[str, Any], raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = resolve_repo_root(cfg) / path
    return path.resolve()


def _numeric_or_text(value: Any) -> tuple[int, Any]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (0, value)
    if isinstance(value, float):
        return (0, value)
    return (1, str(value))


def _optional_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _required_str(cfg: Mapping[str, Any], key: str) -> str:
    value = cfg.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"visualization.offline_video_stream.{key} must be a non-empty string.")
    return value


def _str_cfg(cfg: Mapping[str, Any], key: str, *, default: str) -> str:
    value = cfg.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"visualization.offline_video_stream.{key} must be a non-empty string.")
    return value


def _int_cfg(cfg: Mapping[str, Any], key: str, *, default: int) -> int:
    value = cfg.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"visualization.offline_video_stream.{key} must be an int.")
    return value


def _optional_int_cfg(cfg: Mapping[str, Any], key: str) -> int | None:
    value = cfg.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"visualization.offline_video_stream.{key} must be an int.")
    return value


def _float_cfg(cfg: Mapping[str, Any], key: str, *, default: float) -> float:
    value = cfg.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"visualization.offline_video_stream.{key} must be numeric.")
    return float(value)


def _bool_cfg(cfg: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = cfg.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"visualization.offline_video_stream.{key} must be bool.")
    return value
