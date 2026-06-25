#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe WAM-Policy v2 visualization runtime.")
    parser.add_argument(
        "probe",
        choices=[
            "probe-static-source",
            "probe-import",
            "probe-video-encoder",
            "probe-vae-decode",
            "probe-teacher-student-sampling",
            "probe-decode-and-encode",
        ],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--require-external-exists", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from wam_policy.utils.config import apply_overrides, load_config
    from wam_policy.utils.external import bootstrap_external_repos

    args = build_parser().parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.overrides)
    bootstrap_external_repos(cfg, require_exists=args.require_external_exists)
    output_dir = _probe_output_dir(cfg, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probes: dict[str, Callable[[Mapping[str, Any], Path], Mapping[str, Any]]] = {
        "probe-static-source": _probe_static_source,
        "probe-import": _probe_import,
        "probe-video-encoder": _probe_video_encoder,
        "probe-vae-decode": _probe_vae_decode,
        "probe-teacher-student-sampling": _probe_teacher_student_sampling,
        "probe-decode-and-encode": _probe_decode_and_encode,
    }
    return _run_probe(args.probe, probes[args.probe], cfg, output_dir)


def _run_probe(
    name: str,
    fn: Callable[[Mapping[str, Any], Path], Mapping[str, Any]],
    cfg: Mapping[str, Any],
    output_dir: Path,
) -> int:
    summary_path = output_dir / f"{name}_summary.json"
    try:
        summary = dict(fn(cfg, output_dir))
        summary.update({"probe": name, "status": "passed"})
        _write_json(summary_path, summary)
        print(f"PASSED:{name}:summary={summary_path}")
        return 0
    except Exception as exc:
        summary = {
            "probe": name,
            "status": "failed",
            "failed_stage": _failed_stage(name),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "adapter_location": "src/wam_policy/adapters/lingbot_va.py",
        }
        _write_json(summary_path, summary)
        print(f"FAILED:{name}:summary={summary_path}", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _probe_static_source(cfg: Mapping[str, Any], output_dir: Path) -> Mapping[str, Any]:
    from wam_policy.visualization.static import static_visualization_check

    return {
        "offline": static_visualization_check(cfg, "offline-video-stream"),
        "robotwin": static_visualization_check(cfg, "robotwin-rollout"),
        "output_dir": str(output_dir),
    }


def _probe_import(cfg: Mapping[str, Any], output_dir: Path) -> Mapping[str, Any]:
    del output_dir
    import torch
    import wam_policy
    from wam_policy.adapters.lingbot_va import LingBotVAAdapter
    from wam_policy.visualization.video import probe_video_encoder

    adapter = LingBotVAAdapter(cfg)
    try:
        from wan_va.modules.utils import load_vae
    except Exception as exc:
        raise RuntimeError(f"LingBot/Wan VAE import failed: {exc}") from exc
    try:
        import wandb  # noqa: F401

        wandb_import = True
    except ImportError:
        wandb_import = False
    return {
        "wam_policy_module": str(Path(wam_policy.__file__).resolve()),
        "torch_version": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "torch_cuda_available": torch.cuda.is_available(),
        "lingbot_repo": str(adapter.repo_path),
        "cuda_dependency_scan": _scan_lingbot_cuda_dependencies(adapter.repo_path),
        "vae_loader": repr(load_vae),
        "wandb_import": wandb_import,
        "encoder_probe_callable": repr(probe_video_encoder),
    }


def _probe_video_encoder(cfg: Mapping[str, Any], output_dir: Path) -> Mapping[str, Any]:
    del cfg
    from wam_policy.visualization.video import probe_video_encoder

    result = probe_video_encoder(output_dir, fps=10)
    return {"encoding": result.to_summary()}


def _probe_vae_decode(cfg: Mapping[str, Any], output_dir: Path) -> Mapping[str, Any]:
    import torch
    from wam_policy.adapters.lingbot_va import LingBotVAAdapter

    adapter = LingBotVAAdapter(cfg)
    vae = adapter.load_video_vae()
    z_clean = _load_first_manifest_latent(cfg)
    if z_clean.ndim == 4:
        z_clean = z_clean.unsqueeze(0)
    frames = adapter.decode_video_latents(
        vae,
        z_clean.to(device=adapter.device, dtype=adapter.torch_dtype),
    )
    tensor_path = output_dir / "vae_decode_probe_frames.pt"
    torch.save(frames, tensor_path)
    return {
        "failed_stage": None,
        "frames_shape": list(frames.shape),
        "frames_path": str(tensor_path),
        "device": str(adapter.device),
        "dtype": str(adapter.torch_dtype),
    }


def _probe_teacher_student_sampling(cfg: Mapping[str, Any], output_dir: Path) -> Mapping[str, Any]:
    import torch
    from wam_policy.adapters.checkpoint import load_student_checkpoint
    from wam_policy.adapters.lingbot_va import LingBotVAAdapter
    from wam_policy.models.lingbot_student import LingBotVideoFlowMapStudent
    from wam_policy.visualization.offline import sample_student_video_flowmap

    vis_cfg = _offline_vis_cfg(cfg)
    adapter = LingBotVAAdapter(cfg)
    teacher = adapter.load_pretrained_video_action_model().to(
        device=adapter.device,
        dtype=adapter.torch_dtype,
    )
    student = LingBotVideoFlowMapStudent(cfg, adapter)
    student.build()
    student.to(device=adapter.device, dtype=adapter.torch_dtype)
    load_student_checkpoint(
        student,
        _required_str(vis_cfg, "student_checkpoint_path"),
        strict=bool(vis_cfg.get("checkpoint_strict", False)),
        map_location="cpu",
    )
    z_clean, context = _load_first_manifest_sample(cfg)
    if z_clean.ndim == 4:
        z_clean = z_clean.unsqueeze(0)
    z_noise = torch.randn_like(z_clean, device=adapter.device, dtype=adapter.torch_dtype)
    teacher_latents = adapter.sample_teacher_video(
        teacher,
        z_noise,
        context,
        num_steps=int(vis_cfg.get("teacher_steps", 25)),
        snr_shift=float(vis_cfg.get("teacher_snr_shift", 5.0)),
        guidance_scale=float(vis_cfg.get("teacher_guidance_scale", 1.0)),
    )
    student_latents = sample_student_video_flowmap(
        student,
        z_noise,
        context,
        num_steps=int(vis_cfg.get("student_steps", 4)),
    )
    teacher_path = output_dir / "teacher_sampling_probe_latents.pt"
    student_path = output_dir / "student_sampling_probe_latents.pt"
    torch.save(teacher_latents.detach().cpu(), teacher_path)
    torch.save(student_latents.detach().cpu(), student_path)
    return {
        "teacher_latents_shape": list(teacher_latents.shape),
        "student_latents_shape": list(student_latents.shape),
        "teacher_latents_path": str(teacher_path),
        "student_latents_path": str(student_path),
    }


def _probe_decode_and_encode(cfg: Mapping[str, Any], output_dir: Path) -> Mapping[str, Any]:
    from wam_policy.adapters.lingbot_va import LingBotVAAdapter
    from wam_policy.visualization.video import write_video_artifact

    adapter = LingBotVAAdapter(cfg)
    vae = adapter.load_video_vae()
    z_clean = _load_first_manifest_latent(cfg)
    if z_clean.ndim == 4:
        z_clean = z_clean.unsqueeze(0)
    frames = adapter.decode_video_latents(
        vae,
        z_clean.to(device=adapter.device, dtype=adapter.torch_dtype),
    )
    result = write_video_artifact(frames[0], output_dir / "decode_and_encode_probe.mp4", fps=10)
    return {
        "frames_shape": list(frames.shape),
        "encoding": result.to_summary(),
    }


def _load_first_manifest_latent(cfg: Mapping[str, Any]) -> Any:
    z_clean, _ = _load_first_manifest_sample(cfg)
    return z_clean


def _load_first_manifest_sample(cfg: Mapping[str, Any]) -> tuple[Any, Any]:
    from wam_policy.data.video_latent_dataset import VideoLatentManifestDataset

    dataset = VideoLatentManifestDataset.from_config(cfg)
    if len(dataset) == 0:
        raise ValueError("Video manifest is empty.")
    sample = dataset[0]
    return sample["z_clean"], sample["context"]


def _probe_output_dir(cfg: Mapping[str, Any], raw_path: str | None) -> Path:
    from wam_policy.utils.paths import resolve_repo_root

    if raw_path is not None:
        path = Path(raw_path).expanduser()
    else:
        path = resolve_repo_root(cfg) / "results/v2/visualization_probe"
    if not path.is_absolute():
        path = resolve_repo_root(cfg) / path
    return path.resolve()


def _offline_vis_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    visualization = cfg.get("visualization")
    if not isinstance(visualization, Mapping):
        return {}
    offline = visualization.get("offline_video_stream")
    return offline if isinstance(offline, Mapping) else {}


def _required_str(cfg: Mapping[str, Any], key: str) -> str:
    value = cfg.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"visualization.offline_video_stream.{key} is required.")
    return value


def _failed_stage(probe_name: str) -> str:
    return {
        "probe-import": "visualization_import",
        "probe-video-encoder": "video_encoding_probe",
        "probe-vae-decode": "vae_decode",
        "probe-teacher-student-sampling": "teacher_student_sampling",
        "probe-decode-and-encode": "decode_and_encode",
    }.get(probe_name, "static_source")


def _scan_lingbot_cuda_dependencies(repo_path: Path) -> list[dict[str, Any]]:
    patterns = (
        ".cuda(",
        "torch.device(\"cuda",
        "torch.device('cuda",
        "flash_attn",
        "flash_attn_interface",
        "CUDAExtension",
        "cupy",
    )
    scan_roots = [
        repo_path / "wan_va" / "modules",
        repo_path / "wan_va" / "wan_va_server.py",
    ]
    matches: list[dict[str, Any]] = []
    for root in scan_roots:
        files = [root] if root.is_file() else sorted(root.rglob("*.py")) if root.is_dir() else []
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if any(pattern in line for pattern in patterns):
                    matches.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "text": line.strip(),
                        }
                    )
    return matches


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
