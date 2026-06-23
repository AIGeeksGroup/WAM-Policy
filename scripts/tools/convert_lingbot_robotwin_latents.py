#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


DEFAULT_SOURCE_GLOB = "latents/**/*.pth"
DEFAULT_MAX_SAMPLES = 16
REQUIRED_KEYS = {
    "latent",
    "latent_num_frames",
    "latent_height",
    "latent_width",
    "text_emb",
}
CONTEXT_KEYS = (
    "frame_ids",
    "start_frame",
    "end_frame",
    "fps",
    "ori_fps",
)


@dataclass(frozen=True)
class ConvertedSample:
    sample_id: str
    source_path: Path
    z_clean_path: Path
    context_path: Path
    manifest_item: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert LingBot-VA RoboTwin WAN 2.2 latent .pth files into WAM-Policy "
            "z_clean/context split files plus a JSONL manifest."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="WAM-Policy repo root. Relative manifest/output paths are resolved under it.",
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="Root of the downloaded robbyant/robotwin-clean-and-aug-lerobot dataset.",
    )
    parser.add_argument(
        "--source-glob",
        default=DEFAULT_SOURCE_GLOB,
        help=f"Glob under source-root for LingBot latent files. Default: {DEFAULT_SOURCE_GLOB}",
    )
    parser.add_argument(
        "--output-root",
        default="datasets/latents/robotwin_video_ffm_debug",
        help="Directory for converted z_clean/context files.",
    )
    parser.add_argument(
        "--manifest",
        default="datasets/manifests/video_latents_debug.jsonl",
        help="Output JSONL manifest path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=(
            f"Maximum samples to convert. Default: {DEFAULT_MAX_SAMPLES}. "
            "Use 0 to convert all matched samples."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output files and manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and validate samples without writing converted files or manifest.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    source_root = _resolve_path(args.source_root, repo_root)
    output_root = _resolve_path(args.output_root, repo_root)
    manifest_path = _resolve_path(args.manifest, repo_root)

    samples = convert_lingbot_robotwin_latents(
        repo_root=repo_root,
        source_root=source_root,
        source_glob=args.source_glob,
        output_root=output_root,
        manifest_path=manifest_path,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    action = "validated" if args.dry_run else "converted"
    print(f"{action}_samples: {len(samples)}")
    print(f"source_root: {source_root}")
    print(f"output_root: {output_root}")
    print(f"manifest: {manifest_path}")
    return 0


def convert_lingbot_robotwin_latents(
    *,
    repo_root: Path,
    source_root: Path,
    source_glob: str = DEFAULT_SOURCE_GLOB,
    output_root: Path,
    manifest_path: Path,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[ConvertedSample]:
    repo_root = repo_root.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    output_root = _resolve_under_repo(output_root, repo_root)
    manifest_path = _resolve_under_repo(manifest_path, repo_root)

    _validate_inputs(
        source_root=source_root,
        source_glob=source_glob,
        output_root=output_root,
        manifest_path=manifest_path,
        max_samples=max_samples,
        overwrite=overwrite,
        dry_run=dry_run,
    )

    source_paths = _find_source_paths(source_root, source_glob, max_samples=max_samples)
    if not source_paths:
        raise FileNotFoundError(
            f"No LingBot latent .pth files matched {source_glob!r} under {source_root}."
        )

    converted: list[ConvertedSample] = []
    manifest_items: list[dict[str, Any]] = []
    for source_path in source_paths:
        sample = _convert_one(
            repo_root=repo_root,
            source_root=source_root,
            source_path=source_path,
            output_root=output_root,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        converted.append(sample)
        manifest_items.append(sample.manifest_item)

    if not dry_run:
        _write_manifest(manifest_path, manifest_items, overwrite=overwrite)
    return converted


def _convert_one(
    *,
    repo_root: Path,
    source_root: Path,
    source_path: Path,
    output_root: Path,
    overwrite: bool,
    dry_run: bool,
) -> ConvertedSample:
    item = _load_lingbot_latent(source_path)
    z_clean = _latent_to_z_clean(item, source_path)
    prompt = _optional_text(item.get("text"))
    context = _build_context(
        item,
        source_root=source_root,
        source_path=source_path,
        text=prompt,
    )

    relative_source = source_path.relative_to(source_root)
    relative_stem = _output_relative_stem(relative_source)
    sample_id = _sample_id(relative_stem)
    output_dir = output_root / relative_stem.parent
    z_clean_path = output_dir / f"{relative_stem.name}_z.pt"
    context_path = output_dir / f"{relative_stem.name}_context.pt"

    manifest_item = {
        "sample_id": sample_id,
        "z_clean": _relative_posix(z_clean_path, repo_root),
        "context": _relative_posix(context_path, repo_root),
        "prompt": prompt,
        "source": _relative_posix(source_path, repo_root),
        "shape": list(z_clean.shape),
        "dtype": _dtype_name(z_clean.dtype),
    }

    if not dry_run:
        _ensure_can_write(z_clean_path, overwrite=overwrite)
        _ensure_can_write(context_path, overwrite=overwrite)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(z_clean, z_clean_path)
        torch.save(context, context_path)

    return ConvertedSample(
        sample_id=sample_id,
        source_path=source_path,
        z_clean_path=z_clean_path,
        context_path=context_path,
        manifest_item=manifest_item,
    )


def _validate_inputs(
    *,
    source_root: Path,
    source_glob: str,
    output_root: Path,
    manifest_path: Path,
    max_samples: int,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError(f"source-root must be an existing directory: {source_root}")
    if Path(source_glob).is_absolute():
        raise ValueError("source-glob must be relative to source-root.")
    if max_samples < 0:
        raise ValueError("max-samples must be >= 0.")
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"output-root must be a directory: {output_root}")
    if manifest_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Manifest already exists. Use --overwrite: {manifest_path}")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite and not dry_run:
        raise FileExistsError(f"Output root is not empty. Use --overwrite: {output_root}")


def _find_source_paths(source_root: Path, source_glob: str, max_samples: int) -> list[Path]:
    paths = sorted(path.resolve() for path in source_root.glob(source_glob) if path.is_file())
    if max_samples:
        return paths[:max_samples]
    return paths


def _load_lingbot_latent(path: Path) -> dict[str, Any]:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError(f"LingBot latent file must contain a dict: {path}")

    missing = sorted(REQUIRED_KEYS.difference(loaded))
    if missing:
        raise ValueError(f"LingBot latent file is missing required keys {missing}: {path}")
    return loaded


def _latent_to_z_clean(item: dict[str, Any], source_path: Path) -> torch.Tensor:
    latent = item["latent"]
    text_emb = item["text_emb"]
    if not isinstance(latent, torch.Tensor):
        raise TypeError(f"'latent' must be a torch.Tensor: {source_path}")
    if latent.ndim != 2:
        raise ValueError(f"'latent' must have shape [F*H*W, C], got {tuple(latent.shape)}.")
    if not isinstance(text_emb, torch.Tensor):
        raise TypeError(f"'text_emb' must be a torch.Tensor: {source_path}")

    frames = _required_positive_int(item["latent_num_frames"], "latent_num_frames", source_path)
    height = _required_positive_int(item["latent_height"], "latent_height", source_path)
    width = _required_positive_int(item["latent_width"], "latent_width", source_path)
    expected_tokens = frames * height * width
    if latent.shape[0] != expected_tokens:
        raise ValueError(
            f"'latent' token count must equal F*H*W={expected_tokens}, "
            f"got {latent.shape[0]}: {source_path}"
        )

    channels = latent.shape[1]
    return latent.reshape(frames, height, width, channels).permute(3, 0, 1, 2).contiguous()


def _build_context(
    item: dict[str, Any],
    *,
    source_root: Path,
    source_path: Path,
    text: str,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "text_emb": item["text_emb"],
        "text": text,
        "source_latent": source_path.relative_to(source_root).as_posix(),
    }
    for key in CONTEXT_KEYS:
        context[key] = item.get(key)
    return context


def _write_manifest(
    manifest_path: Path,
    manifest_items: list[dict[str, Any]],
    *,
    overwrite: bool,
) -> None:
    _ensure_can_write(manifest_path, overwrite=overwrite)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item, sort_keys=True) for item in manifest_items]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists. Use --overwrite: {path}")


def _required_positive_int(value: Any, name: str, source_path: Path) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"'{name}' must be a positive int in {source_path}.")
    return value


def _sample_id(relative_stem: Path) -> str:
    return "__".join(relative_stem.parts)


def _output_relative_stem(relative_source: Path) -> Path:
    stem = relative_source.with_suffix("")
    if stem.parts and stem.parts[0] == "latents":
        return Path(*stem.parts[1:])
    return stem


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved.resolve()


def _resolve_under_repo(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _relative_posix(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
