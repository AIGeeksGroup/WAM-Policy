#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any, Mapping, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload WAM visualization outputs to W&B.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary", action="append", default=[])
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", default="allow")
    parser.add_argument("--mode", default="online")
    parser.add_argument("--dir", default="wandb")
    parser.add_argument("--enabled", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Visualization output directory does not exist: {output_dir}")
    tar_path = _tar_output_dir(output_dir)
    print(f"visualization_tar_gz: {tar_path}")

    videos = _collect_videos([Path(path) for path in args.summary], output_dir)
    upload_summary_path = output_dir / "wandb_upload_summary.json"
    if not _wandb_enabled(args.enabled):
        _write_json(
            upload_summary_path,
            {
                "wandb_upload": "skipped",
                "tar_gz": str(tar_path),
                "video_keys": sorted(videos),
            },
        )
        print("wandb_upload: skipped")
        return 0

    try:
        wandb = importlib.import_module("wandb")
        init_kwargs: dict[str, Any] = {
            "project": args.project,
            "mode": args.mode,
            "dir": str(Path(args.dir).expanduser().resolve()),
            "resume": args.resume,
            "config": {
                "visualization_output_dir": str(output_dir),
                "visualization_tar_gz": str(tar_path),
            },
        }
        if args.entity:
            init_kwargs["entity"] = args.entity
        if args.name:
            init_kwargs["name"] = args.name
        if args.run_id:
            init_kwargs["id"] = args.run_id

        run = wandb.init(**init_kwargs)
        payload = {}
        for key, path in videos.items():
            suffix = path.suffix.lower().lstrip(".")
            if suffix in {"mp4", "gif"}:
                payload[key] = wandb.Video(str(path), format=suffix)
        if payload:
            wandb.log(payload)
            print(f"wandb_video_keys: {sorted(payload)}")
        artifact = wandb.Artifact(
            name=_artifact_name(args.run_id, output_dir),
            type="visualization-artifacts",
            metadata={"output_dir": str(output_dir), "tar_gz": str(tar_path)},
        )
        artifact.add_file(str(tar_path), name=tar_path.name)
        run.log_artifact(artifact)
        wandb.finish()
    except Exception as exc:
        _write_json(
            upload_summary_path,
            {
                "wandb_upload": "failed",
                "tar_gz": str(tar_path),
                "output_dir": str(output_dir),
                "video_keys": sorted(videos),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        print("wandb_upload: failed")
        print(f"visualization_output_dir: {output_dir}")
        print(f"visualization_tar_gz: {tar_path}")
        print(f"error: {type(exc).__name__}: {exc}")
        return 1
    _write_json(
        upload_summary_path,
        {
            "wandb_upload": "succeeded",
            "tar_gz": str(tar_path),
            "video_keys": sorted(payload),
            "artifact": artifact.name,
        },
    )
    print("wandb_upload: succeeded")
    return 0


def _collect_videos(summary_paths: list[Path], output_dir: Path) -> dict[str, Path]:
    videos: dict[str, Path] = {}
    for raw_path in summary_paths:
        path = raw_path.expanduser()
        if not path.is_absolute():
            path = output_dir / path
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if not isinstance(summary, Mapping):
            continue
        raw_videos = summary.get("videos")
        if not isinstance(raw_videos, Mapping):
            continue
        for key, raw_video_path in raw_videos.items():
            if not isinstance(key, str) or not isinstance(raw_video_path, str):
                continue
            video_path = Path(raw_video_path).expanduser()
            if not video_path.is_absolute():
                video_path = output_dir / video_path
            if video_path.exists():
                videos[key] = video_path.resolve()
    return videos


def _tar_output_dir(output_dir: Path) -> Path:
    tar_path = output_dir.with_suffix(".tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(output_dir.parent))
    return tar_path.resolve()


def _wandb_enabled(enabled: str) -> bool:
    if enabled in {"1", "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"}:
        return True
    if enabled in {"0", "false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF"}:
        return False
    if enabled in {"auto", "Auto", "AUTO"}:
        return bool(os.environ.get("WANDB_API_KEY"))
    raise ValueError("--enabled must be bool-like or auto.")


def _artifact_name(run_id: str | None, output_dir: Path) -> str:
    raw = run_id or output_dir.name or "unknown"
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in raw)
    return f"wam-policy-visualization-{safe}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
