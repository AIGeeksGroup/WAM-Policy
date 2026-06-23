#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path
from typing import Any, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload WAM-Policy admin diagnostic logs to a Weights & Biases artifact."
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", default="online")
    parser.add_argument("--dir", default="wandb")
    parser.add_argument("--resume", default="allow")
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--slurm-out", default=None)
    parser.add_argument("--slurm-err", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    upload_admin_logs_to_wandb(
        job_id=args.job_id,
        log_dir=Path(args.log_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        project=args.project,
        name=args.name,
        run_id=args.run_id,
        mode=args.mode,
        wandb_dir=Path(args.dir),
        resume=args.resume,
        exit_code=args.exit_code,
        entity=args.entity,
        slurm_out=Path(args.slurm_out) if args.slurm_out else None,
        slurm_err=Path(args.slurm_err) if args.slurm_err else None,
    )
    return 0


def upload_admin_logs_to_wandb(
    *,
    job_id: str,
    log_dir: Path,
    checkpoint_dir: Path,
    project: str,
    name: str,
    run_id: str,
    mode: str,
    wandb_dir: Path,
    resume: str,
    exit_code: int,
    entity: str | None = None,
    slurm_out: Path | None = None,
    slurm_err: Path | None = None,
) -> list[Path]:
    wandb = importlib.import_module("wandb")
    log_dir = log_dir.expanduser().resolve()
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    wandb_dir = wandb_dir.expanduser().resolve()
    wandb_dir.mkdir(parents=True, exist_ok=True)

    log_files = _collect_log_files(
        job_id=job_id,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        slurm_out=slurm_out,
        slurm_err=slurm_err,
    )

    init_kwargs: dict[str, Any] = {
        "project": project,
        "name": name,
        "id": run_id,
        "resume": resume,
        "mode": mode,
        "dir": str(wandb_dir),
        "job_type": "admin-diagnostic",
        "tags": ["video_ffm", "admin_logs"],
        "config": {
            "slurm_job_id": job_id,
            "admin_log_dir": str(log_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "exit_code": exit_code,
        },
    }
    if entity:
        init_kwargs["entity"] = entity

    run = wandb.init(**init_kwargs)
    artifact = wandb.Artifact(
        name=_artifact_name(job_id),
        type="admin-logs",
        metadata={
            "slurm_job_id": job_id,
            "exit_code": exit_code,
            "log_dir": str(log_dir),
            "checkpoint_dir": str(checkpoint_dir),
        },
    )
    for path in log_files:
        artifact.add_file(str(path), name=path.name)

    run.log_artifact(artifact)
    wandb.finish()
    return log_files


def _collect_log_files(
    *,
    job_id: str,
    log_dir: Path,
    checkpoint_dir: Path,
    slurm_out: Path | None,
    slurm_err: Path | None,
) -> list[Path]:
    files: list[Path] = []
    status_file = log_dir / "status.txt"
    if status_file.is_file():
        files.append(status_file)
    if log_dir.is_dir():
        files.extend(sorted(log_dir.glob("*.log")))

    checkpoint_listing = _write_checkpoint_listing(job_id, log_dir, checkpoint_dir)
    if checkpoint_listing is not None:
        files.append(checkpoint_listing)

    for path in (slurm_out, slurm_err):
        if path is not None and path.expanduser().is_file():
            files.append(path.expanduser().resolve())

    return _dedupe_existing_files(files)


def _write_checkpoint_listing(job_id: str, log_dir: Path, checkpoint_dir: Path) -> Path | None:
    if not log_dir.is_dir():
        return None

    listing_path = log_dir / f"checkpoint_files_{job_id}.txt"
    lines: list[str] = []
    if checkpoint_dir.is_dir():
        for path in sorted(checkpoint_dir.iterdir()):
            if path.is_file():
                lines.append(f"{path} {path.stat().st_size} bytes")
    else:
        lines.append(f"checkpoint_dir_missing: {checkpoint_dir}")
    listing_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return listing_path


def _dedupe_existing_files(paths: Sequence[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _artifact_name(job_id: str) -> str:
    safe_job_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", job_id).strip("-")
    return f"wam-video-ffm-admin-logs-{safe_job_id or 'unknown'}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
