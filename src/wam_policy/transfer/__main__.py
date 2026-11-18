"""Stage 1.1 Cross-Morphology Policy Transfer (teacher SFT).

    # single node, N GPUs (CUDA or ROCm); paper: 8 ranks, configs/stage1_1_transfer/teacher_sft.json
    torchrun --standalone --nproc_per_node N -m wam_policy.transfer train --config CONFIG \\
        --cache-root CACHE --transformer NATIVE9_TRANSFORMER --output-dir RUN [--base-model BASE]
    # continue an interrupted run (same config and number of ranks)
    torchrun ... -m wam_policy.transfer train --config CONFIG ... --resume RUN/checkpoints/step_XXXXXXXX
    # release teacher format (wam_policy.json + transformer/), optional upload
    python -m wam_policy.transfer export --checkpoint RUN/checkpoints/step_00005000 --output TEACHER \\
        [--upload-repo USER/REPO --path-in-repo teacher]

A completed run also exports ``RUN/teacher`` automatically.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from wam_policy.common import site


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m wam_policy.transfer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    for key in ("cache-root", "transformer", "output-dir", "base-model", "run-name"):
        train.add_argument("--" + key)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--accumulation", type=int)
    train.add_argument("--device", choices=("cuda", "cpu"))
    train.add_argument("--attention", choices=("flex", "reference"))
    train.add_argument("--wandb-mode", choices=("disabled", "offline", "online"))
    train.add_argument("--resume", help="complete training checkpoint directory, or 'latest'")
    train.add_argument("--stop-after", type=int, help="stop (with a checkpoint) after this optimizer step")
    export = sub.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--upload-repo", default=None, help="HF model repo (default: WAM_UPLOAD_REPO; empty = no upload)")
    export.add_argument("--path-in-repo", default="")
    args = parser.parse_args(argv)

    if args.command == "train":
        from wam_policy.transfer.config import load_config
        from wam_policy.transfer.engine import run_training
        overrides = {k: getattr(args, k) for k in ("cache_root", "transformer", "output_dir", "base_model",
                                                    "run_name", "max_steps", "accumulation", "device",
                                                    "attention", "wandb_mode")}
        config = load_config(args.config, **overrides)
        resume = args.resume
        if resume == "latest":
            from wam_policy.transfer.checkpoint import latest_checkpoint
            resume = latest_checkpoint(Path(config.output_dir) / "checkpoints")
            if resume is None:
                raise FileNotFoundError(f"No complete checkpoint under {config.output_dir}/checkpoints")
        result = run_training(config, resume=resume, stop_after=args.stop_after)
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps(result, indent=2, default=str))
    else:
        from wam_policy.transfer.checkpoint import export_teacher, upload
        output = export_teacher(args.checkpoint, args.output)
        result = {"teacher": str(output)}
        repo = args.upload_repo if args.upload_repo is not None else site.get("WAM_UPLOAD_REPO")
        if repo:
            result["upload"] = upload(output, repo, args.path_in_repo)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
