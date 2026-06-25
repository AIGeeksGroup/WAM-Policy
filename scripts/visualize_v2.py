#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WAM-Policy v2 visualization entrypoint.")
    parser.add_argument(
        "mode",
        choices=["offline-video-stream", "robotwin-rollout"],
        help="Visualization mode to run.",
    )
    parser.add_argument("--config", required=True, help="Path to a visualization YAML config.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value with dotted keys.",
    )
    parser.add_argument(
        "--dry-run",
        choices=["static"],
        default=None,
        help="Run static config checks without constructing runtime objects.",
    )
    parser.add_argument(
        "--require-external-exists",
        action="store_true",
        help="Require configured external repos to exist during bootstrap.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from wam_policy.utils.config import apply_overrides, load_config
    from wam_policy.utils.external import bootstrap_external_repos
    from wam_policy.utils.paths import resolve_repo_root

    args = build_parser().parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.overrides)
    repo_root = resolve_repo_root(cfg)
    external_paths = bootstrap_external_repos(
        cfg,
        require_exists=args.require_external_exists and args.dry_run != "static",
    )

    if args.dry_run == "static":
        from wam_policy.visualization.static import static_visualization_check

        result = static_visualization_check(cfg, args.mode)
        print("Static visualization dry-run passed")
        print(f"repo_root: {repo_root}")
        for path in external_paths:
            print(f"external_repo: {path}")
        print(f"mode: {args.mode}")
        print(f"output_dir: {result['output_dir']}")
        return 0

    if args.mode == "offline-video-stream":
        from wam_policy.visualization.offline import run_offline_video_stream_visualization

        result = run_offline_video_stream_visualization(cfg)
    elif args.mode == "robotwin-rollout":
        from wam_policy.visualization.robotwin import run_robotwin_rollout_visualization

        result = run_robotwin_rollout_visualization(cfg)
    else:
        raise ValueError(f"Unknown visualization mode: {args.mode}")

    print(f"output_dir: {result['output_dir']}")
    print(f"metadata_path: {result['metadata_path']}")
    wandb_run_url = result.get("wandb_run_url")
    if wandb_run_url:
        print(f"wandb_run_url: {wandb_run_url}")
    for key, path in sorted(result["videos"].items()):
        print(f"video_{key}: {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
