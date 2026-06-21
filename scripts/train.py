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
    parser = argparse.ArgumentParser(description="WAM-Policy training entrypoint.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value with dotted keys, for example runtime.device=cuda.",
    )
    parser.add_argument(
        "--dry-run",
        choices=["static"],
        default=None,
        help="Run static config and dispatch checks without constructing runtime objects.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from wam_policy.opd.pipelines.v2_anyflow_dual_head import run as run_v2_pipeline
    from wam_policy.utils.config import apply_overrides, load_config
    from wam_policy.utils.external import bootstrap_external_repos
    from wam_policy.utils.paths import resolve_repo_root

    args = build_parser().parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.overrides)
    repo_root = resolve_repo_root(cfg)
    external_paths = bootstrap_external_repos(cfg, require_exists=False)

    dry_run = args.dry_run == "static"
    result = run_v2_pipeline(cfg, dry_run=dry_run)

    if dry_run:
        print("Static dry-run passed")
        print(f"repo_root: {repo_root}")
        for path in external_paths:
            print(f"external_repo: {path}")
        if isinstance(result, dict):
            print(f"pipeline: {result.get('pipeline')}")
            print(f"stage: {result.get('stage')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
