"""Stage 1.1 data preparation.

    python -m wam_policy.data download --snapshot DIR            # the ten QLM-Bench releases
    python -m wam_policy.data inspect  --snapshot DIR [--cache DIR]
    python -m wam_policy.data fit      --snapshot DIR --cache DIR   # train-split action normalizer
    python -m wam_policy.data cache    --snapshot DIR --cache DIR --base-model DIR [--device cuda]

``--base-model`` is a LingBot-VA base snapshot (``vae/``, ``text_encoder/``, ``tokenizer/``).
The dataset repository/revision default to ``QLM_DATASET_REPO`` / ``QLM_DATASET_REVISION``.
"""
from __future__ import annotations

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m wam_policy.data", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {name: sub.add_parser(name) for name in ("download", "inspect", "fit", "cache")}
    for name, p in commands.items():
        p.add_argument("--snapshot", required=True, help="local dataset snapshot directory")
        p.add_argument("--releases", nargs="+", help="subset of release or task names (default: all ten)")
        if name in ("inspect", "fit", "cache"):
            p.add_argument("--cache", required=name != "inspect", help="frozen cache directory")
    commands["download"].add_argument("--repo")
    commands["download"].add_argument("--revision")
    commands["cache"].add_argument("--base-model", required=True)
    commands["cache"].add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if args.command == "download":
        from wam_policy.data.qlm_bench import download
        result = download(args.snapshot, repo=args.repo, revision=args.revision, names=args.releases)
    elif args.command == "inspect":
        from wam_policy.data.prepare import inspect
        result = inspect(args.snapshot, args.releases)
        if args.cache:
            from wam_policy.data.frozen_cache import FrozenEpisodes
            result["cache"] = FrozenEpisodes(args.cache).counts()
    elif args.command == "fit":
        from wam_policy.data.prepare import fit
        result = fit(args.snapshot, args.cache, args.releases)
    else:
        from wam_policy.data.frozen_cache import build_cache
        result = build_cache(args.snapshot, args.cache, args.base_model, device=args.device, names=args.releases)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
