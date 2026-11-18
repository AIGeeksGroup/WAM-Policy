"""Stage 1.1 RGB + proprioception ablation.

  convert  native9 transformer -> native9 + proprioception transformer (initial weights)
  data     10D states + train-only state normalizer (needs the Stage 1.1 frozen cache)
  train    whole-episode SFT (run under torchrun; see scripts/ablations/proprioception/train.sh)
  export   training checkpoint -> teacher directory (wam_policy.json + transformer/)
  serve    qlm-live-v1 policy server for QLM-Bench Standard100
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m wam_policy.ablations.proprioception", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert")
    convert.add_argument("--native9-transformer", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--symlink", action="store_true", help="link unchanged native9 shards instead of copying")
    data = sub.add_parser("data")
    data.add_argument("--config", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--run-name", default="proprioception-sft5000")
    train.add_argument("--resume", help="training checkpoint directory (.../checkpoints/step_XXXXXXXX)")
    train.add_argument("--max-steps", type=int, help="stop earlier than the config's max_steps")
    export = sub.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    from .serving import add_arguments
    add_arguments(sub.add_parser("serve"))
    args = parser.parse_args(argv)

    if args.command == "convert":
        from .conversion import convert as run
        result = run(args.native9_transformer, args.output, symlink=args.symlink)
    elif args.command == "data":
        from .config import load_config
        from .data import prepare_states
        result = prepare_states(load_config(args.config))
    elif args.command == "train":
        from .config import load_config
        from .engine import run_training
        result = run_training(load_config(args.config), run_name=args.run_name, resume=args.resume,
                              max_steps=args.max_steps)
    elif args.command == "export":
        from .export import export_teacher
        result = export_teacher(args.checkpoint, args.output)
    else:
        from .serving import serve
        return serve(args)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
