"""Stage 1.2 OPSD (on-policy self-distillation of the low-NFE student).

  python -m wam_policy.opsd cache  --config configs/stage1_2_opsd/teacher_cache.json --output RUN_DIR
  python -m wam_policy.opsd train  --config configs/stage1_2_opsd/opsd.json --output RUN_DIR [--resume CKPT]
  python -m wam_policy.opsd export --config RUN_DIR/config.json --checkpoint RUN_DIR/learner/checkpoints/step_N --output DIR
  python -m wam_policy.opsd finalize-cache --cache RUN_DIR/learner/cache
  python -m wam_policy.opsd smoke  --output DIR      # tiny CPU self-check, no weights or simulator

Common options: --resource KEY=VALUE (resources, overrides ${ENV} placeholders), --set FIELD=VALUE
(config fields, e.g. --set world_size=1), --dry-run (print the process plan). rank-train/rank-cache
are the per-rank entry points the launcher starts under torchrun.
"""
import argparse
import json


def add_config(p, *, experts=False):
    p.add_argument("--config", required=True)
    p.add_argument("--resource", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    if experts:
        p.add_argument("--expert", action="append", default=[], metavar="TASK=DIR",
                       help="Stage 2.1 specialist directory routed to TASK")


# Resources a command does not read; their placeholders may stay unset.
UNUSED = {"cache": ("offline_cache", "initialized_student"), "merge": ("offline_cache", "initialized_student")}


def load(args):
    import sys
    from wam_policy.opsd.config import Config
    experts = None
    if getattr(args, "expert", None):
        from wam_policy.mopd.experts import expert_arguments
        experts = expert_arguments(args.expert)
    config = Config.load(args.config, overrides=args.set, resources=args.resource, experts=experts,
                         unused=UNUSED.get(args.command, ()))
    if config.resources.get("experts") and not args.command.startswith("rank-"):
        from wam_policy.mopd.experts import routing_table
        print("M-OPD teacher routing (other tasks: plain teacher):", file=sys.stderr)
        for task, row in routing_table(config).items():
            print(f"  {task} -> {row['specialist']} (round {row['round']})", file=sys.stderr)
    return config


def parser(prog=None, *, experts=False):
    p = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest="command", required=True)
    c = commands.add_parser("cache", help="Build the teacher target cache (teacher acts in the QLM streams)")
    add_config(c, experts=experts)
    c.add_argument("--output", required=True)
    c.add_argument("--contexts", type=int, help="Default: config cache_contexts")
    c.add_argument("--dry-run", action="store_true")
    c = commands.add_parser("train", help="Distill the student (torchrun learner + QLM workers)")
    add_config(c, experts=experts)
    c.add_argument("--output", required=True)
    c.add_argument("--resume")
    c.add_argument("--stop-after", type=int, help="Stop after this many updates (saves a checkpoint)")
    c.add_argument("--dry-run", action="store_true")
    c = commands.add_parser("export", help="Export a checkpoint's student (release student format)")
    add_config(c, experts=experts)
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--variant", choices=("raw", "ema"), default="ema")
    c.add_argument("--output", required=True)
    c = commands.add_parser("finalize-cache", help="Write the manifest of an interrupted cache build")
    c.add_argument("--cache", required=True)
    c = commands.add_parser("smoke", help="Tiny CPU self-check")
    c.add_argument("--output", required=True)
    for name in ("rank-train", "rank-cache"):
        c = commands.add_parser(name)
        add_config(c)
        c.add_argument("--output", required=True)
        c.add_argument("--resume")
        c.add_argument("--stop-after", type=int)
        c.add_argument("--contexts", type=int)
    return p


def export(config, checkpoint, variant, output):
    import torch
    from pathlib import Path
    from wam_policy.opsd.checkpoint import read_manifest, check_resume_config
    from wam_policy.opsd.export import export_student
    manifest = read_manifest(checkpoint)
    check_resume_config(manifest, config)
    state = torch.load(Path(checkpoint) / "state.pt", map_location="cpu", weights_only=False, mmap=True)
    adapter = state["student"] if variant == "raw" else state["student_ema"]
    if adapter is None:
        raise ValueError("The checkpoint precedes the EMA start step; export --variant raw")
    return export_student(adapter, output, config=config, variant=variant, updates=manifest["student_step"])


def run(args, smoke):
    if args.command == "finalize-cache":
        from wam_policy.opsd.targets import finalize_cache
        return finalize_cache(args.cache)
    if args.command == "smoke":
        return smoke(args.output)
    config = load(args)
    if args.command == "rank-train":
        from wam_policy.opsd.engine import train
        return train(config, args.output, resume=args.resume, stop_after=args.stop_after)
    if args.command == "rank-cache":
        from wam_policy.opsd.engine import build_target_cache
        return build_target_cache(config, args.output, contexts=args.contexts)
    if args.command == "export":
        return export(config, args.checkpoint, args.variant, args.output)
    from wam_policy.opsd.launch import launch
    if args.command == "cache":
        return launch(config, args.output, execute=not args.dry_run,
                      cache_contexts=args.contexts or config.cache_contexts)
    return launch(config, args.output, execute=not args.dry_run, stop_after=args.stop_after, resume=args.resume)


def main(argv=None):
    from wam_policy.opsd.smoke import tiny_smoke
    args = parser("python -m wam_policy.opsd").parse_args(argv)
    print(json.dumps(run(args, tiny_smoke), indent=1, default=str))


if __name__ == "__main__":
    main()
