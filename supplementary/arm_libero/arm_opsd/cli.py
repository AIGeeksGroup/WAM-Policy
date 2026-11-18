"""Training commands: python -m arm_opsd.cli <command> (console script: arm-opsd).

  download-base-model   robbyant/lingbot-va-posttrain-libero-long at the paper's revision
  prepare-prompts       T5 embeddings of the ten LIBERO-Long instructions (from the reset manifest)
  build-teacher-cache   teacher rollouts -> teacher cache (torchrun, method.world_size ranks)
  train                 one student (torchrun, method.world_size ranks)
  export                arm_libero.json + student_adapter.pt from a training checkpoint
  tiny-smoke            the whole training chain on a tiny random LingBot-VA, no weights or physics
"""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="arm-opsd", description="OPSD on the arm WAM LingBot-VA (LIBERO-Long)")
    sub = parser.add_subparsers(dest="command", required=True)

    def config_args(p, prompts=True):
        p.add_argument("--config", required=True, help="configs/students/<variant>.json (or a resolved-config.json)")
        p.add_argument("--site", help="site file with the machine paths (configs/site.example.json)")
        if prompts:
            p.add_argument("--prompts", required=True, help="prompt bank from prepare-prompts")

    p = sub.add_parser("download-base-model")
    p.add_argument("--output", required=True)
    p.add_argument("--revision", help="default: the revision used in the paper")
    p = sub.add_parser("prepare-prompts")
    config_args(p, prompts=False)
    p.add_argument("--output", required=True)
    p = sub.add_parser("build-teacher-cache")
    config_args(p)
    p.add_argument("--output", required=True)
    p.add_argument("--contexts", type=int, default=5000)
    p = sub.add_parser("train")
    config_args(p)
    p.add_argument("--resume", help="checkpoints/round-NNNNNN of an earlier run of the same config")
    p = sub.add_parser("export")
    config_args(p, prompts=False)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--variant", choices=("ema", "raw"), default="ema")
    p = sub.add_parser("tiny-smoke")
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args(argv)

    if args.command == "tiny-smoke":
        from arm_opsd.testing.tiny import tiny_smoke
        print(json.dumps(tiny_smoke(args.output, device=args.device, rounds=args.rounds), indent=2))
        return
    if args.command == "download-base-model":
        from arm_opsd.assets.base_model import download_base_model
        from arm_opsd.config import BASE_REVISION
        print(json.dumps(download_base_model(args.output, revision=args.revision or BASE_REVISION), indent=2))
        return
    from arm_opsd.config import ArmConfig
    config = ArmConfig.load(args.config, site=args.site)
    if args.command == "train":
        from arm_opsd.runtime import train
        train(args.config, args.prompts, site=args.site, resume=args.resume)
    elif args.command == "build-teacher-cache":
        from arm_opsd.runtime import build_target_cache
        print(json.dumps(build_target_cache(args.config, args.prompts, args.output, contexts=args.contexts,
                                            site=args.site), indent=2))
    elif args.command == "prepare-prompts":
        from arm_opsd.assets.base_model import prepare_prompt_bank
        from arm_opsd.assets.resets import load_manifest
        manifest = load_manifest(config.machine.reset_manifest)
        print(json.dumps(prepare_prompt_bank(config.machine.model_root, manifest["cases"], args.output,
                                             device=config.machine.device), indent=2))
    elif args.command == "export":
        from arm_opsd.runtime import export_checkpoint
        print(json.dumps(export_checkpoint(args.config, args.checkpoint, args.output, variant=args.variant,
                                           site=args.site, device=config.machine.device), indent=2))


if __name__ == "__main__":
    main()
