"""Stage 2.2 M-OPD (task-routed specialist teachers distilled into the OPSD student).

Order: build the OPSD teacher cache first (``python -m wam_policy.opsd cache``), then

  python -m wam_policy.mopd cache --config configs/stage2_2_mopd/mopd.json --output CACHE_DIR \\
      --previous-cache OPSD_CACHE_RUN/learner/cache \\
      --expert push_box=SPECIALIST_DIR --expert open_room_door=SPECIALIST_DIR
  python -m wam_policy.mopd train --config configs/stage2_2_mopd/mopd.json --output RUN_DIR \\
      --resource offline_cache=CACHE_DIR/cache --resource initialized_student=OPSD_STUDENT_DIR \\
      --expert push_box=... --expert open_room_door=...

``cache`` collects only the expert tasks with their specialist teachers and merges them into the
OPSD cache (``CACHE_DIR/cache``); ``merge`` repeats the merge of an existing collection. The other
sub-commands are those of ``python -m wam_policy.opsd``; experts may also come from the config.
"""
import json
import sys

from wam_policy.opsd.__main__ import add_config, load, parser, run


def main(argv=None):
    p = parser("python -m wam_policy.mopd", experts=True)
    p.description = __doc__
    commands = next(a for a in p._actions if a.dest == "command")
    c = commands.choices["cache"]
    c.add_argument("--previous-cache", required=True, help="The OPSD teacher cache the specialist contexts extend")
    m = commands.add_parser("merge", help="Merge a finished specialist collection into the OPSD cache")
    add_config(m, experts=True)
    m.add_argument("--previous-cache", required=True)
    m.add_argument("--collected", required=True)
    m.add_argument("--output", required=True)
    args = p.parse_args(argv)
    from wam_policy.mopd.smoke import tiny_smoke
    if args.command in ("cache", "merge", "train"):
        config = load(args)
        if not config.resources.get("experts"):
            sys.exit("M-OPD needs specialist teachers: --expert TASK=DIR or resources.experts in the config")
        from wam_policy.mopd import expert_cache
        if args.command == "cache":
            result = expert_cache.build(config, args.output, args.previous_cache, execute=not args.dry_run)
        elif args.command == "merge":
            result = expert_cache.merge_report(args.previous_cache, args.collected, args.output, config)
        else:
            if not config.resources.get("initialized_student"):
                sys.exit("M-OPD continues the OPSD student: pass --resource initialized_student=<student dir>")
            result = run(args, tiny_smoke)
    else:
        result = run(args, tiny_smoke)
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
