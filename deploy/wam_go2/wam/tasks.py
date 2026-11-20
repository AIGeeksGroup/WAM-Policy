"""The ten QLM-Bench tasks the robot can run (config/wam_tasks.json).

Standard library only: the shell scripts call it with the system python3.

    python3 -m wam_go2.wam.tasks list
    python3 -m wam_go2.wam.tasks get <task> instruction|entry_ee|task_profile_version

``instruction`` is the exact training prompt (the policy server selects the cached
prompt embedding by it), ``entry_ee`` the front-left foot target the controller
enters POLICY with (every training episode starts from RAMBO's default command),
``task_profile_version`` the QLM-Bench runtime profile reported at reset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[2]
CATALOGUE = DEPLOY / "config" / "wam_tasks.json"
FIELDS = ("instruction", "entry_ee", "task_profile_version")


class TaskError(ValueError):
    pass


def load(path: Path = CATALOGUE) -> dict:
    return json.loads(Path(path).read_text())


def ids(catalogue: dict | None = None) -> list:
    return [entry["id"] for entry in (catalogue or load())["tasks"]]


def task(task_id: str, catalogue: dict | None = None) -> dict:
    catalogue = catalogue or load()
    for entry in catalogue["tasks"]:
        if entry["id"] == task_id:
            return entry
    raise TaskError(f"unknown task {task_id!r} (tasks: {', '.join(ids(catalogue))})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    get = sub.add_parser("get")
    get.add_argument("task")
    get.add_argument("field", choices=FIELDS)
    args = ap.parse_args(argv)
    try:
        if args.command == "list":
            for entry in load()["tasks"]:
                print(f"{entry['id']}\t{entry['instruction']}")
            return 0
        value = task(args.task)[args.field]
    except TaskError as error:
        print(f"tasks: {error}", file=sys.stderr)
        return 2
    print(" ".join(f"{v:g}" for v in value) if isinstance(value, list) else value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
