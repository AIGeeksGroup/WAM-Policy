"""Check the released model directories before the policy server is started.

Loading a WAM-Policy model takes minutes; a wrong directory would only fail after
that. This check reads the small ``wam_policy.json`` files of the release
(format ``wam-policy-checkpoint-v1``, see ``src/wam_policy/models/metadata.py``)
and the task's cached prompt embedding, and confirms that the server would take
the images this bridge sends (``dual-rgb-360p-v1``: both views 640x360).

Standard library only, so the launcher can run it with any ``python3``.

    python3 -m wam_go2.wam.model_check --policy opsd_student --teacher <dir> --adapter <dir> \\
        --prompt-root <dir> --task push_box
    # exit 0: consistent; 3: problems (one per line); 2: bad arguments
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from . import settings, tasks

FORMAT = "wam-policy-checkpoint-v1"
STUDENT_METHOD = {"opsd_student": "opsd", "mopd_student": "mopd"}
ADAPTER_WEIGHTS = {"student": "student_adapter.pt", "specialist": "adapter.pt"}


def _metadata(directory: Path, problems: List[str], what: str) -> Optional[dict]:
    path = Path(directory) / "wam_policy.json"
    if not path.is_file():
        problems.append(f"{what}: no wam_policy.json in {directory}")
        return None
    try:
        meta = json.loads(path.read_text())
    except ValueError as error:
        problems.append(f"{what}: {path} is not JSON ({error})")
        return None
    if meta.get("format") != FORMAT:
        problems.append(f"{what}: {path} is not a {FORMAT} file")
        return None
    return meta


def problems(policy: str, teacher: Path, adapter: Optional[Path], prompt_root: Path, task: str) -> List[str]:
    """Lines describing what is wrong ([] = the server can be started for this task)."""
    found: List[str] = []
    if policy not in settings.POLICIES:
        return [f"policy {policy!r} is not one of {', '.join(settings.POLICIES)}"]
    try:
        entry = tasks.task(task)
    except tasks.TaskError as error:
        return [str(error)]

    meta = _metadata(teacher, found, "teacher")
    if meta is not None:
        if meta.get("kind") != "teacher":
            found.append(f"teacher: {teacher} holds a {meta.get('kind')!r} model, not the Stage 1.1 teacher")
        elif meta.get("rgb_profile") != settings.RGB_PROFILE:
            found.append(f"teacher: rgb_profile is {meta.get('rgb_profile')!r}, the bridge sends {settings.RGB_PROFILE}")

    if policy == "teacher":
        if adapter is not None:
            found.append("policy teacher takes no adapter")
    elif adapter is None:
        found.append(f"policy {policy} needs an adapter directory")
    else:
        meta = _metadata(adapter, found, "adapter")
        if meta is not None:
            kind = meta.get("kind")
            if policy in STUDENT_METHOD:
                if kind != "student" or meta.get("method") != STUDENT_METHOD[policy]:
                    found.append(f"adapter: {adapter} is a {kind} ({meta.get('method')}), "
                                 f"not a {STUDENT_METHOD[policy]} student")
            elif kind != "specialist":
                found.append(f"adapter: {adapter} is a {kind}, not a specialist")
            elif meta.get("task") != task:
                found.append(f"adapter: the specialist {adapter} was trained for {meta.get('task')!r}, not {task!r}")
            weights = ADAPTER_WEIGHTS.get(kind)
            if weights and not (Path(adapter) / weights).is_file():
                found.append(f"adapter: {weights} missing in {adapter}")

    embedding = Path(prompt_root) / task / "embedding.pt"
    if not embedding.is_file():
        found.append(f"prompt embedding of {task} missing: {embedding}")
    if not entry.get("instruction"):
        found.append(f"task {task} has no instruction in config/wam_tasks.json")
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", required=True)
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--prompt-root", required=True, type=Path)
    ap.add_argument("--task", required=True)
    args = ap.parse_args(argv)
    lines = problems(args.policy, args.teacher, args.adapter, args.prompt_root, args.task)
    for line in lines:
        print(f"model_check: {line}")
    return 3 if lines else 0


if __name__ == "__main__":
    raise SystemExit(main())
