"""Enumerate every instruction a LIBERO-Pro plan can show, using the official parser.

Runs in the LIBERO-Pro environment. S21 replaces the instruction from reset; D21
switches it once mid-episode. The model environment embeds the resulting texts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arm_opsd.sim.libero_pro.plan import episode_key, load_plan


def instructions_for(bddl):
    from libero.libero.envs import bddl_utils
    from libero.libero.envs.nonphysical_perturbations import (NonPhysicalPerturbationRuntime,
                                                              build_execution_plan, default_asset_root)
    parsed = bddl_utils.robosuite_parse_problem(str(bddl))
    original = parsed["language_instruction"]
    if isinstance(original, list):  # Same join as BDDLBaseDomain.language_instruction.
        original = " ".join(str(value) for value in original)
    plan = build_execution_plan(parsed.get("perturbation_config", {}), default_asset_root())
    runtime = NonPhysicalPerturbationRuntime(plan)
    texts = [runtime.language_instruction(str(original), 0)]
    if plan.case_id == "D21":
        runtime._dynamic_event_step = 0
        texts.append(runtime.language_instruction(str(original), 1))
    return {"original": str(original), "reset": texts[0], "texts": texts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.task_manifest)
    manifest = json.loads(manifest_path.read_text())
    tasks = {(t["suite"], int(t["base_task_id"]), "base" if t["case_id"] == "BASE" else t["case_id"]): t
             for t in manifest["tasks"]}
    plan = load_plan(args.plan)
    rows = {}
    for spec in plan["episodes"]:
        task = tasks[(spec["suite"], int(spec["task_id"]), spec["variant"])]
        bddl = manifest_path.parent / task["bddl_path"]
        rows[episode_key(spec)] = {**instructions_for(bddl), "bddl_path": task["bddl_path"]}
    payload = {"schema": "arm-opsd-libero-pro-instructions-v1", "plan_id": plan["plan_id"],
               "episodes": rows, "texts": sorted({t for row in rows.values() for t in row["texts"]})}
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"episodes": len(rows), "texts": len(payload["texts"])}))


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
