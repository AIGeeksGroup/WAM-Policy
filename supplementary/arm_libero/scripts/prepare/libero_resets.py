#!/usr/bin/env python
"""Prepare the LIBERO-Long reset cases (train 100 / validation 20 / test 50 per task) with the CPU worker.
The test cases are the first 50 official init states of each task. Does not run a policy."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arm_opsd.sim.libero.worker_client import WorkerClient
from arm_opsd.assets.resets import (SPLIT_COUNTS, load_official_states, reset_seed,
                                          save_manifest, task_descriptors)


def write_progress(path, value):
    temporary = path.with_suffix(".json.pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--worker-env", required=True)
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--count-per-task", type=int, help="cases per split and task (default: train 100, validation 20, test 50)")
    parser.add_argument("--splits", nargs="+", choices=tuple(SPLIT_COUNTS), default=tuple(SPLIT_COUNTS),
                        help="Evaluation-only hosts need just test; training requires all three")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to replace a reset manifest directory")
    staging = output.with_name(output.name + ".partial")
    staging.mkdir(parents=True)
    (staging / "launch.json").write_text(json.dumps({"pid": os.getpid(), "argv": sys.argv,
        "started_unix": time.time(), "model_loaded": False, "policy_execution": False}, indent=2))
    counts = ({s: args.count_per_task for s in args.splits} if args.count_per_task
              else {s: SPLIT_COUNTS[s] for s in args.splits})
    cases, seen = [], set()
    with (staging / "worker.stderr.log").open("wb") as log, WorkerClient(
            python=args.python, env=json.loads(Path(args.worker_env).read_text()), stderr=log) as client:
        # Test is fixed first; train/validation collisions are rejected without looking at policy SR.
        for split in [s for s in ("test", "train", "validation") if s in counts]:
            for task in task_descriptors(args.libero_root):
                task_id = task["task_id"]
                official = load_official_states(task["official_init_file"]) if split == "test" else None
                if official is not None and len(official) < counts[split]:
                    raise ValueError(f"task{task_id}: fewer than {counts[split]} distinct official cases")
                accepted, attempt = 0, 0
                while accepted < counts[split]:
                    if attempt >= counts[split] * 20:
                        raise RuntimeError("reset pool bounded attempts exhausted")
                    case = {**task, "split": split,
                            "reset_seed": reset_seed(args.master_seed, split, task_id, attempt)}
                    if official is not None:
                        case["init_state"] = official[accepted]
                        case["official_index"] = accepted
                    else:
                        case["capture_random_init"] = True
                    case_id = f"{split}-task{task_id:02d}-case{accepted:04d}"
                    attempt += 1
                    # Invalid official cases fail; train/validation only reject protocol-invalid resets.
                    try:
                        response = client.reset(case, episode_id=f"{case_id}-attempt{attempt}")
                    except Exception as error:
                        with (staging / "candidates.jsonl").open("a") as events:
                            events.write(json.dumps({"case_id": case_id, "attempt": attempt,
                                "reset_seed": case["reset_seed"], "status": "failed",
                                "error": f"{type(error).__name__}: {error}"}) + "\n")
                        raise  # A failed client is not silently recreated or an official case replaced.
                    info = response["reset_info"]
                    with (staging / "candidates.jsonl").open("a") as events:
                        events.write(json.dumps({"case_id": case_id, "attempt": attempt,
                            "reset_seed": case["reset_seed"],
                            "status": "duplicate" if info["reset_key"] in seen else "accepted"}) + "\n")
                    if info["reset_key"] in seen:
                        if official is not None:
                            raise ValueError("duplicate physical official test case")
                        continue
                    seen.add(info["reset_key"])
                    case.pop("capture_random_init", None)
                    case.update({k: info[k] for k in ("init_state", "fixtures", "reset_key")}, case_id=case_id)
                    cases.append(case)
                    accepted += 1
                    write_progress(staging / "progress.json", {"complete": False,
                        "accepted_total": len(cases), "expected_total": sum(counts.values()) * 10,
                        "split": split, "task_id": task_id, "accepted_in_task": accepted,
                        "updated_unix": time.time()})
                    print(case_id, flush=True)
    save_manifest(staging, cases, counts=counts, master_seed=args.master_seed)
    (staging / "preparation.json").write_text(json.dumps({"count_per_task": counts,
        "case_count": len(cases), "no_policy_execution": True}, indent=2))
    write_progress(staging / "progress.json", {"complete": True, "accepted_total": len(cases),
        "expected_total": sum(counts.values()) * 10, "updated_unix": time.time()})
    staging.rename(output)


if __name__ == "__main__":
    main()
