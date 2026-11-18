#!/usr/bin/env python
"""Environment check of the CPU LIBERO worker: real reset, render and chunk receipts on every task
(no model). Writes check.json and one image per task and camera."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from arm_opsd.sim.libero.worker_client import WorkerClient
from arm_opsd.assets.resets import load_official_states, reset_seed, task_descriptors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True, help="independent CPU Python")
    parser.add_argument("--worker-env", required=True)
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="*", type=int, default=list(range(10)))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = task_descriptors(args.libero_root)
    evidence = {"schema": "arm-opsd-libero-cpu-check-v1", "model_loaded": False,
                "policy_evaluation": False, "tasks": [], "passed": False}
    try:
        with (output / "worker.stderr.log").open("wb") as log, WorkerClient(
                python=args.python, env=json.loads(Path(args.worker_env).read_text()), stderr=log) as client:
            evidence["initial_health"] = client.health()
            for task_id in args.tasks:
                task = tasks[task_id]
                states = load_official_states(task["official_init_file"])
                case = {**task, "reset_seed": reset_seed(42, "env-check", task_id, 0), "init_state": states[0]}
                episode_id = f"env-check-task{task_id}"
                initial = client.reset(case, episode_id)
                for camera, array in initial["observation"].items():
                    if np.std(array.astype(np.float64)) <= 1:
                        raise RuntimeError(f"invalid/blank scene image: {task_id}/{camera}")
                    from PIL import Image
                    Image.fromarray(array).save(output / f"task{task_id}-{camera.split('.')[-1]}.png")
                before = client.status()
                time.sleep(0.05)
                after = client.status()
                if before["timestep"] != after["timestep"] or not np.array_equal(before["state"], after["state"]):
                    raise RuntimeError("physics advanced while waiting")
                receipt = client.step_chunk(np.zeros((12, 7)), episode_id=episode_id,
                                            chunk_id=0, actor_version=0, request_id=f"task{task_id}-chunk0")
                duplicate = client.step_chunk(np.zeros((12, 7)), episode_id=episode_id,
                                              chunk_id=0, actor_version=0, request_id=f"task{task_id}-chunk0")
                if receipt["receipt_id"] != duplicate["receipt_id"] or not np.array_equal(
                        receipt["applied_actions"], duplicate["applied_actions"]):
                    raise RuntimeError("a repeated request was executed again")
                second_count = 0
                if not receipt["terminal"]:
                    second = client.step_chunk(np.zeros((16, 7)), episode_id=episode_id,
                                               chunk_id=1, actor_version=1)
                    second_count = second["executed_count"]
                evidence["tasks"].append({"task_id": task_id, "task_name": task["task_name"],
                    "instruction": task["instruction"], "official_init_count": len(states),
                    "initial_steps": initial["timestep"], "first_chunk_executed": receipt["executed_count"],
                    "second_chunk_executed": second_count, "repeated_request_not_executed": True,
                    "osc_controls_finite": all(d.get("osc_actuator_control_finite", False) for d in receipt["cpu_diagnostics"]),
                    "first_chunk_control_abs_max": max(d["actuator_control_abs_max"] for d in receipt["cpu_diagnostics"]),
                    "waiting_physics_frozen": True})
                (output / "check.json").write_text(json.dumps(evidence, indent=2) + "\n")
            evidence["passed"] = True
    except Exception as error:
        evidence["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "check.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"passed": True, "tasks": len(evidence["tasks"]), "output": str(output)}))


if __name__ == "__main__":
    main()
