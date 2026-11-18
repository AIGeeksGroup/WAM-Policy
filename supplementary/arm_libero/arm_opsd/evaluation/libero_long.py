"""Standard LIBERO-Long evaluation on one GPU (supplementary tables of the arm WAM on LIBERO, column Long).

Ten tasks x the first fifty official init states, on the CPU LIBERO worker (OSMesa): the official
client's reset (five zero-action settle steps), 128x128 views, the native chunk protocol and
timestep < 800. Policy loading, sampler, observation encoder and action codec are those of the
LIBERO-Pro2.0 harness; only the environment differs. Each case is written once, so a rerun resumes.
"""
import json
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
import torch

from arm_opsd.assets.base_model import PromptBank
from arm_opsd.assets.resets import load_test_manifest, materialize_case
from arm_opsd.config import ArmContract, canonical_hash
from arm_opsd.evaluation.episode import evaluate_case
from arm_opsd.evaluation.policies import ARMS, NFE, arm_sampler, load_policy
from arm_opsd.evaluation.paired_stats import paired_report, wilson_interval

EPISODES_PER_TASK = 50


def case_seed(master_seed, case_id):
    return int(canonical_hash([master_seed, "evaluation", case_id])[:15], 16)


def summarize_long(destination, cases, *, label):
    """summary.json and result.txt of one results directory."""
    destination = Path(destination)
    rows = [json.loads(p.read_text()) for c in cases if (p := destination / f"{c['case_id']}.json").exists()]
    names = {c["task_id"]: c.get("task_name") or c["instruction"] for c in cases}
    per_task = {task: [int(r["success"]) for r in rows if r["task_id"] == task] for task in sorted(names)}
    complete = len(rows) == len(cases)
    successes = sum(int(r["success"]) for r in rows)
    task_sr = {str(t): float(np.mean(v)) for t, v in per_task.items() if v}
    macro = float(np.mean(list(task_sr.values()))) if task_sr else None
    summary = {"total": len(cases), "completed": len(rows), "complete": complete, "successes": successes,
               "macro_sr": macro, "macro_sr_wilson95": wilson_interval(successes, len(rows)), "task_sr": task_sr,
               "timeouts": sum(int(r.get("timeout", False)) for r in rows),
               "episode_wall_seconds_mean": float(np.mean([r["episode_wall_seconds"] for r in rows])) if rows else None}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    state = "complete" if complete else f"PARTIAL {len(rows)}/{len(cases)}"
    lines = ["=" * 78, f"Standard LIBERO-Long: {label}, {state}"]
    if macro is not None:
        low, high = summary["macro_sr_wilson95"]
        lines.append(f"Success rate: {100 * macro:.1f}%  ({successes}/{len(rows)}, Wilson 95% {100 * low:.1f}-{100 * high:.1f})")
    for task, values in per_task.items():
        rate = f"{100 * np.mean(values):5.1f}%" if values else "    -"
        lines.append(f"  task {task}: {rate}  {sum(values):2d}/{len(values):2d}  {names[task]}")
    lines.append("=" * 78)
    text = "\n".join(lines) + "\n"
    (destination / "result.txt").write_text(text)
    return summary, text


def load_long_rows(directory):
    """Every case of one complete standard LIBERO-Long results directory, keyed by case_id."""
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    if not summary.get("complete"):
        raise ValueError(f"Incomplete standard LIBERO-Long result: {directory}")
    rows = {}
    for path in sorted(directory.glob("test-task*-case*.json")):
        if ".failed-" in path.name or ".protocol-invalid-" in path.name:
            continue
        row = json.loads(path.read_text())
        rows[row["case_id"]] = row
    if len(rows) != summary["total"]:
        raise ValueError(f"Expected {summary['total']} cases in {directory}, found {len(rows)}")
    return rows


def libero_long_report(baseline_dir, candidate_dir, *, bootstrap_samples=10000, seed=42):
    """Candidate minus baseline on the same 500 cases: per task, macro delta with a task-stratified paired
    bootstrap, discordant counts and the exact McNemar test."""
    a, b = load_long_rows(baseline_dir), load_long_rows(candidate_dir)
    if a.keys() != b.keys():
        raise ValueError("The two results cover different cases")
    keys = sorted(a)
    report = paired_report([a[k] for k in keys], [b[k] for k in keys], bootstrap_samples=bootstrap_samples, seed=seed)
    result = {"baseline": str(baseline_dir), "candidate": str(candidate_dir), **report}
    low, high = result["macro_delta_ci95"]
    lines = ["=" * 78, f"Standard LIBERO-Long paired: {candidate_dir} vs {baseline_dir}",
             f"Success: {100 * result['candidate_macro_sr']:.1f}% vs {100 * result['baseline_macro_sr']:.1f}%",
             f"Delta: {100 * result['macro_delta']:+.1f} pp, 95% {100 * low:+.1f} to {100 * high:+.1f}; "
             f"+{result['improved']} / -{result['regressed']}, McNemar p = {result['mcnemar_exact_p']:.3g}"]
    for task, row in result["tasks"].items():
        lines.append(f"  task {task}: {100 * row['candidate_sr']:5.1f}% vs {100 * row['baseline_sr']:5.1f}%  "
                     f"(+{row['improved']} / -{row['regressed']})")
    lines.append("=" * 78)
    result["text"] = "\n".join(lines) + "\n"
    return result


def evaluate_libero_long(*, model_root, prompts, reset_manifest, cpu_python, worker_env, policy, output_dir,
                         student=None, seed=42, max_episodes=None, worker_timeout=180., episodes_per_task=EPISODES_PER_TASK,
                         video_cases_per_task=2, role_override=None, encoder_override=None, prompt_override=None,
                         worker_factory=None):
    """role_override, encoder_override, prompt_override and worker_factory replace the real model and
    environment (tiny smokes only)."""
    from arm_opsd.runtime import json_value
    from arm_opsd.evaluation.libero_pro import check_same_run
    contract = ArmContract()
    manifest = load_test_manifest(reset_manifest, episodes_per_task=episodes_per_task)
    manifest_dir = Path(reset_manifest).parent
    cases = [row for row in manifest["cases"] if row["split"] == "test"]
    if role_override is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Standard LIBERO-Long policy evaluation needs a GPU")
        role, student_info = load_policy(policy, model_root, student)
    else:
        role, student_info = role_override, None
    prompt_lookup = prompt_override or PromptBank(prompts)
    if encoder_override is None:
        from diffusers import AutoencoderKLWan
        from arm_opsd.model.codec import ObservationEncoder
        encoder = ObservationEncoder(AutoencoderKLWan.from_pretrained(str(Path(model_root) / "vae"),
            torch_dtype=torch.bfloat16, local_files_only=True).to(role.device))
    else:
        encoder = encoder_override
    arm = ARMS[policy]
    description = {"benchmark": "libero_long", "policy": policy, "label": arm.label, "nfe": NFE[policy],
                   "guidance": {"video": arm.video_cfg, "action": arm.action_cfg}, "student": student_info,
                   "master_seed": seed, "plan_id": manifest["content_key"]}
    sampler = arm_sampler(policy, contract)
    destination = Path(output_dir) / policy
    destination.mkdir(parents=True, exist_ok=True)
    check_same_run(destination / "run.json", description)
    if worker_factory is None:
        from arm_opsd.sim.libero.worker_client import WorkerClient
        environment = json.loads(Path(worker_env).read_text())
        worker_factory = lambda stderr: WorkerClient(cpu_python, env=environment, stderr=stderr, timeout=worker_timeout)
    done = 0
    pending = [row for row in cases if not (destination / f"{row['case_id']}.json").exists()]
    with (destination / "worker.stderr.log").open("ab") as stderr, worker_factory(stderr) as worker:
        for row in pending:
            if max_episodes is not None and done >= max_episodes:
                break
            key = row["case_id"]
            case = materialize_case(row, manifest_dir)
            writer = None
            started = time.perf_counter()
            try:
                if row.get("official_index", 10**9) < video_cases_per_task:
                    import imageio.v2 as imageio
                    writer = imageio.get_writer(str(destination / f"{key}.mp4"), fps=20)
                def video_sink(observation):
                    writer.append_data(np.concatenate([observation[k] for k in contract.cameras], axis=1))
                result = evaluate_case(role, worker, encoder, prompt_lookup, case, resource_hash="libero-long",
                    generation_seed=case_seed(seed, key), sampler=sampler,
                    video_sink=video_sink if writer is not None else None)
            except Exception as exc:
                (destination / f"{key}.failed-{uuid4().hex}.json").write_text(json.dumps(
                    {"case_id": key, "error": repr(exc)}, indent=2) + "\n")
                raise
            finally:
                if writer is not None:
                    writer.close()
            result.update(official_index=row.get("official_index"), task_name=row.get("task_name"),
                          episode_wall_seconds=time.perf_counter() - started, policy=policy, split="test")
            (destination / f"{key}.json").write_text(json.dumps(json_value(result), indent=2) + "\n")
            done += 1
    label = arm.label if student_info is None else f"{arm.label}, {Path(student_info['path']).name}"
    summary, text = summarize_long(destination, cases, label=label)
    print(text, flush=True)
    return summary
