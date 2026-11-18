"""LIBERO-Pro2.0 evaluation on one GPU and its official-group aggregation (supplementary tables of the arm WAM on LIBERO).

The official LIBERO-Pro2.0 server owns tasks, perturbations, physics, EGL rendering and success; the
policy runs the same closed-loop episode as on LIBERO-Long (``evaluate_case``). The plan is the official
test split restricted to LIBERO-10 (held-out tasks 5 and 8): 84 task instances x 30 seeds = 2,520
episodes (scripts/eval/protocol.sh). Each finished episode is written once, so a rerun resumes.
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import socket
import time
from uuid import uuid4

import numpy as np
import torch

from arm_opsd.config import ArmContract, canonical_hash
from arm_opsd.evaluation.episode import evaluate_case
from arm_opsd.evaluation.policies import ARMS, NFE, arm_sampler, load_policy
from arm_opsd.evaluation.paired_stats import instance_delta_ci, paired_report
from arm_opsd.sim.libero_pro.plan import HELD_OUT, episode_key, load_plan, spec_matches, variant_group

PROMPT_SCHEMA = "arm-opsd-libero-pro-prompts-v1"
INSTRUCTION_SCHEMA = "arm-opsd-libero-pro-instructions-v1"
TEXT_SHAPE = (1, 512, 4096)


def load_instructions(path, plan):
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != INSTRUCTION_SCHEMA or payload.get("plan_id") != plan["plan_id"]:
        raise ValueError("Instruction list belongs to another plan")
    if set(payload["episodes"]) != set(plan["keys"]):
        raise ValueError("Instruction list does not cover the plan exactly")
    return payload


def prepare_libero_pro_prompts(model_root, instructions_path, plan_path, output, *, device="cuda"):
    """Embed every reachable instruction once (S21/D21 change the text), so evaluation never loads T5."""
    from arm_opsd.assets.base_model import text_embedder
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    instructions = load_instructions(instructions_path, load_plan(plan_path))
    embed = text_embedder(model_root, device=device)
    payload = {"schema": PROMPT_SCHEMA, "negative": embed(""),
               "texts": {text: embed(text) for text in instructions["texts"]}}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {"texts": len(payload["texts"]), "output": str(output)}


class LiberoProPromptBank:
    """Prompt embeddings keyed by the exact instruction text."""

    def __init__(self, path):
        self.payload = torch.load(path, map_location="cpu", weights_only=True)
        if self.payload.get("schema") != PROMPT_SCHEMA:
            raise ValueError(f"{path} is not an {PROMPT_SCHEMA} prompt bank")
        if tuple(self.payload["negative"].shape) != TEXT_SHAPE:
            raise ValueError("Negative prompt embedding must be [1, 512, 4096]")

    def lookup(self, text):
        if text not in self.payload["texts"]:
            raise KeyError(f"Instruction has no precomputed embedding: {text!r}")
        return self.payload["texts"][text], self.payload["negative"]

    def bind(self, episode):
        # evaluate_case passes the reset-time instruction; the server's current one wins.
        return lambda task_id, instruction: self.lookup(episode.instruction)


def generation_seed(master_seed, spec):
    return int(canonical_hash([master_seed, "libero-pro", spec["suite"], int(spec["task_id"]),
                               str(spec["variant"]).upper(), int(spec["init_state_id"]),
                               int(spec["seed"])])[:15], 16)


def run_description(policy, student, *, seed, plan, run_id):
    arm = ARMS[policy]
    return {"benchmark": "libero_pro2", "policy": policy, "label": arm.label, "nfe": NFE[policy],
            "guidance": {"video": arm.video_cfg, "action": arm.action_cfg}, "student": student,
            "master_seed": seed, "plan_id": plan["plan_id"], "run_id": run_id}


def check_same_run(path, description):
    """Results of one directory come from one policy, student, seed and plan; resuming another is refused."""
    if path.exists():
        recorded = json.loads(path.read_text())
        keys = ("policy", "student", "master_seed", "plan_id")
        if {k: recorded.get(k) for k in keys} != json.loads(json.dumps({k: description[k] for k in keys})):
            raise ValueError(f"{path.parent} holds results of another policy/student/seed/plan; use another --output")
    else:
        path.write_text(json.dumps(description, indent=2) + "\n")


def evaluate_libero_pro(*, model_root, prompts, plan_path, instructions_path, run_id, policy, output_dir,
                        pro_python, student=None, seed=42, runner_id=None, video_variants=("BASE",),
                        max_episodes=None, role_override=None, encoder_override=None):
    """Claim and evaluate this policy's leased episodes of one submitted LIBERO-Pro2.0 run.
    ``role_override``/``encoder_override`` replace the real model (tiny smokes only)."""
    from arm_opsd.sim.libero_pro.client import ProEpisode, ProWorkerClient
    from arm_opsd.runtime import json_value
    contract = ArmContract()
    plan = load_plan(plan_path)
    instructions = load_instructions(instructions_path, plan)
    by_key = dict(zip(plan["keys"], plan["episodes"]))
    if role_override is None:
        if not torch.cuda.is_available():
            raise RuntimeError("LIBERO-Pro2.0 policy evaluation needs a CUDA/ROCm GPU")
        role, student_info = load_policy(policy, model_root, student)
    else:
        role, student_info = role_override, None
    bank = LiberoProPromptBank(prompts)
    if encoder_override is None:
        from diffusers import AutoencoderKLWan
        from arm_opsd.model.codec import ObservationEncoder
        encoder = ObservationEncoder(AutoencoderKLWan.from_pretrained(str(Path(model_root) / "vae"),
            torch_dtype=torch.bfloat16, local_files_only=True).to(role.device))
    else:
        encoder = encoder_override
    description = run_description(policy, student_info, seed=seed, plan=plan, run_id=run_id)
    sampler = arm_sampler(policy, contract)
    destination = Path(output_dir) / policy
    destination.mkdir(parents=True, exist_ok=True)
    runner_id = runner_id or f"{socket.gethostname()}-{policy}-{os.getpid()}"
    done = 0
    with (destination / "worker.stderr.log").open("ab") as stderr, ProWorkerClient(pro_python, stderr=stderr) as worker:
        health = worker.health()
        check_same_run(destination / "run.json", {**description, "libero_pro": health,
                                                  "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        while max_episodes is None or done < max_episodes:
            claim = worker.claim(run_id, runner_id)
            if claim["status"] == "done":
                break
            if claim["status"] == "queued":
                time.sleep(float(claim["retry_after_seconds"]))
                continue
            spec = claim["episode_spec"]
            key = episode_key(spec)
            path = destination / f"{key}.json"
            if key not in by_key or not spec_matches(by_key[key], spec) or path.exists():
                worker.request("ABANDON", {"job_id": claim["job_id"]})
                raise ValueError(f"Leased episode {key} is not a pending episode of this plan")
            if claim["instruction"] != instructions["episodes"][key]["reset"]:
                worker.request("ABANDON", {"job_id": claim["job_id"]})
                raise ValueError(f"Reset instruction differs from the enumerated list: {key}")
            claim["case_id"] = key
            episode = ProEpisode(worker, claim)
            case = {"case_id": key, "task_id": int(spec["task_id"]), "instruction": claim["instruction"]}
            writer = None
            started = time.perf_counter()
            try:
                if str(spec["variant"]).upper() in video_variants:
                    import imageio.v2 as imageio
                    writer = imageio.get_writer(str(destination / f"{key}.mp4"), fps=20)
                def video_sink(observation):
                    writer.append_data(np.concatenate([observation[k] for k in contract.cameras], axis=1))
                result = evaluate_case(role, episode, encoder, bank.bind(episode), case,
                    resource_hash="libero-pro2", generation_seed=generation_seed(seed, spec),
                    sampler=sampler, video_sink=video_sink if writer is not None else None)
                if writer is not None:
                    writer.close()
                    writer = None
                    result["video"] = {"path": f"{key}.mp4", "fps": 20, "frames": result["policy_actions"] + 1}
                server = episode.complete()
            except Exception as exc:
                (destination / f"{key}.failed-{uuid4().hex}.json").write_text(json.dumps(
                    {"case_id": key, "error": repr(exc)}, indent=2) + "\n")
                try:
                    episode.abandon()
                except Exception:
                    pass
                raise
            finally:
                if writer is not None:
                    writer.close()
            result.update(episode_spec=spec, variant=str(spec["variant"]).upper(), group=variant_group(spec["variant"]),
                          held_out=int(spec["task_id"]) in HELD_OUT.get(spec["suite"], ()),
                          instruction_reset=claim["instruction"], libero_pro=episode.stats,
                          libero_pro_job={k: claim[k] for k in ("run_id", "job_id", "job_index", "lease_id")},
                          libero_pro_result=server, episode_wall_seconds=time.perf_counter() - started,
                          policy=policy)
            path.write_text(json.dumps(json_value(result), indent=2) + "\n")
            done += 1
    return summarize(destination, plan)


def summarize(destination, plan):
    rows = [json.loads(p.read_text()) for k in plan["keys"] if (p := Path(destination) / f"{k}.json").exists()]
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(int(row["success"]))
    summary = {"completed": len(rows), "total": plan["rollout_count"], "complete": len(rows) == plan["rollout_count"],
               "task_sr": {str(t): float(np.mean(v)) for t, v in sorted(by_task.items())},
               "episode_sr": float(np.mean([int(r["success"]) for r in rows])) if rows else None,
               "episode_wall_seconds_mean": float(np.mean([r["episode_wall_seconds"] for r in rows])) if rows else None}
    summary["macro_sr"] = float(np.mean(list(summary["task_sr"].values()))) if rows else None
    (Path(destination) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load_libero_pro_evaluation(directory, plan):
    """Every plan episode of one results directory; failures stay in the denominator."""
    directory = Path(directory)
    rows = []
    for key in plan["keys"]:
        path = directory / f"{key}.json"
        if not path.exists():
            raise ValueError(f"Incomplete evaluation in {directory}: missing {key}")
        row = json.loads(path.read_text())
        if row["case_id"] != key:
            raise ValueError(f"{path} holds episode {row['case_id']}")
        rows.append(row)
    return rows


def capped(rows, limit=520):
    """The same episodes, scored as if the package default step limit of LIBERO-10 (520) applied."""
    return [{**r, "success": bool(r["success"]) and r["policy_actions"] <= limit} for r in rows]


def official_aggregate(rows):
    """The official formula: a case's SR is its per-task SR mean; static/runtime are means over cases."""
    by_case = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_case[row["variant"]][row["task_id"]].append(int(row["success"]))
    cases = {case: {"sr": float(np.mean([np.mean(v) for v in tasks.values()])),
                    "n": sum(len(v) for v in tasks.values()),
                    "task_sr": {str(t): float(np.mean(v)) for t, v in sorted(tasks.items())}}
             for case, tasks in sorted(by_case.items())}
    mean = lambda prefix: float(np.mean([v["sr"] for k, v in cases.items() if k.startswith(prefix)]))
    return {"base_sr": cases["BASE"]["sr"], "static_sr": mean("S"), "runtime_sr": mean("D"), "cases": cases}


def libero_pro_official_summary(directory, plan_path):
    """BASE / static / runtime group rates of one policy, at the plan's step limit (800, the paper's) and at 520."""
    plan = load_plan(plan_path)
    rows = load_libero_pro_evaluation(directory, plan)
    run = Path(directory) / "run.json"
    return {"plan_id": plan["plan_id"], "selection": plan["selection"], "n": len(rows),
            "run": json.loads(run.read_text()) if run.exists() else None,
            "plan_limit": official_aggregate(rows), "official_limit_520": official_aggregate(capped(rows))}


def official_summary_text(summary, label=None):
    label = label or (summary.get("run") or {}).get("label", "policy")
    lines = ["=" * 78, f"LIBERO-Pro2.0: {label}, {summary['n']} episodes"]
    for name, title in (("plan_limit", "800 steps"), ("official_limit_520", "520 steps")):
        row = summary[name]
        lines.append(f"  {title}: BASE {100 * row['base_sr']:.1f}%  static {100 * row['static_sr']:.1f}%  "
                     f"runtime {100 * row['runtime_sr']:.1f}%")
    lines.append("=" * 78)
    return "\n".join(lines) + "\n"


def official_paired(rows_a, rows_b, *, bootstrap_samples, seed):
    """Candidate minus baseline in the official aggregation, resampling whole perturbation cases."""
    a, b = official_aggregate(rows_a)["cases"], official_aggregate(rows_b)["cases"]
    delta = {case: b[case]["sr"] - a[case]["sr"] for case in a}
    rng = np.random.default_rng(seed)

    def scope(kinds):
        values = np.asarray([v for case, v in delta.items() if case[0] in kinds])
        draws = values[rng.integers(0, len(values), size=(bootstrap_samples, len(values)))].mean(1)
        return {"cases": len(values), "mean_delta": float(values.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist()}
    return {"base_delta": delta["BASE"], "perturbed": scope("SD"), "static": scope("S"), "runtime": scope("D"),
            "cases": delta}


SCOPES = {"overall": lambda r: True, **{g: (lambda r, g=g: r["group"] == g) for g in ("base", "static", "dynamic")}}


def libero_pro_report(baseline_dir, candidate_dir, plan_path, *, bootstrap_samples=10000, seed=42):
    """Candidate minus baseline on the same 2,520 episodes: task-macro delta with an instance-bootstrap
    interval, discordant counts and McNemar p (overall and per group), and the official-group deltas."""
    plan = load_plan(plan_path)
    a = load_libero_pro_evaluation(baseline_dir, plan)
    b = load_libero_pro_evaluation(candidate_dir, plan)

    def paired(rows_a, rows_b):
        report = paired_report(rows_a, rows_b, bootstrap_samples=bootstrap_samples, seed=seed)
        report.pop("macro_delta_ci95")
        report["paired_macro_delta_ci95"] = instance_delta_ci(rows_a, rows_b, bootstrap_samples=bootstrap_samples, seed=seed)
        report["baseline_episode_sr"] = float(np.mean([int(r["success"]) for r in rows_a]))
        report["candidate_episode_sr"] = float(np.mean([int(r["success"]) for r in rows_b]))
        return report

    def subset(predicate):
        keep = [i for i, row in enumerate(a) if predicate(row)]
        return [a[i] for i in keep], [b[i] for i in keep]

    result = {"plan_id": plan["plan_id"], "baseline": str(baseline_dir), "candidate": str(candidate_dir),
              "official": {"plan_limit": official_paired(a, b, bootstrap_samples=bootstrap_samples, seed=seed),
                           "official_limit_520": official_paired(capped(a), capped(b),
                                                                 bootstrap_samples=bootstrap_samples, seed=seed)},
              "overall": paired(a, b),
              "groups": {g: paired(*subset(SCOPES[g])) for g in ("base", "static", "dynamic")},
              "official_limit_520": paired(capped(a), capped(b))}
    variants = defaultdict(lambda: [0, 0, 0, 0, 0])
    for x, y in zip(a, b):
        row = variants[x["variant"]]
        row[0] += 1
        row[1] += int(x["success"])
        row[2] += int(y["success"])
        row[3] += int(y["success"] and not x["success"])
        row[4] += int(x["success"] and not y["success"])
    result["variants"] = {v: {"n": n, "baseline_sr": s0 / n, "candidate_sr": s1 / n, "improved": up, "regressed": down}
                          for v, (n, s0, s1, up, down) in sorted(variants.items())}
    return result
