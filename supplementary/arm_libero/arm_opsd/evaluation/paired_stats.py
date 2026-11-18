"""Paired statistics of two policies evaluated on the same episodes (supplementary table 'Paired comparisons on LIBERO').

Two episodes form a pair when they share the case (LIBERO-Long: task and official init state;
LIBERO-Pro2.0: task instance and seed). Reported per comparison: the difference of success rates,
the discordant counts (+: only the first policy succeeds, -: only the second), the exact two-sided
McNemar test on these counts, and a bootstrap 95% interval of the difference.
"""
from collections import defaultdict
import math

import numpy as np


def mcnemar_exact(improved, regressed):
    """Two-sided exact McNemar p-value on the discordant pairs."""
    n = improved + regressed
    if n == 0:
        return 1.
    tail = sum(math.comb(n, k) for k in range(min(improved, regressed) + 1)) / 2 ** n
    return min(1., 2 * tail)


def wilson_interval(successes, total, z=1.96):
    if not total:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [centre - half, centre + half]


def check_pairs(baseline, candidate):
    """Rows of two evaluations keyed by case_id; every case must carry the same task and generation seed."""
    a, b = {x["case_id"]: x for x in baseline}, {x["case_id"]: x for x in candidate}
    if len(a) != len(baseline) or len(b) != len(candidate) or a.keys() != b.keys():
        raise ValueError("Paired evaluations must cover the same distinct cases")
    for key in a:
        if (a[key]["task_id"], a[key]["generation_seed"]) != (b[key]["task_id"], b[key]["generation_seed"]):
            raise ValueError(f"Paired case {key} has another task or generation seed")
    return a, b


def paired_report(baseline, candidate, *, bootstrap_samples=10000, seed=42):
    """Per-task success rates, discordant counts and a task-stratified case bootstrap of the macro delta."""
    a, b = check_pairs(baseline, candidate)
    groups = defaultdict(list)
    for key in sorted(a):
        groups[a[key]["task_id"]].append((int(a[key]["success"]), int(b[key]["success"])))
    rng = np.random.default_rng(seed)
    samples = np.zeros(bootstrap_samples)
    tasks = {}
    for task, rows in sorted(groups.items()):
        values = np.asarray(rows)
        delta = values[:, 1] - values[:, 0]
        samples += delta[rng.integers(0, len(rows), size=(bootstrap_samples, len(rows)))].mean(1) / len(groups)
        tasks[str(task)] = {"n": len(rows), "baseline_sr": float(values[:, 0].mean()),
                            "candidate_sr": float(values[:, 1].mean()),
                            "improved": int((delta == 1).sum()), "regressed": int((delta == -1).sum())}
    improved = sum(v["improved"] for v in tasks.values())
    regressed = sum(v["regressed"] for v in tasks.values())
    baseline_macro = float(np.mean([x["baseline_sr"] for x in tasks.values()]))
    candidate_macro = float(np.mean([x["candidate_sr"] for x in tasks.values()]))
    return {"n": len(a), "tasks": tasks, "baseline_macro_sr": baseline_macro, "candidate_macro_sr": candidate_macro,
            "macro_delta": candidate_macro - baseline_macro,
            "macro_delta_ci95": np.quantile(samples, [.025, .975]).tolist(),
            "improved": improved, "regressed": regressed, "mcnemar_exact_p": mcnemar_exact(improved, regressed)}


def instance_delta_ci(baseline, candidate, *, bootstrap_samples=10000, seed=42):
    """LIBERO-Pro2.0: paired task-macro interval resampling whole task instances (variants) within each task,
    since the seeds of one instance are not independent. None when a task has a single instance."""
    by_task = defaultdict(lambda: defaultdict(list))
    for x, y in zip(baseline, candidate):
        by_task[x["task_id"]][x["variant"]].append(int(y["success"]) - int(x["success"]))
    if any(len(instances) < 2 for instances in by_task.values()):
        return None
    rng = np.random.default_rng(seed)
    samples = np.zeros(bootstrap_samples)
    for instances in by_task.values():
        deltas = np.asarray([np.mean(v) for v in instances.values()])
        samples += deltas[rng.integers(0, len(deltas), size=(bootstrap_samples, len(deltas)))].mean(1) / len(by_task)
    return np.quantile(samples, [.025, .975]).tolist()
