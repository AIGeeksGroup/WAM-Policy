"""Trial tables and success rates from QLM-Bench ``result.json`` files.

    python -m wam_policy.evaluation.trials --result <run>/result.json [...] --output <dir>

Each Standard100 run (one policy seed) contributes 100 trials; three runs with policy seeds
42, 43 and 44 form the 300-episode protocol used for the paper tables.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

OUTCOMES = ("success", "failure", "timeout")
COLUMNS = ("trial_id", "case_id", "task_id", "policy_seed", "outcome", "termination_reason",
           "simulation_time_s", "completion_time_s")


def rows(result):
    """One row per expected trial with a terminal outcome (first terminal attempt)."""
    if isinstance(result, (str, Path)):
        result = json.loads(Path(result).read_text())
    selected = {}
    for row in result["outcomes"]:
        if row["outcome"] in OUTCOMES and row["trial_id"] not in selected:
            selected[row["trial_id"]] = row
    table = []
    for trial in result["run_manifest"]["expected_trials"]:
        row = selected.get(trial["trial_id"])
        if row is None:
            continue
        seconds = row.get("simulation_time_ns", 0) / 1e9
        table.append({"trial_id": row["trial_id"], "case_id": row["case_id"], "task_id": row["task_id"],
                      "policy_seed": int(row["policy_seed"]), "outcome": row["outcome"],
                      "termination_reason": row.get("termination_reason"), "simulation_time_s": seconds,
                      "completion_time_s": seconds if row["outcome"] == "success" else None})
    return table, len(result["run_manifest"]["expected_trials"])


def read_csv(path):
    with Path(path).open(newline="") as stream:
        table = list(csv.DictReader(stream))
    for row in table:
        row["policy_seed"] = int(row["policy_seed"])
    return table


def load(paths):
    """Rows of several result.json / trials.csv files, keyed by (case_id, policy_seed)."""
    table = {}
    for path in paths:
        items = read_csv(path) if str(path).endswith(".csv") else rows(path)[0]
        for row in items:
            key = (row["case_id"], row["policy_seed"])
            if key in table:
                raise ValueError(f"Duplicate trial {key} in {path}")
            table[key] = row
    return table


def summary(table, expected=None):
    """Success rates overall, per task and per policy seed. ``success_rate_std_over_seeds`` is the
    sample standard deviation (ddof=1) of the per-seed success rates -- the paper's "±" for seeds
    42/43/44, whose per-seed successes out of 100 are the "Seeds" column."""
    by_task, by_seed = {}, {}
    for row in table:
        by_task.setdefault(row["task_id"], []).append(row)
        by_seed.setdefault(row["policy_seed"], []).append(row)
    rate = lambda items: sum(r["outcome"] == "success" for r in items) / len(items) if items else None
    rates = [rate(v) for _, v in sorted(by_seed.items())]
    std = None
    if len(rates) > 1:
        mean = sum(rates) / len(rates)
        std = (sum((x - mean) ** 2 for x in rates) / (len(rates) - 1)) ** 0.5
    return {"trials": len(table), "expected_trials": expected, "complete": expected in (None, len(table)),
            "success_rate": rate(table), "successes": sum(r["outcome"] == "success" for r in table),
            "seeds": sorted(by_seed),
            "by_seed": {str(s): {"trials": len(v), "successes": sum(r["outcome"] == "success" for r in v),
                                 "success_rate": rate(v)} for s, v in sorted(by_seed.items())},
            "success_rate_std_over_seeds": std,
            "tasks": {t: {"trials": len(v), "success_rate": rate(v),
                          "outcomes": {o: sum(r["outcome"] == o for r in v) for o in OUTCOMES}}
                      for t, v in sorted(by_task.items())}}


def write_csv(path, table):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({k: ("" if r.get(k) is None else r[k]) for k in COLUMNS} for r in table)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result", nargs="+", required=True, help="QLM-Bench result.json of each run")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    table, expected, seen = [], 0, set()
    for path in a.result:
        items, count = rows(path)
        for row in items:
            key = (row["case_id"], row["policy_seed"])
            if key in seen:
                raise SystemExit(f"Duplicate trial {key} in {path}")
            seen.add(key)
        table += items
        expected += count
    write_csv(a.output / "trials.csv", table)
    report = summary(table, expected)
    (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['successes']}/{report['trials']} successful (expected {expected}); "
          + ", ".join(f"{t}: {v['success_rate']:.2f}" for t, v in report["tasks"].items()))


if __name__ == "__main__":
    main()
