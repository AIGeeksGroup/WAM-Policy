"""Paired comparison of two policies on the same Standard100 trials (exact McNemar test).

    python -m wam_policy.evaluation.compare --new <result.json|trials.csv> ... \
        --baseline <result.json|trials.csv> ... --output compare.json

Trials pair by (case_id, policy_seed): every Standard100 case has the same initial state and
budget for every policy, and the policy seed fixes the sampling noise order. With seeds 42/43/44
the test runs over 300 pairs; the pairs of one case are not independent across seeds, so the
per-seed tests are reported as well.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

from wam_policy.evaluation.trials import load


def mcnemar(b, c):
    """Exact two-sided McNemar p-value: binomial(b + c, 1/2) on the discordant pairs."""
    n = b + c
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def tally(keys, new, base):
    won = lambda row: row["outcome"] == "success"
    gains = [k for k in keys if won(new[k]) and not won(base[k])]
    losses = [k for k in keys if won(base[k]) and not won(new[k])]
    return {"pairs": len(keys), "new_success": sum(won(new[k]) for k in keys),
            "baseline_success": sum(won(base[k]) for k in keys), "gains": len(gains), "losses": len(losses),
            "mcnemar_exact_p": mcnemar(len(gains), len(losses))}


def compare(new, base, allow_partial=False):
    unpaired = (len(set(new) - set(base)), len(set(base) - set(new)))
    if any(unpaired) and not allow_partial:
        raise ValueError(f"New and baseline cover different (case, seed) pairs (unpaired {unpaired[0]} new, "
                         f"{unpaired[1]} baseline); pass --allow-partial to compare the intersection")
    keys = sorted(set(new) & set(base))
    if not keys:
        raise ValueError("No common (case_id, policy_seed) pairs")
    seeds = sorted({k[1] for k in keys})
    tasks = sorted({new[k]["task_id"] for k in keys})
    return {"total": tally(keys, new, base),
            "by_seed": {str(s): tally([k for k in keys if k[1] == s], new, base) for s in seeds},
            "by_task": {t: tally([k for k in keys if new[k]["task_id"] == t], new, base) for t in tasks},
            "unpaired_new": len(set(new) - set(base)), "unpaired_baseline": len(set(base) - set(new))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new", nargs="+", required=True)
    p.add_argument("--baseline", nargs="+", required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--allow-partial", action="store_true", help="compare only the common (case, seed) pairs")
    a = p.parse_args(argv)
    result = compare(load(a.new), load(a.baseline), a.allow_partial)
    if a.output:
        a.output.write_text(json.dumps(result, indent=2) + "\n")
    t = result["total"]
    print(f"paired {t['pairs']}: new {t['new_success']} vs baseline {t['baseline_success']}, "
          f"+{t['gains']}/-{t['losses']}, McNemar p={t['mcnemar_exact_p']:.3g}")


if __name__ == "__main__":
    main()
