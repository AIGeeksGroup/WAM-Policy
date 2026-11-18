"""Mean duration of one inference call (the paper's "Time" column) from policy-server logs.

    python -m wam_policy.evaluation.latency <service log dir> [...]

Reads ``requests.jsonl`` written by ``python -m wam_policy.inference.serve --log-dir`` (the
default Standard100 runner always passes one) and reports the mean / median wall time of
``predict`` requests, i.e. one 16-command chunk (sampling with the history cache, without the
feedback encoding of the next observation). The paper measured 0.159 s per chunk on an
RTX 5070 Ti with merged OPSD LoRA, a CPU VAE, and CUDA synchronization around sampling.
"""
import argparse
import json
from pathlib import Path


def predict_durations(directory):
    out = []
    path = Path(directory) / "requests.jsonl"
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("method") == "predict" and row.get("error") is None:
            out.append(float(row["duration_s"]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log_dirs", nargs="+")
    a = p.parse_args(argv)
    dirs = [d for d in a.log_dirs if Path(d).is_dir() and (Path(d) / "requests.jsonl").is_file()]
    values = sorted(v for d in dirs for v in predict_durations(d))
    if not values:
        raise SystemExit("no successful predict requests found")
    median = values[len(values) // 2] if len(values) % 2 else 0.5 * (values[len(values) // 2 - 1] + values[len(values) // 2])
    print(json.dumps({"predict_calls": len(values), "mean_s": sum(values) / len(values), "median_s": median}))


if __name__ == "__main__":
    main()
