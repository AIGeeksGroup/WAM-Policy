"""Default QLM-Bench Standard100 evaluation: one policy service and one client per policy seed.

    python -m wam_policy.evaluation.standard100 --policy opsd_student \
        --teacher <stage1_1_teacher> --adapter <stage1_2_opsd_student> \
        --base-model <lingbot-va-base> --prompt-root <prompt_embeddings> \
        --resource-root <QLM-Bench Standard100 resources> \
        --qlm-python <python of the QLM-Bench environment> [--qlm-src <QLM-Bench>/src] \
        --seeds 42,43,44 --output <dir> [--vae-device cpu]

For every policy seed this starts a fresh policy server (``python -m wam_policy.inference.serve``,
or ``python -m wam_policy.ablations.proprioception serve`` for ``--policy proprioception_teacher``)
in this interpreter, runs ``python -m qlm_bench.benchmark run`` in the QLM-Bench environment (MuJoCo
on EGL), and stops the server. A fresh server per seed (and per retry) keeps each client's episodes
in a clean single-session server, as in the paper runs. Seed 42 is the Standard100
protocol; seeds 43 and 44 are its repeats (300 episodes in total).

A seed whose ``result.json`` is complete is skipped. An incomplete one (infrastructure error) is
retried up to ``--attempts`` times with a fresh server, passing the previous ``result.json`` as
``--previous-result`` so finished episodes are kept. Server and client run with
``OMP_NUM_THREADS``/``MKL_NUM_THREADS`` = ``--threads`` (4, as in the paper runs) and
``TOKENIZERS_PARALLELISM=false``. This is the non-accelerated path; see
``wam_policy.evaluation.run_parallel`` for the multi-session variant.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from wam_policy.evaluation.trials import OUTCOMES, rows, summary, write_csv
from wam_policy.inference.policies import POLICIES

ABLATION_POLICY = "proprioception_teacher"


def complete(result_path):
    if not result_path.is_file():
        return False
    table, expected = rows(result_path)
    return len(table) == expected and all(r["outcome"] in OUTCOMES for r in table)


def thread_env(args, base=None):
    env = dict(os.environ if base is None else base)
    env.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads), TOKENIZERS_PARALLELISM="false")
    return env


def start_service(args, log_dir):
    module = ["wam_policy.ablations.proprioception", "serve"] if args.policy == ABLATION_POLICY else \
        ["wam_policy.inference.serve", "--policy", args.policy]
    command = [sys.executable, "-m", *module, "--teacher", str(args.teacher), "--base-model", str(args.base_model),
               "--port", str(args.port), "--log-dir", str(log_dir), "--device", args.device]
    for flag, value in (("--adapter", args.adapter), ("--prompt-root", args.prompt_root),
                        ("--vae-device", args.vae_device), ("--video-steps", args.video_steps),
                        ("--action-steps", args.action_steps), ("--offload-blocks", args.offload_blocks)):
        if value is not None:
            command += [flag, str(value)]
    env = thread_env(args)
    if args.qlm_src:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [env.get("PYTHONPATH"), str(args.qlm_src)]))
    log = (log_dir.parent / f"{log_dir.name}.log").open("w")
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    ready = log_dir / "ready.json"
    while not ready.exists():
        if process.poll() is not None:
            raise RuntimeError(f"policy service exited with {process.returncode}; see {log.name}")
        time.sleep(2)
    return process


def stop_service(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_client(args, seed, output, previous=None):
    command = [str(args.qlm_python), "-m", "qlm_bench.benchmark", "run", "--resource-root", str(args.resource_root),
               "--policy-port", str(args.port), "--output", str(output), "--policy-seed", str(seed),
               "--policy-name", args.policy]
    if previous is not None:
        command += ["--previous-result", str(previous)]
    if args.cases:
        command += ["--cases", args.cases]
    if args.record_video:
        command += ["--record-video", "--ffmpeg", args.ffmpeg, "--ffprobe", args.ffprobe]
    env = thread_env(args, {k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    if args.qlm_src:
        env["PYTHONPATH"] = str(args.qlm_src)
    env.setdefault("MUJOCO_GL", "egl")
    with (output.parent / f"{output.name}.log").open("w") as log:
        return subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def run_seed(args, seed):
    """Fresh server per attempt; resume an incomplete result with --previous-result."""
    target = args.output / f"seed{seed}"
    for attempt in range(1, args.attempts + 1):
        previous = None
        if target.exists():
            kept = args.output / f"seed{seed}.attempt-{int(time.time())}"
            target.rename(kept)
            if (kept / "result.json").is_file():
                previous = kept / "result.json"
        service = None
        try:
            service = start_service(args, args.output / f"service-seed{seed}-{time.strftime('%Y%m%dT%H%M%S')}")
            code = run_client(args, seed, target, previous)
            print(f"seed {seed} attempt {attempt}: client exit code {code}", flush=True)
        finally:
            stop_service(service)
        if complete(target / "result.json"):
            return True
    return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, choices=(*POLICIES, ABLATION_POLICY))
    p.add_argument("--teacher", required=True, type=Path,
                   help="Stage 1.1 teacher (or the proprioception ablation teacher for proprioception_teacher)")
    p.add_argument("--adapter", type=Path)
    p.add_argument("--base-model", required=True, type=Path)
    p.add_argument("--prompt-root", type=Path)
    p.add_argument("--video-steps", type=int)
    p.add_argument("--action-steps", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--vae-device", default="cpu", help="default cpu, as in the paper's single-service runs")
    p.add_argument("--offload-blocks", type=int)
    p.add_argument("--resource-root", required=True, type=Path)
    p.add_argument("--qlm-python", required=True, type=Path)
    p.add_argument("--qlm-src", type=Path, help="QLM-Bench src directory (if not pip-installed)")
    p.add_argument("--seeds", default="42", help="comma-separated policy seeds, e.g. 42,43,44")
    p.add_argument("--cases", help="comma-separated case ids (a subset is not a Standard100 score)")
    p.add_argument("--attempts", type=int, default=3, help="attempts per seed after infrastructure errors")
    p.add_argument("--threads", type=int, default=4, help="OMP/MKL threads of server and client (paper: 4)")
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    p.add_argument("--port", type=int, default=8862)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    if args.policy == ABLATION_POLICY and args.adapter is not None:
        p.error("proprioception_teacher takes no --adapter")
    if args.policy == ABLATION_POLICY and args.offload_blocks:
        p.error("--offload-blocks is not supported for proprioception_teacher")
    if args.policy == ABLATION_POLICY:
        args.offload_blocks = None
    seeds = [int(s) for s in args.seeds.split(",")]
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        if not complete(args.output / f"seed{seed}" / "result.json") and not run_seed(args, seed):
            print(f"seed {seed}: incomplete after {args.attempts} attempts", flush=True)
    table, expected = [], 0
    for seed in seeds:
        result = args.output / f"seed{seed}" / "result.json"
        if result.is_file():
            items, count = rows(result)
            table += items
            expected += count
    write_csv(args.output / "trials.csv", table)
    report = summary(table, expected)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("trials", "expected_trials", "successes", "success_rate",
                                              "success_rate_std_over_seeds")}))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
