"""Accelerated QLM-Bench Standard100 evaluation: one resident multi-session service, N concurrent clients.

    python -m wam_policy.evaluation.run_parallel --policy {teacher,specialist,opsd_student,mopd_student} \
        --teacher <stage1_1_teacher> [--adapter <specialist or student dir>] \
        --base-model <lingbot-va-base> --prompt-root <prompt_embeddings> \
        --resource-root <QLM-Bench Standard100 resources> \
        --qlm-python <python of the QLM-Bench environment> [--qlm-src <QLM-Bench>/src] \
        --seeds 42,43,44 [--tasks push_box,...] [--cases id,...] --clients 8 --output <dir>

The teacher and specialists are served by :mod:`wam_policy.evaluation.batched_service` (row-batched
prediction across sessions), students by :mod:`wam_policy.evaluation.multi_session_service`; both
keep the VAE on the GPU. The service runs in this interpreter; every (case, policy seed) gets its own
``python -m qlm_bench.benchmark run --cases <case> --policy-seed <seed>`` client in the QLM-Bench
environment (its own MuJoCo/RAMBO/render process and its own durable record), at most ``--clients``
at a time (QLM-Bench gives every trial and attempt its own transport session, so runs of the same
case with different seeds may overlap). Output: ``<output>/seed<S>/<case>/`` (the client's ``result.json``, ``progress/``,
``durable-outcomes/``, ...) and ``<case>.log``; a (case, seed) whose ``result.json`` is complete is
skipped, an incomplete directory is renamed ``*.interrupted-<time>`` and rerun. At the end the
per-trial outcomes of all (case, seed) runs are merged into ``<output>/trials.csv`` and
``summary.json`` (seed 42 = Standard100; 42/43/44 = the 300-episode protocol).
Use ``--jobs case:seed,...`` to select individual pairs, for example when splitting trials across
machines.
Results differ from the single-session :mod:`wam_policy.evaluation.standard100` path only by bf16
rounding of batched GEMMs (teacher/specialist) and the VAE device.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from wam_policy.evaluation.standard100 import complete
from wam_policy.evaluation.trials import rows, summary, write_csv
from wam_policy.inference.policies import POLICIES

SUITE = Path("benchmarks") / "mujoco-standard-test100-v1"
BATCHED = ("teacher", "specialist")


def log(message):
    print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}", flush=True)


def read_cases(resource_root):
    """[(task_id, case_id)] of the Standard100 suite, in manifest order."""
    suite = Path(resource_root) / SUITE
    manifest = json.loads((suite / "manifest.json").read_text())
    cases = []
    for case_id in manifest["case_ids"]:
        record = json.loads((suite / "cases" / f"{case_id}.json").read_text())
        if record["case_id"] != case_id:
            raise ValueError(f"Case file {case_id}.json holds {record['case_id']}")
        cases.append((record["task_id"], case_id))
    return cases


def plan_jobs(cases, seeds, tasks=None, selected=None):
    """Task by task; within a task all cases of the first seed, then the next (keeps a case's seeds apart)."""
    order = list(dict.fromkeys(t for t, _ in cases))
    if tasks:
        unknown = set(tasks) - set(order)
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")
        order = [t for t in order if t in tasks]
    if selected:
        unknown = set(selected) - {c for _, c in cases}
        if unknown:
            raise ValueError(f"Unknown Standard100 cases: {sorted(unknown)}")
    unique_seeds = dict.fromkeys(seeds)
    return [(task, case, seed) for task in order for seed in unique_seeds for t, case in cases
            if t == task and (not selected or case in selected)]


def plan_explicit_jobs(cases, pairs):
    """Select case/seed pairs in the order given, running each pair once."""
    tasks_by_case = {case: task for task, case in cases}
    jobs = []
    for pair in pairs.split(","):
        case, _, seed = pair.strip().rpartition(":")
        case = case.strip()
        try:
            jobs.append((tasks_by_case[case], case, int(seed)))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid job {pair!r}. Expected a known case ID and integer seed (CASE:SEED).") from exc
    return list(dict.fromkeys(jobs))


def start_service(args, log_dir):
    module = "batched_service" if args.policy in BATCHED else "multi_session_service"
    command = [sys.executable, "-u", "-m", f"wam_policy.evaluation.{module}", "--policy", args.policy,
               "--teacher", str(args.teacher), "--base-model", str(args.base_model), "--port", str(args.port),
               "--log-dir", str(log_dir), "--device", args.device, "--dtype", args.dtype]
    options = [("--adapter", args.adapter), ("--prompt-root", args.prompt_root), ("--vae-device", args.vae_device)]
    if args.policy in BATCHED:
        options += [("--video-steps", args.video_steps), ("--action-steps", args.action_steps),
                    ("--max-batch", args.max_batch), ("--max-wait-s", args.max_wait_s)]
    elif args.video_steps or args.action_steps:
        raise SystemExit("A student runs the 1/2 sampler it was distilled for; do not set --video/--action-steps")
    for flag, value in options:
        if value is not None:
            command += [flag, str(value)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.qlm_src:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [env.get("PYTHONPATH"), str(args.qlm_src)]))
    log_file = log_dir.parent / f"{log_dir.name}.log"
    process = subprocess.Popen(command, env=env, stdout=log_file.open("w"), stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True)
    log(f"{module} pid {process.pid}, log {log_file}")
    while not (log_dir / "ready.json").exists():
        if process.poll() is not None:
            raise RuntimeError(f"policy service exited with {process.returncode}; see {log_file}")
        time.sleep(2)
    log("policy service ready")
    return process


def start_client(args, case, seed, target):
    command = [str(args.qlm_python), "-u", "-m", "qlm_bench.benchmark", "run",
               "--resource-root", str(args.resource_root), "--policy-port", str(args.port),
               "--output", str(target), "--policy-seed", str(seed), "--cases", case,
               "--policy-name", args.policy_name or args.policy]
    if args.record_video:
        command += ["--record-video", "--ffmpeg", args.ffmpeg, "--ffprobe", args.ffprobe]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if args.qlm_src:
        env["PYTHONPATH"] = str(args.qlm_src)
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(args.client_threads))
    env.setdefault("MUJOCO_GL", "egl")
    log_file = target.parent / f"{target.name}.log"
    return subprocess.Popen(command, env=env, stdout=log_file.open("w"), stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def outcome_of(target):
    try:
        table, _ = rows(target / "result.json")
        return " ".join(f"{r['outcome']} {r['termination_reason']} sim {r['simulation_time_s']:.2f}s" for r in table)
    except (OSError, ValueError, KeyError):
        return "no result"


def stop(process, grace=60):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(grace)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)


def run_jobs(args, todo):
    stopping = []

    def on_signal(signum, frame):
        stopping.append(signum)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    service = start_service(args, args.output / f"service-{time.strftime('%Y%m%dT%H%M%S')}")
    pending, running, done = list(todo), {}, 0
    try:
        while pending or running:
            if service.poll() is not None and not stopping:
                log(f"policy service exited with {service.returncode}; not launching more clients")
                stopping.append("service")
            while pending and len(running) < args.clients and not stopping:
                task, case, seed = pending.pop(0)
                target = args.output / f"seed{seed}" / case
                if complete(target / "result.json"):
                    continue
                if target.exists():
                    target.rename(target.with_name(f"{target.name}.interrupted-{int(time.time())}"))
                target.parent.mkdir(parents=True, exist_ok=True)
                running[start_client(args, case, seed, target)] = (task, case, seed, target, time.monotonic())
            for process in list(running):
                if process.poll() is None:
                    continue
                task, case, seed, target, started = running.pop(process)
                done += 1
                log(f"[{done}/{len(todo)}] {task} {case} seed {seed} rc={process.returncode} {outcome_of(target)} "
                    f"wall {time.monotonic() - started:.0f}s")
            if stopping:
                for process in running:   # a killed client writes no result.json; it reruns on resume
                    stop(process, grace=30)
                if all(p.poll() is not None for p in running):
                    break
            time.sleep(1)
    finally:
        for process in running:
            stop(process, grace=30)
        stop(service)
    return not stopping


def merge(args, jobs):
    table, expected = [], 0
    for _, case, seed in jobs:
        result = args.output / f"seed{seed}" / case / "result.json"
        if result.is_file():
            items, count = rows(result)
            table += items
            expected += count
        else:
            expected += 1
    table.sort(key=lambda r: (r["policy_seed"], r["task_id"], r["case_id"]))
    write_csv(args.output / "trials.csv", table)
    report = summary(table, expected)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("trials", "expected_trials", "successes", "success_rate")}))
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, choices=POLICIES)
    p.add_argument("--teacher", required=True, type=Path, help="Stage 1.1 teacher directory")
    p.add_argument("--adapter", type=Path, help="specialist or student (opsd/mopd) directory")
    p.add_argument("--base-model", required=True, type=Path, help="LingBot-VA base snapshot")
    p.add_argument("--prompt-root", type=Path, help="<prompt_root>/<task>/embedding.pt")
    p.add_argument("--video-steps", type=int, help="teacher/specialist only (default from metadata, 25)")
    p.add_argument("--action-steps", type=int, help="teacher/specialist only (default from metadata, 50)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--vae-device", default=None, help="default: same as --device (accelerated setting)")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--max-batch", type=int, default=8, help="teacher/specialist: rows per batched prediction")
    p.add_argument("--max-wait-s", type=float, default=1.5, help="teacher/specialist: batching wait")
    p.add_argument("--resource-root", required=True, type=Path,
                   help="holds benchmarks/mujoco-standard-test100-v1/ and assets/")
    p.add_argument("--qlm-python", required=True, type=Path, help="python of the QLM-Bench environment")
    p.add_argument("--qlm-src", type=Path, help="QLM-Bench src directory (if not pip-installed)")
    p.add_argument("--seeds", default="42", help="comma-separated policy seeds, e.g. 42,43,44")
    p.add_argument("--tasks", help="comma-separated task subset (a subset is not a Standard100 score)")
    p.add_argument("--cases", help="comma-separated case subset (a subset is not a Standard100 score)")
    p.add_argument("--jobs", help="comma-separated CASE:SEED pairs (overrides --cases, --tasks and --seeds)")
    p.add_argument("--clients", type=int, default=8, help="concurrent QLM clients")
    p.add_argument("--client-threads", type=int, default=2, help="OMP threads per client")
    p.add_argument("--policy-name", help="free label written by the client (default: --policy)")
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    p.add_argument("--port", type=int, default=8870)
    p.add_argument("--output", required=True, type=Path)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (args.policy == "teacher") != (args.adapter is None):
        raise SystemExit("teacher takes no --adapter; specialist and students require one")
    if args.clients < 1:
        raise SystemExit("--clients >= 1 required")
    cases = read_cases(args.resource_root)
    try:
        if args.jobs is not None:
            jobs = plan_explicit_jobs(cases, args.jobs)
        else:
            seeds = [int(s) for s in args.seeds.split(",") if s]
            jobs = plan_jobs(cases, seeds, tasks=args.tasks.split(",") if args.tasks else None,
                             selected=set(args.cases.split(",")) if args.cases else None)
    except ValueError as exc:
        p.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    todo = [job for job in jobs if not complete(args.output / f"seed{job[2]}" / job[1] / "result.json")]
    log(f"{len(jobs)} (case, seed) runs, {len(jobs) - len(todo)} already complete, {len(todo)} to run")
    if todo:
        run_jobs(args, todo)
    report = merge(args, jobs)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
