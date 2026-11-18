"""Run one GPU stage (pool / collect / eval) as N shard processes, one per GPU, and merge their outputs.

Every shard is ``python -m wam_policy.task_rl <stage> --run R ... --shard i --nshards N`` with one GPU visible
(HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES set to the shard's device) and the CPU affinity restricted to its own block
of cores (the allowed cores split evenly, or ``cores_per_gpu`` each); its scene workers inherit the block. If any shard
fails, the others are stopped and the stage exits non-zero (every stage resumes from what is on disk). Merge: results /
episodes in a fixed order (start-list position or unit order), the per-shard meta files collected with the stage wall
time.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from wam_policy.task_rl.core import run as R


def visible_ids():
    base = os.environ.get('HIP_VISIBLE_DEVICES') or os.environ.get('CUDA_VISIBLE_DEVICES')
    return [x for x in base.split(',') if x != ''] if base else None


def launch(run, stage, args, gpus, gpu_map, cores_per_gpu, logdir):
    cores = sorted(os.sched_getaffinity(0))
    per = max(1, len(cores) // gpus)
    if cores_per_gpu and cores_per_gpu < per:
        per = cores_per_gpu
    ids = visible_ids()
    procs = []
    logdir.mkdir(parents=True, exist_ok=True)
    for i in range(gpus):
        logical = gpu_map[i]
        dev = ids[logical] if ids else str(logical)
        env = {**os.environ, 'HIP_VISIBLE_DEVICES': dev, 'CUDA_VISIBLE_DEVICES': dev, 'TASK_RL_SHARD_CORES': str(per)}
        block = cores[i * per:(i + 1) * per] or cores
        cmd = [sys.executable, '-m', 'wam_policy.task_rl', stage, '--run', str(run.root), *args,
               '--shard', str(i), '--nshards', str(gpus)]
        log = open(logdir / f'{stage}-s{i:02d}.log', 'a')
        log.write(f'# {time.strftime("%Y-%m-%dT%H:%M:%S")} gpu {dev} cores {block[0]}-{block[-1]} ({len(block)})\n'); log.flush()
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      preexec_fn=lambda b=block: (os.setsid(), os.sched_setaffinity(0, b))))
    failed = None
    try:
        while any(p.poll() is None for p in procs):
            for i, p in enumerate(procs):
                if p.poll() not in (None, 0):
                    failed = (i, p.returncode)
                    raise RuntimeError
            time.sleep(5)
        bad = [(i, p.returncode) for i, p in enumerate(procs) if p.returncode]
        if bad:
            failed = bad[0]
            raise RuntimeError
    except (RuntimeError, KeyboardInterrupt):
        for p in procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in procs:
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        raise SystemExit(f'{stage}: shard {failed} failed; see {logdir}/{stage}-s*.log')


def _write_jsonl(path, rows):
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows)); os.replace(tmp, path)


def merge_collect(run, args, gpus, wall):
    out = run.path(args.out)
    starts = R.load_json(run.path(args.starts))
    pos = {s['id']: i for i, s in enumerate(starts)}
    seen = {}
    for i in range(gpus):
        for r in R.jsonl(out / f'results-s{i:02d}.jsonl'):
            seen[r['candidate']] = r        # a resumed shard may have appended a candidate once; keep one row per id
    rows = sorted(seen.values(), key=lambda r: (pos[r['start']], r['k']))
    expected = {f"{s['id']}-c{j}" for s in starts for j in range(args.k)}
    missing = expected - set(seen)
    if missing:
        raise SystemExit(f'collect merge: {len(missing)} candidates missing, e.g. {sorted(missing)[:3]}')
    _write_jsonl(out / 'results.jsonl', rows)
    metas = [R.load_json(out / f'meta-s{i:02d}.json') for i in range(gpus)]
    R.write_json(out / 'meta.json', {'kind': metas[0]['kind'], 'k': args.k, 'nshards': gpus, 'stage_wall_seconds': wall,
                                     'total_seconds': wall, 'load_seconds': max(m['load_seconds'] for m in metas),
                                     'per_start': [x for m in metas for x in m['per_start']],
                                     'gpu_peak': {kk: max(m['gpu_peak'][kk] for m in metas) for kk in metas[0]['gpu_peak']},
                                     'shards': metas})
    print(f'collect merged: {len(rows)} candidates, {gpus} shards, {wall:.0f} s', flush=True)


def merge_pool(run, args, gpus, wall):
    out = run.root / 'pool'
    eps = sorted({r['unit']: r for i in range(gpus) for r in R.jsonl(out / f'episodes-s{i:02d}.jsonl')}.values(),
                 key=lambda r: r['order'])
    bds = sorted({b['id']: b for i in range(gpus) for b in R.jsonl(out / f'boundaries-s{i:02d}.jsonl')}.values(),
                 key=lambda b: (b['order'], b['groups']))
    _write_jsonl(out / 'episodes.jsonl', eps)
    _write_jsonl(out / 'boundaries.jsonl', bds)
    metas = [R.load_json(out / f'meta-s{i:02d}.json') for i in range(gpus)]
    R.write_json(out / 'meta.json', {'nshards': gpus, 'stage_wall_seconds': wall, 'shards': metas, 'episodes': len(eps),
                                     'boundaries': len(bds), 'success': int(sum(e['success'] for e in eps))})
    print(f'pool merged: {len(eps)} episodes, {len(bds)} boundaries, {wall:.0f} s', flush=True)


def merge_eval(run, args, gpus, wall):
    out = run.path(args.out)
    eps = sorted({r['unit']: r for i in range(gpus) for r in R.jsonl(out / f'episodes-s{i:02d}.jsonl')}.values(),
                 key=lambda r: r['order'])
    _write_jsonl(out / 'episodes.jsonl', eps)
    metas = [R.load_json(out / f'meta-s{i:02d}.json') for i in range(gpus)]
    n, s = len(eps), int(sum(e['success'] for e in eps))
    term = {}
    for e in eps:
        term[str(e['terminal'])] = term.get(str(e['terminal']), 0) + 1
    R.write_json(out / 'summary.json', {'episodes': n, 'success': s, 'success_rate': s / n if n else None, 'terminals': term,
                                        'lora': metas[0].get('lora'), 'scenes': metas[0].get('scenes'),
                                        'reps': metas[0].get('reps'), 'nshards': gpus, 'stage_wall_seconds': wall,
                                        'episode_wall_s_median': sorted(e['wall_s'] for e in eps)[n // 2] if n else None,
                                        'shards': metas})
    print(f'eval merged: {s}/{n} success, {wall:.0f} s', flush=True)


MERGE = {'pool': merge_pool, 'collect': merge_collect, 'eval': merge_eval}


def sharded(run, stage, args, rest, *, gpus, gpu_map=None, cores_per_gpu=0):
    """Launch ``stage`` with the remaining CLI arguments ``rest`` on ``gpus`` shards and merge."""
    gpu_map = gpu_map or list(range(gpus))
    if len(gpu_map) != gpus:
        raise SystemExit('gpu map needs one entry per shard')
    logdir = (run.root / 'pool' / 'logs') if stage == 'pool' else run.path(args.out) / 'logs'
    t0 = time.monotonic()
    launch(run, stage, rest, gpus, gpu_map, cores_per_gpu, Path(logdir))
    MERGE[stage](run, args, gpus, time.monotonic() - t0)
