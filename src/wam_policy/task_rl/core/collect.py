"""Candidate windows (GPU): K candidates per start, each closed loop for one window with window-shared noise.

Sampling policy = theta_old: the frozen teacher with the switchable LoRA in mode ``old`` loaded from ``--lora`` (round
0: no file, mode ``off`` = the teacher itself). 25/50, no CFG. Noise: alpha = 1, i.e. each candidate draws one d_k per
modality and uses it as the initial noise of every chunk of its window.

Rows. The candidates of a start run together: the start's history is rebuilt once by the one-row sampler
(``history.rebuild``), its K/V cache and VAE streaming cache are copied into one row per candidate, and the rows advance
in lock step: one row-batched sampling per chunk, every row's scene worker executes its 16 commands at the same time,
one row-batched VAE encode and commit. A batch holds rows // k starts (which must share the in-window history
depth); it is a function of the start list only and shards take whole batches, so any shard count produces the same
batches. Stored per chunk, from what the sampler actually used: the history it committed before the chunk (start prefix
once per start in ``starts/<id>.pt``, the candidate's own committed groups in its file), the clean video latent and
clean action it generated (bf16), the requested and executed commands; per candidate every privileged row (restore
row + one per command) for the reward. A candidate that terminates inside the window stops there; its sampled chunks
stay training samples and its rows are held at the last row for the reward.

Candidate ids and seeds depend only on (kind, round, start id, k).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from wam_policy.task_rl.core import run as R
from wam_policy.task_rl.core import scene as S


def cand_seed(kind, round_name, start_id, k):
    return R.seed_of('probe', start_id, k) if kind == 'probe' else R.seed_of('cand', round_name, start_id, k)


def start_batches(starts, per):
    """Consecutive starts, ``per`` to a batch; a batch never mixes in-window history depths."""
    out = []
    for s in starts:
        depth = min(s['groups'] + 1, 4)
        if out and len(out[-1]) < per and min(out[-1][0]['groups'] + 1, 4) == depth:
            out[-1].append(s)
        else:
            out.append([s])
    return out


def run_rows(run, workers, builder, bcodec, sampler, rcodec, text, rows, kind, round_name, out):
    """One lock-step window for ``rows`` = [(start, k)] (consecutive rows of a start). Returns records, timing,
    per-start rebuild seconds, wall."""
    import torch
    from wam_policy.task_rl.core import history
    from wam_policy.task_rl.core.sampler import stack_persistent
    H = run.profile.window_chunks
    t0 = time.monotonic()
    timing = {'restore': 0., 'rebuild': 0., 'predict': 0., 'physics': 0., 'feedback': 0.}
    n = len(rows)
    starts = []
    for st, _ in rows:
        if not starts or starts[-1]['id'] != st['id']:
            starts.append(st)
    counts = [sum(1 for s, _ in rows if s['id'] == st['id']) for st in starts]
    # physics restore of every row runs on the workers while the GPU rebuilds the histories
    pending = workers.start(range(n), lambda env, st, k: env.restore_state(st['case'], st['model'], str(run.root / st['state']),
                                                                         f"w-{st['id']}-c{k}"),
                            [r[0] for r in rows], [r[1] for r in rows])
    sampler.load_starts(text, counts)
    rcodec.reset()
    n0, rebuild_s = [], {}
    for st, c in zip(starts, counts):
        ts = time.monotonic()
        record = history.load(run.root / st['record']).prefix(st['groups'])
        history.rebuild(builder, bcodec, record, text, seed=R.seed_of('rebuild', st['id']))
        lat, act = builder.history_tensors()
        if kind == 'train':
            torch.save({'latents': lat, 'actions': act, 'groups': st['groups'], 'start': st['id']}, out / 'starts' / f"{st['id']}.pt")
        sampler.put_start(builder)
        rcodec.put_start(bcodec, c)
        n0 += [len(builder.hist_latents)] * c
        rebuild_s[st['id']] = time.monotonic() - ts
    seeds = [cand_seed(kind, round_name, st['id'], k) for st, k in rows]
    sampler.arm(seeds, stack_persistent(seeds, sampler.device), float(run.config['alpha_shared_noise']))
    rcodec.arm()
    timing['rebuild'] = time.monotonic() - t0
    for f in pending:
        f.result()
    timing['restore'] = time.monotonic() - t0
    acc = [dict(z=[], u=[], req=[], ex=[], lat=[], act=[], prefix_len=[], terminal=None, term_ex=None, wall=None) for _ in rows]
    alive = list(range(n))
    for j in range(H):
        t = time.monotonic()
        _, _, requested = sampler.endpoints()
        timing['predict'] += time.monotonic() - t
        for pos, i in enumerate(alive):
            a = acc[i]
            a['prefix_len'].append(n0[i] + j)
            a['z'].append(sampler.generated['video'][pos:pos + 1].clone()); a['u'].append(sampler.generated['action'][pos:pos + 1].clone())
            a['req'].append(requested[pos].copy())
        t = time.monotonic()
        results = workers.run(alive, lambda env, req: env.execute(req), requested)
        timing['physics'] += time.monotonic() - t
        keep = []
        for pos, (i, r) in enumerate(zip(alive, results)):
            if r['terminal'] is not None:
                # the worker pads a terminal chunk with the prepared (never-run) commands; keep only the run ones
                acc[i]['terminal'] = r['terminal']
                acc[i]['term_ex'] = [np.asarray(c['executed'], np.float32) for c in r['commands'] if c.get('status') == 'completed']
                acc[i]['wall'] = time.monotonic() - t0
            elif r['completed_count'] != 16 or len(r['samples']) != 4:
                raise ValueError('Incomplete ordinary real feedback')
            else:
                keep.append(pos)
        if not keep:
            break
        t = time.monotonic()
        if len(keep) < len(alive):
            sampler.keep(keep); rcodec.keep(keep)
            alive, results = [alive[p] for p in keep], [results[p] for p in keep]
        executed = [S.executed_of(r) for r in results]
        latent = rcodec.push(S.stack_views([S.boundary_views(r['samples']) for r in results]))
        lat, act = sampler.commit(latent, executed)
        for pos, i in enumerate(alive):
            acc[i]['ex'].append(executed[pos]); acc[i]['lat'].append(lat[pos:pos + 1].clone()); acc[i]['act'].append(act[pos:pos + 1].clone())
        timing['feedback'] += time.monotonic() - t
    wall = time.monotonic() - t0
    recs = []
    for i, (st, k) in enumerate(rows):
        a = acc[i]
        terminal, term_ex = a['terminal'], a['term_ex']
        env = workers.envs[i]
        recs.append(dict(
            video=torch.cat(a['z'], 0), action=torch.cat(a['u'], 0),
            new_latents=torch.cat(a['lat'], 2) if a['lat'] else torch.zeros(1, 48, 0, 24, 20, dtype=torch.bfloat16),
            new_actions=torch.cat(a['act'], 2) if a['act'] else torch.zeros(1, 9, 0, 16, 1, dtype=torch.bfloat16),
            prefix_len=torch.tensor(a['prefix_len']), start_groups=n0[i], requested=np.stack(a['req']),
            executed=np.stack(a['ex']) if a['ex'] else np.zeros((0, 16, 9), np.float32),
            terminal_executed=None if term_ex is None else (np.stack(term_ex) if term_ex else np.zeros((0, 9), np.float32)),
            rows=env.rows(), columns=env.columns(), seed=seeds[i], d_seed=seeds[i],
            terminal=None if terminal is None else {kk: terminal[kk] for kk in ('reason', 'tick')},
            success=bool(terminal is not None and terminal.get('success', False)),
            candidate=f"{st['id']}-c{k}", start=st['id'], k=k, kind=kind))
    tims = [{**timing, 'wall': acc[i]['wall'] or wall, 'rows': n, 'starts': len(starts)} for i in range(n)]
    return recs, tims, rebuild_s, wall


def main(run, args):
    import torch
    from wam_policy.task_rl import adapters as A
    from wam_policy.task_rl.core import sampler as SM
    out = run.path(args.out)
    for d in ('cands', 'starts'):
        (out / d).mkdir(parents=True, exist_ok=True)
    tag = f's{args.shard:02d}'
    t0 = time.monotonic()
    bundle, text, builder, sampler, rcodec, label = SM.load_policy(run, args.lora)
    bcodec = bundle.codec
    if args.shard == 0 or not (out / 'text.pt').exists():
        torch.save(text.detach().to(torch.bfloat16).cpu(), out / f'text.pt.{tag}')
        os.replace(out / f'text.pt.{tag}', out / 'text.pt')
    load_s = time.monotonic() - t0
    all_starts = R.load_json(run.path(args.starts))
    rows_per = run.rows
    batches = R.shard_of(start_batches(all_starts, max(1, rows_per // args.k)), args.shard, args.nshards)
    starts = [s for b in batches for s in b]
    res_path = out / f'results-{tag}.jsonl'
    done = {r['candidate'] for r in R.jsonl(res_path)}
    # one unit of work = at most `rows` rows of one batch (a start with more candidates than rows is split)
    units = []
    for b in batches:
        rows = [(st, k) for st in b for k in range(args.k) if f"{st['id']}-c{k}" not in done]
        units += R.batches_of(rows, rows_per)
    workers = S.Workers(run, run.root / 'states', max((len(u) for u in units), default=0))
    per_start, batch_log = {}, []
    print('policy', label, 'loaded', round(load_s, 1), 's;', len(starts), 'starts', len(units), 'row batches of <=',
          rows_per, 'rows', flush=True)
    try:
        for rows in units:
            for attempt in (1, 2):
                try:
                    recs, tims, rebuild_s, wall = run_rows(run, workers, builder, bcodec, sampler, rcodec, text, rows,
                                                           args.kind, args.round_name, out)
                    break
                except S.INFRA as exc:
                    print(f'infrastructure error {rows[0][0]["id"]} attempt {attempt}: {exc!r}', flush=True)
                    if attempt == 2:
                        raise
                    workers.restart()
            for rec, timing in zip(recs, tims):
                torch.save(rec, out / 'cands' / f"{rec['candidate']}.pt")
                R.append_jsonl(res_path, dict(candidate=rec['candidate'], start=rec['start'], k=rec['k'], seed=rec['seed'],
                                              chunks=int(rec['video'].shape[0]),
                                              terminal=None if rec['terminal'] is None else rec['terminal']['reason'],
                                              success=rec['success'], timing=timing))
            ids = list(rebuild_s)
            for sid in ids:      # a start's share of its batch (a start split over several row batches accumulates)
                e = per_start.setdefault(sid, {'start': sid, 'rebuild_s': 0., 'wall_s': 0.})
                e['rebuild_s'] += rebuild_s[sid]; e['wall_s'] += wall / len(ids)
            keys = ('rebuild', 'restore', 'predict', 'physics', 'feedback')
            batch_log.append({'starts': ids, 'rows': len(rows), 'wall_s': wall, **{k: tims[0][k] for k in keys}})
            print(json.dumps({'starts': ids, 'rows': len(rows), 'wall_s': round(wall, 1),
                              **{k: round(tims[0][k], 2) for k in keys}, 'elapsed_s': round(time.monotonic() - t0)}), flush=True)
    finally:
        workers.close()
    meta = {'kind': args.kind, 'k': args.k, 'lora': None if args.lora is None else str(args.lora),
            'alpha': run.config['alpha_shared_noise'], 'nfe': R.NFE, 'shard': args.shard, 'nshards': args.nshards,
            'starts': [s['id'] for s in starts], 'gl': run.gl, 'render_mode': run.config['render_mode'], 'rows': rows_per,
            'scene_workers': len(workers.envs), 'load_seconds': load_s, 'total_seconds': time.monotonic() - t0,
            'per_start': list(per_start.values()), 'batches': batch_log, 'gpu_peak': R.peak_gib(),
            'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'hip': getattr(torch.version, 'hip', None),
            'lora_mode': A.lora_modules(bundle.model)[0].switch['mode']}
    mpath = out / f'meta-{tag}.json'
    prev = R.load_json(mpath) if mpath.exists() else None
    if prev:
        meta['previous_runs'] = prev.get('previous_runs', []) + [{k: prev[k] for k in ('total_seconds', 'per_start')}]
    R.write_json(mpath, meta)
    print('done', meta['gpu_peak'], round(meta['total_seconds'], 1), flush=True)
