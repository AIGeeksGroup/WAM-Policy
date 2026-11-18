"""Start pool (GPU): the teacher from reset on the training scenes, closed loop; starts captured per task profile.

Units: the first ``--cases`` scenes of the profile's seeded scene order, times ``--reps`` teacher seeds per scene (rep 0
seed seed_of('pool', case), rep r > 0 seed_of('pool', case, r) and the unit suffix -r<r>). ``--shard i --nshards n``
takes unit batches i, i+n, ... (one process per GPU; ``parallel`` merges the per-shard files into pool/episodes.jsonl
and pool/boundaries.jsonl in unit order).
"""
from __future__ import annotations

import json
import time

from wam_policy.task_rl.core import run as R
from wam_policy.task_rl.core import rollout as RO
from wam_policy.task_rl.core import scene as S


def units(run, n_cases, reps):
    out = []
    order = run.profile.scene_order()
    for r in range(reps):
        for c in order[:n_cases]:
            unit = c['case_id'] if r == 0 else f"{c['case_id']}-r{r}"
            seed = R.seed_of('pool', c['case_id']) if r == 0 else R.seed_of('pool', c['case_id'], r)
            out.append(dict(case=c, unit=unit, rep=r, seed=seed, order=len(out)))
    return out


def main(run, args):
    out = run.root
    for d in ('pool/history', 'pool/episodes', 'states/pool'):
        (out / d).mkdir(parents=True, exist_ok=True)
    tag = f's{args.shard:02d}'
    import torch
    from wam_policy.task_rl import adapters as A
    from wam_policy.task_rl.core import sampler as SM
    t0 = time.monotonic()
    bundle, text, _, sampler, rcodec, _ = SM.load_policy(run)
    load_s = time.monotonic() - t0
    eps_path, bnd_path = out / f'pool/episodes-{tag}.jsonl', out / f'pool/boundaries-{tag}.jsonl'
    done = {r['unit']: r for r in R.jsonl(eps_path)}
    cases = args.cases if args.cases is not None else run.config['pool_cases']
    reps = args.reps if args.reps is not None else run.config['pool_reps']
    batches, per = RO.unit_batches(run, units(run, cases, reps), args.shard, args.nshards)
    workers = S.Workers(run, out / 'states', per if batches else 0)
    print('teacher loaded', round(load_s, 1), 's; shard', args.shard, '/', args.nshards, 'units', sum(map(len, batches)),
          'rows per batch', per, 'done', len(done), flush=True)

    def each(u, s, bds):
        s['order'], s['rep'] = u['order'], u['rep']
        for bd in bds:
            bd['order'] = u['order']
            R.append_jsonl(bnd_path, bd)
        R.append_jsonl(eps_path, s)
        done[u['unit']] = s
        print(json.dumps({k: s.get(k) for k in ('order', 'unit', 'chunks', 'terminal', 'success', 'min_D',
                                                'first_contact_tick', 'eligible', 'wall_s', 'chunk_wall_median_s')}), flush=True)

    try:
        RO.run_batches(run, workers, sampler, rcodec, text, batches, done, out, each,
                       max_chunks=run.profile.pool_max_chunks)
    finally:
        workers.close()
    meta = {'shard': args.shard, 'nshards': args.nshards, 'cases': cases, 'reps': reps, 'gl': run.gl,
            'render_mode': run.config['render_mode'], 'rows': per, 'load_seconds': load_s,
            'total_seconds': time.monotonic() - t0, 'nfe': R.NFE, 'gpu_peak': R.peak_gib(), 'torch': torch.__version__,
            'hip': getattr(torch.version, 'hip', None), 'gpu': torch.cuda.get_device_name(0),
            'lora_params': A.n_params(bundle.model)}
    R.write_json(out / f'pool/meta-{tag}.json', meta)
    print('done', meta['gpu_peak'], round(meta['total_seconds'], 1), flush=True)


def select(run, args):
    """Start split (CPU): pool/selection.json + probe/starts.json, or rounds/<round>/starts.json with ``--round``."""
    profile = run.profile
    if args.round is None:
        sel = profile.select()
        R.write_json(run.root / 'pool/selection.json', sel)
        R.write_json(run.root / 'probe/starts.json', sel['probe'])
        print(json.dumps({k: v for k, v in sel.items() if k not in ('probe', 'train', 'teacher')}, indent=1))
        print('probe starts', len(sel['probe']), 'train starts', len(sel['train']))
    else:
        sel = R.load_json(run.root / 'pool/selection.json')
        st = profile.round_starts(sel, args.round)
        R.write_json(run.root / 'rounds' / args.round / 'starts.json', st)
        print(args.round, len(st), 'starts', len({s['case_id'] for s in st}), 'scenes')
