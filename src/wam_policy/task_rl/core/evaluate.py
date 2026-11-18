"""In-training full-episode evaluation (GPU): a policy from reset on the run's scenes until the runtime's own terminal.

This is the held-out check of the training chain (success rate after every ``eval_every`` rounds), not the paper's
Standard100 benchmark. Policy: the teacher (``--lora`` absent), a round's ``old-lora.pt`` (``--lora``, mode old) or the
current adapters of a round's ``state.pt`` (``--lora-current``). Scenes: ``probe`` / ``train`` (the run's split),
``all`` (the profile's scene order) or an integer N (its first N). ``--reps R`` runs every scene R times with seeds
seed_of('eval', case, rep), the same for every policy, so two policies are paired per (scene, rep).
"""
from __future__ import annotations

import json
import time

from wam_policy.task_rl.core import run as R
from wam_policy.task_rl.core import rollout as RO
from wam_policy.task_rl.core import scene as S


def eval_units(run, scenes, reps):
    order = run.profile.scene_order()
    if scenes in ('probe', 'train'):
        sel = R.load_json(run.root / 'pool/selection.json')
        ids = sel['probe_scenes' if scenes == 'probe' else 'train_scenes']
        by = {c['case_id']: c for c in order}
        cases = [by[i] for i in ids]
    elif scenes == 'all':
        cases = order
    else:
        cases = order[:int(scenes)]
    out = []
    for r in range(reps):
        for c in cases:
            out.append(dict(case=c, unit=f"{c['case_id']}-e{r}", rep=r, seed=R.seed_of('eval', c['case_id'], r), order=len(out)))
    return out


def main(run, args):
    import torch
    from wam_policy.task_rl.core import sampler as SM
    out = run.path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f's{args.shard:02d}'
    t0 = time.monotonic()
    lora = args.lora if args.lora is not None else args.lora_current
    _, text, _, sampler, rcodec, label = SM.load_policy(run, lora, current=args.lora_current is not None)
    load_s = time.monotonic() - t0
    eps_path = out / f'episodes-{tag}.jsonl'
    done = {r['unit']: r for r in R.jsonl(eps_path)}
    batches, per = RO.unit_batches(run, eval_units(run, args.scenes, args.reps), args.shard, args.nshards)
    workers = S.Workers(run, out / 'states', per if batches else 0)
    print('policy', label, 'loaded', round(load_s, 1), 's; units', sum(map(len, batches)), 'rows per batch', per,
          'done', len(done), flush=True)

    def each(u, s, _):
        s.update(order=u['order'], rep=u['rep'], policy=label)
        R.append_jsonl(eps_path, s)
        print(json.dumps({k: s.get(k) for k in ('order', 'unit', 'chunks', 'terminal', 'success', 'final_D', 'min_D',
                                                'wall_s', 'chunk_wall_median_s')}), flush=True)

    try:
        RO.run_batches(run, workers, sampler, rcodec, text, batches, done, out, each, capture=False, save_history=False,
                       sub='.', max_chunks=run.profile.eval_max_chunks)
    finally:
        workers.close()
    R.write_json(out / f'meta-{tag}.json', {'shard': args.shard, 'nshards': args.nshards, 'lora': label, 'scenes': args.scenes,
                                             'reps': args.reps, 'gl': run.gl, 'render_mode': run.config['render_mode'],
                                             'rows': per, 'load_seconds': load_s, 'total_seconds': time.monotonic() - t0,
                                             'gpu_peak': R.peak_gib(), 'gpu': torch.cuda.get_device_name(0)})
