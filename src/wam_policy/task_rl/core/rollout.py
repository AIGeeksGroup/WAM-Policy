"""Closed-loop episodes from reset, several in lock step (start pool and full-episode evaluation).

Policy: the teacher (25/50, no CFG, ordinary independent noise per row) or a round's adapters. Each episode runs
until the runtime's own terminal (success / fall / case-horizon timeout) or a chunk cap. Rows: one row-batched
sampling per chunk, every row's scene worker executing at the same time, one row-batched VAE encode and commit; a row
leaves the batch at its terminal. Row-batched and one-row sampling agree to bf16 rounding.

Start capture is task specific (``profile.episode_tracker``): push_box persists the live state at every eligible
chunk boundary, open_room_door persists one boundary from the worker's in-memory ring once the handle contact is
seen. Per episode: the history record (``history.save``), every privileged row from reset, the outcome.
"""
from __future__ import annotations

import time

import numpy as np

from wam_policy.task_rl.core import history
from wam_policy.task_rl.core import scene as S
from wam_policy.task_rl.core.history import chunk_views
from wam_policy.task_rl.core.run import TICK_S


def run_cases(run, workers, sampler, rcodec, text, units, out, *, max_chunks, capture=True, save_history=True,
              sub='pool'):
    """``units``: dicts with case (case_id, case, model), unit, seed. Returns [(summary, boundaries)] per unit."""
    profile = run.profile
    n = len(units)
    t0 = time.monotonic()
    first = workers.run(range(n), lambda env, u: env.reset(u['case']['case'], u['case']['model'], f"{sub}-{u['unit']}"), units)
    records = []
    for b in first:
        r = history.HistoryRecord(); r.initial(b['rgb']); records.append(r)
    rcodec.reset()
    latents = rcodec.push(S.stack_views([chunk_views(r.frames, [0]) for r in records]))
    sampler.reset_rows(latents, text, [u['seed'] for u in units])     # alpha 0: ordinary independent noise per row
    cols = workers.envs[0].columns()
    trackers = [profile.episode_tracker(u, out / 'states' / 'pool', capture) for u in units]
    st = [dict(contact_seen=False, first_contact_tick=None, terminal=None, chunks=max_chunks, wall=None, timing=None,
               chunk_wall=[]) for _ in units]
    timing = {'predict': 0., 'physics': 0., 'feedback': 0., 'capture': 0.}
    if capture:
        for u in units:
            (out / 'states' / 'pool' / u['unit']).mkdir(parents=True, exist_ok=True)
    alive = list(range(n))
    for k in range(max_chunks):
        tc = time.monotonic()
        t = time.monotonic(); _, _, req = sampler.endpoints(); timing['predict'] += time.monotonic() - t
        t = time.monotonic()
        results = workers.run(alive, lambda env, c, tr: tr.execute(env, c), req, [trackers[i] for i in alive])
        timing['physics'] += time.monotonic() - t
        keep = []
        for pos, (i, r) in enumerate(zip(alive, results)):
            s = st[i]
            rows = np.asarray(r['privileged']['rows'], np.float64)
            if 'fl_object_contact' in cols:
                fc = rows[:, cols.index('fl_object_contact')] > 0.5
                if not s['contact_seen'] and fc.any():
                    s['contact_seen'], s['first_contact_tick'] = True, int(rows[np.argmax(fc), 0])
            trackers[i].observe(rows, cols)
            if r['terminal'] is not None:
                s['terminal'], s['chunks'], s['wall'], s['timing'] = r['terminal'], k + 1, time.monotonic() - t0, dict(timing)
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
        sampler.commit(latent, executed)
        for i, r, e in zip(alive, results, executed):
            records[i].chunk(r['samples'], e)
        timing['feedback'] += time.monotonic() - t
        if capture:
            t = time.monotonic()
            cap = [(i, int(r['end_tick'])) for i, r in zip(alive, results)
                   if trackers[i].capture_now(int(r['end_tick']), st[i]['contact_seen'])]
            if cap:
                sids = [f"{units[i]['unit']}-k{k + 1:02d}" for i, _ in cap]
                paths = [out / 'states' / 'pool' / units[i]['unit'] / f'{sid}.pt' for (i, _), sid in zip(cap, sids)]
                workers.run([i for i, _ in cap], lambda env, path: env.capture_state(path), [str(p) for p in paths])
                for (i, tick), sid, path in zip(cap, sids, paths):
                    u = units[i]
                    trackers[i].boundaries.append(dict(id=sid, case_id=u['case']['case_id'], unit=u['unit'], groups=k + 1,
                                                       tick=tick, t_s=tick * TICK_S, state=str(path.relative_to(out)),
                                                       D=trackers[i].last_distance))
            timing['capture'] += time.monotonic() - t
        for i in alive:
            st[i]['chunk_wall'].append(time.monotonic() - tc)
    outs = []
    for i, u in enumerate(units):
        s, record, case, unit = st[i], records[i], u['case'], u['unit']
        rec_path = out / sub / 'history' / f'{unit}.npz'
        if save_history:
            rec_path.parent.mkdir(parents=True, exist_ok=True)
            if rec_path.exists():
                rec_path.unlink(); rec_path.with_suffix('.json').unlink(missing_ok=True)
            history.save(rec_path, record)
        allrows = workers.envs[i].rows()
        (out / sub / 'episodes').mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / sub / 'episodes' / f'{unit}.npz', privileged=allrows, columns=np.asarray(cols))
        terminal = s['terminal']
        bds = trackers[i].boundaries
        summary = dict(case_id=case['case_id'], unit=unit, seed=u['seed'], chunks=s['chunks'],
                       terminal=None if terminal is None else terminal['reason'],
                       success=bool(terminal is not None and terminal.get('success', False)),
                       final_tick=int(allrows[-1, 0]), first_contact_tick=s['first_contact_tick'],
                       **trackers[i].summary(allrows, cols),
                       eligible=len(bds), record=str(rec_path.relative_to(out)) if save_history else None,
                       record_groups=record.groups, wall_s=s['wall'] or time.monotonic() - t0,
                       timing=s['timing'] or dict(timing), rows=n,
                       chunk_wall_median_s=float(np.median(s['chunk_wall'])) if s['chunk_wall'] else None)
        for bd in bds:
            bd['record'] = summary['record']; bd['case'] = case['case']; bd['model'] = case['model']
        outs.append((summary, bds))
    return outs


def run_batches(run, workers, sampler, rcodec, text, batches, done, out, each, **kw):
    """Run every batch (skipping finished units; one retry after a scene-worker failure) and hand each episode to
    ``each(unit, summary, boundaries)``."""
    for batch in batches:
        todo = [u for u in batch if u['unit'] not in done]
        if not todo:
            continue
        for attempt in (1, 2):
            try:
                outs = run_cases(run, workers, sampler, rcodec, text, todo, out, **kw)
                break
            except S.INFRA as exc:
                print(f'infrastructure error {todo[0]["unit"]} attempt {attempt}: {exc!r}', flush=True)
                if attempt == 2:
                    raise
                workers.restart()
        for u, (s, bds) in zip(todo, outs):
            each(u, s, bds)


def unit_batches(run, all_units, shard, nshards):
    """Episodes that run together: min(rows, ceil(units / shards)) consecutive units per batch, batches dealt to the
    shards in turn."""
    from wam_policy.task_rl.core.run import batches_of, shard_of
    per = max(1, min(run.rows, -(-len(all_units) // nshards)))
    return shard_of(batches_of(all_units, per), shard, nshards), per
