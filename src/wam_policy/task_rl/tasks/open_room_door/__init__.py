"""open_room_door task profile of Stage 2.1.

Contact metric X6 (``metric``): f = FL foot, g = the static handle point of the case, door yaw psi from the reset row;
dx/dy/dz = f - g in the door frame (dx + toward the door). tc = first row at or after the lift (first dz > 0) with
hypot(dx, dz) <= 3.4 cm and -6.5 <= dy <= 7.0 cm; X6 = mean dx over rows tc+64 .. tc+96, in cm (None without contact
or with fewer rows). X6 >= -0.55 cm counts as a correct contact.

Starts. One per teacher episode: the boundary 4 chunks before the chunk of the first handle contact, persisted from
the scene worker's in-memory ring while the episode runs (episodes capped at ``max_chunks`` = 66). The split is by
scene: a scene is usable when one of its episodes gave a start; in the seeded scene order the first ``probe_scenes``
usable scenes are the probe set, the next ``train_scenes`` (default: all the rest) the training scenes; probe starts
= every start of the probe scenes, training starts = every start of the training scenes in one seeded permutation;
round r takes entries [n r, n r + n) cyclically with n = min(starts_per_round, training starts).

Window: 13 chunks per candidate; only the first ``train_chunks`` = 7 (4 before the contact chunk, the contact chunk,
2 after) are training samples, the rest are scored only. Reward r = min(X6, +0.3 cm); a candidate without X6 gets the
group's lowest r; a group without any X6 is skipped and stays out of the scale. Groups with a range below
``skip_range_cm`` (0.5 cm) are skipped.

Stop rules: push_box's structure and thresholds with the probe reading replaced by the paired r difference against
the teacher probe (cm); "more than half of the groups skipped" is a normal end (``saturated``: groups whose eight
candidates all reach the cap). Full-episode evaluation runs to the case's own 60 s horizon (no chunk cap).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from wam_policy.task_rl.core import run as R
from wam_policy.task_rl.tasks import base as B

BEFORE = 4                       # start = contact chunk - 1 - BEFORE
CORRECT_CM, CAP_CM = -0.55, 0.3
HORIZON_TICKS = 30000            # 60 s case horizon
EVAL_CHUNKS_DEFAULT = -(-HORIZON_TICKS // R.CHUNK_TICKS) + 1   # 189: past the horizon, the runtime's timeout ends it


def metric(rows, cols, goal):
    """X6 of rows from reset (row i = the state after command i; row 0 = reset)."""
    col = {str(c): i for i, c in enumerate(cols)}
    f = rows[:, [col['fl_foot_x'], col['fl_foot_y'], col['fl_foot_z']]] - np.asarray(goal, np.float64)
    psi = 2 * math.atan2(rows[0, col['object_qz']], rows[0, col['object_qw']]) - rows[0, col['progress']]
    c, s = math.cos(psi), math.sin(psi)
    dx, dy, dz = c * f[:, 0] + s * f[:, 1], -s * f[:, 0] + c * f[:, 1], f[:, 2]
    up = np.flatnonzero(dz > 0)
    tc = None
    if len(up):
        hit = np.flatnonzero((np.arange(len(rows)) >= up[0]) & (np.hypot(dx, dz) <= 0.034) & (dy >= -0.065) & (dy <= 0.070))
        tc = int(hit[0]) if len(hit) else None
    x6 = float(dx[tc + 64:tc + 97].mean() * 100) if tc is not None and tc + 96 < len(rows) else None
    return dict(dx=dx, tc=tc, x6=x6, contact_chunk=None if tc is None else (tc - 1) // R.CH + 1)   # chunk 1-based


class DoorTracker(B.EpisodeTracker):
    """Every chunk keeps its post-chunk snapshot in the worker's ring until the first handle contact is seen on the
    rows from reset; then the boundary BEFORE chunks before the contact chunk is persisted (``persist_ring``)."""

    def __init__(self, unit, capture_dir, capture, run):
        super().__init__(unit, capture_dir, capture)
        self.run = run
        self.found = False
        self.goal = json.loads(Path(unit['case']['case']).read_text())['goal']['position']

    def execute(self, env, commands):
        active = self.capture and not self.found
        r = env.execute(commands, ring=active)
        if active:
            m = metric(env.rows(), env.columns(), self.goal)
            if m['tc'] is not None:
                self.found = True
                groups = m['contact_chunk'] - 1 - BEFORE
                if groups >= 1:
                    unit = self.unit['unit']
                    sid = f'{unit}-k{groups:02d}'
                    path = self.capture_dir / unit / f'{sid}.pt'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    rc = env.env.persist_ring(groups - 1, str(path), {'start': sid, 'contact_chunk': m['contact_chunk'],
                                                                      'tc_row': m['tc']})
                    tick = int(rc['absolute_tick'])
                    self.boundaries.append(dict(id=sid, groups=groups, tick=tick, tc_row=m['tc'],
                                                contact_chunk=m['contact_chunk'], state=str(path.relative_to(self.run.root)),
                                                case_id=self.unit['case']['case_id'], unit=unit, t_s=tick * R.TICK_S, D=None))
        return r


class OpenRoomDoor(B.TaskProfile):
    name = 'open_room_door'
    reward_unit = 'cm'
    CONFIG_DEFAULTS = {'window_chunks': 13, 'train_chunks': 7, 'skip_range_cm': 0.5, 'max_chunks': 66, 'pool_reps': 4,
                       'pool': {'probe_scenes': 8, 'train_scenes': None}}
    # integer form of the paper run's seed labels (scene order, training-start order)
    SCENE_ORDER_SEED = int(
        '726c2d76342d6f70656e2d646f6f722d6e66742d706f6f6c9f17b3d9fde1c79098ead09069ea96ade8fe4cafdce578b686faa8ca4e061982f446'
        '4354235ce6ee8f561d90e9ff02e659626b05a28c00c9cdbb15afcdec9d47', 16)
    TRAIN_ORDER_SEED = int(
        '726c2d76342d6f70656e2d646f6f722d6e66742d747261696e2d6f726465729bf290a46f72a25d808cd6a5f1e28052afc9f44d3584b8e1ba568a'
        '0af11d4e669c144fcda18bed30c6902a01d173e3bd37bc9bfb59ad74ad5990687f4f4ba345', 16)
    probe_reference_runs = ('pre-a',)      # one seed repeats bit for bit on the CPU renderer: no second teacher run

    def train_chunks(self, chunks):
        return min(chunks, int(self.cfg['train_chunks']))

    @property
    def pool_max_chunks(self):
        return int(self.cfg.get('max_chunks') or 66)

    @property
    def eval_max_chunks(self):
        return int(self.cfg.get('eval_max_chunks') or EVAL_CHUNKS_DEFAULT)

    def cases(self):
        out = super().cases()
        for c in out:
            c['goal'] = json.loads(Path(c['case']).read_text())['goal']['position']
        return out

    def goals(self):
        return {c['case']: c['goal'] for c in self.cases()}

    def episode_tracker(self, unit, capture_dir, capture):
        return DoorTracker(unit, capture_dir, capture, self.run)

    # ---- starts
    def teacher_rows(self):
        cases = {c['case_id']: c for c in self.cases()}
        out = []
        for e in sorted(R.jsonl(self.run.root / 'pool/episodes.jsonl'), key=lambda r: r['order']):
            priv, cols = self.base_rows(e['unit'])
            m = metric(priv, cols, cases[e['case_id']]['goal'])
            out.append(dict(unit=e['unit'], case_id=e['case_id'], order=e['order'], rep=e.get('rep', 0), chunks=e['chunks'],
                            terminal=e['terminal'], success=e['success'], tc=m['tc'], contact_chunk=m['contact_chunk'],
                            x6=m['x6'], correct=bool(m['x6'] is not None and m['x6'] >= CORRECT_CM)))
        return out

    @staticmethod
    def baseline(rows):
        return {'episodes': len(rows), 'success': int(sum(t['success'] for t in rows)),
                'with_contact': int(sum(t['tc'] is not None for t in rows)), 'with_x6': int(sum(t['x6'] is not None for t in rows)),
                'contact_correct': int(sum(t['correct'] for t in rows)),
                'correct_x_success': two_by_two(rows),
                'terminal': {str(k): int(sum(t['terminal'] == k for t in rows)) for k in {t['terminal'] for t in rows}}}

    def select(self):
        pc = {'probe_scenes': 8, 'train_scenes': None, **self.cfg.get('pool', {})}
        teach = self.teacher_rows()
        bds = {b['unit']: b for b in R.jsonl(self.run.root / 'pool/boundaries.jsonl')}
        by_scene = {}
        for t in teach:                                   # episode order: every scene's rep 0 first, then rep 1, ...
            if t['unit'] in bds:
                by_scene.setdefault(t['case_id'], []).append(bds[t['unit']])
        usable = list(dict.fromkeys(t['case_id'] for t in teach if t['case_id'] in by_scene))
        n_p = pc['probe_scenes']
        probe_scenes = usable[:n_p]
        train_scenes = usable[n_p:] if pc['train_scenes'] is None else usable[n_p:n_p + pc['train_scenes']]
        probe = [b for c in probe_scenes for b in by_scene[c]]
        train = [b for c in train_scenes for b in by_scene[c]]
        train = [train[i] for i in self.train_order(len(train))]
        ps, ts = set(probe_scenes), set(train_scenes)
        return {'probe_scenes': probe_scenes, 'train_scenes': train_scenes, 'probe': probe, 'train': train,
                'episodes': len(teach), 'usable_episodes': len(bds), 'usable_scenes': len(usable),
                'teacher_baseline': self.baseline(teach),
                'teacher_baseline_probe_scenes': self.baseline([t for t in teach if t['case_id'] in ps]),
                'teacher_baseline_train_scenes': self.baseline([t for t in teach if t['case_id'] in ts]),
                'teacher': teach, 'pool_config': pc}

    def round_starts(self, sel, rnd):
        if rnd.startswith('smoke'):
            return self.smoke_starts(sel, rnd)
        r, T = int(rnd[1:]), sel['train']
        n = min(self.cfg['starts_per_round'], len(T))
        return [T[(r * n + i) % len(T)] for i in range(n)]

    # ---- reward
    def candidate_rewards(self, collect_dir, starts):
        import torch
        by_id = {s['id']: s for s in starts}
        goal = self.goals()
        base_cache, out = {}, {}
        for row in R.jsonl(Path(collect_dir) / 'results.jsonl'):
            st = by_id[row['start']]
            unit = st['unit']
            if unit not in base_cache:
                base_cache[unit] = self.base_rows(unit)
            base, cols = base_cache[unit]
            rec = torch.load(Path(collect_dir) / 'cands' / f"{row['candidate']}.pt", weights_only=False, map_location='cpu')
            if list(rec['columns']) != cols:
                raise ValueError('Column mismatch')
            priv, i0 = B.splice(base, np.asarray(rec['rows'], np.float64))
            m = metric(priv, cols, goal[st['case']])
            chunks = int(rec['video'].shape[0])
            train = self.train_chunks(chunks)
            req = rec['requested']
            out[row['candidate']] = dict(start=row['start'], case_id=st['case_id'], k=row['k'], x6=m['x6'],
                                         tc_rel=None if m['tc'] is None else m['tc'] - i0,
                                         contact_chunk_rel=None if m['tc'] is None else (m['tc'] - i0 - 1) // R.CH + 1,
                                         correct=bool(m['x6'] is not None and m['x6'] >= CORRECT_CM),
                                         r=None if m['x6'] is None else min(m['x6'], CAP_CM),
                                         success=bool(rec['success']),
                                         terminal=None if rec['terminal'] is None else rec['terminal']['reason'],
                                         chunks=chunks, train_chunks=train, truncated=int(self.cfg['train_chunks']) - train,
                                         first_chunk=req[0][:, :6].tolist(), window_mean=req[:, :, :6].mean((0, 1)).tolist())
        return fill_missing(out)

    def pooled_sd(self, cands):
        groups = {}
        for c in cands.values():
            if not c['group_without_x6']:
                groups.setdefault(c['start'], []).append(c['r'])
        ss = sum(float(((np.array(v) - np.mean(v)) ** 2).sum()) for v in groups.values())
        dof = sum(len(v) - 1 for v in groups.values())
        return ((ss / dof) ** .5 if dof else None), dof

    def score(self, cands, scale):
        skip = self.cfg['skip_range_cm']
        tc = int(self.cfg['train_chunks'])
        groups = {}
        for cid, c in cands.items():
            groups.setdefault(c['start'], []).append(cid)
        rho, per = {}, {}
        for sid, ids in sorted(groups.items()):
            ids = sorted(ids, key=lambda x: cands[x]['k'])
            if all(cands[i]['group_without_x6'] for i in ids):
                per[sid] = dict(n=len(ids), skipped=True, no_x6=len(ids), reason='no candidate with X6')
                continue
            r = np.array([cands[i]['r'] for i in ids])
            per[sid] = dict(n=len(ids), mean_cm=float(r.mean()), sd_cm=float(r.std(ddof=1)), range_cm=float(np.ptp(r)),
                            skipped=bool(np.ptp(r) < skip), correct=int(sum(cands[i]['correct'] for i in ids)),
                            no_x6=int(sum(cands[i]['no_x6'] for i in ids)), successes=int(sum(cands[i]['success'] for i in ids)))
            if scale is not None:
                w = B.group_weights(r, scale, skip)
                if w is not None:
                    per[sid]['rho'] = [float(x) for x in w]
                    per[sid]['clipped'] = int(sum((x <= 0.0) | (x >= 1.0) for x in w))
                    rho.update({i: float(x) for i, x in zip(ids, w)})
        sd, dof = self.pooled_sd(cands)
        x6 = np.array([c['x6'] for c in cands.values() if c['x6'] is not None])
        r = np.array([c['r'] for c in cands.values() if not c['group_without_x6']])
        used = [c for cid, c in cands.items() if cid in rho]
        agg = dict(candidates=len(cands), groups=len(groups), skipped_groups=int(sum(v['skipped'] for v in per.values())),
                   used_candidates=len(rho), correct=int(sum(c['correct'] for c in cands.values())),
                   correct_fraction=float(np.mean([c['correct'] for c in cands.values()])),
                   no_x6=int(sum(c['no_x6'] for c in cands.values())),
                   x6_quantiles_cm=dict(zip(('p0', 'p10', 'p25', 'p50', 'p75', 'p90', 'p100'),
                                            np.percentile(x6, [0, 10, 25, 50, 75, 90, 100]).round(3).tolist())) if len(x6) else None,
                   mean_x6_cm=float(x6.mean()) if len(x6) else None, mean_reward_cm=float(r.mean()) if len(r) else None,
                   reward_at_cap=int(sum(c['x6'] is not None and c['x6'] >= CAP_CM for c in cands.values())),
                   pooled_within_sd_cm=sd, dof=dof, scale_cm=scale, skip_range_cm=skip,
                   median_group_range_cm=float(np.median([v['range_cm'] for v in per.values() if 'range_cm' in v]))
                   if any('range_cm' in v for v in per.values()) else None,
                   terminated=int(sum(c['terminal'] is not None for c in cands.values())),
                   terminals={str(t): int(sum(c['terminal'] == t for c in cands.values())) for t in {c['terminal'] for c in cands.values()}},
                   truncated_candidates=int(sum(c['truncated'] > 0 for c in used)), truncated_chunks=int(sum(c['truncated'] for c in used)),
                   training_samples=int(sum(c['train_chunks'] for c in used)),
                   training_samples_expected=len(used) * tc - int(sum(c['truncated'] for c in used)),
                   successes_in_window=int(sum(c['success'] for c in cands.values())),
                   clipped_fraction=float(np.mean([(x <= 0) or (x >= 1) for x in rho.values()])) if rho else None,
                   rho_mean_abs_dev=float(np.mean([abs(2 * x - 1) for x in rho.values()])) if rho else None)
        return agg, per, rho

    # ---- probe
    def probe_compare(self, ref, run):
        ids = sorted(set(ref) & set(run))
        cr = np.array([run[i]['correct'] for i in ids], float); cf = np.array([ref[i]['correct'] for i in ids], float)
        scene = [run[i]['case_id'] for i in ids]
        both = [i for i in ids if run[i]['x6'] is not None and ref[i]['x6'] is not None]
        d = np.array([min(run[i]['x6'], CAP_CM) - min(ref[i]['x6'], CAP_CM) for i in both])
        out = dict(n=len(ids), correct_fraction=float(cr.mean()), teacher_correct_fraction=float(cf.mean()),
                   correct_diff=float((cr - cf).mean()), correct_diff_se_scene_cluster=B.cluster_se(cr - cf, scene),
                   became_correct=int(((cr == 1) & (cf == 0)).sum()), became_incorrect=int(((cr == 0) & (cf == 1)).sum()),
                   r_pairs=len(both), r_missing=len(ids) - len(both),
                   r_mean_cm=float(np.mean([min(run[i]['x6'], CAP_CM) for i in both])) if both else None,
                   teacher_r_mean_cm=float(np.mean([min(ref[i]['x6'], CAP_CM) for i in both])) if both else None,
                   r_diff_mean_cm=float(d.mean()) if len(d) else None,
                   r_diff_se_scene_cluster_cm=B.cluster_se(d, [run[i]['case_id'] for i in both]) if len(d) > 1 else None,
                   r_diff_identical=int((d == 0).sum()) if len(d) else 0,
                   positive=int((d > 0).sum()), negative=int((d < 0).sum()),
                   successes=int(sum(run[i]['success'] for i in ids)), teacher_successes=int(sum(ref[i]['success'] for i in ids)))
        out.update(B.command_shift(run, ref, ids))
        return out

    # ---- round table
    def eval_summary(self, d):
        """X6 and outcome of every episode of eval dir d; written to d/task-summary.json."""
        cases = {c['case_id']: c for c in self.cases()}
        eps = []
        for e in sorted(R.jsonl(d / 'episodes.jsonl'), key=lambda r: r['order']):
            z = np.load(d / 'episodes' / f"{e['unit']}.npz")
            m = metric(z['privileged'], [str(c) for c in z['columns']], cases[e['case_id']]['goal'])
            eps.append(dict(unit=e['unit'], case_id=e['case_id'], rep=e.get('rep'), chunks=e['chunks'], terminal=e['terminal'],
                            success=bool(e['success']), seconds_sim=float(z['privileged'][-1, 0]) * R.TICK_S, tc=m['tc'],
                            contact_chunk=m['contact_chunk'], x6=m['x6'],
                            correct=bool(m['x6'] is not None and m['x6'] >= CORRECT_CM)))
        s = B.load_or_none(d / 'summary.json') or {}
        x6 = [e['x6'] for e in eps if e['x6'] is not None]
        out = {'name': d.name, 'policy': s.get('lora'), 'scenes': s.get('scenes'), 'reps': s.get('reps'), 'episodes': len(eps),
               'success': int(sum(e['success'] for e in eps)),
               'success_rate': (sum(e['success'] for e in eps) / len(eps)) if eps else None,
               'contact_correct': int(sum(e['correct'] for e in eps)), 'no_x6': len(eps) - len(x6),
               'mean_x6_cm': float(np.mean(x6)) if x6 else None, 'correct_x_success': two_by_two(eps),
               'terminals': s.get('terminals'), 'max_chunks': max((e['chunks'] for e in eps), default=None),
               'stage_wall_seconds': s.get('stage_wall_seconds'), 'episodes_detail': eps}
        R.write_json(d / 'task-summary.json', out)
        return out

    def round_row(self, rd, ref):
        sc, lm, rb, pr = (B.load_or_none(rd / f) for f in ('scores.json', 'learn-metrics.json', 'readback.json', 'probe/scores.json'))
        agg = sc['aggregate']
        row = {'round': rd.name, **{k: agg.get(k) for k in (
            'candidates', 'groups', 'skipped_groups', 'used_candidates', 'correct_fraction', 'no_x6', 'mean_x6_cm',
            'mean_reward_cm', 'x6_quantiles_cm', 'reward_at_cap', 'pooled_within_sd_cm', 'scale_cm', 'median_group_range_cm',
            'terminated', 'truncated_candidates', 'truncated_chunks', 'training_samples', 'clipped_fraction',
            'rho_mean_abs_dev', 'successes_in_window')}}
        if lm:
            row.update(B.learn_columns(lm))
            row.update(grad_norm_steps=[s['grad_norm'] for s in lm['steps']], seconds_per_sample=lm['seconds_per_sample'],
                       gpu_peak_update=lm['gpu_peak_update'])
        if rb:
            row['readback_ok'] = rb.get('ok')
        if pr and ref:
            pc = self.probe_compare(ref, pr['candidates'])
            row.update(probe=pc, probe_correct=pc['correct_fraction'], probe_teacher_correct=pc['teacher_correct_fraction'],
                       probe_correct_diff=pc['correct_diff'], probe_correct_se=pc['correct_diff_se_scene_cluster'],
                       probe_r_cm=pc['r_mean_cm'], probe_diff_cm=pc['r_diff_mean_cm'], probe_se_cm=pc['r_diff_se_scene_cluster_cm'],
                       probe_pos=pc['positive'], probe_neg=pc['negative'], probe_successes=pc['successes'],
                       first_chunk_shift_mean=pc['first_chunk_shift_mean'])
        row['stage_seconds'], row['round_seconds'] = B.stage_seconds(rd)
        return row

    def stop_rules(self, rows, ref_successes):
        cfg, fired = self.cfg, []
        se = lambda r: r.get('probe_se_cm') or 0.
        pr = [r for r in rows if r.get('probe_diff_cm') is not None]
        for i in range(2, len(pr)):
            if all(pr[j]['probe_diff_cm'] < pr[j - 1]['probe_diff_cm'] - 2 * se(pr[j]) for j in (i - 1, i)):
                fired.append(('regress', pr[i]['round'], 'probe r diff fell by > 2 SE in two consecutive rounds'))
        for r in pr:
            if ref_successes is not None and r['probe_successes'] <= ref_successes - 2:
                fired.append(('regress', r['round'], f"probe in-window successes {r['probe_successes']} <= teacher {ref_successes} - 2"))
        tr = [r['mean_reward_cm'] for r in rows]
        for i in range(3, len(tr)):
            if tr[i] < tr[i - 1] < tr[i - 2] < tr[i - 3]:
                fired.append(('regress', rows[i]['round'], 'training-group mean r fell three rounds in a row'))
        for r in rows:
            if r.get('first_chunk_shift_mean'):
                big = {k: v for k, v in r['first_chunk_shift_mean'].items() if abs(v) > 0.1}
                if big and not (r.get('probe_diff_cm') is not None and r['probe_diff_cm'] > se(r)):
                    fired.append(('runaway', r['round'], f'first-chunk shift {big} without a probe gain'))
            if (r.get('reg_mean') or 0) > 0.02:
                fired.append(('runaway', r['round'], f"reference regulariser {r['reg_mean']:.4f} > 0.02"))
            if r['skipped_groups'] > r['groups'] / 2:
                fired.append(('saturated', r['round'], f"{r['skipped_groups']}/{r['groups']} groups skipped (normal end)"))
        if len(pr) >= 4 and all(abs(pr[j]['probe_diff_cm'] - pr[j - 1]['probe_diff_cm']) < se(pr[j]) for j in range(len(pr) - 3, len(pr))):
            fired.append(('plateau', pr[-1]['round'], 'probe r diff changed < 1 SE in each of the last three rounds'))
        if len(rows) >= cfg.get('max_rounds', 15):
            fired.append(('plateau', rows[-1]['round'], f"max_rounds {cfg.get('max_rounds', 15)} reached"))
        return fired

    def round_table(self, fail_on_stop=False):
        O = self.run.root
        ref_s = B.load_or_none(O / 'probe/r00-pre-a/scores.json')
        ref = ref_s['candidates'] if ref_s else None
        rows = [self.round_row(rd, ref) for rd in sorted((O / 'rounds').glob('r[0-9][0-9]')) if (rd / 'scores.json').exists()]
        ref_successes = int(sum(c['success'] for c in ref.values())) if ref else None
        evals = {d.name: self.eval_summary(d) for d in sorted((O / 'eval').glob('*')) if (d / 'summary.json').exists()}
        sel = B.load_or_none(O / 'pool/selection.json')
        fired = self.stop_rules(rows, ref_successes)
        res = {'rounds': rows, 'teacher_probe': None if ref is None else
               {'candidates': len(ref), 'correct_fraction': sum(c['correct'] for c in ref.values()) / len(ref),
                'successes': ref_successes},
               'teacher_pool': None if sel is None else {k: sel[k] for k in ('teacher_baseline', 'teacher_baseline_probe_scenes',
                                                                              'teacher_baseline_train_scenes') if k in sel},
               'stop_rules_fired': fired,
               'evals': {k: {kk: v[kk] for kk in v if kk != 'episodes_detail'} for k, v in evals.items()}}
        R.write_json(O / 'results/round-table.json', res)
        cols = [('round', None), ('groups', None), ('skipped_groups', None), ('correct_fraction', 3), ('mean_x6_cm', 3),
                ('mean_reward_cm', 3), ('reward_at_cap', None), ('samples', None), ('grad_norm_mean', 4), ('grad_probe_video', 4),
                ('grad_probe_action', 4), ('reg_mean', 5), ('disp_current', 3), ('disp_old', 3), ('probe_correct', 3),
                ('probe_teacher_correct', 3), ('probe_diff_cm', 3), ('probe_se_cm', 3), ('probe_pos', None), ('probe_neg', None),
                ('probe_successes', None), ('readback_ok', None), ('round_seconds', 0)]
        md = ['# Round table (open_room_door)', '', f'Run: `{O}`', '']
        if res['teacher_probe']:
            md += [f"Teacher probe: {len(ref)} candidates, contact correct {res['teacher_probe']['correct_fraction']:.3f}, "
                   f"in-window successes {ref_successes}.", '']
        if res['teacher_pool'] and 'teacher_baseline' in res['teacher_pool']:
            tb = res['teacher_pool']['teacher_baseline']
            md += [f"Teacher start pool ({tb['episodes']} episodes, capped at max_chunks): success {tb['success']}, contact "
                   f"correct {tb['contact_correct']}, correct x success {json.dumps(tb.get('correct_x_success'))}.", '']
        md += ['| ' + ' | '.join(c for c, _ in cols) + ' |', '|' + '---|' * len(cols)]
        for r in rows:
            md.append('| ' + ' | '.join(B.fmt(r.get(c), nd) if nd is not None else B.fmt(r.get(c)) for c, nd in cols) + ' |')
        md.append('')
        if evals:
            md += ['| eval | policy | episodes | success | contact correct | correct+success | correct+failure | '
                   'incorrect+success | incorrect+failure | no X6 | mean X6 cm |', '|---|---|---|---|---|---|---|---|---|---|---|']
            for k, v in evals.items():
                q = v['correct_x_success']
                md.append(f"| {k} | {v['policy']} | {v['episodes']} | {v['success']} | {v['contact_correct']} | "
                          f"{q['correct_success']} | {q['correct_failure']} | {q['incorrect_success']} | "
                          f"{q['incorrect_failure']} | {v['no_x6']} | {B.fmt(v['mean_x6_cm'], 3)} |")
            md.append('')
        md += ['Stop rules fired:' if fired else 'Stop rules fired: none'] + [f'- {k} at {r}: {why}' for k, r, why in fired]
        (O / 'results/round-table.md').write_text('\n'.join(md) + '\n')
        print('\n'.join(md))
        return res, (3 if fail_on_stop and fired else 0)


def fill_missing(cands):
    """No X6 -> the lowest r of the group; a group without any X6 -> r = 0 for all (skipped, out of the scale)."""
    groups = {}
    for c in cands.values():
        groups.setdefault(c['start'], []).append(c)
    for g in groups.values():
        have = [c['r'] for c in g if c['r'] is not None]
        for c in g:
            c['no_x6'] = c['r'] is None
            c['group_without_x6'] = not have
            if c['r'] is None:
                c['r'] = min(have) if have else 0.0
    return cands


def two_by_two(rows):
    return {f"{'correct' if c else 'incorrect'}_{'success' if s else 'failure'}":
            int(sum(bool(r['correct']) == c and bool(r['success']) == s for r in rows)) for c in (True, False) for s in (True, False)}


PROFILE = OpenRoomDoor
