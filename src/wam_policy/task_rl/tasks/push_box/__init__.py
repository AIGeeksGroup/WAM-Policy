"""push_box task profile of Stage 2.1.

Starts. The teacher runs from reset on the training scenes to its own terminal (success / fall / 24 s timeout,
capped at 76 chunks). Every chunk boundary that is (a) after the first FL-foot/box contact, (b) before the first row
with task distance D < 0.10 m and (c) at least one window before the horizon is persisted (``capture_state``). Scenes
with at least 3 such boundaries are usable; in the seeded scene order the first ``probe_scenes`` usable scenes are the
probe set and the next ``train_scenes`` the training scenes. Probe: ``probe_per_scene`` boundaries per scene (evenly
spread); training: up to ``train_per_scene`` per scene, filled round-robin up to ``train_min``; one seeded permutation;
round r uses entries [n r, n r + n) cyclically (n = starts_per_round).

Window: 6 chunks, every chunk trains. Reward: ``rewards.PushReward`` (metres). Group: one start, K candidates; skipped
when the reward range is below ``skip_range_m`` (4 mm). rho = 0.5 + 0.5 clip((r - group mean) / scale, -1, 1), scale =
the pooled within-group sd frozen at round 0.

Stop rules (SE = scene-clustered SE of the paired probe difference against the round-0 teacher probe):
  regress   probe diff lower than the previous round's by more than 2 SE in two consecutive rounds;
            probe in-window successes two or more below the teacher's;
            training-group mean reward lower than the previous round three rounds in a row
  runaway   |first-chunk mean shift| > 0.1 normalizer units on any channel while the probe diff is not above 1 SE;
            reference regulariser (round mean) > 0.02; more than half of the groups skipped
  plateau   probe diff changed by less than 1 SE in each of the last three rounds; or max_rounds reached
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from wam_policy.task_rl.core import run as R
from wam_policy.task_rl.tasks import base as B
from wam_policy.task_rl.tasks.push_box import rewards as RW

D_NEAR = 0.10
HORIZON_TICKS = 12000      # 24 s case horizon
MAX_CHUNKS = 76
POOL_DEFAULT = {'probe_scenes': 8, 'train_scenes': 40, 'probe_per_scene': 3, 'train_per_scene': 5, 'train_min': 200,
                'start_after_contact': True}


class PushTracker(B.EpisodeTracker):
    def __init__(self, unit, capture_dir, capture, profile):
        super().__init__(unit, capture_dir, capture)
        self.profile = profile
        self.near_seen = False
        self.first_near_tick = None

    def observe(self, rows, cols):
        D = RW.D_from_rows(rows, cols)
        if not self.near_seen and (D < D_NEAR).any():
            self.near_seen, self.first_near_tick = True, int(rows[np.argmax(D < D_NEAR), 0])
        self.last_distance = float(D[-1])

    def capture_now(self, end_tick, contact_seen):
        after_contact = contact_seen or not self.profile.pool_cfg()['start_after_contact']
        return (after_contact and not self.near_seen
                and end_tick + self.profile.window_chunks * R.CHUNK_TICKS <= HORIZON_TICKS)

    def summary(self, allrows, cols):
        D = RW.D_from_rows(allrows, cols)
        return dict(final_D=float(D[-1]), min_D=float(D.min()), first_near_tick=self.first_near_tick)


class PushBox(B.TaskProfile):
    name = 'push_box'
    reward_unit = 'm'
    CONFIG_DEFAULTS = {'window_chunks': 6, 'skip_range_m': 0.004, 'pool_reps': 2, 'pool': dict(POOL_DEFAULT)}
    # integer form of the paper run's seed labels (scene order, training-start order)
    SCENE_ORDER_SEED = int(
        '726c2d76342d707573682d6e66742d70696c6f742d32303236313131342d706f6f6c1e7fd5de0f67fafdfe9e0f21fec427e3fe235ced4743012278'
        'c0732f8486e853c3a4f24f15dd0f09ff34c7640517d85ff83373d6c01de2266eaa6bcc471e3a0e', 16)
    TRAIN_ORDER_SEED = int(
        '726c2d76342d707573682d6e66742d70696c6f742d32303236313131342d747261696e2d6f7264657253c749834fb5849a05570d8ce61aba8b2a'
        '442d9184c1a548b22809577ef81bd28846f082ea5a7be6dcde8bbda95660defac2f784c024eebd1ad909e3116557ab', 16)

    @property
    def pool_max_chunks(self):
        return MAX_CHUNKS

    @property
    def eval_max_chunks(self):
        return MAX_CHUNKS

    def pool_cfg(self):
        return {**POOL_DEFAULT, **self.cfg.get('pool', {})}

    def episode_tracker(self, unit, capture_dir, capture):
        return PushTracker(unit, capture_dir, capture, self)

    # ---- starts
    def select(self):
        pc = self.pool_cfg()
        eps = sorted(R.jsonl(self.run.root / 'pool/episodes.jsonl'), key=lambda r: r['order'])
        bds = {}
        for b in R.jsonl(self.run.root / 'pool/boundaries.jsonl'):
            bds.setdefault(b['case_id'], []).append(b)
        usable = list(dict.fromkeys(e['case_id'] for e in eps if len(bds.get(e['case_id'], [])) >= 3))
        probe_scenes = usable[:pc['probe_scenes']]
        train_scenes = usable[pc['probe_scenes']:pc['probe_scenes'] + pc['train_scenes']]
        spread = _spread
        probe = [b for c in probe_scenes for b in spread(sorted(bds[c], key=lambda b: b['groups']), pc['probe_per_scene'])]
        chosen = {c: spread(sorted(bds[c], key=lambda b: b['groups']), pc['train_per_scene']) for c in train_scenes}
        while sum(map(len, chosen.values())) < pc['train_min']:
            added = False
            for c in train_scenes:
                rest = [b for b in sorted(bds[c], key=lambda b: b['groups']) if b not in chosen[c]]
                if rest and sum(map(len, chosen.values())) < pc['train_min']:
                    chosen[c].append(rest[len(rest) // 2]); added = True
            if not added:
                break
        train = [b for c in train_scenes for b in sorted(chosen[c], key=lambda b: b['groups'])]
        train = [train[i] for i in self.train_order(len(train))]
        outcome = {}
        for e in eps:
            outcome.setdefault(e['case_id'], []).append(bool(e['success']))
        return {'probe_scenes': probe_scenes, 'train_scenes': train_scenes, 'probe': probe, 'train': train,
                'usable_scenes': len(usable), 'episodes': len(eps),
                'teacher_baseline': {'episodes': len(eps), 'success': int(sum(e['success'] for e in eps)),
                                     'terminal': {str(t): int(sum(e['terminal'] == t for e in eps))
                                                  for t in {e['terminal'] for e in eps}}},
                'probe_scene_outcomes': {c: outcome[c] for c in probe_scenes},
                'train_scene_outcomes': {c: outcome[c] for c in train_scenes}, 'pool_config': pc}

    def round_starts(self, sel, rnd):
        if rnd.startswith('smoke'):
            return self.smoke_starts(sel, rnd)
        r, n, T = int(rnd[1:]), self.cfg['starts_per_round'], sel['train']
        return [T[(r * n + i) % len(T)] for i in range(n)]

    # ---- reward
    def candidate_rewards(self, collect_dir, starts):
        import torch
        Rw = RW.PushReward(self.window_chunks)
        base_cache, out = {}, {}
        by_id = {s['id']: s for s in starts}
        for row in R.jsonl(Path(collect_dir) / 'results.jsonl'):
            st = by_id[row['start']]
            unit = st.get('unit', st['case_id'])
            if unit not in base_cache:
                base_cache[unit] = self.base_rows(unit)
            base, cols = base_cache[unit]
            rec = torch.load(Path(collect_dir) / 'cands' / f"{row['candidate']}.pt", weights_only=False, map_location='cpu')
            if list(rec['columns']) != cols:
                raise ValueError('Column mismatch')
            w = RW.candidate_reward(Rw, base, rec['rows'], cols)
            req = rec['requested']
            out[row['candidate']] = dict(start=row['start'], case_id=st['case_id'], k=row['k'], r=float(w['r']),
                                         dev=float(w['dev']), prog=float(w['prog']), ds=float(w['ds']),
                                         dlat=float(w['dlat']), dyaw=float(w['dyaw']), padded=w['padded'],
                                         success=bool(rec['success']),
                                         terminal=None if rec['terminal'] is None else rec['terminal']['reason'],
                                         chunks=int(rec['video'].shape[0]),
                                         first_chunk=req[0][:, :6].tolist(), window_mean=req[:, :, :6].mean((0, 1)).tolist(),
                                         final_D=float(RW.D_from_rows(rec['rows'][-1:], cols)[0]))
        return out

    def pooled_sd(self, cands):
        groups = {}
        for c in cands.values():
            groups.setdefault(c['start'], []).append(c['r'])
        ss = sum(float(((np.array(v) - np.mean(v)) ** 2).sum()) for v in groups.values())
        dof = sum(len(v) - 1 for v in groups.values())
        return (ss / dof) ** .5, dof

    def score(self, cands, scale):
        skip_range = self.cfg['skip_range_m']
        groups = {}
        for cid, c in cands.items():
            groups.setdefault(c['start'], []).append(cid)
        rho, per = {}, {}
        for sid, ids in sorted(groups.items()):
            ids = sorted(ids, key=lambda x: cands[x]['k'])
            r = np.array([cands[i]['r'] for i in ids])
            w = None if scale is None else B.group_weights(r, scale, skip_range)
            per[sid] = dict(n=len(ids), mean_mm=float(r.mean() * 1e3), sd_mm=float(r.std(ddof=1) * 1e3) if len(r) > 1 else 0.,
                            range_mm=float(np.ptp(r) * 1e3), skipped=bool(np.ptp(r) < skip_range),
                            successes=int(sum(cands[i]['success'] for i in ids)))
            if w is not None:
                per[sid]['rho'] = [float(x) for x in w]
                per[sid]['clipped'] = int(sum((x <= 0.0) | (x >= 1.0) for x in w))
                rho.update({i: float(x) for i, x in zip(ids, w)})
        sd, dof = self.pooled_sd(cands)
        r = np.array([c['r'] for c in cands.values()])
        agg = dict(candidates=len(cands), groups=len(groups), skipped_groups=int(sum(v['skipped'] for v in per.values())),
                   used_candidates=len(rho), mean_reward_mm=float(r.mean() * 1e3), pooled_within_sd_mm=sd * 1e3, dof=dof,
                   scale_mm=None if scale is None else scale * 1e3,
                   median_group_sd_mm=float(np.median([v['sd_mm'] for v in per.values()])),
                   median_group_range_mm=float(np.median([v['range_mm'] for v in per.values()])),
                   successes_in_window=int(sum(c['success'] for c in cands.values())),
                   padded=int(sum(c['padded'] for c in cands.values())),
                   clipped_fraction=float(np.mean([(x <= 0) or (x >= 1) for x in rho.values()])) if rho else None,
                   rho_mean_abs_dev=float(np.mean([abs(2 * x - 1) for x in rho.values()])) if rho else None)
        return agg, per, rho

    # ---- probe
    def probe_compare(self, ref, run):
        ids = sorted(set(ref) & set(run))
        d = np.array([run[i]['r'] - ref[i]['r'] for i in ids])
        out = dict(n=len(ids), mean_reward_mm=float(np.mean([run[i]['r'] for i in ids]) * 1e3),
                   ref_mean_reward_mm=float(np.mean([ref[i]['r'] for i in ids]) * 1e3),
                   diff_mean_mm=float(d.mean() * 1e3),
                   diff_se_mm=float(d.std(ddof=1) / len(d) ** .5 * 1e3) if len(d) > 1 else None,
                   diff_sd_mm=float(d.std(ddof=1) * 1e3) if len(d) > 1 else None,
                   diff_abs_median_mm=float(np.median(np.abs(d)) * 1e3),
                   successes=int(sum(run[i]['success'] for i in ids)), ref_successes=int(sum(ref[i]['success'] for i in ids)),
                   positive=int((d > 0).sum()), negative=int((d < 0).sum()))
        out['diff_se_start_cluster_mm'] = (B.cluster_se(d, [run[i]['start'] for i in ids]) or 0) * 1e3
        scene = lambda c: c.get('case_id') or re.sub(r'(-r\d+)?-k\d+$', '', c['start'])
        out['diff_se_scene_cluster_mm'] = (B.cluster_se(d, [scene(run[i]) for i in ids]) or 0) * 1e3
        out.update(B.command_shift(run, ref, ids))
        return out

    # ---- round table
    def round_row(self, rd, ref):
        sc, lm, pr = (B.load_or_none(rd / f) for f in ('scores.json', 'learn-metrics.json', 'probe/scores.json'))
        agg = sc['aggregate']
        row = {'round': rd.name, 'train_mean_reward_mm': agg['mean_reward_mm'], 'median_group_sd_mm': agg['median_group_sd_mm'],
               'groups': agg['groups'], 'skipped_groups': agg['skipped_groups'], 'clipped_fraction': agg['clipped_fraction'],
               'in_window_successes_train': agg['successes_in_window']}
        if lm:
            row.update(B.learn_columns(lm))
        if pr and ref:
            pc = self.probe_compare(ref, pr['candidates'])
            row.update(probe_reward_mm=pc['mean_reward_mm'], probe_diff_mm=pc['diff_mean_mm'],
                       probe_se_mm=pc['diff_se_scene_cluster_mm'], probe_pos=pc['positive'], probe_neg=pc['negative'],
                       probe_successes=pc['successes'], first_chunk_shift_mean=pc['first_chunk_shift_mean'])
        row['stage_seconds'], row['round_seconds'] = B.stage_seconds(rd)
        return row

    def stop_rules(self, rows, ref_successes):
        cfg, fired = self.cfg, []
        pr = [r for r in rows if 'probe_diff_mm' in r]
        for i in range(2, len(pr)):
            if all(pr[j]['probe_diff_mm'] < pr[j - 1]['probe_diff_mm'] - 2 * pr[j]['probe_se_mm'] for j in (i - 1, i)):
                fired.append(('regress', pr[i]['round'], 'probe diff fell by > 2 SE in two consecutive rounds'))
        for r in pr:
            if ref_successes is not None and r['probe_successes'] <= ref_successes - 2:
                fired.append(('regress', r['round'], f"probe in-window successes {r['probe_successes']} <= teacher {ref_successes} - 2"))
        tr = [r['train_mean_reward_mm'] for r in rows]
        for i in range(3, len(tr)):
            if tr[i] < tr[i - 1] < tr[i - 2] < tr[i - 3]:
                fired.append(('regress', rows[i]['round'], 'training-group mean reward fell three rounds in a row'))
        for r in rows:
            if 'first_chunk_shift_mean' in r:
                big = {k: v for k, v in r['first_chunk_shift_mean'].items() if abs(v) > 0.1}
                if big and not r['probe_diff_mm'] > r['probe_se_mm']:
                    fired.append(('runaway', r['round'], f'first-chunk shift {big} without a probe gain'))
            if (r.get('reg_mean') or 0) > 0.02:
                fired.append(('runaway', r['round'], f"reference regulariser {r['reg_mean']:.4f} > 0.02"))
            if r['skipped_groups'] > r['groups'] / 2:
                fired.append(('runaway', r['round'], f"{r['skipped_groups']}/{r['groups']} groups skipped"))
        if len(pr) >= 4 and all(abs(pr[j]['probe_diff_mm'] - pr[j - 1]['probe_diff_mm']) < pr[j]['probe_se_mm']
                                for j in range(len(pr) - 3, len(pr))):
            fired.append(('plateau', pr[-1]['round'], 'probe diff changed < 1 SE in each of the last three rounds'))
        if len(rows) >= cfg.get('max_rounds', 15):
            fired.append(('plateau', rows[-1]['round'], f"max_rounds {cfg.get('max_rounds', 15)} reached"))
        return fired

    def round_table(self, fail_on_stop=False):
        O = self.run.root
        ref_s, rep_s = B.load_or_none(O / 'probe/r00-pre-a/scores.json'), B.load_or_none(O / 'probe/r00-pre-b/scores.json')
        ref = ref_s['candidates'] if ref_s else None
        rows = [self.round_row(rd, ref) for rd in sorted((O / 'rounds').glob('r[0-9][0-9]')) if (rd / 'scores.json').exists()]
        repeat = self.probe_compare(ref, rep_s['candidates']) if ref and rep_s else None
        ref_successes = self.probe_compare(ref, ref)['successes'] if ref else None
        evals = {d.name: B.load_or_none(d / 'summary.json') for d in sorted((O / 'eval').glob('*')) if (d / 'summary.json').exists()}
        fired = self.stop_rules(rows, ref_successes)
        res = {'rounds': rows, 'repeat_noise': repeat, 'teacher_probe_successes': ref_successes, 'stop_rules_fired': fired,
               'evals': {k: {kk: v.get(kk) for kk in ('episodes', 'success', 'success_rate', 'lora', 'scenes', 'reps')}
                         for k, v in evals.items()}}
        R.write_json(O / 'results/round-table.json', res)
        cols = [('round', None), ('train_mean_reward_mm', 1), ('median_group_sd_mm', 1), ('skipped_groups', None),
                ('clipped_fraction', 2), ('samples', None), ('grad_norm_mean', 4), ('reg_mean', 5), ('disp_current', 3),
                ('disp_old', 3), ('probe_diff_mm', 2), ('probe_se_mm', 2), ('probe_pos', None), ('probe_neg', None),
                ('probe_successes', None), ('round_seconds', 0)]
        lines = ['| ' + ' | '.join(c for c, _ in cols) + ' | vy shift | fl_y shift |', '|' + '---|' * (len(cols) + 2)]
        for r in rows:
            sh = r.get('first_chunk_shift_mean') or {}
            lines.append('| ' + ' | '.join(B.fmt(r.get(c), nd) if nd is not None else B.fmt(r.get(c)) for c, nd in cols)
                         + f" | {B.fmt(sh.get('vy'), 4)} | {B.fmt(sh.get('fl_y'), 4)} |")
        md = ['# Round table (push_box)', '', f'Run: `{O}`', '']
        if repeat:
            md += [f"Same-seed repeat noise (r00 pre-b vs pre-a): {repeat['diff_mean_mm']:+.2f} ± "
                   f"{repeat['diff_se_scene_cluster_mm']:.2f} mm (scene-clustered SE); teacher probe in-window successes "
                   f"{ref_successes}.", '']
        md += lines + ['']
        if evals:
            md += ['| eval | success | episodes | policy |', '|---|---|---|---|']
            md += [f"| {k} | {v['success']} | {v['episodes']} | {v.get('lora')} |" for k, v in evals.items()] + ['']
        md += ['Stop rules fired:' if fired else 'Stop rules fired: none'] + [f'- {k} at {r}: {why}' for k, r, why in fired]
        (O / 'results/round-table.md').write_text('\n'.join(md) + '\n')
        print('\n'.join(md))
        return res, (3 if fail_on_stop and fired else 0)

    #: round 0 runs the teacher probe twice (reference and same-seed repeat)
    probe_reference_runs = ('pre-a', 'pre-b')


def _spread(items, n):
    if len(items) <= n:
        return list(items)
    idx = np.unique(np.round(np.linspace(0, len(items) - 1, n)).astype(int))
    return [items[i] for i in idx]


PROFILE = PushBox
