"""Optional experiment tracking of a run: the round table rows (and in-training evaluations) as one metrics series.

Always writes ``results/tracking/metrics.jsonl``; mirrors to W&B only when enabled (``WANDB_MODE`` online/offline,
``WANDB_ENTITY`` / ``WANDB_PROJECT`` from the site configuration or the arguments).
"""
from __future__ import annotations

from wam_policy.task_rl.core import run as R

SCALARS = ('train_mean_reward_mm', 'mean_reward_cm', 'probe_diff_mm', 'probe_se_mm', 'probe_diff_cm', 'probe_se_cm',
           'probe_correct', 'probe_successes', 'grad_norm_mean', 'reg_mean', 'disp_current', 'disp_old',
           'skipped_groups', 'samples', 'round_seconds')


def main(run, args):
    from wam_policy.common.tracking import Tracker
    res, _ = run.profile.round_table()
    tracker = Tracker(run.root / 'results/tracking', name=args.name or run.root.name, group=run.task, job_type='task-rl',
                      config={'task': run.task, **run.config}, mode=args.wandb_mode, entity=args.wandb_entity,
                      project=args.wandb_project, step_metric='round')
    for row in res['rounds']:
        tracker.log({k: row[k] for k in SCALARS if isinstance(row.get(k), (int, float))}, step=int(row['round'][1:]))
    for name, ev in res['evals'].items():
        if ev.get('success_rate') is not None and name.endswith('-old'):
            tracker.log({'eval_success_rate': ev['success_rate']}, step=int(name.split('-')[0][1:]))
    tracker.finish(summary={'rounds': len(res['rounds']), 'stop_rules_fired': len(res['stop_rules_fired'])})
