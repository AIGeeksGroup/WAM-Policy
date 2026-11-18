"""Window reward, group statistics and the weights rho of one collection (CPU).

The reward and the group rule are the task profile's. Scale: ``--scale-mode freeze`` computes the pooled within-group
sd of this collection, sqrt(sum_g sum_i (r_gi - mean_g)^2 / sum_g (n_g - 1)), and writes it to ``scale.json`` (refuses
to overwrite; round 0); ``use`` reads scale.json; ``provisional`` computes it without writing; ``none`` gives no weights
(probe runs). rho = 0.5 + 0.5 * clip((r - group mean) / scale, -1, 1), one rho for all chunks of a candidate.
"""
from __future__ import annotations

import json

from wam_policy.task_rl.core import run as R


def main(run, args):
    profile = run.profile
    starts = R.load_json(run.path(args.starts))
    cands = profile.candidate_rewards(run.path(args.collect), starts)
    sd, dof = profile.pooled_sd(cands)
    scale_path = run.root / 'scale.json'
    key = profile.scale_key
    if args.scale_mode == 'freeze':
        if scale_path.exists():
            raise SystemExit('scale.json exists; refusing to overwrite the frozen scale')
        R.write_json(scale_path, {key: sd, 'dof': dof, 'source': str(args.collect), 'rule': 'pooled within-group sd, ddof 1'})
        scale = sd
    elif args.scale_mode == 'use':
        scale = R.load_json(scale_path)[key]
    elif args.scale_mode == 'provisional':
        scale = sd
    else:
        scale = None
    agg, per, rho = profile.score(cands, scale)
    agg['scale_mode'] = args.scale_mode
    R.write_json(run.path(args.out), {'aggregate': agg, 'groups': per, 'rho': rho, 'candidates': cands})
    print(json.dumps(agg, indent=1))
