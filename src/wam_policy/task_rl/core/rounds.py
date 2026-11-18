"""Round driver: one training round stage by stage, and several rounds in a row.

A round (``round rNN``): select its starts -> [r00: teacher full-episode eval when eval_every > 0; teacher probe
(reference, and for push_box a same-seed repeat)] -> collect K candidate windows per start with theta_old -> window
reward and rho (r00 freezes the reward scale) -> DiffusionNFT update + EMA -> probe with the new theta_old -> full-episode
eval after every ``eval_every``-th round -> probe comparison -> readback -> round table. Each stage is skipped when its
output exists, so a round resumes after a failure. GPU stages use ``gpus`` GPUs: collection, probe and eval as shards,
the update data-parallel (torchrun).

``rounds A B`` runs rounds A..B, each continuing from the previous round's state and old adapters, and stops at the first
failing stage or, after a round, when a stop rule of the round table fires (``STOPPED`` is written). With ``deadline``
(epoch seconds) a round is not started when less than ``round_budget`` seconds (default 1.3 x the longest finished
round, else 3 h) remain, so a batch job ends cleanly at a round boundary and the next job continues.
"""
from __future__ import annotations

import collections
import json
import subprocess
import sys
import time
from datetime import datetime

from wam_policy.task_rl.core import run as R


def _now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


class Driver:
    def __init__(self, run, gpus=1, cores_per_gpu=0):
        self.run, self.gpus, self.cores = run, gpus, cores_per_gpu
        (run.root / 'logs').mkdir(parents=True, exist_ok=True)

    def cli(self, *args):
        return [sys.executable, '-m', 'wam_policy.task_rl', args[0], '--run', str(self.run.root), *map(str, args[1:])]

    def gpu(self, *args):
        return self.cli(*args, '--gpus', self.gpus, '--cores-per-gpu', self.cores)

    def call(self, log, *cmds):
        with open(self.run.root / 'logs' / log, 'a') as f:
            for cmd in cmds:
                f.write(f'# {_now()} {" ".join(cmd)}\n'); f.flush()
                if subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode:
                    raise SystemExit(f'failed: {" ".join(cmd)} (see {self.run.root / "logs" / log})')

    def stage(self, rnd, name, done, *cmds):
        rd = self.run.root / 'rounds' / rnd
        if done.exists():
            print(f'[{rnd}] {name}: done', flush=True)
            return
        t0 = time.time()
        print(f'[{rnd}] {name}: start {_now()}', flush=True)
        self.call(f'{rnd}-{name}.log', *cmds)
        sec = int(time.time() - t0)
        R.append_jsonl(rd / 'stage-times.jsonl', {'round': rnd, 'stage': name, 'seconds': sec, 'gpus': self.gpus, 'end': _now()})
        print(f'[{rnd}] {name}: {sec} s', flush=True)

    def round(self, rnd, prev):
        run, O, cfg = self.run, self.run.root, self.run.config
        rd = O / 'rounds' / rnd
        rd.mkdir(parents=True, exist_ok=True)
        if not (rd / 'starts.json').exists():
            self.call(f'{rnd}-select.log', self.cli('select', '--round', rnd))
        lora = [] if prev is None else ['--lora', f'rounds/{prev}/old-lora.pt']
        state = 'none' if prev is None else f'rounds/{prev}/state.pt'
        scale = 'freeze' if rnd == 'r00' else 'use'
        pk, ev, es, er = cfg['probe_k'], int(cfg['eval_every']), cfg['eval_scenes'], cfg['eval_reps']
        if ev > 0 and rnd == 'r00':
            self.stage(rnd, 'eval-teacher', O / 'eval/teacher/summary.json',
                       self.gpu('eval', '--out', 'eval/teacher', '--scenes', es, '--reps', er))
        if rnd == 'r00':
            for tag in run.profile.probe_reference_runs:
                d = f'probe/r00-{tag}'
                self.stage(rnd, f'probe-{tag}', O / d / 'scores.json',
                           self.gpu('collect', '--out', d, '--starts', 'probe/starts.json', '--kind', 'probe', '--k', pk),
                           self.cli('score', '--collect', d, '--starts', 'probe/starts.json', '--out', f'{d}/scores.json',
                                    '--scale-mode', 'none'))
        self.stage(rnd, 'collect', rd / 'collect/meta.json',
                   self.gpu('collect', '--out', f'rounds/{rnd}/collect', '--starts', f'rounds/{rnd}/starts.json',
                            '--kind', 'train', '--k', cfg['candidates'], '--round-name', rnd, *lora))
        self.stage(rnd, 'score', rd / 'scores.json',
                   self.cli('score', '--collect', f'rounds/{rnd}/collect', '--starts', f'rounds/{rnd}/starts.json',
                            '--out', f'rounds/{rnd}/scores.json', '--scale-mode', scale))
        learn = self.cli('learn', '--round', rnd, '--collect', f'rounds/{rnd}/collect', '--scores', f'rounds/{rnd}/scores.json',
                         '--state-in', state, '--state-out', f'rounds/{rnd}/state.pt')
        if self.gpus > 1:
            learn = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(self.gpus),
                     '-m', 'wam_policy.task_rl', *learn[3:]]
        self.stage(rnd, 'learn', rd / 'state.pt', learn)
        self.stage(rnd, 'probe', rd / 'probe/scores.json',
                   self.gpu('collect', '--out', f'rounds/{rnd}/probe', '--starts', 'probe/starts.json', '--kind', 'probe',
                            '--k', pk, '--lora', f'rounds/{rnd}/old-lora.pt'),
                   self.cli('score', '--collect', f'rounds/{rnd}/probe', '--starts', 'probe/starts.json',
                            '--out', f'rounds/{rnd}/probe/scores.json', '--scale-mode', 'none'))
        n = int(rnd[1:])
        if ev > 0 and (n + 1) % ev == 0:
            self.stage(rnd, 'eval', O / f'eval/{rnd}-old/summary.json',
                       self.gpu('eval', '--out', f'eval/{rnd}-old', '--scenes', es, '--reps', er,
                                '--lora', f'rounds/{rnd}/old-lora.pt'))
        self.call('probe-compare.log', self.cli('probe'))
        self.stage(rnd, 'readback', rd / 'readback.json',
                   self.cli('readback', '--round', rnd, '--prev-state', 'none' if prev is None else prev))
        subprocess.run(self.cli('table'), stdout=open(O / 'logs/round-table.log', 'w'), stderr=subprocess.STDOUT)
        print(f'[{rnd}] round complete {_now()}', flush=True)

    def rounds(self, first, last, deadline=None, round_budget=None):
        O = self.run.root
        if (O / 'STOPPED').exists():
            print('STOPPED exists (a stop rule fired earlier); remove it to continue')
            return
        for i in range(first, last + 1):
            rnd, prev = f'r{i:02d}', (None if i == 0 else f'r{i - 1:02d}')
            if prev and not (O / 'rounds' / prev / 'old-lora.pt').exists():
                raise SystemExit(f'missing rounds/{prev}/old-lora.pt')
            if (O / 'rounds' / rnd / 'readback.json').exists():
                print(f'[{rnd}] already complete')
                continue
            if deadline:
                t = collections.Counter()
                for r in R.jsonl_many(O.glob('rounds/r*/stage-times.jsonl')):
                    t[r['round']] += r['seconds']
                longest = max(t.values()) if t else 0
                budget = round_budget or (longest * 13 // 10 if longest > 0 else 10800)
                left = int(deadline - time.time())
                if left < budget:
                    print(f'[{rnd}] not started: {left}s left < budget {budget}s')
                    return
            self.round(rnd, prev)
            with open(O / 'logs/round-table.log', 'w') as f:
                code = subprocess.run(self.cli('table', '--fail-on-stop'), stdout=f, stderr=subprocess.STDOUT).returncode
            if code:
                print(f'[{rnd}] stop rule fired (see {O / "results/round-table.md"})')
                (O / 'STOPPED').touch()
                return


def probe_compare(run, args):
    """Every probe run of the run against the round-0 teacher probe (results/probe-summary.json)."""
    O, profile = run.root, run.profile
    ref = R.load_json(O / 'probe/r00-pre-a/scores.json')['candidates']
    res = {'ref': 'probe/r00-pre-a/scores.json', 'runs': {}}
    if (O / 'probe/r00-pre-b/scores.json').exists():
        res['repeat_noise'] = profile.probe_compare(ref, R.load_json(O / 'probe/r00-pre-b/scores.json')['candidates'])
    for p in sorted(O.glob('rounds/r*/probe/scores.json')):
        res['runs'][str(p.relative_to(O))] = profile.probe_compare(ref, R.load_json(p)['candidates'])
    R.write_json(O / 'results/probe-summary.json', res)
    print(json.dumps(res, indent=1))
