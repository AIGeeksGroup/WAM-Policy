"""Stage 2.1 Task-Specific RL command line: ``python -m wam_policy.task_rl <command> ...``.

  prepare   unpack a task's QLM-Bench inputs into a data root; write the cases directory      (CPU)
  init      create a run directory (task, config, teacher, base model, prompts, cases, QLM)   (CPU)
  pool      teacher episodes from reset on the training scenes; starts captured               (GPU, sharded)
  select    start split (pool/selection.json, probe/starts.json) or a round's starts          (CPU)
  round     one training round (collect, score, learn, probe, eval, readback, table)          (driver)
  rounds    rounds A..B with the stop rules                                                   (driver)
  collect / score / learn / probe / eval / readback / table    the single stages a round runs
  export    a round's adapters in the release specialist format                               (CPU)
  publish   export + optional upload (--repo or WAM_UPLOAD_REPO)                              (CPU)
  track     round table rows to results/tracking (and W&B when WANDB_MODE is online/offline)  (CPU)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _gpu_args(p):
    p.add_argument('--gpus', type=int, default=1, help='shard processes, one GPU each')
    p.add_argument('--gpu-map', default=None, help='comma list: logical GPU of each shard (default 0..N-1)')
    p.add_argument('--cores-per-gpu', type=int, default=0)
    p.add_argument('--shard', type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument('--nshards', type=int, default=1, help=argparse.SUPPRESS)


def _forward(args, names):
    """CLI arguments of a shard process, rebuilt from the parsed values."""
    out = []
    for name in names:
        v = getattr(args, name.replace('-', '_'))
        if v is None or v is False:
            continue
        out += [f'--{name}'] if v is True else [f'--{name}', str(v)]
    return out


def parser():
    p = argparse.ArgumentParser(prog='python -m wam_policy.task_rl', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)

    q = sub.add_parser('prepare')
    q.add_argument('--task', required=True)
    q.add_argument('--inputs', required=True, help='<QLM-Bench>/inputs/task_rl/<task>')
    q.add_argument('--data-root', required=True)
    q.add_argument('--out', required=True, help='cases directory written for the run')
    q.add_argument('--reuse', default=None, help='hard-link (or copy) missing scene files from a tree with the same layout')
    q.add_argument('--download', action='store_true', help='download missing scene files from the dataset repository')
    q.add_argument('--repo', default=None)
    q.add_argument('--revision', default=None)
    q.add_argument('--controller', default=None,
                   help='RAMBO model_2000.pt to use instead of the one in the inputs (default: $RAMBO_CONTROLLER if set)')

    q = sub.add_parser('init')
    q.add_argument('--run', required=True)
    q.add_argument('--task', required=True)
    q.add_argument('--config', required=True, help='JSON config (configs/stage2_1_task_rl/*.json)')
    q.add_argument('--set', action='append', default=[], metavar='KEY=JSON', help='override one config field')
    q.add_argument('--teacher', required=True, help='Stage 1.1 teacher directory (wam_policy.json + transformer/)')
    q.add_argument('--base-model', required=True, help='LingBot-VA base snapshot (vae/)')
    q.add_argument('--prompt-root', required=True,
                   help='cached prompt embeddings <root>/<task>/embedding.pt (HF prompt_embeddings/), as in the paper runs')
    q.add_argument('--cases', required=True, help='cases directory written by prepare')
    q.add_argument('--qlm-python', required=True, help='interpreter of the QLM-Bench (MuJoCo, CPU) environment')
    q.add_argument('--qlm-src', required=True, help='<QLM-Bench>/src')
    q.add_argument('--gl', choices=('osmesa', 'egl'), default='osmesa', help='scene-worker renderer backend')
    q.add_argument('--osmesa-lib', default=None, help='directory prepended to the scene workers\' LD_LIBRARY_PATH')
    q.add_argument('--lp-threads', default=None, help='llvmpipe threads per scene worker (default auto)')

    q = sub.add_parser('pool'); q.add_argument('--run', required=True); _gpu_args(q)
    q.add_argument('--cases', type=int, default=None, help='first N scenes of the scene order (default config pool_cases)')
    q.add_argument('--reps', type=int, default=None, help='teacher seeds per scene (default config pool_reps)')

    q = sub.add_parser('select'); q.add_argument('--run', required=True); q.add_argument('--round', default=None)

    q = sub.add_parser('collect'); q.add_argument('--run', required=True); _gpu_args(q)
    q.add_argument('--out', required=True); q.add_argument('--starts', required=True)
    q.add_argument('--kind', choices=('train', 'probe'), required=True)
    q.add_argument('--k', type=int, default=8)
    q.add_argument('--lora', default=None, help="a round's old-lora.pt (mode old)")
    q.add_argument('--round-name', default='r00')

    q = sub.add_parser('score'); q.add_argument('--run', required=True)
    q.add_argument('--collect', required=True); q.add_argument('--starts', required=True); q.add_argument('--out', required=True)
    q.add_argument('--scale-mode', choices=('freeze', 'use', 'provisional', 'none'), required=True)

    q = sub.add_parser('learn'); q.add_argument('--run', required=True)
    q.add_argument('--round', required=True); q.add_argument('--collect', required=True); q.add_argument('--scores', required=True)
    q.add_argument('--state-in', default='none'); q.add_argument('--state-out', required=True)

    q = sub.add_parser('probe'); q.add_argument('--run', required=True)

    q = sub.add_parser('eval'); q.add_argument('--run', required=True); _gpu_args(q)
    q.add_argument('--out', required=True); q.add_argument('--scenes', default='probe'); q.add_argument('--reps', type=int, default=1)
    q.add_argument('--lora', default=None, help="old-lora.pt: mode old")
    q.add_argument('--lora-current', default=None, help="state.pt: its current adapters, run in mode old")

    q = sub.add_parser('readback'); q.add_argument('--run', required=True)
    q.add_argument('--round', required=True); q.add_argument('--prev-state', default='none'); q.add_argument('--dir', default=None)

    q = sub.add_parser('table'); q.add_argument('--run', required=True); q.add_argument('--fail-on-stop', action='store_true')

    for name in ('round', 'rounds'):
        q = sub.add_parser(name); q.add_argument('--run', required=True)
        q.add_argument('--gpus', type=int, default=1); q.add_argument('--cores-per-gpu', type=int, default=0)
        if name == 'round':
            q.add_argument('round'); q.add_argument('prev', help='previous round or none')
        else:
            q.add_argument('first', type=int); q.add_argument('last', type=int)
            q.add_argument('--deadline', type=float, default=None, help='epoch seconds; no round starts too close to it')
            q.add_argument('--round-budget', type=int, default=None)

    for name in ('export', 'publish'):
        q = sub.add_parser(name); q.add_argument('--run', required=True)
        q.add_argument('--round', required=True); q.add_argument('--output', required=True)
        if name == 'publish':
            q.add_argument('--repo', default=None); q.add_argument('--path-in-repo', default=None)

    q = sub.add_parser('track'); q.add_argument('--run', required=True); q.add_argument('--name', default=None)
    q.add_argument('--wandb-mode', default=None); q.add_argument('--wandb-entity', default=None)
    q.add_argument('--wandb-project', default=None)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    cmd = args.command
    if cmd == 'learn' and 'LOCAL_RANK' in os.environ:
        from wam_policy.task_rl.core.run import isolate_gpu
        isolate_gpu(int(os.environ['LOCAL_RANK']))          # before torch is imported: one GPU per rank
    from wam_policy.task_rl.core import run as R
    if cmd == 'prepare':
        from wam_policy.task_rl.core import prepare
        return prepare.main(args)
    if cmd == 'init':
        config = json.loads(Path(args.config).read_text())
        for kv in args.set:
            k, v = kv.split('=', 1)
            config[k] = json.loads(v)
        run = R.init_run(args.run, task=args.task, config=config, teacher=args.teacher, base_model=args.base_model,
                         prompt_root=args.prompt_root, cases=args.cases, qlm_python=args.qlm_python, qlm_src=args.qlm_src,
                         gl=args.gl, osmesa_lib=args.osmesa_lib, lp_threads=args.lp_threads)
        print('run', run.root, 'task', run.task, 'rows', run.rows, 'render_mode', run.config['render_mode'])
        return
    run = R.Run(args.run)
    if cmd in ('pool', 'collect', 'eval') and args.shard is None:
        from wam_policy.task_rl.core import parallel
        names = {'pool': ['cases', 'reps'],
                 'collect': ['out', 'starts', 'kind', 'k', 'lora', 'round-name'],
                 'eval': ['out', 'scenes', 'reps', 'lora', 'lora-current']}[cmd]
        gpu_map = [int(x) for x in args.gpu_map.split(',')] if args.gpu_map else None
        return parallel.sharded(run, cmd, args, _forward(args, names), gpus=args.gpus, gpu_map=gpu_map,
                                cores_per_gpu=args.cores_per_gpu)
    if cmd == 'pool':
        from wam_policy.task_rl.core import pool
        return pool.main(run, args)
    if cmd == 'select':
        from wam_policy.task_rl.core import pool
        return pool.select(run, args)
    if cmd == 'collect':
        from wam_policy.task_rl.core import collect
        return collect.main(run, args)
    if cmd == 'score':
        from wam_policy.task_rl.core import score
        return score.main(run, args)
    if cmd == 'learn':
        from wam_policy.task_rl.core import learn
        return learn.main(run, args)
    if cmd == 'probe':
        from wam_policy.task_rl.core import rounds
        return rounds.probe_compare(run, args)
    if cmd == 'eval':
        from wam_policy.task_rl.core import evaluate
        return evaluate.main(run, args)
    if cmd == 'readback':
        from wam_policy.task_rl.core import readback
        return readback.main(run, args)
    if cmd == 'table':
        _, code = run.profile.round_table(args.fail_on_stop)
        sys.exit(code)
    if cmd in ('round', 'rounds'):
        from wam_policy.task_rl.core.rounds import Driver
        d = Driver(run, args.gpus, args.cores_per_gpu)
        if cmd == 'round':
            return d.round(args.round, None if args.prev == 'none' else args.prev)
        return d.rounds(args.first, args.last, args.deadline, args.round_budget)
    if cmd in ('export', 'publish'):
        from wam_policy.task_rl.core import publish
        if cmd == 'export':
            return publish.export(run, args.round, args.output)
        return publish.publish(run, args.round, args.output, args.repo, args.path_in_repo)
    if cmd == 'track':
        from wam_policy.task_rl.core import track
        return track.main(run, args)


if __name__ == '__main__':
    main()
