"""One round of the window-level DiffusionNFT update, on one GPU or data-parallel on several (torchrun).

One frozen teacher backbone (bf16, flex attention, activation checkpointing) with the switchable LoRA: current trains,
old (EMA) and off (= the teacher, the reference) are evaluated without grad on the same weights. The round's training
chunks (``objective.round_index``) are shuffled once with the round seed and split into ``steps_per_round``
mini-batches (order[i::S]); each mini-batch is one AdamW step on sum(per-sample loss) / batch size with grad clipping.
After the last step: old <- eta * old + (1 - eta) * current. The per-modality gradient norms are probed on the first
``probe_samples`` samples of the round at the round's starting weights.

Data parallel (WORLD_SIZE = W > 1, ``python -m torch.distributed.run --nproc_per_node W -m wam_policy.task_rl learn``):
every rank holds the same weights and optimizer state; in each mini-batch rank r takes groups[r::W] (groups of
``learn_batch`` samples of equal history length; one sample each by default), accumulates loss_i / len(batch) over its
share, the LoRA gradients are summed over ranks (one flat all_reduce), and every rank applies the identical clip + AdamW
step. Sample order, renoising seeds (per sample id), batches and rho do not depend on W. Each rank sees exactly one GPU
(set before torch is imported). Rank 0 gathers the per-sample records and writes every output.

Outputs: ``state.pt`` (current + old adapters, optimizer, scheduler, step count, config), ``old-lora.pt`` (the old
adapters = the sampling policy of the next round and the round's specialist), ``learn-metrics.json``,
``learn-samples.jsonl.gz``.
"""
from __future__ import annotations

import gzip
import json
import os
import random
import time
from datetime import timedelta


def build(run, state_in, device):
    import torch
    from wam_policy.models.checkpoint_io import load_model
    from wam_policy.task_rl import adapters as A
    from wam_policy.task_rl.core import run as R
    from wan_va.distributed.fsdp import apply_ac
    cfg = run.config
    model = load_model(run.teacher / 'transformer', dtype=torch.bfloat16, device=device, attn_mode='flex')
    A.inject(model, rank=cfg['lora_rank'], alpha=cfg['lora_alpha'])
    apply_ac(model)
    params = A.trainable(model)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=cfg['learning_rate'], betas=(.9, .95), eps=1e-8, weight_decay=0., fused=False, foreach=False)
    warm = cfg['warmup_steps']
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min((step + 1) / warm, 1.))
    steps = 0
    if state_in is not None:
        s = torch.load(state_in, map_location='cpu', weights_only=False)
        if R.algorithm_fields(s.get('config')) != R.algorithm_fields(cfg):
            raise SystemExit(f'{state_in} was trained with another configuration; a run resumes only with its own')
        A.load_lora_state(model, s['lora'])
        opt.load_state_dict(s['optimizer']); sched.load_state_dict(s['scheduler']); steps = s['steps']
    return model, params, opt, sched, steps


class Dist:
    def __init__(self):
        import torch.distributed as dist
        self.world = int(os.environ.get('WORLD_SIZE', '1'))
        self.rank = int(os.environ.get('RANK', '0'))
        self.dist = dist if self.world > 1 else None
        if self.dist:
            dist.init_process_group(os.environ.get('TASK_RL_DIST_BACKEND', 'nccl'),     # nccl = RCCL on ROCm
                                    timeout=timedelta(seconds=int(os.environ.get('TASK_RL_DIST_TIMEOUT', '7200'))))

    def sum_(self, t):
        if self.dist:
            self.dist.all_reduce(t, op=self.dist.ReduceOp.SUM)
        return t

    def gather(self, obj):
        if not self.dist:
            return [obj]
        out = [None] * self.world
        self.dist.all_gather_object(out, obj)
        return out

    def close(self):
        if self.dist:
            self.dist.barrier(); self.dist.destroy_process_group()


def reduce_grads(dd, params):
    import torch
    if dd.world == 1:
        return
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])
    dd.sum_(flat)
    o = 0
    for p in params:
        n = p.numel()
        g = flat[o:o + n].view_as(p)
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)
        o += n


def checksum(params):
    import torch
    with torch.no_grad():
        return float(sum(p.double().sum() + p.double().abs().sum() * 1e-3 for p in params))


def main(run, args):
    import torch
    from wam_policy.task_rl import adapters as A
    from wam_policy.task_rl.core import objective as N
    from wam_policy.task_rl.core import run as R
    cfg = run.config
    out = run.path(args.state_out)
    dd = Dist()
    device = torch.device('cuda:0')
    t0 = time.monotonic()
    model, params, opt, sched, steps0 = build(run, None if args.state_in == 'none' else run.path(args.state_in), device)
    load_s = time.monotonic() - t0
    model._old_is_zero = all(not m.old_B.any() for m in A.lora_modules(model))
    scores = R.load_json(run.path(args.scores))
    index = N.round_index(scores)
    store = N.SampleStore(run.path(args.collect))
    rng = random.Random(f"{cfg['seed']}:{args.round}:order")
    order = list(range(len(index))); rng.shuffle(order)
    S = cfg['steps_per_round']
    batches = [order[i::S] for i in range(S)] if len(order) >= S else [[i] for i in order]
    if dd.rank == 0:
        print(f'{args.round}: {len(index)} chunk samples, {len(batches)} steps, batch {len(batches[0]) if batches else 0}, '
              f'world {dd.world}, load {load_s:.0f} s', flush=True)
    seed = lambda sid: R.seed_of(cfg['seed'], args.round, sid, 'nft')
    disp0 = {'current': A.displacement(model, 'current'), 'old': A.displacement(model, 'old')}
    probe = []
    tp = time.monotonic()
    for n, i in enumerate(order[:cfg['probe_samples']]):
        if n % dd.world != dd.rank:
            continue
        e = index[i]
        g, rec = N.term_grad_norms(model, store.get(e), e['rho'], seed(e['sid']), cfg, device)
        probe.append({'n': n, 'id': e['sid'], 'rho': e['rho'], **{f'grad_{k}': v for k, v in g.items()},
                      **{k: rec[k] for k in ('pv', 'pa', 'regv', 'rega')}})
    for q in params:
        q.grad = None
    probe = sorted([x for part in dd.gather(probe) for x in part], key=lambda x: x['n'])
    probe_s = time.monotonic() - tp
    torch.cuda.reset_peak_memory_stats()
    log, records = [], []
    for b, batch in enumerate(batches):
        ts = time.monotonic()
        opt.zero_grad(set_to_none=True)
        acc = torch.zeros(4, dtype=torch.float64, device=device)   # pv, pa, reg, count
        tc = time.monotonic()
        for g in N.groups_of(batch, index, store, run.learn_batch)[dd.rank::dd.world]:
            es = [index[i] for i in g]
            loss, recs = N.terms_group(model, [store.get(e) for e in es], [e['rho'] for e in es],
                                       [seed(e['sid']) for e in es], cfg, device)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite NFT loss ' + es[0]['sid'])
            (loss / len(batch)).backward()
            for e, rec in zip(es, recs):
                rec.update(id=e['sid'], rho=e['rho'], step=steps0 + b, rank=dd.rank, group=len(es))
                records.append(rec)
                acc += torch.tensor([rec['pv'], rec['pa'], rec['regv'] + rec['rega'], 1.], dtype=torch.float64, device=device)
        compute_s = time.monotonic() - tc
        reduce_grads(dd, params)
        dd.sum_(acc)
        norm = torch.nn.utils.clip_grad_norm_(params, cfg['grad_clip'])
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient norm')
        lr = opt.param_groups[0]['lr']
        opt.step(); sched.step()
        torch.cuda.synchronize()
        cnt = float(acc[3])
        row = {'step': steps0 + b, 'batch': len(batch), 'lr': lr, 'grad_norm': float(norm), 'seconds': time.monotonic() - ts,
               'compute_seconds_rank0': compute_s, 'pv': float(acc[0]) / cnt, 'pa': float(acc[1]) / cnt, 'reg': float(acc[2]) / cnt}
        if dd.world > 1 and (b % 8 == 7 or b == len(batches) - 1):
            cs = dd.gather(checksum(params))
            row['rank_checksum_spread'] = max(cs) - min(cs)
            if max(cs) != min(cs):
                raise RuntimeError(f'ranks diverged at step {b}: {cs}')
        log.append(row)
        if dd.rank == 0:
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    peak = dd.gather(R.peak_gib())
    records = sorted([r for part in dd.gather(records) for r in part], key=lambda r: (r['step'], r['id']))
    steps = steps0 + len(log)
    disp_cur_pre_ema = A.displacement(model, 'current')
    A.ema_old(model, cfg['ema'])
    final_cs = dd.gather(checksum(params))
    if max(final_cs) != min(final_cs):
        raise RuntimeError(f'ranks diverged at the end: {final_cs}')
    if dd.rank == 0:
        nrec = max(1, len(records))
        upd = sum(l['seconds'] for l in log)
        metrics = {'round': args.round, 'learn_batch': run.learn_batch, 'world_size': dd.world, 'samples': len(index),
                   'steps_this_round': len(log), 'optimizer_steps_total': steps,
                   'batch_size': len(batches[0]) if batches else 0, 'load_seconds': load_s, 'probe_seconds': probe_s,
                   'update_seconds': upd, 'seconds_per_sample': upd / nrec, 'gpu_seconds_per_sample': upd * dd.world / nrec,
                   'gpu_peak_update': peak[0], 'gpu_peak_update_per_rank': peak, 'grad_probe': probe,
                   'grad_probe_mean': {k: sum(x[f'grad_{k}'] for x in probe) / max(1, len(probe)) for k in ('video', 'action', 'reference')},
                   'displacement_before': disp0, 'displacement_current': disp_cur_pre_ema,
                   'displacement_old_after_ema': A.displacement(model, 'old'),
                   'displacement_current_minus_old_after_ema': A.displacement(model, 'current', minus='old'),
                   'final_rank_checksums': final_cs, 'steps': log, 'lr_next': opt.param_groups[0]['lr'], 'ema': cfg['ema'],
                   'config': cfg}
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'lora': A.lora_state(model, which=('old',))}, out.with_name('old-lora.pt'))
        R.write_json(out.with_name('learn-metrics.json'), metrics)
        with gzip.open(out.with_name('learn-samples.jsonl.gz'), 'wt') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')
        # state.pt last: the round driver treats it as the learn stage's completion marker
        torch.save({'lora': A.lora_state(model), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
                    'steps': steps, 'round': args.round, 'config': cfg, 'world_size': dd.world}, out)
        print('done', json.dumps({k: metrics[k] for k in ('samples', 'world_size', 'steps_this_round', 'update_seconds',
                                                           'seconds_per_sample', 'gpu_peak_update', 'grad_probe_mean',
                                                           'displacement_current', 'displacement_old_after_ema')}), flush=True)
    dd.close()
