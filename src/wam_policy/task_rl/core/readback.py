"""Read every record of one round back from disk (CPU).

Hard checks (a failure stops the round): every file of the round is present (candidate records, start prefixes,
scores, ``state.pt``, ``old-lora.pt``), the stored tensors have the expected shapes (chunk length 16, 9D commands,
48-channel two-view latents, one privileged row per executed command) and are finite, and every training sample
rebuilds with consistent shapes. Only then ``readback.json`` is written (it marks the round complete); otherwise
``readback-failed.json`` is written and the stage exits non-zero, so a rerun repeats the readback.

Diagnostics (recorded in ``readback.json``, never a stop condition): committed actions vs. the normalizer-encoded
executed commands, requested commands vs. the decoded clean actions, zero force channels, the reward recomputed from
the stored rows vs. scores.json, the training-sample count, the round-0 EMA identity.
"""
from __future__ import annotations

import json

import numpy as np

from wam_policy.task_rl.core import run as R


def _finite(*tensors):
    import torch
    return all(bool(torch.isfinite(t.float()).all()) for t in tensors)


def main(run, args):
    import torch
    from wam_policy.common.action_space import ActionNormalizer
    from wam_policy.models import metadata
    from wam_policy.task_rl import adapters as A
    from wam_policy.task_rl.core import objective as N
    profile = run.profile
    nm = metadata.read(run.teacher, 'teacher')['action_normalizer']
    norm = ActionNormalizer(offset=tuple(nm['offset']), scale=tuple(nm['scale']))
    d = run.path(args.dir) if args.dir else run.root / 'rounds' / args.round
    col = d / 'collect'
    missing = [str(p.relative_to(run.root)) for p in (d / 'starts.json', d / 'scores.json', col / 'results.jsonl',
                                                      col / 'text.pt', d / 'state.pt', d / 'old-lora.pt') if not p.exists()]
    hard, diag, n, nchunks, st_cache = [], [], 0, 0, {}
    rows = R.jsonl(col / 'results.jsonl')
    for row in rows:
        cp = col / 'cands' / f"{row['candidate']}.pt"
        sp = col / 'starts' / f"{row['start']}.pt"
        if not cp.exists() or not sp.exists():
            missing += [str(p.relative_to(run.root)) for p in (cp, sp) if not p.exists()]
            continue
        rec = torch.load(cp, weights_only=False, map_location='cpu')
        n += 1
        if rec['start'] not in st_cache:
            st_cache[rec['start']] = torch.load(sp, weights_only=True)
        st = st_cache[rec['start']]
        k = rec['video'].shape[0]; nchunks += k
        g = rec['new_actions'].shape[2]
        checks = {
            'shapes': (tuple(rec['video'].shape) == (k, 48, 1, 24, 20) and tuple(rec['action'].shape) == (k, 9, 1, 16, 1)
                       and tuple(rec['new_latents'].shape) == (1, 48, g, 24, 20)
                       and tuple(rec['new_actions'].shape) == (1, 9, g, 16, 1)
                       and tuple(rec['requested'].shape) == (k, 16, 9) and tuple(rec['executed'].shape) == (g, 16, 9)
                       and st['latents'].shape[2] == st['actions'].shape[2]),
            'rows': (rec['rows'].shape[0] == 1 + 16 * k) if rec['terminal'] is None else (1 + 16 * (k - 1) < rec['rows'].shape[0] <= 1 + 16 * k),
            'finite': _finite(rec['video'], rec['action'], rec['new_latents'], rec['new_actions'], st['latents'], st['actions'])
                      and bool(np.isfinite(rec['rows']).all()),
        }
        if not all(checks.values()):
            hard.append({'candidate': row['candidate'], **checks})
            continue
        info = {
            'dtype_bf16': rec['video'].dtype == torch.bfloat16 and rec['action'].dtype == torch.bfloat16 and rec['new_latents'].dtype == torch.bfloat16,
            'prefix_len': [int(x) for x in rec['prefix_len']] == [st['latents'].shape[2] + j for j in range(k)],
            'start_actions_group0_zero': not st['actions'][:, :, 0].float().abs().sum().item(),
            'committed_force_zero': not rec['new_actions'][:, 6:].float().abs().sum().item(),
            'executed_committed_only': g == k - (rec['terminal'] is not None),
        }
        if g:
            enc = torch.from_numpy(np.stack([norm.encode(e) for e in rec['executed'][:g]]))
            stored = rec['new_actions'][0, :, :, :, 0].permute(1, 2, 0).float()      # groups,16,9
            info['committed_equals_encoded_executed'] = bool(torch.equal(stored, enc.to(torch.bfloat16).float()))
        dec = np.stack([norm.decode(rec['action'][j, :, 0, :, 0].float().numpy().T) for j in range(k)])
        dec[:, :, 6:] = 0
        info['requested_equals_decoded_action'] = bool(np.allclose(dec, rec['requested'], atol=0, rtol=0))
        if not all(info.values()):
            diag.append({'candidate': row['candidate'], **info})
    res = {'round': args.round, 'candidates': n, 'chunks': nchunks, 'missing_files': missing, 'failed_candidates': hard,
           'diagnostics': {'candidates_with_differences': diag}}
    shape_ok = True
    if not missing and not hard:
        starts, scores = R.load_json(d / 'starts.json'), R.load_json(d / 'scores.json')
        store, index = N.SampleStore(col), N.round_index(scores)
        for e in index:
            s = store.get(e)
            shape_ok &= (s.prefix.latents.shape[2] == s.prefix.actions.shape[2] and tuple(s.video.shape) == (1, 48, 1, 24, 20)
                         and tuple(s.action.shape) == (1, 9, 1, 16, 1))
        cands = profile.candidate_rewards(col, starts)
        res['diagnostics'].update(
            reward_recompute_max_abs_diff=max((abs(cands[c]['r'] - scores['candidates'][c]['r']) for c in cands), default=0.),
            training_samples=len(index),
            expected_training_samples=sum(profile.train_chunks(int(scores['candidates'][c]['chunks'])) for c in scores['rho']))
        s = torch.load(d / 'state.pt', weights_only=False, map_location='cpu')
        state = {'has_lora': any(k.endswith('lora_A') for k in s['lora']), 'has_old': any(k.endswith('old_A') for k in s['lora']),
                 'finite': all(bool(torch.isfinite(v).all()) for v in s['lora'].values()), 'steps': s['steps']}
        shape_ok &= state['has_lora'] and state['has_old'] and state['finite']
        if args.prev_state == 'none':
            eta, errs = s['config']['ema'], []
            for i, A0 in A.initial_A(s['lora']).items():
                errs.append(float((s['lora'][f'{i}.old_A'] - (eta * A0 + (1 - eta) * s['lora'][f'{i}.lora_A'])).abs().max()))
                errs.append(float((s['lora'][f'{i}.old_B'] - (1 - eta) * s['lora'][f'{i}.lora_B']).abs().max()))
            res['diagnostics']['ema_max_abs_err'] = max(errs)
        res['state'] = state
    res['training_sample_shapes_ok'] = bool(shape_ok)
    res['ok'] = bool(not missing and not hard and shape_ok)
    (d / 'readback-failed.json').unlink(missing_ok=True)
    R.write_json(d / ('readback.json' if res['ok'] else 'readback-failed.json'), res)
    print(json.dumps({k: v for k, v in res.items() if k != 'failed_candidates'}, indent=1, default=str))
    if not res['ok']:
        raise SystemExit('readback failed: ' + str(d / 'readback-failed.json'))
