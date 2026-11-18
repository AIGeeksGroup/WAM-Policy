"""DiffusionNFT objective of the window-level update (paper eq. rl_loss).

Per stored chunk (every training chunk of a candidate, the candidate's rho):
    implicit positive / negative velocity branches v_old -/+ beta (v_cur - v_old), each an adaptive-weighted (detached
    per-sample mean |x0_hat - x0|) squared error of the clean endpoint, mixed by rho, divided by beta; plus the
    reference regulariser ||v_cur - v_ref||^2, v_ref = the frozen teacher (adapters off). Video and action each reduce
    to a per-element mean, then
        L = policy_video + policy_action + w_ref * (reg_video + reg_action)        (video : action = 1 : 1)
    the action force channels 6-8 are masked out of the policy term, the regulariser keeps all 9.
Forward: the whole stored prefix (clean history, the sampled clean video as the current video condition, the renoised
current chunk) through the flex-masked training forward of the backbone. The three roles are the one backbone with the
LoRA switch: current (grad), old and off (= reference) without grad.

Renoising: sigma ~ U(0, 1) and Gaussian noise, one per modality per sample from the sample's own generator; the noisy
tensor is rounded to bf16 before both the model and the loss, so the loss sees exactly the model's input.

Several samples per forward (``learn_batch`` = b > 1): samples of equal history length are stacked on the batch
dimension (``velocity_rows``); renoising stays per sample, rho and sigma per sample, and the group's loss is the sum of
the per-sample losses, so the update equals the sample-by-sample one up to rounding.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

from wam_policy.task_rl import adapters as A

ACTION = (1, 9, 1, 16, 1)


# ------------------------------------------------------------------ the loss
def masked_mean(value, mask):
    mask = torch.broadcast_to(mask.to(device=value.device, dtype=torch.bool), value.shape)
    dims = tuple(range(1, value.ndim)); count = mask.sum(dims)
    if torch.any(count == 0):
        raise ValueError('Empty valid sample')
    return torch.where(mask, value, torch.zeros_like(value)).sum(dims) / count


def nft_loss(clean, noisy, sigma, current, old, reference, rho, mask, *, beta=.1, epsilon=1e-5, reference_mask=None):
    """Per-sample (policy, regulariser) of the noise-minus-clean velocity parameterisation."""
    if not 0 < beta or not 0 < epsilon:
        raise ValueError('Positive NFT coefficients required')
    for value in (clean, noisy, current, old, reference, rho, sigma):
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite NFT input')
    if torch.any((rho < 0) | (rho > 1)) or torch.any((sigma < 0) | (sigma > 1)):
        raise ValueError('Rho/sigma out of range')
    if not (clean.shape == noisy.shape == current.shape == old.shape == reference.shape):
        raise ValueError('NFT tensor shape mismatch')
    if rho.numel() != clean.shape[0] or sigma.numel() != clean.shape[0]:
        raise ValueError('Exactly one rho and sigma per sample required')
    clean = clean.detach().float(); noisy = noisy.detach().float(); old = old.detach().float()
    scale = sigma.detach().float().reshape(-1, *([1] * (clean.ndim - 1)))
    estimates = [noisy - scale * ((1 - beta) * old + beta * current.float()),
                 noisy - scale * ((1 + beta) * old - beta * current.float())]
    losses = []
    for estimate in estimates:
        denominator = masked_mean((estimate - clean).abs(), mask).detach().clamp_min(epsilon)
        losses.append(masked_mean((estimate - clean).square(), mask) / denominator)
    policy = (rho.detach().reshape(-1) * losses[0] + (1 - rho.detach().reshape(-1)) * losses[1]) / beta
    regularizer = masked_mean((current.float() - reference.detach().float()).square(),
                              mask if reference_mask is None else reference_mask)
    return policy, regularizer


def renoise(clean, generator):
    x = clean.detach().float()
    sigma = torch.rand((x.shape[0],), generator=generator, device=x.device)
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    s = sigma.reshape(-1, *([1] * (x.ndim - 1)))
    return (1 - s) * x + s * noise, sigma


# ------------------------------------------------------------------ training forward
def build_inputs(sample, noisy_video, noisy_action, sigma_video, sigma_action, device):
    from wan_va.utils import get_mesh_id
    h = sample.prefix
    # Clean history is entirely actual. Current clean video is the saved sampling condition.
    z = torch.cat((h.latents, sample.video_condition), 2).to(device)
    a = torch.cat((h.actions, torch.zeros(ACTION)), 2).to(device)
    zv = torch.cat((h.latents.to(device), noisy_video), 2)
    av = torch.cat((h.actions.to(device), noisy_action), 2)
    count = z.shape[2]
    tv = torch.zeros((1, count), device=device); ta = torch.zeros_like(tv)
    tv[:, -1] = sigma_video * 1000.; ta[:, -1] = sigma_action * 1000.
    grid_v = get_mesh_id(count, 12, 10, 0, 1, 0, action=False).to(device)[None]
    grid_a = get_mesh_id(count, 16, 1, 1, 1, 0, action=True).to(device)[None]
    text = h.text.to(device)
    return {'latent_dict': dict(noisy_latents=zv, latent=z, timesteps=tv, cond_timesteps=torch.zeros_like(tv), grid_id=grid_v,
                                text_emb=text),
            'action_dict': dict(noisy_latents=av, latent=a, timesteps=ta, cond_timesteps=torch.zeros_like(ta), grid_id=grid_a,
                                text_emb=text),
            'chunk_size': 1, 'window_size': 8}


def velocity_pair(model, sample, noisy_video, noisy_action, sigma_video, sigma_action):
    from wan_va.utils import data_seq_to_patch
    device = noisy_video.device
    inputs = build_inputs(sample, noisy_video, noisy_action, sigma_video, sigma_action, device)
    video, action = model(inputs, train_mode=True)
    count = sample.prefix.latents.shape[2] + 1
    video = data_seq_to_patch((1, 2, 2), video, count, 24, 20, batch_size=1)[:, :, -1:]
    action = action.transpose(1, 2).reshape(1, 9, count, 16, 1)[:, :, -1:]
    return video, action


def velocity_rows(model, samples, zv, av, sv, sa):
    """``build_inputs`` + ``velocity_pair`` for b samples of equal prefix length, one row each."""
    from wan_va.utils import data_seq_to_patch, get_mesh_id
    device, b = zv.device, len(samples)
    hl = torch.cat([s.prefix.latents for s in samples]).to(device)
    ha = torch.cat([s.prefix.actions for s in samples]).to(device)
    z = torch.cat((hl, torch.cat([s.video_condition for s in samples]).to(device)), 2)
    a = torch.cat((ha, torch.zeros((b, 9, 1, 16, 1), device=device)), 2)
    zf, af = torch.cat((hl, zv), 2), torch.cat((ha, av), 2)
    count = z.shape[2]
    tv = torch.zeros((b, count), device=device); ta = torch.zeros_like(tv)
    tv[:, -1] = sv * 1000.; ta[:, -1] = sa * 1000.
    grid_v = get_mesh_id(count, 12, 10, 0, 1, 0, action=False).to(device)[None].repeat(b, 1, 1)
    grid_a = get_mesh_id(count, 16, 1, 1, 1, 0, action=True).to(device)[None].repeat(b, 1, 1)
    text = torch.cat([s.prefix.text for s in samples]).to(device)
    inputs = {'latent_dict': dict(noisy_latents=zf, latent=z, timesteps=tv, cond_timesteps=torch.zeros_like(tv), grid_id=grid_v, text_emb=text),
              'action_dict': dict(noisy_latents=af, latent=a, timesteps=ta, cond_timesteps=torch.zeros_like(ta), grid_id=grid_a, text_emb=text),
              'chunk_size': 1, 'window_size': 8}
    video, action = model(inputs, train_mode=True)
    video = data_seq_to_patch((1, 2, 2), video, count, 24, 20, batch_size=b)[:, :, -1:]
    action = action.transpose(1, 2).reshape(b, 9, count, 16, 1)[:, :, -1:]
    return video, action


# ------------------------------------------------------------------ round samples
def round_index(scores):
    """Every training chunk of every non-skipped candidate, in order (scores['rho'] order, chunk ascending), without
    loading tensors: dicts (cid, j, rho, sid). A candidate trains its first ``train_chunks`` chunks (all by default)."""
    out = []
    for cid, rho in scores['rho'].items():
        c = scores['candidates'][cid]
        for j in range(int(c.get('train_chunks', c['chunks']))):
            out.append(dict(cid=cid, j=j, rho=float(rho), sid=f'{cid}-j{j}'))
    return out


class SampleStore:
    """Builds one training sample on demand, caching the per-start prefix and the per-candidate record."""

    def __init__(self, collect_dir):
        self.dir = Path(collect_dir)
        self.text = torch.load(self.dir / 'text.pt', weights_only=True)
        self.starts, self.cands = {}, {}

    def record(self, cid):
        if cid not in self.cands:
            self.cands[cid] = torch.load(self.dir / 'cands' / f'{cid}.pt', weights_only=False, map_location='cpu')
        return self.cands[cid]

    def _start(self, sid):
        if sid not in self.starts:
            self.starts[sid] = torch.load(self.dir / 'starts' / f'{sid}.pt', weights_only=True, map_location='cpu')
        return self.starts[sid]

    def count(self, e):
        """Groups in the sample's training sequence (start prefix + its own committed chunks + the current one)."""
        return self._start(e['cid'].rsplit('-c', 1)[0])['latents'].shape[2] + e['j'] + 1

    def get(self, e):
        rec = self.record(e['cid'])
        st = self._start(rec['start'])
        n0, j = st['latents'].shape[2], e['j']
        if rec['start_groups'] != n0:
            raise ValueError('Candidate recorded from another start prefix')
        if int(rec['prefix_len'][j]) != n0 + j:
            raise ValueError('Prefix length mismatch')
        lat = torch.cat((st['latents'], rec['new_latents'][:, :, :j]), 2)
        act = torch.cat((st['actions'], rec['new_actions'][:, :, :j]), 2)
        return SimpleNamespace(prefix=SimpleNamespace(latents=lat, actions=act, text=self.text),
                               video=rec['video'][j:j + 1], action=rec['action'][j:j + 1],
                               video_condition=rec['video'][j:j + 1], id=e['sid'])


# ------------------------------------------------------------------ per-sample terms
def amp(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


def renoised(sample, seed, device):
    g = torch.Generator(device=device).manual_seed(int(seed))
    z = sample.video.to(device).float(); a = sample.action.to(device).float()
    zv, sv = renoise(z, g); av, sa = renoise(a, g)
    return z, a, zv.to(torch.bfloat16).float(), sv, av.to(torch.bfloat16).float(), sa


def roles(model, sample, zv, av, sv, sa):
    """Old and reference (off) velocities, no grad."""
    device = zv.device
    with torch.no_grad(), amp(device):
        A.set_mode(model, 'old'); ov, oa = velocity_pair(model, sample, zv, av, sv, sa)
        if getattr(model, '_old_is_zero', False):
            rv, ra = ov, oa          # old B = 0 everywhere: old == off bit for bit, skip the forward
        else:
            A.set_mode(model, 'off'); rv, ra = velocity_pair(model, sample, zv, av, sv, sa)
    A.set_mode(model, 'current')
    return ov, oa, rv, ra


def _masks(z, a):
    vm = torch.ones_like(z, dtype=torch.bool); am = torch.ones_like(a, dtype=torch.bool); am[:, 6:] = False
    return vm, am


def terms(model, sample, rho, seed, cfg, device):
    """(policy_video, policy_action, regulariser) scalars with graph on current, plus a float record."""
    z, a, zv, sv, av, sa = renoised(sample, seed, device)
    ov, oa, rv, ra = roles(model, sample, zv, av, sv, sa)
    A.set_mode(model, 'current')
    with amp(device):
        cv, ca = velocity_pair(model, sample, zv, av, sv, sa)
    vm, am = _masks(z, a)
    r = torch.tensor([rho], device=device)
    pv, regv = nft_loss(z, zv, sv, cv, ov, rv, r, vm, beta=cfg['beta'], epsilon=cfg['nft_epsilon'])
    pa, rega = nft_loss(a, av, sa, ca, oa, ra, r, am, beta=cfg['beta'], epsilon=cfg['nft_epsilon'],
                        reference_mask=torch.ones_like(am))
    rec = {'pv': float(pv.detach()), 'pa': float(pa.detach()), 'regv': float(regv.detach()), 'rega': float(rega.detach()),
           'sv': float(sv), 'sa': float(sa),
           'cur_old_v': float((cv.detach().float() - ov.float()).square().mean()),
           'cur_old_a': float((ca.detach().float() - oa.float())[:, :6].square().mean())}
    return pv.sum(), pa.sum(), (regv + rega).sum(), rec


def groups_of(batch, index, store, b):
    """The mini-batch's samples in groups of at most b with equal history length: a function of the mini-batch and b
    only (never of the rank count). b = 1: one sample per group, in the mini-batch's order."""
    if b <= 1:
        return [[i] for i in batch]
    out, opened = [], {}
    for i in batch:
        c = store.count(index[i])
        g = opened.setdefault(c, [])
        g.append(i)
        if len(g) == b:
            out.append(opened.pop(c))
    return out + list(opened.values())


def terms_group(model, samples, rhos, seeds, cfg, device):
    """Sum over the group of (policy_video + policy_action + w_ref * regulariser), with graph on current, and one float
    record per sample. One sample: ``terms``."""
    if len(samples) == 1:
        pv, pa, reg, rec = terms(model, samples[0], rhos[0], seeds[0], cfg, device)
        return pv + pa + cfg['reference_weight'] * reg, [rec]
    parts = [renoised(s, sd, device) for s, sd in zip(samples, seeds)]
    z, a, zv, sv, av, sa = (torch.cat([p[i] for p in parts]) for i in range(6))
    with torch.no_grad(), amp(device):
        A.set_mode(model, 'old'); ov, oa = velocity_rows(model, samples, zv, av, sv, sa)
        if getattr(model, '_old_is_zero', False):
            rv, ra = ov, oa
        else:
            A.set_mode(model, 'off'); rv, ra = velocity_rows(model, samples, zv, av, sv, sa)
    A.set_mode(model, 'current')
    with amp(device):
        cv, ca = velocity_rows(model, samples, zv, av, sv, sa)
    vm, am = _masks(z, a)
    r = torch.tensor(rhos, device=device)
    pv, regv = nft_loss(z, zv, sv, cv, ov, rv, r, vm, beta=cfg['beta'], epsilon=cfg['nft_epsilon'])
    pa, rega = nft_loss(a, av, sa, ca, oa, ra, r, am, beta=cfg['beta'], epsilon=cfg['nft_epsilon'],
                        reference_mask=torch.ones_like(am))
    dv = (cv.detach().float() - ov.float()).square().flatten(1).mean(1)
    da = (ca.detach().float() - oa.float())[:, :6].square().flatten(1).mean(1)
    recs = [{'pv': float(pv[i].detach()), 'pa': float(pa[i].detach()), 'regv': float(regv[i].detach()), 'rega': float(rega[i].detach()),
             'sv': float(sv[i]), 'sa': float(sa[i]), 'cur_old_v': float(dv[i]), 'cur_old_a': float(da[i])}
            for i in range(len(samples))]
    return (pv + pa + cfg['reference_weight'] * (regv + rega)).sum(), recs


def term_grad_norms(model, sample, rho, seed, cfg, device):
    """Separate gradient norms (over the current LoRA) of the video policy term, the action policy term and the
    weighted regulariser, at the same renoising (one forward per term)."""
    params = A.trainable(model)
    out, rec = {}, None
    for k, name in enumerate(('video', 'action', 'reference')):
        pv, pa, reg, rec = terms(model, sample, rho, seed, cfg, device)
        value = (pv, pa, reg * cfg['reference_weight'])[k]
        if not value.requires_grad:
            out[name] = 0.
            continue
        g = torch.autograd.grad(value, params, allow_unused=True)
        out[name] = float(torch.norm(torch.stack([x.float().norm() for x in g if x is not None]))) if any(x is not None for x in g) else 0.
    return out, rec
