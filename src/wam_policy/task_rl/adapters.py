"""Specialist LoRA adapters of Stage 2.1 Task-Specific RL.

Training (``SwitchLoRALinear``): one frozen base Linear plus two adapter sets on the ten Linear layers of every
transformer block (``metadata.SPECIALIST_TARGETS``; rank 32, alpha 32): ``current`` (trainable parameters) and
``old`` (buffers, the EMA policy theta_old). A switch shared by the whole model selects ``current`` / ``old`` /
``off``; ``off`` is the frozen backbone, i.e. the Stage 1.1 teacher theta_0 used as the reference. One backbone
therefore serves the three roles of the DiffusionNFT update.

Adapters are indexed by their position in ``model.modules()`` order (block-major, targets in the order above):
``{'<i>.lora_A', '<i>.lora_B', '<i>.old_A', '<i>.old_B'}``. A released specialist (``kind: specialist``) stores the
old adapters of one round as ``adapter.pt = {'lora': {'<i>.old_A', '<i>.old_B'}}``.

Inference (``attach_specialist``): the frozen old adapters on a loaded teacher, y = W x + B (A x) * alpha / rank,
with A and B cast to the activation dtype, as in the paper's Standard100 evaluation of the specialists.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from wam_policy.models import metadata

TARGETS = metadata.SPECIALIST_TARGETS
RANK, LORA_ALPHA = 32, 32
INIT_SEED = 20261114
MODES = ('current', 'old', 'off')
PROMPT_SIDE = ('attn2.to_k', 'attn2.to_v')


# ------------------------------------------------------------------ training: switchable LoRA
class SwitchLoRALinear(nn.Module):
    def __init__(self, base, switch, rank=RANK, alpha=LORA_ALPHA):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError('LoRA target must be Linear')
        self.base = base.requires_grad_(False)
        self.switch, self.rank, self.alpha, self.scale = switch, rank, alpha, alpha / rank
        self.prompt_side = False     # attn2.to_k / to_v: the input is the prompt, identical for every sampled row
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=torch.float32))
        self.register_buffer('old_A', torch.zeros(rank, base.in_features, device=dev, dtype=torch.float32))
        self.register_buffer('old_B', torch.zeros(base.out_features, rank, device=dev, dtype=torch.float32))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        if self.prompt_side and x.shape[0] > 1 and self.switch.get('rows_share_prompt'):
            return self._apply(x[:1]).expand(x.shape[0], -1, -1)
        return self._apply(x)

    def _apply(self, x):
        y = self.base(x)
        mode = self.switch['mode']
        if mode == 'off':
            return y
        a, b = (self.lora_A, self.lora_B) if mode == 'current' else (self.old_A, self.old_B)
        return y + F.linear(F.linear(x, a), b) * self.scale


def _check_targets(model):
    for block in model.blocks:
        for name in TARGETS:
            if not isinstance(block.get_submodule(name), nn.Linear):
                raise ValueError('Unexpected LoRA target ' + name)


def inject(model, *, seed=INIT_SEED, rank=RANK, alpha=LORA_ALPHA):
    """Install SwitchLoRALinear on every target; A kaiming-uniform (seeded), B = 0, old = current."""
    if any(isinstance(m, SwitchLoRALinear) for m in model.modules()):
        raise ValueError('LoRA already installed')
    _check_targets(model)
    model.requires_grad_(False)
    switch = {'mode': 'off'}
    g = torch.Generator(device='cpu').manual_seed(seed)
    for block in model.blocks:
        for name in TARGETS:
            parent, leaf = name.rsplit('.', 1)
            module = block.get_submodule(parent)
            lin = SwitchLoRALinear(getattr(module, leaf), switch, rank, alpha)
            lin.prompt_side = name in PROMPT_SIDE
            bound = 1 / math.sqrt(lin.lora_A.shape[1])   # kaiming_uniform_(a=sqrt(5)) bound for fan_in
            with torch.no_grad():
                lin.lora_A.copy_(torch.empty(lin.lora_A.shape).uniform_(-bound, bound, generator=g))
                lin.old_A.copy_(lin.lora_A); lin.old_B.copy_(lin.lora_B)
            setattr(module, leaf, lin)
    model._lora_switch = switch
    return switch


def initial_A(state_like, seed=INIT_SEED):
    """The A matrices ``inject`` draws, recomputed from the shapes of a state dict (readback of round 0)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    out, i = {}, 0
    while f'{i}.lora_A' in state_like:
        shape = state_like[f'{i}.lora_A'].shape
        bound = 1 / math.sqrt(shape[1])
        out[i] = torch.empty(shape).uniform_(-bound, bound, generator=g)
        i += 1
    return out


def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, SwitchLoRALinear)]


def set_mode(model, mode):
    if mode not in MODES:
        raise ValueError(mode)
    model._lora_switch['mode'] = mode


def trainable(model):
    return [p for m in lora_modules(model) for p in (m.lora_A, m.lora_B)]


def n_params(model):
    return sum(p.numel() for p in trainable(model))


@torch.no_grad()
def ema_old(model, eta):
    """old <- eta * old + (1 - eta) * current, on A and B separately."""
    for m in lora_modules(model):
        m.old_A.mul_(eta).add_(m.lora_A.detach(), alpha=1 - eta)
        m.old_B.mul_(eta).add_(m.lora_B.detach(), alpha=1 - eta)


@torch.no_grad()
def displacement(model, which='current', minus=None):
    """sqrt(sum_adapters ||scale * B A||_F^2) via tr((B^T B)(A A^T)); ``minus``: of (which - minus) as full products."""
    total = 0.
    for m in lora_modules(model):
        pairs = {'current': (m.lora_A, m.lora_B), 'old': (m.old_A, m.old_B)}
        a, b = (t.float() for t in pairs[which])
        if minus is None:
            total += float(m.scale ** 2 * ((b.T @ b) * (a @ a.T)).sum())
        else:
            a2, b2 = (t.float() for t in pairs[minus])
            # ||B A - B2 A2||^2 = ||BA||^2 + ||B2A2||^2 - 2 tr((B^T B2)(A2 A^T))
            v = ((b.T @ b) * (a @ a.T)).sum() + ((b2.T @ b2) * (a2 @ a2.T)).sum() - 2 * ((b.T @ b2) * (a2 @ a.T).T).sum()
            total += float(m.scale ** 2 * v)
    return max(total, 0.) ** .5


def lora_state(model, which=('current', 'old')):
    out = {}
    for i, m in enumerate(lora_modules(model)):
        if 'current' in which:
            out[f'{i}.lora_A'] = m.lora_A.detach().cpu().clone(); out[f'{i}.lora_B'] = m.lora_B.detach().cpu().clone()
        if 'old' in which:
            out[f'{i}.old_A'] = m.old_A.detach().cpu().clone(); out[f'{i}.old_B'] = m.old_B.detach().cpu().clone()
    return out


@torch.no_grad()
def load_lora_state(model, state, which=('current', 'old')):
    mods = lora_modules(model)
    if 'current' in which and len([k for k in state if k.endswith('lora_A')]) != len(mods):
        raise ValueError('LoRA inventory mismatch')
    for i, m in enumerate(mods):
        if 'current' in which:
            m.lora_A.copy_(state[f'{i}.lora_A']); m.lora_B.copy_(state[f'{i}.lora_B'])
        if 'old' in which:
            m.old_A.copy_(state[f'{i}.old_A']); m.old_B.copy_(state[f'{i}.old_B'])


@torch.no_grad()
def load_old_into(model, state):
    """Sampling side: only the old adapters matter (mode 'old')."""
    mods = lora_modules(model)
    if len([k for k in state if k.endswith('old_A')]) != len(mods):
        raise ValueError(f'Adapter holds {len(state) // 2} modules, the model {len(mods)}')
    for i, m in enumerate(mods):
        m.old_A.copy_(state[f'{i}.old_A']); m.old_B.copy_(state[f'{i}.old_B'])


def load_adapter(path):
    """The ``{'<i>.old_A', '<i>.old_B'}`` dict of an ``adapter.pt`` / round ``old-lora.pt`` (or a specialist dir)."""
    path = Path(path)
    if path.is_dir():
        meta = metadata.read(path, 'specialist')
        path = metadata.weights(path, meta)
    return torch.load(path, map_location='cpu', weights_only=True)['lora']


# ------------------------------------------------------------------ release format
def write_specialist(out, *, task, adapter_path, round_name, rank=RANK, alpha=LORA_ALPHA, notes=None):
    """A released specialist directory: ``wam_policy.json`` (kind specialist) + ``adapter.pt`` (the old adapters)."""
    import shutil
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(adapter_path, map_location='cpu', weights_only=True)
    if set(state) != {'lora'} or any(not (k.endswith('.old_A') or k.endswith('.old_B')) for k in state['lora']):
        raise ValueError(f'{adapter_path} is not an old-adapter file')
    import hashlib
    weights = out / metadata.WEIGHTS['specialist']
    shutil.copyfile(adapter_path, weights)
    h = hashlib.sha256()
    with open(weights, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    # weights_sha256: computed once here, so a consumer's cache key (e.g. M-OPD expert targets) changes with the weights
    meta = {'kind': 'specialist', 'task': task, 'lora': {'rank': rank, 'alpha': alpha, 'targets': list(TARGETS)},
            'adapter_keys': 'old', 'sampling': {'video_steps': 25, 'action_steps': 50}, 'round': round_name,
            'weights_sha256': {weights.name: h.hexdigest()},
            'notes': {'paper_name': f'Stage 2.1 task-specific RL specialist ({task})',
                      'adapter_layout': "{'lora': {'<i>.old_A', '<i>.old_B'}}, i = adapted Linear in module order",
                      **(notes or {})}}
    metadata.write(out, meta)
    return out


# ------------------------------------------------------------------ inference: frozen specialist on the teacher
class SpecialistLinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        dev = base.weight.device
        self.register_buffer('old_A', torch.zeros(rank, base.in_features, device=dev))
        self.register_buffer('old_B', torch.zeros(base.out_features, rank, device=dev))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.old_A.to(x.dtype)), self.old_B.to(x.dtype)) * self.scale


@torch.no_grad()
def attach_specialist(model, specialist_dir):
    """Install the frozen old adapters of a specialist directory on a loaded teacher (25/50 inference).

    Returns the specialist metadata with ``displacement`` (Frobenius norm of sum B A) added.
    """
    meta = metadata.read(specialist_dir, 'specialist')
    if meta['adapter_keys'] != 'old':
        raise ValueError('Only old-adapter specialists are released')
    rank, alpha, targets = meta['lora']['rank'], meta['lora']['alpha'], tuple(meta['lora']['targets'])
    state = torch.load(metadata.weights(specialist_dir, meta), map_location='cpu', weights_only=True)['lora']
    if any(isinstance(m, (SpecialistLinear, SwitchLoRALinear)) for m in model.modules()):
        raise ValueError('An adapter is already installed')
    for block in model.blocks:
        for name in targets:
            parent, leaf = name.rsplit('.', 1)
            module = block.get_submodule(parent)
            base = getattr(module, leaf)
            if not isinstance(base, nn.Linear):
                raise ValueError('Unexpected adapter target ' + name)
            setattr(module, leaf, SpecialistLinear(base, rank, alpha))
    mods = [m for m in model.modules() if isinstance(m, SpecialistLinear)]
    if len(state) != 2 * len(mods):
        raise ValueError(f'Adapter holds {len(state)} tensors for {len(mods)} adapted layers')
    norm = 0.
    for i, m in enumerate(mods):
        m.old_A.copy_(state[f'{i}.old_A']); m.old_B.copy_(state[f'{i}.old_B'])
        norm += float((m.old_B.float() @ m.old_A.float()).pow(2).sum())
    model.eval().requires_grad_(False)
    return {**meta, 'displacement': norm ** .5}
