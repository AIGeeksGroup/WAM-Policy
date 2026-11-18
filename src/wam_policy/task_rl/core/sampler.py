"""Policy loading and the samplers of Stage 2.1: history-recording one-row sampler and the row-batched sampler.

* ``load_policy``: the Stage 1.1 teacher bundle (``wam_policy.inference.loader``), bf16, attention ``torch``, VAE
  encoder on the GPU (decoder moved to the CPU: candidate video is never decoded), the task prompt embedding; the
  switchable LoRA installed and set to ``off`` (= the teacher) or ``old`` (a round's old adapters).
* ``RecordingSampler``: ``VideoActionSampler`` that records what it committed (latents and normalized actions, bf16
  as fed to the model). A start's history is rebuilt by this one-row sampler (``history.rebuild``); the recorded
  prefix is a training input, so the learner sees bit for bit the sampler's condition.
* ``RowSampler``: B rows in lock step (the K candidates of a start, several starts, or several episodes from reset).
  The model's K/V cache has a batch dimension; its slot table is shared by the rows, so rows that advance together fit
  one cache. Each row keeps its own noise generator and persistent noise (same seeds and draws as one row), its own
  absolute group index for the position encoding, and its own VAE streaming cache (``RowCodec``). A start rebuilt by
  the one-row sampler is copied into its rows. Rows share one prompt, so its cross-attention projections are computed
  for one row and expanded. Row-batched and one-row sampling agree to bf16 rounding, not bit for bit.

Window-shared initial noise (candidate windows): each candidate k draws one persistent noise d_k per modality and every
chunk of its window starts from eps = sqrt(1 - alpha^2) xi_t + alpha d_k (alpha = 1 in the paper).
"""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch

from wam_policy.common.flow_config import FlowConfig
from wam_policy.common.tasks import INSTRUCTIONS
from wam_policy.inference.sampler import ACTION_NOISE_SHAPE, SamplingConfig, VideoActionSampler
from wam_policy.task_rl import adapters as A
from wam_policy.task_rl.core.run import NFE

VIDEO_SHAPE = (1, 48, 1, 24, 20)


def load_bundle(run):
    """Teacher bundle and the task's prompt embedding [1, 512, 4096]."""
    from wam_policy.inference.loader import load_inference_bundle
    if run.prompt_root is None:
        raise SystemExit('run.json has no prompt_root: Task RL uses the cached prompt embeddings (init --prompt-root)')
    bundle = load_inference_bundle(checkpoint=run.teacher, base_model=run.base_model, prompt_root=run.prompt_root,
                                   tasks=(run.task,), device='cuda:0', vae_device='cuda:0', dtype='bfloat16')
    bundle.codec.vae.decoder.to('cpu')
    bundle.model.eval().requires_grad_(False)
    text = bundle.prompt_encoder(INSTRUCTIONS[run.task])['text_emb']
    return bundle, text


def make_sampler(bundle, *, lora=None, mode='off', config=None):
    """RecordingSampler on the bundle's model, with the switchable LoRA installed (optionally loaded) and frozen."""
    model = bundle.model
    if not any(isinstance(m, A.SwitchLoRALinear) for m in model.modules()):
        cfg = config or {}
        A.inject(model, rank=cfg.get('lora_rank', A.RANK), alpha=cfg.get('lora_alpha', A.LORA_ALPHA))
    if lora is not None:
        A.load_old_into(model, lora)
    model.requires_grad_(False)
    A.set_mode(model, mode)
    return RecordingSampler(model, bundle.normalizer, SamplingConfig.preset(NFE))


def make_rows(bundle):
    """RowSampler + RowCodec on the bundle's model and VAE (call after ``make_sampler``)."""
    s = RowSampler(bundle.model, bundle.normalizer, SamplingConfig.preset(NFE))
    s.cache_name = 'wam-rows'
    bundle.model._lora_switch['rows_share_prompt'] = True
    return s, RowCodec(bundle.codec)


def load_policy(run, lora_path=None, *, current=False):
    """(bundle, text, one-row builder, row sampler, row codec, label) for the teacher or a round's adapters.
    ``lora_path``: a round's ``old-lora.pt`` (mode 'old'); with ``current`` a round's ``state.pt`` whose current
    adapters are run in mode 'old'."""
    bundle, text = load_bundle(run)
    lora, label = None, 'teacher'
    if lora_path is not None:
        if current:
            st = torch.load(run.path(lora_path), map_location='cpu', weights_only=False)['lora']
            lora = {k.replace('lora_A', 'old_A').replace('lora_B', 'old_B'): v for k, v in st.items() if 'lora_' in k}
            label = f'current:{lora_path}'
        else:
            lora = A.load_adapter(run.path(lora_path))
            label = f'old:{lora_path}'
    builder = make_sampler(bundle, lora=lora, mode='old' if lora is not None else 'off', config=run.config)
    sampler, rcodec = make_rows(bundle)
    return bundle, text, builder, sampler, rcodec, label


def persistent_noise(seed, device, dtype=torch.float32):
    """d_k for one candidate: one standard normal per modality."""
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    return {'video': torch.randn(VIDEO_SHAPE, generator=g).to(device, dtype),
            'action': torch.randn(ACTION_NOISE_SHAPE, generator=g).to(device, dtype)}


def stack_persistent(seeds, device):
    parts = [persistent_noise(s, device) for s in seeds]
    return {k: torch.cat([p[k] for p in parts]) for k in ('video', 'action')}


def _amp(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


class RecordingSampler(VideoActionSampler):
    def __init__(self, model, normalizer, config):
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('The sampling policy must be frozen')
        if config.nfe not in ('25/50', '1/2') or config.window_units != 8:
            raise ValueError('Task RL samples at 25/50 (or 1/2) with window 8')
        super().__init__(model, normalizer, config)
        if self.device.type == 'cuda':
            self.dtype = torch.bfloat16
        self.alpha = 0.0
        self.persistent = None
        self.hist_latents, self.hist_actions = [], []
        self.generated = {}

    def _forward(self, *args, **kwargs):
        with _amp(self.device):
            return super()._forward(*args, **kwargs)

    def reset(self, latent, text_emb, *, seed, empty_text_emb=None):
        super().reset(latent, text_emb, seed=seed, empty_text_emb=empty_text_emb)
        self.hist_latents = [latent.to(self.device, self.dtype).detach().cpu().clone()]
        self.hist_actions = [torch.zeros(*ACTION_NOISE_SHAPE, dtype=self.dtype)]

    def _commit_normalized(self, latent, value):
        self.hist_latents.append(latent.to(self.device, self.dtype).detach().cpu().clone())
        self.hist_actions.append(value.to(self.device, self.dtype).detach().cpu().clone())
        return super()._commit_normalized(latent, value)

    def history_tensors(self):
        """(latents [1,48,G+1,24,20], actions [1,9,G+1,16,1]) exactly as committed (bf16)."""
        return torch.cat(self.hist_latents, 2), torch.cat(self.hist_actions, 2)

    def _mix(self, value, action):
        if not self.alpha or self.persistent is None:
            return value
        d = self.persistent['action' if action else 'video'].to(self.device)
        return ((1.0 - self.alpha ** 2) ** 0.5 * value.float() + self.alpha * d.float()).to(self.dtype)


class RowSampler(RecordingSampler):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rows, self.gens, self.shift, self._grids = 0, [], [], {}

    # ---- model call: every row its own absolute group (position encoding), one shared logical group (cache window)
    def _grid(self, group, action):
        from wan_va.utils import get_mesh_id
        out = []
        for sh in self.shift:
            key = (group + sh, action)
            if key not in self._grids:
                self._grids[key] = get_mesh_id(1, 16 if action else 12, 1 if action else 10, int(action), 1, group + sh,
                                               action=action).to(self.device)
            out.append(self._grids[key])
        return torch.stack(out)

    def _forward(self, value, timestep, *, group, action=False, update=0):
        b = value.shape[0]
        if b != self.rows:
            raise ValueError('Row count mismatch')
        payload = {'noisy_latents': value, 'text_emb': self.text,
                   'timesteps': torch.full((b, 1), float(timestep), device=self.device),
                   'grid_id': self._grid(group, action)}
        for cache in self._caches():
            cache['query_logical_id'] = 2 * group + int(action)
            cache['window_units'] = self.config.window_units
        with _amp(self.device):
            return self.model(payload, update_cache=update, cache_name=self.cache_name, action_mode=action)

    def _sample(self, shape, *, steps, shift, action, guidance=1., noise_offset=None):
        """Flow sampling with a row dimension: one generator draw per row, the window-shared mix, no CFG."""
        from wan_va.utils import FlowMatchScheduler, data_seq_to_patch
        b = self.rows
        value = torch.cat([torch.randn(shape, generator=g, device=self.device, dtype=self.dtype) for g in self.gens])
        value = self._mix(value, action)
        scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
        scheduler.set_timesteps(steps)
        for t in scheduler.timesteps:
            pred = self._forward(value, t, group=self.group, action=action)
            if action:
                pred = pred.transpose(1, 2).reshape(b, 9, 1, 16, 1)
            else:
                pred = data_seq_to_patch((1, 2, 2), pred, 1, 24, 20, batch_size=b)
            value = scheduler.step(pred, t, value).to(self.dtype)
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite flow sampling result")
        self._forward(value, 0, group=self.group, action=action, update=1)
        self.generated['action' if action else 'video'] = value.detach().cpu().clone()
        return value

    @torch.inference_mode()
    def endpoints(self):
        """One chunk for every row: (clean video [B], clean action [B], requested commands: list of (16, 9))."""
        if not self.ready or self.pending:
            raise ValueError('Confirmed history required')
        flow = FlowConfig()
        z = self._sample(VIDEO_SHAPE, steps=self.config.video_steps, shift=flow.video_snr_shift, action=False)
        u = self._sample(ACTION_NOISE_SHAPE, steps=self.config.action_steps, shift=flow.action_snr_shift, action=True)
        flat = u[:, :, 0, :, 0].float().cpu().numpy()
        requested = []
        for r in range(self.rows):
            x = self.normalizer.decode(flat[r].T)
            x[:, 6:] = 0
            if not np.isfinite(x).all():
                raise FloatingPointError('Nonfinite decoded physical commands')
            requested.append(x)
        self.pending = True
        return z, u, requested

    @torch.inference_mode()
    def commit(self, latents, executed):
        """Commit the real feedback of every row; returns what was fed to the model (bf16, CPU): latents, actions."""
        if not self.pending:
            raise ValueError('No pending action prediction')
        vals = []
        for e in executed:
            e = np.asarray(e, np.float32)
            if e.shape != (16, 9):
                raise ValueError('Only a complete execution-confirmed group may commit')
            vals.append(torch.from_numpy(self.normalizer.encode(e).T.copy()).reshape(*ACTION_NOISE_SHAPE))
        lat = latents.to(self.device, self.dtype)
        val = torch.cat(vals).to(self.device, self.dtype)
        if tuple(lat.shape) != (self.rows, 48, 1, 24, 20) or not torch.isfinite(lat).all():
            raise ValueError('One finite dual-view causal latent group per row required')
        self.model.clear_pred_cache(self.cache_name)
        self._forward(lat, 0, group=self.group, update=2)
        self._forward(val, 0, group=self.group, action=True, update=2)
        self.group += 1
        minimum = 2 * max(0, self.group - self.config.window_units // 2)
        for cache in self._caches():
            expired = cache['mask'] & (cache['logical_id'] < minimum)
            cache['mask'][expired] = False
            cache['id'][expired] = -1
        self.pending = False
        return lat.detach().cpu().clone(), val.detach().cpu().clone()

    # ---- filling the rows
    def _open(self, rows, text):
        self.clear()
        self.rows = rows
        self.text = text.to(self.device, self.dtype).expand(rows, -1, -1)
        self.model.create_empty_cache(self.cache_name, self.config.window_units + 4, 120, 16, self.device, self.dtype, rows)
        for cache in self._caches():
            cache['logical_id'] = torch.full_like(cache['id'], -1)
        self.generated, self.gens, self.shift = {}, [], [0] * rows
        self.alpha, self.persistent = 0.0, None

    @torch.inference_mode()
    def reset_rows(self, latents, text, seeds):
        """Episodes from reset (``VideoActionSampler.reset`` per row): the first latent, the zero action placeholder."""
        rows = latents.shape[0]
        self._open(rows, text)
        self.gens = [torch.Generator(device=self.device).manual_seed(int(s)) for s in seeds]
        self._forward(latents.to(self.device, self.dtype), 0, group=0, update=2)
        self._forward(torch.zeros(rows, 9, 1, 16, 1, device=self.device, dtype=self.dtype), 0, group=0, action=True, update=2)
        self.group, self.ready = 1, True

    @torch.inference_mode()
    def load_starts(self, text, counts):
        """Open ``sum(counts)`` rows; then ``put_start`` once per start (after its one-row rebuild) and ``arm``."""
        self._open(sum(counts), text)
        self._counts, self._filled, self._nstart, self._layout = list(counts), 0, 0, None
        self.group = self.config.window_units // 2      # shared logical group of every row at the start of the window

    @torch.inference_mode()
    def put_start(self, builder):
        """Copy the rebuilt observed K/V of the one-row sampler into the next start's rows. The live slots are written
        in (logical unit, slot) order with logical units relative to the start's group, so starts at different absolute
        groups share one slot table; the absolute group stays in the row's position encoding (``shift``)."""
        r0, n = self._filled, self._counts[self._nstart]
        shift = builder.group - self.group
        src0 = builder._caches()[0]
        live = (src0['mask'] & ~src0['is_pred']).nonzero(as_tuple=False).squeeze(-1)
        if int(src0['mask'].sum()) != live.numel():
            raise ValueError('Start rebuilt with a pending prediction')
        logical = src0['logical_id'][live]
        order = live[torch.argsort(logical * (src0['id'].numel() + 1) + live)]
        rel = src0['logical_id'][order] - 2 * shift
        if self._layout is None:
            self._layout = rel.clone()
        elif self._layout.shape != rel.shape or not torch.equal(self._layout, rel):
            raise ValueError('Starts of one batch need the same in-window history depth')
        m = order.numel()
        for dst, src in zip(self._caches(), builder._caches()):
            dst['k'][r0:r0 + n, :m] = src['k'][:, order]
            dst['v'][r0:r0 + n, :m] = src['v'][:, order]
            if r0 == 0:
                dst['mask'][:] = False; dst['is_pred'][:] = False; dst['id'][:] = -1; dst['logical_id'][:] = -1
                dst['mask'][:m] = True
                dst['id'][:m] = rel            # one id per committed unit, increasing with time
                dst['logical_id'][:m] = rel
        for r in range(r0, r0 + n):
            self.shift[r] = shift
        self._filled += n
        self._nstart += 1

    def arm(self, seeds, persistent, alpha):
        """Per-row generators (the candidate seeds) and persistent noise ({'video': [B,...], 'action': [B,...]})."""
        if self._filled != self.rows or len(seeds) != self.rows:
            raise ValueError('Rows not filled')
        self.gens = [torch.Generator(device=self.device).manual_seed(int(s)) for s in seeds]
        self.alpha, self.persistent = alpha, persistent
        self.pending, self.ready = False, True

    @torch.inference_mode()
    def keep(self, idx):
        """Drop the rows not in ``idx`` (positions among the current rows); the pending prediction stays consistent."""
        i = torch.as_tensor(idx, device=self.device, dtype=torch.long)
        for cache in self._caches():
            cache['k'] = cache['k'][i]
            cache['v'] = cache['v'][i]
        self.text = self.text[:len(idx)]
        self.gens = [self.gens[j] for j in idx]
        self.shift = [self.shift[j] for j in idx]
        if self.persistent is not None:
            self.persistent = {k: v[i] for k, v in self.persistent.items()}
        self.rows = len(idx)


class RowCodec:
    """A second ``DualViewVAE`` on the same frozen VAE whose streaming caches hold one row per candidate / episode."""

    def __init__(self, codec):
        from wam_policy.common.video import DualViewVAE
        self.codec = DualViewVAE(codec.vae, rgb_profile=codec.rgb_profile)
        self._states, self._counts = [], []

    def reset(self):
        self.codec.reset()
        self._states, self._counts = [], []

    def push(self, views):
        return self.codec.push(views)

    @torch.inference_mode()
    def put_start(self, builder_codec, count):
        """Remember the one-row codec state of a rebuilt start for ``count`` rows."""
        enc = {k: [t.clone() if torch.is_tensor(t) else t for t in s.feat_cache] for k, s in builder_codec.streams.items()}
        buf = {k: (None if v is None else v.clone()) for k, v in builder_codec.buffers.items()}
        if builder_codec.first or builder_codec.batch_size != 1:
            raise ValueError('One-row codec state after a rebuild required')
        self._states.append((enc, buf)); self._counts.append(count)

    @torch.inference_mode()
    def arm(self):
        def rows(values):
            if not torch.is_tensor(values[0]):
                return values[0]
            return torch.cat([v.expand(c, *v.shape[1:]) for v, c in zip(values, self._counts)])
        for k, s in self.codec.streams.items():
            s.feat_cache = [rows([st[0][k][i] for st in self._states]) for i in range(len(self._states[0][0][k]))]
        self.codec.buffers = {k: (None if self._states[0][1][k] is None else rows([st[1][k] for st in self._states]))
                              for k in self.codec.buffers}
        self.codec.first, self.codec.batch_size = False, sum(self._counts)
        self._states, self._counts = [], []

    @torch.inference_mode()
    def keep(self, idx):
        for s in self.codec.streams.values():
            s.feat_cache = [t[torch.as_tensor(idx, device=t.device)] if torch.is_tensor(t) else t for t in s.feat_cache]
        self.codec.buffers = {k: (None if v is None else v[torch.as_tensor(idx, device=v.device)])
                              for k, v in self.codec.buffers.items()}
        self.codec.batch_size = len(idx)
