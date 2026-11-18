"""Accelerated Standard100 evaluation of the teacher or a specialist: row-batched multi-session service.

    python -m wam_policy.evaluation.batched_service --policy {teacher,specialist} \
        --teacher <stage1_1_teacher> [--adapter <specialist dir>] \
        --base-model <lingbot-va-base snapshot> --prompt-root <prompt_embeddings> \
        [--video-steps 25 --action-steps 50] [--max-batch 8] [--max-wait-s 1.5] [--port 8870] [--log-dir <dir>]

This is the multi-session resident service of the paper's accelerated evaluation (25/50 NFE, no
CFG, window 8, VAE on the GPU). The model (plus the frozen specialist LoRA) loads once through
:func:`wam_policy.inference.policies.build_policy`. Every QLM client session (``session_id``) gets
its own :class:`~wam_policy.inference.session.OnlineSession`: its own sampler (own KV-cache name,
own ``torch.Generator`` seeded by its reset, per-episode policy seed from the reset request), its
own streaming VAE state and its own request de-duplication (the qlm-live-v1 ``RPCServer.dispatch``
rules). ``reset`` / ``feedback`` / ``end`` run per session through the unchanged code.

Only ``predict`` of the sessions waiting at the same time is executed as one row-batched flow
sampling (:func:`predict_rows`): every linear layer runs on the stacked rows, self-attention reads
each row's own cache (gathered once per denoising phase, same window filter), and each session's
cross-attention text K/V is computed once per episode and reused (the same values the single-row
path recomputes at every step). Each row draws its initial noise from its own generator in the
original order (video, then action), so a case's noise and history never depend on the other rows.
Results differ from the single-session ``python -m wam_policy.inference.serve`` only by bf16
rounding of the batched GEMMs (and, with the VAE on the GPU, by its floating-point rounding).
A failed batch is rolled back exactly and every row then runs the single-row prediction.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import queue
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from wam_policy.evaluation.multi_session_service import (MultiSessionService, error_response, run_service,
                                                         service_parser)

BATCHED_POLICIES = ("teacher", "specialist")


# ----------------------------------------------------------------------------------- row-batched attention
class Rows(tuple):
    """``cache_name`` marker for a row-batched forward: one session cache name per row."""


_EPOCH = [0]   # bumped whenever row caches may have changed; invalidates the per-phase gathered history


def _rotary(x, freqs):
    out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    return torch.view_as_real(out * freqs).flatten(3).to(x.dtype)


def _valid(cache):
    m = cache["mask"]
    if "logical_id" in cache:
        lq = cache["query_logical_id"]
        m = m & (cache["logical_id"] >= lq - cache["window_units"]) & (cache["logical_id"] <= lq)
    return m.nonzero(as_tuple=False).squeeze(-1)


def _out(self, hidden, query):
    hidden = hidden.flatten(2, 3).type_as(query)
    return self.to_out[1](self.to_out[0](hidden))


def _cross_rows(self, q, text, rows):
    """Cross-attention with per-session text K/V computed once per episode (same math as recomputing it)."""
    from wan_va.modules.model import custom_sdpa
    query = self.norm_q(self.to_q(q)).unflatten(2, (self.heads, -1))
    store = self.__dict__.setdefault("_text_kv", {})
    ks, vs = [], []
    for r, name in enumerate(rows):
        if name not in store:
            t = text[r:r + 1]
            store[name] = (self.norm_k(self.to_k(t)).unflatten(2, (self.heads, -1)),
                           self.to_v(t).unflatten(2, (self.heads, -1)))
        ks.append(store[name][0])
        vs.append(store[name][1])
    return _out(self, custom_sdpa(query, torch.cat(ks), torch.cat(vs)), query)


def _self_rows(self, q, rotary_emb, update, rows):
    from wan_va.modules.model import custom_sdpa
    query, key, value = self.to_q(q), self.to_k(q), self.to_v(q)
    query = self.norm_q(query).unflatten(2, (self.heads, -1))
    key = self.norm_k(key).unflatten(2, (self.heads, -1))
    value = value.unflatten(2, (self.heads, -1))
    query, key = _rotary(query, rotary_emb), _rotary(key, rotary_emb)
    caches = [self.attn_caches[n] for n in rows]
    if update not in (0, 1, 2):
        raise ValueError("Unknown inference cache lifecycle")
    if update == 0:
        hist = self.__dict__.get("_rows_hist")
        if hist is None or hist[0] != _EPOCH[0]:
            ks, vs = [], []
            for c in caches:
                valid = _valid(c)
                ks.append(c["k"][0, valid])
                vs.append(c["v"][0, valid])
            lens = [x.shape[0] for x in ks]
            if min(lens) == max(lens):
                hist = (_EPOCH[0], torch.stack(ks), torch.stack(vs), None, None)
            else:
                length = max(lens)
                K = ks[0].new_zeros((len(ks), length, *ks[0].shape[1:]))
                V = torch.zeros_like(K)
                mask = torch.zeros((len(ks), length), dtype=torch.bool, device=K.device)
                for r, (k, v) in enumerate(zip(ks, vs)):
                    K[r, :lens[r]] = k
                    V[r, :lens[r]] = v
                    mask[r, :lens[r]] = True
                hist = (_EPOCH[0], K, V, mask, lens)
            self.__dict__["_rows_hist"] = hist
        _, K, V, mask, _ = hist
        K, V = torch.cat((K, key), 1), torch.cat((V, value), 1)
        if mask is None:
            hidden = custom_sdpa(query, K, V)
        else:
            full = torch.cat((mask, mask.new_ones((mask.shape[0], key.shape[1]))), 1)[:, None, None, :]
            hidden = F.scaled_dot_product_attention(query.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2),
                                                    attn_mask=full).transpose(1, 2)
    else:
        # Persistent write (1 = predicted, 2 = observed): per row, exactly the single-row cache lifecycle.
        self.__dict__["_rows_hist"] = None
        outs = []
        for r, (name, c) in enumerate(zip(rows, caches)):
            self.update_cache(name, key[r:r + 1], value[r:r + 1], is_pred=(update == 1))
            valid = _valid(c)
            outs.append(custom_sdpa(query[r:r + 1], c["k"][:, valid], c["v"][:, valid]))
        hidden = torch.cat(outs)
    return _out(self, hidden, query)


def install_attention_patch():
    """Route ``WanAttention.forward`` calls whose ``cache_name`` is :class:`Rows` to the row-batched path."""
    from wan_va.modules import model as M
    if getattr(M.WanAttention, "_rows_patch", False):
        return
    original = M.WanAttention.forward

    def forward(self, q, k, v, rotary_emb, update_cache=0, cache_name="pos"):
        if not isinstance(cache_name, Rows):
            return original(self, q, k, v, rotary_emb, update_cache=update_cache, cache_name=cache_name)
        if self.attn_caches is None:
            return _cross_rows(self, q, k, cache_name)
        return _self_rows(self, q, rotary_emb, update_cache, cache_name)

    M.WanAttention.forward = forward
    M.WanAttention._rows_patch = True


def drop_text_kv(model, name):
    for block in model.blocks:
        block.attn2.__dict__.get("_text_kv", {}).pop(name, None)


def drop_session_tensors(model, name):
    drop_text_kv(model, name)
    for block in model.blocks:
        block.attn1.__dict__["_rows_hist"] = None
        if block.attn1.attn_caches is not None:
            block.attn1.attn_caches.pop(name, None)


# ----------------------------------------------------------------------------------- samplers
def session_sampler_class():
    from wam_policy.inference.sampler import VideoActionSampler

    class SessionSampler(VideoActionSampler):
        """VideoActionSampler on its own cache name; ``predict`` hands out the row-batched result when present."""

        def __init__(self, model, normalizer, config, name):
            super().__init__(model, normalizer, config)
            self.cache_name = name
            self.clear()
            self._batched = None

        def predict(self, **kwargs):
            result, self._batched = self._batched, None
            if result is None:
                return super().predict(**kwargs)
            if isinstance(result, BaseException):
                raise result
            return result

        def batchable(self):
            return self.ready and not self.pending and self.text is not None and self.text.shape[0] == 1

    return SessionSampler


@torch.inference_mode()
def predict_rows(samplers):
    """``VideoActionSampler.predict`` for several sessions at once (same draws, schedule and cache lifecycle)."""
    from wan_va.utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id
    from wam_policy.common.flow_config import FlowConfig
    s0 = samplers[0]
    model, device, dtype, cfg, flow = s0.model, s0.device, s0.dtype, s0.config, FlowConfig()
    if any(s.config != cfg or not s.batchable() for s in samplers):
        raise ValueError("Row batch needs initialized, confirmed, CFG-free sessions with one sampling config")
    if max(cfg.video_guidance, cfg.action_guidance) > 1:
        raise ValueError("Row batching is CFG-free only")
    rows = Rows(s.cache_name for s in samplers)
    if len(set(rows)) != len(rows):
        raise ValueError("Duplicate session in one batch")
    B = len(rows)
    text = torch.cat([s.text for s in samplers])

    def sample(shape, steps, shift, action):
        _EPOCH[0] += 1
        for s in samplers:
            for c in s._caches():
                c["query_logical_id"] = 2 * s.group + int(action)
                c["window_units"] = s.config.window_units
        grid = torch.stack([get_mesh_id(1, 16 if action else 12, 1 if action else 10, int(action), 1, s.group,
                                        action=action) for s in samplers]).to(device)

        def forward(value, t, update):
            payload = {"noisy_latents": value, "text_emb": text,
                       "timesteps": torch.full((B, 1), float(t), device=device), "grid_id": grid}
            return model(payload, update_cache=update, cache_name=rows, action_mode=action)

        value = torch.cat([torch.randn(shape, generator=s.generator, device=device, dtype=dtype) for s in samplers])
        scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
        scheduler.set_timesteps(steps)
        for t in scheduler.timesteps:
            pred = forward(value, t, 0)
            if action:
                pred = pred.transpose(1, 2).reshape(B, 9, 1, 16, 1)
            else:
                pred = data_seq_to_patch((1, 2, 2), pred, 1, 24, 20, batch_size=B)
            value = scheduler.step(pred, t, value).to(dtype)
        # NaN/inf propagate through the remaining steps: checked once per phase (no per-step syncs).
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite flow sampling result")
        forward(value, 0, 1)
        _EPOCH[0] += 1
        return value

    sample((1, 48, 1, 24, 20), cfg.video_steps, flow.video_snr_shift, False)
    actions = sample((1, 9, 1, 16, 1), cfg.action_steps, flow.action_snr_shift, True)
    flat = actions[:, :, 0, :, 0].float().cpu().numpy()
    out = []
    for r, s in enumerate(samplers):
        commands = s.normalizer.decode(flat[r].T)
        commands[:, 6:] = 0
        if commands.shape != (16, 9) or not np.isfinite(commands).all():
            raise ValueError("Finite native9 command group required")
        s.pending = True
        s.last_diagnostics = {"latent_group": s.group, "sampling": asdict(cfg),
                              "history_source": "confirmed_physical_feedback", "row_batch": B}
        out.append(commands)
    return out


@torch.inference_mode()
def rollback_rows(samplers, states):
    """Undo a failed :func:`predict_rows` exactly.

    Before predict a session has no live predicted (``is_pred``) cache entries, and slot allocation
    only looks at the mask, so clearing the predicted entries and restoring the generator gives the
    state the single-row predict starts from."""
    _EPOCH[0] += 1
    for block in samplers[0].model.blocks:
        block.attn1.__dict__["_rows_hist"] = None
    for s, state in zip(samplers, states):
        s.model.clear_pred_cache(s.cache_name)
        s.generator.set_state(state)
        s.pending = False
        s._batched = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------- service
class BatchedService(MultiSessionService):
    """Multi-session service whose concurrent fresh ``predict`` requests run as one row batch."""

    def __init__(self, factory, model, log_dir=None, *, max_batch=8, max_wait_s=1.5, **kwargs):
        super().__init__(factory, log_dir, **kwargs)
        if max_batch < 1 or max_wait_s < 0:
            raise ValueError("--max-batch >= 1 and --max-wait-s >= 0 required")
        self.model, self.max_batch, self.max_wait_s = model, max_batch, max_wait_s
        self.parked = []

    def _live(self):
        now = time.monotonic()
        return [s for s in self.slots.values()
                if s.session.history.state not in ("ENDED", "ERROR", None) and s.session.history.identity is not None
                and now - s.last_seen < 60.]

    def _dispatch(self, slot, request):
        try:
            response = slot.dispatcher.dispatch(request)
        except Exception as error:
            response = error_response(error)
        if slot.session.history.state in ("ENDED", "ERROR") or request.get("method") == "reset":
            # Text K/V are per episode; the KV cache itself is cleared by the session's own lifecycle.
            drop_text_kv(self.model, slot.name)
        return response

    def _fresh_predict(self, slot, request):
        d, s = slot.dispatcher, slot.session
        ident = tuple(request[k] for k in ("session_id", "episode_id", "reset_epoch"))
        if ident != d.identity or d._poisoned or (ident, request["request_id"]) in d._seen:
            return False
        oid = request["payload"].get("observation_id")
        return (s.history.state == "OBSERVED" and oid == s.history.observation_id and oid not in s.predictions
                and s.sampler.batchable())

    def _accept(self, item):
        try:
            self._accept_inner(item)
        except Exception as error:   # never let one request take the whole service down
            traceback.print_exc()
            if not item.done.is_set():
                item.response = error_response(error)
                item.done.set()

    def _accept_inner(self, item):
        request = item.request
        slot = self.slots.get(request.get("session_id")) if isinstance(request, dict) else None
        if slot is not None and slot.parked:
            # Serialize per session exactly like the single-session server: answer the parked predict first.
            slot.deferred.append(item)
            return
        try:
            slot = self._slot_for(request)
        except Exception as error:
            item.response = error_response(error)
            item.done.set()
            return
        if request["method"] == "predict" and self._fresh_predict(slot, request):
            slot.parked = True
            self.parked.append((item, slot))
            return
        item.response = self._dispatch(slot, request)
        item.done.set()

    def _flush(self):
        parked, self.parked = self.parked[:self.max_batch], self.parked[self.max_batch:]
        batch, plain = [], []
        for item, slot in parked:   # re-check right before sampling
            (batch if self._fresh_predict(slot, item.request) and all(slot is not s for _, s in batch)
             else plain).append((item, slot))
        samplers = [slot.session.sampler for _, slot in batch]
        started, error = time.perf_counter(), None
        if samplers:
            states = [s.generator.get_state() for s in samplers]
            try:
                for s, commands in zip(samplers, predict_rows(samplers)):
                    s._batched = commands
            except Exception as caught:
                # Exact rollback, then every row runs the single-row predict (its own result or error).
                error = repr(caught)
                traceback.print_exc()
                del caught
                try:
                    rollback_rows(samplers, states)
                except Exception as failed:
                    traceback.print_exc()
                    for s in samplers:   # cannot restore: these sessions fail, the others are untouched
                        s._batched = RuntimeError(f"Row-batched prediction failed and could not roll back: {failed!r}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        elapsed, now = time.perf_counter() - started, time.monotonic()
        self.log("batches.jsonl", {
            "t": datetime.now(timezone.utc).isoformat(), "rows": len(batch), "unbatched": len(plain),
            "live": len(self._live()), "predict_s": elapsed,
            "oldest_wait_s": max(now - it.arrived for it, _ in parked) - elapsed,
            "sessions": [slot.name for _, slot in batch],
            "memory_peak_allocated": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "error": error})
        for item, slot in batch + plain:
            item.response = self._dispatch(slot, item.request)
            slot.session.sampler._batched = None
            item.done.set()
        for item, slot in batch + plain:
            if slot.parked:
                slot.parked = False
                deferred, slot.deferred = slot.deferred, []
                for later in deferred:
                    self._accept(later)

    def run(self):
        housekeeping = time.monotonic()
        while True:
            timeout = 30.
            if time.monotonic() - housekeeping > 30.:
                housekeeping = time.monotonic()
                try:
                    self._drop_idle()
                except Exception:
                    traceback.print_exc()
            if self.parked:
                oldest = min(it.arrived for it, _ in self.parked)
                timeout = max(0., oldest + self.max_wait_s - time.monotonic())
            try:
                items = [self.queue.get(timeout=timeout)]
            except queue.Empty:
                items = []
            while True:
                try:
                    items.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            for item in items:
                if item is None:
                    return
                self._accept(item)
            if self.parked:
                oldest = min(it.arrived for it, _ in self.parked)
                waiting = {slot.name for _, slot in self.parked}
                live = {s.name for s in self._live()}
                if (len(self.parked) >= self.max_batch or live <= waiting
                        or time.monotonic() - oldest >= self.max_wait_s) and self.queue.empty():
                    self._flush()


def batched_factory(handle):
    """Per-session samplers / VAE streams on the model that ``build_policy`` loaded."""
    from wam_policy.common.video import DualViewVAE
    from wam_policy.inference.policies import PolicyHandle
    from wam_policy.inference.session import OnlineSession
    base = handle.session
    sampler0, codec0 = base.sampler, base.pipeline.codec
    model, Sampler = sampler0.model, session_sampler_class()

    def factory(name):
        sampler = Sampler(model, sampler0.normalizer, sampler0.config, name)
        codec = DualViewVAE(codec0.vae, rgb_profile=codec0.rgb_profile)
        session = OnlineSession(sampler, codec, base.prompt_encoder, policy_seed=base.policy_seed,
                                info=base.info)

        def retire():
            try:
                session.end("replaced_or_idle")
            except Exception:
                pass
            drop_session_tensors(model, name)
        return PolicyHandle(session), retire
    return factory


def parser():
    p = service_parser(__doc__, BATCHED_POLICIES)
    p.add_argument("--video-steps", type=int, help="default from metadata (25)")
    p.add_argument("--action-steps", type=int, help="default from metadata (50)")
    p.add_argument("--max-batch", type=int, default=8, help="most rows in one batched prediction")
    p.add_argument("--max-wait-s", type=float, default=1.5,
                   help="longest a waiting prediction is held for more rows")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    from wam_policy.inference.policies import build_policy
    handle = build_policy(args.policy, teacher=args.teacher, base_model=args.base_model, adapter=args.adapter,
                          prompt_root=args.prompt_root, tasks=tuple(args.tasks.split(",")) if args.tasks else None,
                          video_steps=args.video_steps, action_steps=args.action_steps, device=args.device,
                          vae_device=args.vae_device, dtype=args.dtype, policy_seed=args.policy_seed)
    handle.session.info["service"] = {"kind": "batched", "max_batch": args.max_batch,
                                          "max_wait_s": args.max_wait_s}
    install_attention_patch()
    if args.log_dir:
        args.log_dir.mkdir(parents=True, exist_ok=True)
    service = BatchedService(batched_factory(handle), handle.session.sampler.model, args.log_dir,
                             max_batch=args.max_batch, max_wait_s=args.max_wait_s, rpc_timeout_s=args.timeout_s)
    return run_service(service, args, handle.session.info)


if __name__ == "__main__":
    raise SystemExit(main())
