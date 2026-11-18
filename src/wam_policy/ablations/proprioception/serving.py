"""Serve the proprioception teacher to QLM-Bench (Standard100) over the qlm-live-v1 RPC.

``build_session`` returns a :class:`wam_policy.inference.session.OnlineSession` whose observation
pipeline also reads the measured root-link state of every live sample (QLM-Bench sends it with
``state_semantics`` = :data:`state.STATE_SEMANTICS`) and whose sampler is
:class:`sampler.ProprioceptionSampler`. Prompts are dispatched per episode from the instruction in
the reset request (ten tasks), exactly like the RGB-only teacher.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wam_policy.inference.loader import load_codec, load_prompts, load_teacher
from wam_policy.inference.sampler import SamplingConfig
from wam_policy.inference.session import LiveObservationPipeline, OnlineSession

from .sampler import ProprioceptionSampler
from .state import FIELDS, GROUND_Z, STATE_SEMANTICS, StateNormalizer, select_state


class ProprioceptionPipeline(LiveObservationPipeline):
    def __init__(self, codec, state_normalizer):
        super().__init__(codec)
        self.state_normalizer = state_normalizer

    def validate(self, packet, history, *, initial=False):
        ticks = super().validate(packet, history, initial=initial)
        if packet.get("state_semantics") != STATE_SEMANTICS:
            raise ValueError(f"Live observations must carry root-link state with semantics {STATE_SEMANTICS}")
        if any(set(sample["root_state"]) != set(FIELDS) for sample in packet["samples"]):
            raise ValueError("Every live sample needs the five measured root-link fields")
        return ticks

    def encode(self, packet):
        video = super().encode(packet)
        columns = {key: np.asarray([s["root_state"][key] for s in packet["samples"]]) for key in FIELDS}
        states = self.state_normalizer.encode(select_state(columns, ground_z=GROUND_Z))
        return video, torch.from_numpy(states.T.copy())[None, :, None, :, None]


class ProprioceptionSession(OnlineSession):
    def __init__(self, sampler, codec, prompt_encoder, state_normalizer, *, policy_seed=42, info=None):
        super().__init__(sampler, codec, prompt_encoder, policy_seed=policy_seed, info=info)
        self.pipeline = ProprioceptionPipeline(codec, state_normalizer)


def sampling_config(sampling, window_units=8):
    if isinstance(sampling, SamplingConfig):
        return sampling.validate()
    if isinstance(sampling, str):
        video, action = (int(x) for x in sampling.split("/"))
        return SamplingConfig(video, action, window_units=window_units).validate()
    return SamplingConfig(**{"window_units": window_units, **dict(sampling)}).validate()


def build_session(checkpoint, base_model, prompt_root=None, device="cuda", vae_device=None, sampling="25/50",
                  *, dtype="bfloat16", policy_seed=42, window_units=8):
    """OnlineSession for a released proprioception teacher directory (``wam_policy.json`` + ``transformer/``).

    ``prompt_root`` holds ``<task>/embedding.pt`` (cached T5); without it the frozen T5 of ``base_model``
    is loaded. ``sampling`` is a preset ('25/50' = paper setting), a dict or a ``SamplingConfig``.
    """
    model, action_normalizer, meta = load_teacher(checkpoint, device=device, dtype=dtype, attn_mode="torch")
    if meta.get("proprioception_dim") != 10 or getattr(model, "proprioception_dim", 0) != 10:
        raise ValueError(f"{checkpoint} is not a proprioception teacher (proprioception_dim must be 10)")
    state_normalizer = StateNormalizer.from_record(meta["proprioception_normalizer"])
    config = sampling_config(sampling, window_units)
    codec = load_codec(base_model, device=vae_device or device, dtype=dtype, rgb_profile=meta["rgb_profile"])
    prompts = load_prompts(prompt_root=prompt_root, base_model=base_model, device=device, dtype=dtype)
    info = {"policy": "stage1.1-ablation-proprioception-teacher", "checkpoint": str(checkpoint),
                "sampling": {"video_steps": config.video_steps, "action_steps": config.action_steps,
                             "window_units": config.window_units},
                "dtype": dtype, "device": str(device), "vae_device": str(vae_device or device),
                "runtime": {}}
    return ProprioceptionSession(ProprioceptionSampler(model, action_normalizer, config), codec, prompts,
                                 state_normalizer, policy_seed=policy_seed, info=info)


POLICY = "proprioception_teacher"


class _Handle:
    """Callable RPC handler with a ``session`` attribute (what ``inference.serve.RequestLog`` wraps)."""

    def __init__(self, session):
        self.session = session

    def __call__(self, request):
        return self.session.handle(request)


def serve(args):
    """Same flags and log-dir protocol as ``python -m wam_policy.inference.serve``."""
    from datetime import datetime, timezone
    from qlm_bench.live import RPCServer
    from wam_policy.inference.serve import RequestLog, _write
    sampling = {"video_steps": args.video_steps, "action_steps": args.action_steps}
    session = build_session(args.teacher, args.base_model, args.prompt_root, args.device, args.vae_device,
                            sampling, dtype=args.dtype, policy_seed=args.policy_seed)
    handle = _Handle(session)
    handler = RequestLog(handle, args.log_dir) if args.log_dir else handle
    server = RPCServer(handler, host=args.host, port=args.port, timeout_s=args.timeout_s)
    ready = {"ready": True, "address": list(server.address), "policy": POLICY,
             "started_at_utc": datetime.now(timezone.utc).isoformat()}
    if args.log_dir:
        _write(args.log_dir / "info.json", session.info)
        _write(args.log_dir / "ready.json", ready)
    print(json.dumps(ready), flush=True)
    try:
        server.serve_forever()
    finally:
        try:
            session.end("service_shutdown")
        finally:
            server.close()
    return 0


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--teacher", "--checkpoint", dest="teacher", type=Path, required=True,
                        help="exported proprioception teacher directory (wam_policy.json + transformer/)")
    parser.add_argument("--base-model", type=Path, required=True, help="LingBot-VA base snapshot (vae/ is used)")
    parser.add_argument("--prompt-root", type=Path,
                        help="<prompt_root>/<task>/embedding.pt; without it the frozen T5 of --base-model is loaded")
    parser.add_argument("--video-steps", type=int, default=25)
    parser.add_argument("--action-steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae-device", default=None, help="default: same as --device")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--policy-seed", type=int, default=42,
                        help="fallback seed; a reset request may carry extensions.policy_seed")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8862)
    parser.add_argument("--timeout-s", type=float, default=600.)
    parser.add_argument("--log-dir", type=Path, help="write ready.json, info.json and request/event logs here")
