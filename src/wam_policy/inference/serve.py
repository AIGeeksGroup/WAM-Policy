"""Serve a WAM-Policy model over qlm-live-v1 (QLM-Bench Standard100 and the real robot).

    python -m wam_policy.inference.serve --policy {teacher,specialist,opsd_student,mopd_student} \
        --teacher <stage1_1_teacher> [--adapter <student or specialist dir>] \
        --base-model <lingbot-va-base snapshot> --prompt-root <prompt_embeddings> \
        [--video-steps V --action-steps A] [--vae-device cpu] [--port 8862] [--log-dir <dir>]

The server answers ``reset`` / ``predict`` / ``feedback`` / ``end`` requests of a QLM-Bench
client (``python -m qlm_bench.benchmark run --policy-port <port>``): each observation carries
two 640x360 RGB views and the task instruction; each prediction returns a 16x9 native9 command
chunk (base velocity, front-left foot target, zero force) covering 0.32 s.
One request is served at a time (one episode per server); the accelerated multi-session services
are in :mod:`wam_policy.evaluation`.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from wam_policy.inference.policies import POLICIES, build_policy


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


class RequestLog:
    """Optional JSONL log of requests (method, ids, duration, error) and session events."""

    def __init__(self, handle, directory):
        self.handle, self.directory = handle, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._events, self._cursor = None, 0

    def _append(self, name, value):
        with (self.directory / name).open("a") as stream:
            stream.write(json.dumps(value, default=str) + "\n")

    def _flush_events(self, request_id):
        session = self.handle.session
        if self._events is not session.events:
            self._events, self._cursor = session.events, 0
        for event in session.events[self._cursor:]:
            self._append("events.jsonl", {"request_id": request_id, **event})
            self._cursor += 1

    def __call__(self, request):
        self._flush_events("before")
        started, error = time.perf_counter(), None
        try:
            return self.handle(request)
        except BaseException as caught:
            error = f"{type(caught).__name__}: {caught}"
            raise
        finally:
            self._append("requests.jsonl", {
                "method": request.get("method"),
                **{k: request.get(k) for k in ("request_id", "session_id", "episode_id", "reset_epoch")},
                "duration_s": time.perf_counter() - started, "error": error})
            self._flush_events(request.get("request_id"))


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, choices=POLICIES)
    p.add_argument("--teacher", required=True, type=Path, help="Stage 1.1 teacher directory")
    p.add_argument("--adapter", type=Path, help="student (opsd/mopd) or specialist directory")
    p.add_argument("--base-model", required=True, type=Path, help="LingBot-VA base snapshot (vae/ is used)")
    p.add_argument("--prompt-root", type=Path,
                   help="<prompt_root>/<task>/embedding.pt; without it the frozen T5 of --base-model is loaded")
    p.add_argument("--tasks", help="comma-separated subset of the ten tasks (default: all)")
    p.add_argument("--video-steps", type=int, help="teacher/specialist only (default from metadata, 25)")
    p.add_argument("--action-steps", type=int, help="teacher/specialist only (default from metadata, 50)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--vae-device", default=None, help="default: same as --device; the paper's default "
                                                      "single-service evaluation used cpu")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--offload-blocks", type=int, default=0,
                   help="keep the first N of 30 transformer blocks in CPU memory (small GPUs)")
    p.add_argument("--policy-seed", type=int, default=42,
                   help="fallback seed; a reset request may carry extensions.policy_seed")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8862)
    p.add_argument("--timeout-s", type=float, default=600.)
    p.add_argument("--log-dir", type=Path, help="write ready.json, info.json and request/event logs here")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    from qlm_bench.live import RPCServer
    handle = build_policy(args.policy, teacher=args.teacher, base_model=args.base_model, adapter=args.adapter,
                          prompt_root=args.prompt_root,
                          tasks=tuple(args.tasks.split(",")) if args.tasks else None,
                          video_steps=args.video_steps, action_steps=args.action_steps, device=args.device,
                          vae_device=args.vae_device, dtype=args.dtype, policy_seed=args.policy_seed,
                          offload=args.offload_blocks)
    handler = RequestLog(handle, args.log_dir) if args.log_dir else handle
    server = RPCServer(handler, host=args.host, port=args.port, timeout_s=args.timeout_s)
    ready = {"ready": True, "address": list(server.address), "policy": args.policy,
             "started_at_utc": datetime.now(timezone.utc).isoformat()}
    if args.log_dir:
        _write(args.log_dir / "info.json", handle.session.info)
        _write(args.log_dir / "ready.json", ready)
    print(json.dumps(ready), flush=True)
    try:
        server.serve_forever()
    finally:
        try:
            handle.session.end("service_shutdown")
        finally:
            server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
