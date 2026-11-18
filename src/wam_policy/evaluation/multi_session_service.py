"""Accelerated Standard100 evaluation of a student: one resident model, many concurrent QLM clients.

    python -m wam_policy.evaluation.multi_session_service --policy {opsd_student,mopd_student} \
        --teacher <stage1_1_teacher> --adapter <student dir> \
        --base-model <lingbot-va-base snapshot> --prompt-root <prompt_embeddings> \
        [--vae-device cuda] [--port 8870] [--log-dir <dir>]

This is the multi-session resident service of the paper's accelerated evaluation (1/2-NFE OPSD and
M-OPD students, VAE on the GPU). The model, student LoRA, normalizer and prompt embeddings load once
through :func:`wam_policy.inference.policies.build_policy`. Every QLM client session (``session_id``)
gets its own student sampler (own KV cache, own ``torch.Generator`` seeded by its reset), its own
streaming VAE state (shared frozen weights) and its own request de-duplication (the qlm-live-v1
``RPCServer.dispatch`` rules), so each episode runs the unchanged single-session code path.
Requests of all sessions execute one at a time on a single worker thread (no row batching: a
student prediction is short), so per-episode numerics are those of
``python -m wam_policy.inference.serve``; only the GPU is shared. With the VAE on the GPU instead of
the CPU, results may differ from a CPU-VAE run by floating-point rounding.

The teacher and specialists use the row-batched :mod:`wam_policy.evaluation.batched_service`;
:mod:`wam_policy.evaluation.run_parallel` launches either service plus the clients. This module
also holds the transport pieces both services share.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue
import socketserver
import threading
import time
import traceback

STUDENTS = ("opsd_student", "mopd_student")
LOCAL_HOSTS = ("127.0.0.1", "localhost")


# ----------------------------------------------------------------------------------- shared transport
def error_response(error):
    return {"ok": False, "error_type": type(error).__name__, "error": str(error)}


class Item:
    """One received request waiting for the worker thread."""
    __slots__ = ("request", "response", "done", "arrived")

    def __init__(self, request):
        self.request, self.response, self.done, self.arrived = request, None, threading.Event(), time.monotonic()


def session_dispatcher(handler, timeout_s=600.):
    """A socket-less ``qlm_bench.live.RPCServer``: its ``dispatch`` gives one session the exact
    de-duplication, epoch and poisoning rules of the single-session server."""
    from collections import OrderedDict
    from qlm_bench.live.transport import MAX_FRAME_BYTES, RPCServer
    d = RPCServer.__new__(RPCServer)
    d.handler, d.timeout_s, d.max_frame_bytes = handler, timeout_s, MAX_FRAME_BYTES
    d.max_cached_bytes, d.max_requests = 128 * 1024 * 1024, 10000
    d.identity, d._seen, d._responses, d._cached_bytes = None, {}, OrderedDict(), 0
    d._retired_sessions, d._poisoned = set(), False
    return d


class Slot:
    """Per-session state: the PolicyHandle, its dispatcher and a cleanup callback."""

    def __init__(self, handle, dispatcher, name, retire):
        self.handle, self.session, self.dispatcher, self.name = handle, handle.session, dispatcher, name
        self.retire = retire
        self.last_seen = time.monotonic()
        self.parked = False     # batched service: a fresh predict waits for the next row batch
        self.deferred = []      # batched service: later requests of this slot, accepted after that predict

    def is_retry(self, request):
        """True when ``request`` is a verbatim retry of a request this session already accepted."""
        from qlm_bench.live.transport import dumps
        ident = tuple(request[k] for k in ("session_id", "episode_id", "reset_epoch"))
        digest = hashlib.sha256(dumps(request)).hexdigest()
        return self.dispatcher._seen.get((ident, request["request_id"])) == digest


class MultiSessionService:
    """Serial multi-session service. ``factory(name) -> (PolicyHandle, retire)`` builds one session."""

    def __init__(self, factory, log_dir=None, *, idle_drop_s=120., abandon_s=600., rpc_timeout_s=600.):
        self.factory, self.log_dir = factory, (Path(log_dir) if log_dir else None)
        self.idle_drop_s, self.abandon_s, self.rpc_timeout_s = idle_drop_s, abandon_s, rpc_timeout_s
        self.slots, self.counter, self.queue = {}, 0, queue.Queue()
        self._logs = {}

    def log(self, name, value):
        if self.log_dir is None:
            return
        try:
            if name not in self._logs:
                self._logs[name] = (self.log_dir / name).open("a")
            self._logs[name].write(json.dumps(value, default=str) + "\n")
            self._logs[name].flush()
        except OSError:
            traceback.print_exc()

    # ---- per-session objects (worker thread only)
    def _new_slot(self, session_id):
        self.counter += 1
        name = f"session-{self.counter:05d}"
        handle, retire = self.factory(name)
        return Slot(handle, session_dispatcher(handle, self.rpc_timeout_s), name, retire)

    def _retire(self, session_id):
        slot = self.slots.pop(session_id)
        try:
            slot.retire()
        except Exception:
            traceback.print_exc()

    def _drop_idle(self):
        # Ended sessions after idle_drop_s; sessions abandoned mid-episode (client died without
        # `end`) after abandon_s of silence, far beyond any live client's gap between requests.
        now = time.monotonic()
        for sid, slot in list(self.slots.items()):
            if slot.parked or slot.deferred:
                continue
            quiet = now - slot.last_seen
            if (slot.session.history.state in ("ENDED", "ERROR") and quiet > self.idle_drop_s) or quiet > self.abandon_s:
                self._retire(sid)

    def _slot_for(self, request):
        """The session's slot; a reset that is not a verbatim retry starts a new slot.

        QLM-Bench gives every trial and attempt its own session id; a non-retry reset of a known
        session id (e.g. from an older client) still starts a fresh run of that session."""
        from qlm_bench.live.transport import validate_request
        validate_request(request)
        sid = request["session_id"]
        slot = self.slots.get(sid)
        if slot is not None and request["method"] == "reset" and not slot.is_retry(request):
            self._retire(sid)
            slot = None
        if slot is None:
            if request["method"] != "reset":
                raise ValueError("Unknown session; reset first")
            self._drop_idle()
            slot = self.slots[sid] = self._new_slot(sid)
        slot.last_seen = time.monotonic()
        return slot

    def _accept(self, item):
        import torch
        request = item.request
        try:
            slot = self._slot_for(request)
            started = time.perf_counter()
            item.response = slot.dispatcher.dispatch(request)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.log("requests.jsonl", {"t": round(time.time(), 3), "method": request["method"],
                                        "session": slot.name, "duration_s": round(time.perf_counter() - started, 4),
                                        "live": len(self.slots), "queued": self.queue.qsize()})
        except Exception as error:
            traceback.print_exc()
            item.response = error_response(error)
        item.done.set()

    def run(self):
        housekeeping = time.monotonic()
        while True:
            try:
                item = self.queue.get(timeout=30.)
            except queue.Empty:
                item = False
            if item is None:
                return
            if item:
                self._accept(item)
            if time.monotonic() - housekeeping > 10.:
                housekeeping = time.monotonic()
                try:
                    self._drop_idle()
                except Exception:
                    traceback.print_exc()

    def shutdown(self):
        for sid in list(self.slots):
            self._retire(sid)
        for stream in self._logs.values():
            stream.close()


def serve(service, host, port, timeout_s):
    """Threaded qlm-live-v1 socket front end; every request is handed to ``service.queue``."""
    from qlm_bench.live.transport import MAX_FRAME_BYTES, receive, send

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.settimeout(timeout_s)
            try:
                request = receive(self.request, MAX_FRAME_BYTES)
            except Exception as error:
                response = error_response(error)
            else:
                item = Item(request)
                service.queue.put(item)
                item.done.wait()
                response = item.response
            try:
                send(self.request, response, MAX_FRAME_BYTES)
            except (OSError, EOFError):
                pass

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
        request_queue_size = 128

    if host not in LOCAL_HOSTS:
        raise ValueError("Unauthenticated live RPC is localhost-only")
    server = Server((host, port), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .1}, daemon=True).start()
    return server


def service_parser(description, policies):
    """Model arguments of ``python -m wam_policy.inference.serve`` (same meaning) + transport."""
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, choices=policies)
    p.add_argument("--teacher", required=True, type=Path, help="Stage 1.1 teacher directory")
    p.add_argument("--adapter", type=Path, help="student (opsd/mopd) or specialist directory")
    p.add_argument("--base-model", required=True, type=Path, help="LingBot-VA base snapshot (vae/ is used)")
    p.add_argument("--prompt-root", type=Path,
                   help="<prompt_root>/<task>/embedding.pt; without it the frozen T5 of --base-model is loaded")
    p.add_argument("--tasks", help="comma-separated subset of the ten tasks (default: all)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--vae-device", default=None, help="default: same as --device (the accelerated setting)")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--policy-seed", type=int, default=42,
                   help="fallback seed; a reset request may carry extensions.policy_seed")
    p.add_argument("--host", default="127.0.0.1", choices=LOCAL_HOSTS)
    p.add_argument("--port", type=int, default=8870)
    p.add_argument("--timeout-s", type=float, default=600.)
    p.add_argument("--log-dir", type=Path, help="write ready.json, info.json and per-request logs here")
    return p


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def run_service(service, args, info):
    """Serve until interrupted; writes info.json / ready.json to ``--log-dir``."""
    server = serve(service, args.host, args.port, args.timeout_s)
    ready = {"ready": True, "address": list(server.server_address), "policy": args.policy,
             "started_at_utc": datetime.now(timezone.utc).isoformat()}
    if args.log_dir:
        write_json(args.log_dir / "info.json", info)
        write_json(args.log_dir / "ready.json", ready)
    print(json.dumps(ready), flush=True)
    try:
        service.run()
    finally:
        server.shutdown()
        service.shutdown()
    return 0


# ----------------------------------------------------------------------------------- student service
def student_factory(handle):
    """Per-session student samplers on the model that ``build_policy`` loaded (LoRA attached once)."""
    from wam_policy.common.video import DualViewVAE
    from wam_policy.inference.policies import PolicyHandle
    from wam_policy.inference.session import OnlineSession
    base = handle.session
    actor0, codec0 = base.sampler, base.pipeline.codec

    def factory(name):
        actor = actor0.spawn() if hasattr(actor0, 'spawn') else type(actor0)(
            actor0.runtime, actor0.normalizer, contract=actor0.contract, steps=actor0.steps)
        codec = DualViewVAE(codec0.vae, rgb_profile=codec0.rgb_profile)
        session = OnlineSession(actor, codec, base.prompt_encoder, policy_seed=base.policy_seed,
                                info=base.info)
        return PolicyHandle(session, actor.begin_episode), actor.clear
    return factory


def main(argv=None):
    args = service_parser(__doc__, STUDENTS).parse_args(argv)
    from wam_policy.inference.policies import build_policy
    if args.adapter is None:
        raise SystemExit("--adapter <student dir> is required")
    handle = build_policy(args.policy, teacher=args.teacher, base_model=args.base_model, adapter=args.adapter,
                          prompt_root=args.prompt_root, tasks=tuple(args.tasks.split(",")) if args.tasks else None,
                          device=args.device, vae_device=args.vae_device, dtype=args.dtype,
                          policy_seed=args.policy_seed)
    handle.session.info["service"] = {"kind": "multi_session", "row_batching": False}
    if args.log_dir:
        args.log_dir.mkdir(parents=True, exist_ok=True)
    service = MultiSessionService(student_factory(handle), args.log_dir, rpc_timeout_s=args.timeout_s)
    return run_service(service, args, handle.session.info)


if __name__ == "__main__":
    raise SystemExit(main())
