"""A supervised CPU process. A timed-out request is never resent automatically."""
from __future__ import annotations

import os
from pathlib import Path
import selectors
import struct
import subprocess
import sys
import time
import uuid

from arm_opsd.sim.libero.worker_protocol import MAX_FRAME_BYTES, PROTOCOL_VERSION, ProtocolError, pack, unpack


class WorkerFailure(RuntimeError):
    """The round must abort; execution may be uncertain after process loss."""


class WorkerClient:
    def __init__(self, python=sys.executable, worker_args=(), timeout=120.0,
                 env=None, stderr=None, session_id=None):
        self.timeout = float(timeout)
        self.session_id = session_id or uuid.uuid4().hex
        self.closed = False
        worker_env = dict(os.environ, **(env or {}))
        worker_env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa",
                          LIBGL_ALWAYS_SOFTWARE="true", GALLIUM_DRIVER="llvmpipe",
                          CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
                          ROCR_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                          OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", LP_NUM_THREADS="1",
                          NUMBA_DISABLE_JIT="1", PYTHONFAULTHANDLER="1")
        script = Path(__file__).with_name("cpu_worker.py")
        self.process = subprocess.Popen([str(python), "-u", str(script), *map(str, worker_args)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=stderr, env=worker_env, bufsize=0)

    def _read_exact(self, n, deadline):
        pieces = []
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while n:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise WorkerFailure("CPU worker timeout; request not resent; abort round")
                part = os.read(self.process.stdout.fileno(), n)
                if not part:
                    raise WorkerFailure(f"CPU worker exited ({self.process.poll()}); abort round")
                pieces.append(part)
                n -= len(part)
        return b"".join(pieces)

    def request(self, op, payload=None, request_id=None):
        if self.closed:
            raise WorkerFailure("CPU worker is closed")
        payload = payload or {}
        request_id = request_id or uuid.uuid4().hex
        request = {"protocol": PROTOCOL_VERSION, "session_id": self.session_id,
                   "request_id": request_id, "op": op, "payload": payload,
}
        try:
            raw = pack(request)
            if len(raw) > MAX_FRAME_BYTES:
                raise ProtocolError("IPC frame too large")
            self.process.stdin.write(struct.pack("!I", len(raw)) + raw)
            self.process.stdin.flush()
            deadline = time.monotonic() + self.timeout
            prefix = self._read_exact(4, deadline)
            length, = struct.unpack("!I", prefix)
            if length > MAX_FRAME_BYTES:
                # Text such as b'WARN' means non-protocol bytes reached the frame pipe.
                raise ProtocolError(f"IPC frame too large: length prefix {prefix!r}")
            response = unpack(self._read_exact(length, deadline))
            if response.get("request_id") != request_id:
                raise ProtocolError("mismatched response request ID")
            if not response.get("ok"):
                raise WorkerFailure(response.get("error", "worker rejected request"))
            return response["result"]
        except Exception:
            self.terminate()
            raise

    def health(self):
        return self.request("HEALTH")

    def reset(self, case, episode_id=None, *, rollout_profile=None):
        return self.request("RESET", {"case": case, "episode_id": episode_id or uuid.uuid4().hex,
                                      "rollout_profile": rollout_profile})

    def step_chunk(self, actions, *, episode_id, chunk_id, actor_version, request_id=None):
        return self.request("STEP_CHUNK", {"actions": actions, "episode_id": episode_id,
                            "chunk_id": chunk_id, "actor_version": actor_version}, request_id)

    def status(self):
        return self.request("STATUS")

    def terminate(self):
        if not self.closed:
            self.closed = True
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            for pipe in (self.process.stdin, self.process.stdout):
                pipe.close()

    def close(self):
        if not self.closed:
            try:
                self.request("CLOSE")
            finally:
                self.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
