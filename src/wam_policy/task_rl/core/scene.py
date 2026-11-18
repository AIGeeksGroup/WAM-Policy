"""Client side of the QLM-Bench Task RL scene worker (``python -m qlm_bench.task_rl.scene_worker``).

One worker is one CPU process (MuJoCo + RAMBO + the policy-camera renderer) serving one rollout stream over
length-prefixed JSON frames (4-byte big-endian length + ``qlm_bench.live.transport`` payload). No simulator is imported
here. Operations: reset, restore_state, execute (16 native9 commands), capture_state, persist_ring.

``Workers`` drives N workers in parallel (one per sampled row): the GPU process and its workers alternate (sampling,
then physics + rendering), so they do not compete for the process's cores.
"""
from __future__ import annotations

import os
import selectors
import struct
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SCHEMA = 'qlm-task-rl-scene-ipc-v1'
INFRA = (BrokenPipeError, ConnectionResetError, EOFError, TimeoutError)


class SceneClient:
    def __init__(self, python, qlm_src, controller, state_root, *, environment=None, timeout=600., gl='osmesa',
                 render_mode='full'):
        from qlm_bench.live.transport import MAX_FRAME_BYTES, dumps, loads
        self._dumps, self._loads, self._max = dumps, loads, MAX_FRAME_BYTES
        env = {**os.environ, **(environment or {}), 'PYTHONPATH': str(qlm_src), 'CUDA_VISIBLE_DEVICES': '',
               'MUJOCO_GL': gl, 'PYOPENGL_PLATFORM': gl, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
               'MKL_NUM_THREADS': '1'}
        self._args = [str(python), '-m', 'qlm_bench.task_rl.scene_worker', '--controller', str(controller),
                      '--state-root', str(state_root), '--render-mode', render_mode]
        self._env = env
        self.timeout = timeout
        self.process = subprocess.Popen(self._args, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.session, self.serial, self.chunk = None, 0, 0

    def _read(self, count, deadline):
        value, fd = bytearray(), self.process.stdout.fileno()
        with selectors.DefaultSelector() as sel:
            sel.register(fd, selectors.EVENT_READ)
            while len(value) < count:
                if not sel.select(max(0, deadline - time.monotonic())):
                    raise TimeoutError('QLM scene worker timeout')
                block = os.read(fd, count - len(value))
                if not block:
                    raise EOFError('QLM scene worker exited')
                value.extend(block)
        return bytes(value)

    def request(self, operation, payload):
        self.serial += 1
        raw = self._dumps({'schema': SCHEMA, 'id': f'{self.session}:{self.serial}', 'operation': operation,
                           'session': self.session, 'payload': payload})
        if len(raw) > self._max:
            raise ValueError('Worker request too large')
        self.process.stdin.write(struct.pack('!I', len(raw)) + raw)
        self.process.stdin.flush()
        deadline = time.monotonic() + self.timeout
        count = struct.unpack('!I', self._read(4, deadline))[0]
        if not 0 < count <= self._max:
            raise ValueError('Invalid worker response size')
        response = self._loads(self._read(count, deadline))
        if not response['ok']:
            raise RuntimeError(response['error_type'] + ': ' + response['error'])
        return response['result']

    def _new_session(self, name):
        self.session, self.chunk = f'{name}:{uuid.uuid4().hex}', 0

    def reset(self, case, model, name='s0'):
        self._new_session(name)
        return self.request('reset', {'case': str(case), 'model': str(model)})

    def restore_state(self, case, model, path, name='own'):
        self._new_session(name)
        return self.request('restore_state', {'case': str(case), 'model': str(model), 'path': str(path)})

    def execute(self, commands, ring=False):
        commands = np.asarray(commands, np.float64)
        result = self.request('execute', {'chunk': self.chunk, 'commands': commands.tolist(), 'ring': bool(ring)})
        self.chunk += 1
        return result

    def capture_state(self, path):
        return self.request('capture_state', {'path': str(path)})

    def persist_ring(self, chunk, path, origin=None):
        return self.request('persist_ring', {'chunk': int(chunk), 'path': str(path), 'origin': origin or {}})

    def restart(self):
        self.close()
        self.process = subprocess.Popen(self._args, env=self._env, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.session, self.chunk = None, 0

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()


class RecordingEnv:
    """SceneClient proxy keeping the privileged rows of the current session (restore/reset row, then one per command)."""

    def __init__(self, env):
        self.env, self.payloads = env, []

    def reset(self, *a, **k):
        r = self.env.reset(*a, **k); self.payloads = [r['privileged']]
        return r

    def restore_state(self, *a, **k):
        r = self.env.restore_state(*a, **k); self.payloads = [r['privileged']]
        return r

    def execute(self, commands, ring=False):
        r = self.env.execute(commands, ring=ring); self.payloads.append(r['privileged'])
        return r

    def rows(self):
        return np.asarray([row for p in self.payloads for row in p['rows']], np.float64)

    def columns(self):
        return [str(c) for c in self.payloads[0]['columns']]

    def __getattr__(self, name):
        return getattr(self.env, name)


class Workers:
    """N scene workers driven in parallel: a call fans out on a thread pool and returns when every worker answered."""

    def __init__(self, run, state_root, n):
        Path(state_root).mkdir(parents=True, exist_ok=True)
        controller = run.profile.controller()
        self.envs = [RecordingEnv(SceneClient(run.qlm_python, run.qlm_src, controller, state_root, gl=run.gl,
                                              render_mode=run.config['render_mode'],
                                              environment=run.scene_environment()))
                     for _ in range(n)]
        self.pool = ThreadPoolExecutor(max_workers=max(1, n))

    def start(self, rows, fn, *args):
        """Futures of fn(env[row], *[a[i] for a in args]) for the i-th row."""
        return [self.pool.submit(fn, self.envs[r], *(a[i] for a in args)) for i, r in enumerate(rows)]

    def run(self, rows, fn, *args):
        futures = self.start(rows, fn, *args)
        out, err = [], None
        for f in futures:            # wait for all of them before raising: no request is left in flight
            try:
                out.append(f.result())
            except Exception as exc:
                err = err or exc
        if err is not None:
            raise err
        return out

    def restart(self):
        for e in self.envs:
            e.restart()

    def close(self):
        for e in self.envs:
            e.close()
        self.pool.shutdown(wait=False)


def stack_views(per_row):
    """Codec input of several rows: {camera: [B, n, H, W, 3]} from one {camera: [n, H, W, 3]} per row."""
    return {k: np.stack([v[k] for v in per_row]) for k in per_row[0]}


def boundary_views(samples):
    from wam_policy.task_rl.core.history import chunk_views
    return chunk_views({s['tick']: s['rgb'] for s in samples}, [s['tick'] for s in samples])


def executed_of(result):
    return np.asarray([r['executed'] for r in result['commands']], np.float32)
