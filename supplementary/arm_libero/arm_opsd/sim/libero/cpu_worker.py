"""LIBERO CPU-only physics/rendering service; intentionally independent of wan_va."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import importlib.metadata
import os
from pathlib import Path
import random
import sys
import time

# Direct execution must resolve only this repository's lightweight evaluation package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    # Set numerical-library thread limits before importing NumPy, including direct CLI use.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LP_NUM_THREADS"):
        os.environ[key] = "1"
from arm_opsd.sim.libero.worker_protocol import (CAMERAS, PROTOCOL_VERSION, ProtocolError,
                                               digest, read_frame, write_frame)
from arm_opsd.sim.libero.rollout_contract import validate_profile, TRAIN_PROFILE, SETTLE_STEPS, EVAL_TIMESTEPS
import numpy as np

OBSERVER = "observer"


def configure_cpu():
    required = {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa",
                "LIBGL_ALWAYS_SOFTWARE": "true", "GALLIUM_DRIVER": "llvmpipe"}
    for key, value in required.items():
        if key in os.environ and os.environ[key] != value:
            raise RuntimeError(f"CPU worker requires {key}={value}; refusing fallback")
        os.environ[key] = value
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        os.environ[key] = ""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LP_NUM_THREADS"):
        os.environ[key] = "1"
    # Prevent llvmlite's LLVM from colliding with the Mesa llvmpipe LLVM runtime.
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    affinity = os.environ.get("ARM_OPSD_CPU_AFFINITY", "")
    if affinity:
        try:
            cores = {int(value.strip()) for value in affinity.split(",")}
        except ValueError as error:
            raise RuntimeError("ARM_OPSD_CPU_AFFINITY must be comma-separated CPU integers") from error
        allowed = os.sched_getaffinity(0)
        if not cores or any(core < 0 for core in cores) or not cores.issubset(allowed):
            raise RuntimeError(f"requested CPU affinity {sorted(cores)} outside allowed {sorted(allowed)}")
        os.sched_setaffinity(0, cores)
        if os.sched_getaffinity(0) != cores:
            raise RuntimeError("CPU affinity did not match requested assignment")


def extract_observation(obs):
    result = {}
    for source, target in zip(("agentview_image", "robot0_eye_in_hand_image"), CAMERAS):
        image = np.asarray(obs[source])
        if image.shape != (128, 128, 3) or image.dtype != np.uint8:
            raise ProtocolError(f"invalid {source}: {image.shape}/{image.dtype}")
        result[target] = np.ascontiguousarray(image[::-1])
    if "observer_image" in obs:
        # Diagnostic third-person view for visualization only; never a model camera.
        image = np.asarray(obs["observer_image"])
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ProtocolError(f"invalid observer image: {image.shape}/{image.dtype}")
        result[OBSERVER] = np.ascontiguousarray(image[::-1])
    return result


class LiberoBackend:
    def __init__(self, reset_attempts=5):
        configure_cpu()
        self.env = None
        self.bddl = None
        self.reset_attempts = int(reset_attempts)
        if self.reset_attempts < 1:
            raise ValueError("reset_attempts must be positive")
        self.gl = None
        self.observer = None

    def reset(self, case):
        from libero.libero.envs import OffScreenRenderEnv
        from robosuite.utils.errors import RandomizationError
        bddl = str(Path(case["bddl_file"]).resolve())
        if self.env is None or self.bddl != bddl:
            self.close()
            self.env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128,
                                          camera_widths=128, control_freq=20,
                                          robots=["Panda"], controller="OSC_POSE")
            self.bddl = bddl
        observer = case.get("observer_camera")
        if observer is not None and (not isinstance(observer.get("name"), str)
                                     or not all(type(observer.get(k)) is int and 16 <= observer[k] <= 1024
                                                for k in ("height", "width"))):
            raise ProtocolError("observer_camera requires name and integer height/width")
        self.observer = observer
        seed = int(case["reset_seed"])
        np.random.seed(seed)
        random.seed(seed)
        self.env.seed(seed)
        # Never call ControlEnv.reset(): its finally/continue suppresses exceptions.
        for attempt in range(self.reset_attempts):
            try:
                self.env.env.reset()
                break
            except RandomizationError:
                if attempt + 1 == self.reset_attempts:
                    raise RuntimeError("bounded LIBERO placement reset exhausted")
        if "init_state" not in case and not case.get("capture_random_init"):
            raise ProtocolError("case requires init_state or explicit capture_random_init")
        state = np.asarray(case.get("init_state", self.state()), dtype=np.float64)
        if state.ndim != 1 or not np.isfinite(state).all():
            raise ProtocolError("init_state must be a finite flat array")
        if "fixtures" in case:
            # LIBERO writes these to MjModel; flattened sim state omits them.
            saved = case["fixtures"]
            for name in ("body_pos", "body_quat"):
                target = getattr(self.env.sim.model, name)
                values = np.asarray(saved[name], dtype=np.float64)
                if values.shape != target.shape or not np.isfinite(values).all():
                    raise ProtocolError(f"invalid fixture {name} contract")
                target[:] = values
        self.env.set_init_state(state)
        fixtures = self.fixture_state()
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = self.step(np.zeros(7, dtype=np.float64))
        self.gl = self._gl_info()
        if "llvmpipe" not in self.gl["renderer"].lower():
            raise RuntimeError(f"CPU renderer verification failed: {self.gl}")
        # reset_key tells two physical resets apart (reset preparation refuses duplicate cases).
        reset_info = {"reset_seed": seed, "settle_steps": SETTLE_STEPS, "reset_success": bool(self.env.check_success()),
                      "init_state": state, "fixtures": fixtures,
                      "reset_key": digest({"state": self.state()[1:], "fixtures": fixtures})}
        return obs, reset_info

    def fixture_state(self):
        model = self.env.sim.model
        return {"body_pos": np.asarray(model.body_pos).copy(),
                "body_quat": np.asarray(model.body_quat).copy()}

    def state(self):
        return np.asarray(self.env.get_sim_state(), dtype=np.float64).copy()

    @property
    def timestep(self):
        return int(self.env.env.timestep)

    def step(self, command):
        # Time the original binding calls without altering controller/physics/RGB.
        # The residual includes OSC, observations and Python overhead.
        timings = {"physics_seconds": 0., "render_seconds": 0.}
        sim = self.env.sim
        original = {}
        for method, field in (("step", "physics_seconds"), ("render", "render_seconds")):
            call = getattr(sim, method, None)
            if call is None:
                continue
            original[method] = call
            def measured(*args, _call=call, _field=field, **kwargs):
                start = time.perf_counter()
                try:
                    return _call(*args, **kwargs)
                finally:
                    timings[_field] += time.perf_counter() - start
            setattr(sim, method, measured)
        try:
            obs, reward, done, info = self.env.step(command)
        finally:
            for method, call in original.items():
                setattr(sim, method, call)
        state, control = self.state(), np.asarray(self.env.sim.data.ctrl).copy()
        if not np.isfinite(state).all() or not np.isfinite(control).all():
            raise RuntimeError("non-finite MuJoCo state or OSC actuator control")
        if self.observer is not None:
            # Rendered after the measured step; physics state is not advanced by rendering.
            obs = dict(obs)
            obs["observer_image"] = np.asarray(self.env.sim.render(camera_name=self.observer["name"],
                height=self.observer["height"], width=self.observer["width"]))
        info = dict(info)
        info["cpu_diagnostics"] = {"state_finite": True, "osc_actuator_control_finite": True,
                                   "actuator_control_abs_max": float(np.max(np.abs(control))),
                                   **timings, "binding_timing_available": len(original) == 2}
        return obs, reward, done, info

    def _gl_info(self):
        from OpenGL import GL
        context = getattr(self.env.sim, "_render_context_offscreen", None)
        if context is not None and hasattr(context, "gl_ctx"):
            context.gl_ctx.make_current()
        def get(key):
            raw = GL.glGetString(key)
            if raw is None:
                raise RuntimeError("GL strings unavailable after actual rendering")
            return raw.decode("utf8", errors="replace")
        return {"vendor": get(GL.GL_VENDOR), "renderer": get(GL.GL_RENDERER),
                "version": get(GL.GL_VERSION)}

    def health(self):
        versions = {}
        for name in ("mujoco", "robosuite", "numpy", "PyOpenGL", "msgpack"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        return {"versions": versions, "gl": self.gl, "renderer_verified": self.gl is not None,
                "physics_device": "cpu", "render_device": "cpu", "backend": "osmesa",
                "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT"),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "pid": os.getpid(), "torch_imported": "torch" in sys.modules}

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


class WorkerService:
    """Single-threaded state machine; no physics advances between STEP requests."""
    def __init__(self, backend):
        self.backend = backend
        self.session_id = None
        self.episode_id = None
        self.chunk_index = 0
        self.policy_steps = 0
        self.terminal = True
        self.closed = False
        self.failed = False
        self.episode_ids = set()
        self.receipts = OrderedDict()
        self.request_signatures = {}
        self.chunk_ids = set()
        self.last_request = None

    def handle(self, request):
        request_id = request.get("request_id")
        signature = None
        try:
            if request.get("protocol") != PROTOCOL_VERSION:
                raise ProtocolError("protocol version mismatch")
            if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
                raise ProtocolError("request ID must be a nonempty string of at most 256 characters")
            session = request["session_id"]
            if not isinstance(session, str) or not session or len(session) > 256:
                raise ProtocolError("session ID must be a nonempty string of at most 256 characters")
            if self.session_id is None:
                self.session_id = session
            if session != self.session_id:
                raise ProtocolError("session mismatch")
            payload = request["payload"]
            if not isinstance(payload, dict):
                raise ProtocolError("request payload must be a mapping")
            # A repeated request ID returns the stored response once and never executes physics twice.
            signature = (request["op"], digest(payload))
            if request_id in self.request_signatures and self.request_signatures[request_id] != signature:
                raise ProtocolError("request ID reused with different operation/payload")
            if request_id in self.request_signatures and request_id not in self.receipts:
                raise ProtocolError("old request receipt evicted; request will not be executed again")
            if request_id in self.receipts:
                old_signature, response = self.receipts[request_id]
                if old_signature != signature:
                    raise ProtocolError("request ID reused with different operation/payload")
                return response
            self.request_signatures[request_id] = signature
            if self.closed:
                raise ProtocolError("worker already closed")
            result = self._dispatch(request["op"], payload)
            if request["op"] == "STEP_CHUNK":
                result.update(receipt_id=request_id, session_id=self.session_id)
            self.last_request = request_id
            response = {"ok": True, "request_id": request_id, "result": result}
        except Exception as error:
            response = {"ok": False, "request_id": request_id,
                        "error": f"{type(error).__name__}: {error}"}
        if signature is not None and self.request_signatures.get(request_id) == signature:
            self.receipts[request_id] = (signature, response)
            while len(self.receipts) > 8:
                self.receipts.popitem(last=False)
        return response

    def _dispatch(self, op, payload):
        if op == "HEALTH":
            return self.backend.health()
        if op == "STATUS":
            return {"session_id": self.session_id, "episode_id": self.episode_id,
                    "chunk_index": self.chunk_index, "policy_steps": self.policy_steps,
                    "terminal": self.terminal, "failed": self.failed, "last_request": self.last_request,
                    "timestep": self.backend.timestep if self.episode_id and not self.failed else None,
                    "state": self.backend.state() if self.episode_id and not self.failed else None}
        if op == "CLOSE":
            self.backend.close()
            self.closed = True
            return {"closed": True}
        if self.failed:
            raise ProtocolError("session failed after uncertain environment mutation; start a new worker/session")
        if op == "RESET":
            episode_id = payload["episode_id"]
            profile = validate_profile(payload.get("rollout_profile"), payload["case"].get("task_id"))
            if not self.terminal and getattr(self, "rollout_profile", {}).get("name") == TRAIN_PROFILE:
                raise ProtocolError("Cannot reset a live training episode")
            if not episode_id or episode_id in self.episode_ids:
                raise ProtocolError("RESET requires a fresh episode ID")
            self.episode_ids.add(episode_id)
            try:
                obs, reset_info = self.backend.reset(payload["case"])
                if reset_info.get("reset_success"):
                    raise ProtocolError("reset is already successful after settle; invalid case")
                observation = extract_observation(obs)
            except Exception:
                self.failed = True
                self.terminal = True
                raise
            self.episode_id = episode_id
            self.rollout_profile = profile
            self.chunk_index = 0
            self.policy_steps = 0
            self.terminal = False
            return {"episode_id": episode_id, "observation": observation,
                    "reset_info": reset_info, "timestep": self.backend.timestep,
                    "rollout_profile": profile}
        if op != "STEP_CHUNK":
            raise ProtocolError(f"unsupported operation: {op}")
        if payload["episode_id"] != self.episode_id or self.terminal:
            raise ProtocolError("STEP_CHUNK requires a live matching episode")
        chunk_id = payload["chunk_id"]
        if type(chunk_id) is not int or chunk_id != self.chunk_index:
            raise ProtocolError("chunk_id must equal the next nonnegative integer chunk index")
        if (self.episode_id, chunk_id) in self.chunk_ids:
            raise ProtocolError("chunk already executed under another request ID")
        commands = np.asarray(payload["actions"], dtype=np.float64)
        expected = 12 if self.chunk_index == 0 else 16
        if commands.shape != (expected, 7) or not np.isfinite(commands).all():
            raise ProtocolError(f"expected finite actions[{expected},7]")
        if type(payload["actor_version"]) is not int or payload["actor_version"] < 0:
            raise ProtocolError("invalid actor version")
        training = self.rollout_profile["name"] == TRAIN_PROFILE
        if ((training and self.chunk_index >= self.rollout_profile["max_joint_chunks"])
                or (not training and self.backend.timestep >= EVAL_TIMESTEPS)):
            raise ProtocolError("horizon exhausted before chunk dispatch")
        # Mark in-flight before any env.step: an exception cannot authorize replay.
        self.chunk_ids.add((self.episode_id, chunk_id))
        observations, applied, diagnostics = [], [], []
        success = False
        start_time = time.monotonic()
        try:
            for command in commands:
                obs, _, done, info = self.backend.step(command.copy())
                applied.append(command.copy())
                observations.append(extract_observation(obs))
                diagnostics.append(info.get("cpu_diagnostics", {}))
                self.policy_steps += 1
                if bool(done):
                    success = True
                    break
            final_timestep = self.backend.timestep
        except Exception:
            self.terminal = True  # Unknown execution outcome: caller must abort round.
            self.failed = True
            raise
        timeout = not success and (self.chunk_index + 1 >= self.rollout_profile["max_joint_chunks"]
                                   if training else final_timestep >= EVAL_TIMESTEPS)
        self.terminal = success or timeout
        result = {"episode_id": self.episode_id, "chunk_id": chunk_id,
                  "chunk_index": self.chunk_index, "actor_version": payload["actor_version"],
                  "observations": observations, "applied_actions": np.asarray(applied),
                  "cpu_diagnostics": diagnostics,
                  "requested_actions": commands.copy(), "executed_count": len(applied),
                  "execution_mask": np.arange(expected) < len(applied),
                  "success": success, "timeout": timeout, "terminal": self.terminal,
                  "terminal_reason": ("task_success" if success else
                                      "training_timeout" if timeout and training else
                                      "evaluation_timeout" if timeout else None),
                  "terminal_partial": self.terminal and len(applied) < expected,
                  "rollout_profile": self.rollout_profile,
                  "policy_steps": self.policy_steps, "timestep": final_timestep,
                  "cpu_physics_seconds": sum(d.get("physics_seconds", 0.) for d in diagnostics),
                  "cpu_render_seconds": sum(d.get("render_seconds", 0.) for d in diagnostics),
                  "cpu_step_render_seconds": time.monotonic() - start_time}
        self.chunk_index += 1
        return result


def isolate_protocol_stdout():
    """Return a private writer for the inherited stdout pipe and point fd 1 at stderr.

    MuJoCo, Mesa and other C libraries write to the process-wide fd 1 (``python -u``
    also makes C stdio unbuffered), which ``contextlib.redirect_stdout`` cannot
    intercept. After this call every Python or C write to stdout lands in the
    worker's stderr log; only the returned writer carries protocol frames. Must run
    before LIBERO/robosuite/MuJoCo/OpenGL are imported (they are imported lazily).
    """
    sys.stdout.flush()
    protocol_fd = os.dup(1)  # Non-inheritable: child processes cannot write frames.
    os.dup2(2, 1)
    return os.fdopen(protocol_fd, "wb")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-attempts", type=int, default=5)
    args = parser.parse_args()
    writer = isolate_protocol_stdout()
    reader = sys.stdin.buffer
    service = WorkerService(LiberoBackend(args.reset_attempts))
    try:
        while not service.closed:
            request = read_frame(reader)
            response = service.handle(request)
            write_frame(writer, response)
    except EOFError:
        pass
    finally:
        service.backend.close()


if __name__ == "__main__":
    main()
