"""LIBERO-Pro 2.0 lease driver; runs in the official LIBERO-Pro client environment.

The official server owns BDDL, assets, initial state, perturbations, physics, EGL
rendering and success. This process only claims leased episodes and steps one
model chunk at a time, returning frames in the native worker convention.
Intentionally independent of wan_va and torch.
"""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import sys
import threading
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from arm_opsd.sim.libero.worker_protocol import (CAMERAS, PROTOCOL_VERSION, ProtocolError,
                                               read_frame, write_frame)

SOURCE_CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
IMAGE_SHAPE = (128, 128, 3)
TERMINAL_RUN = {"completed", "failed", "cancelled"}


def isolate_protocol_stdout():
    # Same contract as cpu_worker.isolate_protocol_stdout: frames on a private fd.
    sys.stdout.flush()
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(protocol_fd, "wb")


def convert_observation(observation):
    """Official raw robosuite frames -> native worker keys and orientation."""
    result, resized = {}, []
    for source, target in zip(SOURCE_CAMERAS, CAMERAS):
        image = np.asarray(observation[source])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ProtocolError(f"invalid {source}: {image.shape}/{image.dtype}")
        if image.shape != IMAGE_SHAPE:
            # S16 may change resolution; the model input contract stays 128x128.
            import cv2
            image = cv2.resize(image, IMAGE_SHAPE[1::-1], interpolation=cv2.INTER_AREA)
            resized.append(source)
        result[target] = np.ascontiguousarray(image[::-1])
    return result, resized


class Heartbeat:
    def __init__(self, lease):
        self.lease, self.failure = lease, None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.wait(self.lease.heartbeat_interval_seconds):
            try:
                self.lease.heartbeat()
            except BaseException as error:  # Reported on the next request.
                self.failure = error
                return

    def check(self):
        if self.failure is not None:
            raise RuntimeError(f"lease heartbeat failed: {self.failure!r}")

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)


class ProService:
    def __init__(self, client):
        self.client = client
        self.lease = self.heartbeat = None
        self.job_id = None
        self.chunk_index = 0
        self.policy_steps = 0
        self.terminal = False
        self.instruction = None
        self.low = self.high = None
        self.closed = False

    def handle(self, request):
        try:
            if request.get("protocol") != PROTOCOL_VERSION:
                raise ProtocolError("protocol version mismatch")
            result = self.dispatch(request["op"], request["payload"])
            return {"request_id": request["request_id"], "ok": True, "result": result}
        except Exception as error:
            return {"request_id": request.get("request_id"), "ok": False, "error": repr(error)}

    def dispatch(self, op, payload):
        if op == "HEALTH":
            versions = {}
            for name in ("libero-remote", "libero", "mujoco", "robosuite", "numpy", "msgpack"):
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    versions[name] = None
            return {"python": sys.version.split()[0], "versions": versions,
                    "server": self.client._transport.get_json("/v1/health"),
                    "server_version": self.client._transport.get_json("/v1/version")}
        if op == "CLOSE":
            self.release(abandon=True)
            self.closed = True
            return {"closed": True}
        if op == "CLAIM":
            return self.claim(payload["run_id"], payload["runner_id"])
        if op == "STEP_CHUNK":
            return self.step_chunk(payload)
        if op == "COMPLETE":
            self.require_job(payload)
            if not self.terminal:
                raise ProtocolError("COMPLETE requires a terminal episode")
            self.heartbeat.check()
            return self.release(abandon=False)
        if op == "ABANDON":
            self.require_job(payload)
            return self.release(abandon=True)
        raise ProtocolError(f"unsupported operation: {op}")

    def require_job(self, payload):
        if self.lease is None or payload.get("job_id") != self.job_id:
            raise ProtocolError("operation requires the live leased job")

    def claim(self, run_id, runner_id):
        if self.lease is not None:
            raise ProtocolError("previous lease is still open")
        snapshot = self.client.get_run(run_id)
        if snapshot["status"] in TERMINAL_RUN:
            return {"status": "done", "run": snapshot}
        lease = self.client.claim(run_id, runner_id)
        if lease is None:
            return {"status": "queued", "retry_after_seconds": self.client.last_retry_after_seconds or 1.0}
        self.lease, self.heartbeat = lease, Heartbeat(lease)
        self.job_id, self.chunk_index, self.policy_steps, self.terminal = lease.job_id, 0, 0, False
        observation, info = lease.initial_observation()
        schema = dict(info["action_schema"])
        self.low = np.asarray(schema["low"], dtype=np.float64)
        self.high = np.asarray(schema["high"], dtype=np.float64)
        self.instruction = str(info["language_instruction"])
        frames, resized = convert_observation(observation)
        return {"status": "leased", "run_id": lease.run_id, "job_id": lease.job_id,
                "job_index": lease.job_index, "lease_id": lease.lease_id,
                "episode_spec": lease.episode_spec,
                "action_schema": schema, "instruction": self.instruction,
                "observation_schema_version": info.get("observation_schema_version"),
                "observation": frames, "resized_cameras": resized,
                "session_id": info.get("session_id"), "episode_id": info.get("episode_id")}

    def step_chunk(self, payload):
        self.require_job(payload)
        if self.terminal:
            raise ProtocolError("STEP_CHUNK after a terminal step")
        if payload.get("chunk_id") != self.chunk_index:
            raise ProtocolError("chunk_id must equal the next chunk index")
        commands = np.asarray(payload["actions"], dtype=np.float64)
        expected = 12 if self.chunk_index == 0 else 16
        if commands.shape != (expected, 7) or not np.isfinite(commands).all():
            raise ProtocolError(f"expected finite actions[{expected},7]")
        observations, resized, instructions, step_seconds = [], set(), [], []
        clipped_steps, success, terminated, truncated, executed = 0, False, False, False, 0
        start = time.monotonic()
        for command in commands:
            self.heartbeat.check()
            # robosuite OSC clips inputs to the same bounds; the server rejects them instead.
            action = np.clip(command, self.low, self.high)
            clipped_steps += int(not np.array_equal(action, command))
            observation, _, terminated, truncated, info = self.lease.step(action)
            executed += 1
            self.policy_steps += 1
            frames, names = convert_observation(observation)
            observations.append(frames)
            resized.update(names)
            step_seconds.append(float(info.get("server_step_seconds", 0.)))
            instruction = str(info.get("language_instruction", self.instruction))
            if instruction != self.instruction:
                instructions.append({"policy_step": self.policy_steps, "instruction": instruction})
                self.instruction = instruction
            success = bool(info.get("success", False))
            if terminated or truncated:
                break
        self.terminal = bool(terminated or truncated)
        result = {"job_id": self.job_id, "chunk_id": self.chunk_index, "receipt_id": f"{self.job_id}:{self.chunk_index}",
                  "observations": observations, "applied_actions": commands[:executed].copy(),
                  "requested_actions": commands.copy(), "executed_count": executed,
                  "clipped_steps": clipped_steps, "resized_cameras": sorted(resized),
                  "success": success, "timeout": bool(truncated and not success), "terminal": self.terminal,
                  "terminal_reason": ("task_success" if success else "evaluation_timeout" if truncated
                                      else "terminated" if terminated else None),
                  "policy_steps": self.policy_steps, "instruction": self.instruction,
                  "instruction_changes": instructions, "server_step_seconds": step_seconds,
                  "cpu_physics_seconds": 0., "cpu_render_seconds": 0.,
                  "chunk_wall_seconds": time.monotonic() - start}
        self.chunk_index += 1
        return result

    def release(self, *, abandon):
        if self.lease is None:
            return None
        lease, heartbeat = self.lease, self.heartbeat
        self.lease = self.heartbeat = self.job_id = None
        heartbeat.stop()
        return lease.abandon() if abandon else lease.complete()


def main():
    writer = isolate_protocol_stdout()
    from libero_remote import RemoteRunClient
    url, key = os.environ.get("LIBERO_REMOTE_URL"), os.environ.get("LIBERO_REMOTE_API_KEY")
    if not url or not key:
        raise SystemExit("LIBERO_REMOTE_URL and LIBERO_REMOTE_API_KEY are required in the environment")
    service = ProService(RemoteRunClient(url, timeout=120.0, api_key=key))
    reader = sys.stdin.buffer
    try:
        while not service.closed:
            write_frame(writer, service.handle(read_frame(reader)))
    except EOFError:
        pass
    finally:
        try:
            service.release(abandon=True)
        finally:
            service.client.close()


if __name__ == "__main__":
    main()
