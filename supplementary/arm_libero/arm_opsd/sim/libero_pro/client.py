"""Model-side handle on the LIBERO-Pro lease driver, shaped like the native CPU worker."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import uuid

from arm_opsd.sim.libero.worker_client import WorkerClient


class ProWorkerClient(WorkerClient):
    """Runs pro_worker.py in the LIBERO-Pro client environment over the native framed pipe.

    Server URL and API key are inherited from the environment, never passed as arguments.
    """
    def __init__(self, python, *, timeout=600.0, stderr=None, env=None):
        self.timeout = float(timeout)
        self.session_id = uuid.uuid4().hex
        self.closed = False
        worker_env = dict(os.environ, **(env or {}))
        for key in ("LIBERO_REMOTE_URL", "LIBERO_REMOTE_API_KEY"):
            if not worker_env.get(key):
                raise RuntimeError(f"{key} must be set (source the LIBERO-Pro client.env)")
        # The client process never touches the model GPU or the model environment's packages.
        worker_env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH="", PYTHONNOUSERSITE="1",
                          PYTHONFAULTHANDLER="1")
        script = Path(__file__).with_name("pro_worker.py")
        self.process = subprocess.Popen([str(python), "-u", str(script)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=stderr, env=worker_env, bufsize=0)

    def claim(self, run_id, runner_id):
        return self.request("CLAIM", {"run_id": run_id, "runner_id": runner_id})


class ProEpisode:
    """One leased episode exposing the reset/step_chunk surface used by evaluate_case.

    ``instruction`` follows the server (S21 at reset, D21 mid-episode); the prompt
    lookup reads it, so a changed instruction conditions the next generated chunk.
    """
    def __init__(self, client, claim):
        if claim.get("status") != "leased":
            raise ValueError("ProEpisode requires a leased claim")
        self.client, self.claim = client, claim
        self.job_id = claim["job_id"]
        self.session_id = claim.get("session_id") or claim["job_id"]
        self.instruction = claim["instruction"]
        self.reset_count = 0
        self.stats = {"server_step_seconds": [], "chunk_wall_seconds": [], "clipped_steps": 0,
                      "resized_cameras": sorted(claim.get("resized_cameras", [])), "instruction_changes": []}

    def reset(self, case, episode_id):
        if self.reset_count or case["case_id"] != self.claim["case_id"]:
            raise ValueError("A leased episode is reset exactly once, for its own case")
        self.reset_count += 1
        return {"episode_id": episode_id, "observation": self.claim["observation"], "reset_info": {}}

    def step_chunk(self, commands, *, episode_id, chunk_id, actor_version):
        receipt = self.client.request("STEP_CHUNK", {"job_id": self.job_id, "chunk_id": chunk_id,
                                                     "actions": commands})
        self.instruction = receipt["instruction"]
        self.stats["server_step_seconds"].extend(receipt["server_step_seconds"])
        self.stats["chunk_wall_seconds"].append(receipt["chunk_wall_seconds"])
        self.stats["clipped_steps"] += receipt["clipped_steps"]
        self.stats["resized_cameras"] = sorted(set(self.stats["resized_cameras"]) | set(receipt["resized_cameras"]))
        self.stats["instruction_changes"].extend(receipt["instruction_changes"])
        return receipt

    def complete(self):
        return self.client.request("COMPLETE", {"job_id": self.job_id})

    def abandon(self):
        return self.client.request("ABANDON", {"job_id": self.job_id})
