"""Bounded distributed lifecycle, collective errors and rank-local RNG."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import timedelta
import os
import platform
import random
import numpy as np
import torch
import torch.distributed as dist


class World:
    def __init__(self, device="cuda", *, initialize=True, timeout=1800):
        self.size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.size < 1 or not 0 <= self.rank < self.size:
            raise ValueError("Expected a positive world size and a valid rank")
        self.device = torch.device(device, self.local_rank) if device == "cuda" else torch.device("cpu")
        self.backend_kind = "rocm" if device == "cuda" and torch.version.hip else "cuda" if device == "cuda" else "cpu"
        if device == "cuda":
            if not torch.cuda.is_available() or torch.cuda.device_count() < self.size:
                raise RuntimeError("Insufficient visible GPU allocation")
            torch.cuda.set_device(self.device)
        if self.size > 1 and initialize and not dist.is_initialized():
            dist.init_process_group("nccl" if device == "cuda" else "gloo", timeout=timedelta(seconds=timeout))

    def barrier(self):
        if self.size > 1:
            dist.barrier()

    def all(self, condition):
        t = torch.tensor(int(bool(condition)), device=self.device)
        if self.size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())

    def reduce(self, values, *, maximum=False):
        t = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MAX if maximum else dist.ReduceOp.SUM)
            if not maximum:
                t /= self.size
        return t.cpu().tolist()

    def gather(self, value):
        if self.size == 1:
            return [value]
        result = [None] * self.size
        dist.all_gather_object(result, value)
        return result

    def broadcast(self, value):
        values = [value]
        if self.size > 1:
            dist.broadcast_object_list(values, src=0)
        return values[0]

    def leader(self, fn):
        result = error = None
        if self.rank == 0:
            try:
                result = fn()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        error = self.broadcast(error)
        if error:
            raise RuntimeError("rank0 operation failed: " + error)
        return self.broadcast(result)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "gpu": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["gpu"] is not None:
        if device.type != "cuda":
            raise ValueError("GPU RNG cannot resume on CPU")
        torch.cuda.set_rng_state(state["gpu"], device)


@contextmanager
def isolated_rng(device, seed):
    state = rng_state(device)
    try:
        seed_all(seed)
        yield
    finally:
        restore_rng(state, device)


def environment(world):
    result = {"python": platform.python_version(), "os": platform.platform(), "torch": torch.__version__,
              "hip": torch.version.hip, "cuda_build": torch.version.cuda, "backend": world.backend_kind,
              "world_size": world.size,
              "resources": {k: os.environ.get(k) for k in ("SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE")}}
    if world.device.type == "cuda":
        p = torch.cuda.get_device_properties(world.device)
        result.update(gpu=p.name, gpu_bytes=p.total_memory)
    import importlib.metadata
    for name in ("diffusers", "transformers", "lerobot", "triton", "pytorch-triton-rocm", "wandb"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result
