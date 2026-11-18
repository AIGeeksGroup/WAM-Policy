"""Explicit loss-unit DDP; variable-length environment/replay never enters DDP."""
from contextlib import nullcontext
from datetime import timedelta
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class RoundAborted(RuntimeError):
    pass


class DistributedContext:
    def __init__(self, *, device="cpu", initialize=False, timeout_s=600):
        self.device = torch.device(device)
        if initialize and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            local = int(os.environ.get("LOCAL_RANK", "0"))
            if self.device.type == "cuda":
                torch.cuda.set_device(local)
                self.device = torch.device("cuda", local)
            if not dist.is_initialized():
                dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo",
                                        timeout=timedelta(seconds=timeout_s))
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def wrap(self, unit):
        if self.world_size == 1:
            return unit
        return DistributedDataParallel(unit, device_ids=[self.device.index] if self.device.type == "cuda" else None,
                                       broadcast_buffers=False, find_unused_parameters=False)

    def accumulation(self, wrapper, final):
        return wrapper.no_sync() if not final and isinstance(wrapper, DistributedDataParallel) else nullcontext()

    def consensus(self, error=None, phase="phase"):
        flag = torch.tensor(int(error is None), dtype=torch.int32, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if flag.item() == 0:
            messages = [None] * self.world_size
            if self.world_size > 1:
                dist.all_gather_object(messages, str(error) if error is not None else None)
            else:
                messages[0] = str(error)
            raise RoundAborted(f"{phase}: {messages}") from error

    def finite_gradients(self, model, phase):
        invalid = [name for name, p in model.named_parameters()
                   if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
        self.consensus(FloatingPointError(f"nonfinite grads: {invalid}") if invalid else None, phase)

    def agree(self, value, phase):
        from arm_opsd.config import canonical_hash
        if len({canonical_hash(v) for v in self.gather(value)}) != 1:
            raise RoundAborted(f"{phase}: ranks disagree")

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        out = [None] * self.world_size
        dist.all_gather_object(out, value)
        return out

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()
