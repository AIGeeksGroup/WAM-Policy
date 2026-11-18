"""Whole-query DDP boundaries and collective failure decisions."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from wam_policy.common.distributed import World


class LossUnit(nn.Module):
    def __init__(self, model, compute):
        super().__init__()
        self.model = model
        self.compute = compute

    def forward(self, *args, **kwargs):
        # All trainable model calls happen inside this DDP-visible forward.
        return self.compute(self.model, *args, **kwargs)


def wrap(model, compute, world):
    unit = LossUnit(model, compute)
    if world.size > 1:
        unit = DistributedDataParallel(unit,
                device_ids=[world.local_rank] if world.device.type == "cuda" else None,
                broadcast_buffers=False, find_unused_parameters=False, gradient_as_bucket_view=True)
    return unit


def collective_call(world, fn):
    result, error = None, None
    try:
        result = fn()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if not world.all(error is None):
        raise RuntimeError("Collective abort: " + repr(world.gather(error)))
    return result


def backward_unit(unit, world, *, synchronize, scale, args=(), kwargs=None):
    context = (unit.no_sync() if isinstance(unit, DistributedDataParallel) and not synchronize
               else nullcontext())
    with context:
        loss, metrics, endpoint = collective_call(world, lambda: unit(*args, **(kwargs or {})))
        if not world.all(bool(torch.isfinite(loss))):
            raise FloatingPointError("Nonfinite loss on at least one rank")
        if not synchronize:
            # No reducer collectives run in no_sync: agree on local autograd
            # failures before any rank enters the next DDP forward/backward.
            collective_call(world, lambda: (loss * scale).backward())
        else:
            # Reducer failures inside synchronized backward require the process
            # group's bounded watchdog; don't interleave a second collective.
            (loss * scale).backward()
    return metrics, endpoint


def optimizer_step(model, optimizer, world, clip):
    parameters = [p for p in model.parameters() if p.requires_grad]
    valid = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters)
    if not world.all(valid):
        raise FloatingPointError("Missing/nonfinite LoRA gradients; no optimizer update")
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip)
    if not world.all(bool(torch.isfinite(norm))):
        raise FloatingPointError("Nonfinite global gradient norm")
    collective_call(world, optimizer.step)
    optimizer.zero_grad(set_to_none=True)
    import hashlib
    sample=torch.cat([torch.cat((p.detach().reshape(-1)[:16],p.detach().reshape(-1)[-16:])) for p in parameters]).float().cpu()
    summary=hashlib.sha256(sample.numpy().tobytes()).hexdigest()
    if len(set(world.gather(summary)))!=1:
        raise RuntimeError('LoRA parameter summaries diverged across ranks')
    return float(norm)
