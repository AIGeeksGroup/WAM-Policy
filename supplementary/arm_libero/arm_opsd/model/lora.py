"""Minimal explicit LoRA for the six native shared-block Linear targets."""
import math
from collections.abc import Iterator

import torch
from torch import nn
from torch.nn import functional as F


TARGETS = ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
           "ffn.net.0.proj", "ffn.net.2")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=256, alpha=256, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRA target must be torch.nn.Linear")
        if rank <= 0 or alpha <= 0 or dropout != 0.0:
            raise ValueError("LoRA requires positive rank/alpha and dropout=0")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = int(rank), float(alpha)
        self.scale = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device,
                                               dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device,
                                               dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, value):
        base = self.base(value)
        # Autocast is intentionally disabled for FP32 master adapters. Convert
        # only the result back to the native activation dtype.
        with torch.autocast(device_type=value.device.type, enabled=False):
            delta = F.linear(F.linear(value.float(), self.lora_A), self.lora_B)
        return base + (delta * self.scale).to(base.dtype)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # model.to(BF16) may be called after injection; adapter masters stay FP32.
        for parameter in (self.lora_A, self.lora_B):
            parameter.data = parameter.data.float()
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.float()
        return result


def inject_lora(model: nn.Module, rank=256, alpha=256, dropout=0.0) -> tuple[str, ...]:
    if not hasattr(model, "blocks") or not len(model.blocks):
        raise ValueError("expected a native shared-block transformer")
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("LoRA is already installed")
    model.requires_grad_(False)
    names = []
    for index, block in enumerate(model.blocks):
        for target in TARGETS:
            parent_path, name = target.rsplit(".", 1)
            parent = block.get_submodule(parent_path)
            original = parent.get_submodule(name)
            setattr(parent, name, LoRALinear(original, rank, alpha, dropout))
            names.append(f"blocks.{index}.{target}")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected = {name + suffix for name in names for suffix in (".lora_A", ".lora_B")}
    if trainable != expected:
        raise RuntimeError("trainable parameter manifest differs from exact LoRA targets")
    model.eval()
    return tuple(names)


def lora_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone()
            for name, value in model.named_parameters()
            if name.endswith((".lora_A", ".lora_B"))}


def load_adapter_state_dict(model: nn.Module, state, strict=True):
    parameters = dict(model.named_parameters())
    expected = {name for name in parameters if name.endswith((".lora_A", ".lora_B"))}
    missing, unexpected = expected - set(state), set(state) - expected
    if strict and (missing or unexpected):
        raise ValueError(f"adapter keys differ: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    # Validate all shapes before changing any parameter.
    for name in expected & set(state):
        if state[name].shape != parameters[name].shape:
            raise ValueError(f"adapter shape mismatch: {name}")
    with torch.no_grad():
        for name in expected & set(state):
            parameters[name].copy_(state[name].to(parameters[name]))
    return {"missing_keys": sorted(missing), "unexpected_keys": sorted(unexpected)}

