"""Teacher/student roles and the student LoRA on the shared DiT (no PEFT dependency)."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from wam_policy.models.metadata import STUDENT_TARGETS as TARGETS


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1:
            raise ValueError("LoRA requires a Linear and positive rank")
        self.base = base.requires_grad_(False)
        self.scale = float(alpha) / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base = self.base(x)
        # FP32 masters, activation dtype controlled by the surrounding BF16 autocast.
        update = nn.functional.linear(nn.functional.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return base + (update * self.scale).to(base.dtype)


def inject_lora(model, rank=256, alpha=256):
    model.eval().requires_grad_(False)
    expected = {f"blocks.{i}.{suffix}" for i in range(len(model.blocks)) for suffix in TARGETS}
    modules = dict(model.named_modules())
    if not expected or any(name not in modules or not isinstance(modules[name], nn.Linear) for name in expected):
        raise ValueError("Shared DiT is missing an exact LoRA target")
    for name in sorted(expected):
        parent, leaf = name.rsplit(".", 1)
        model.get_submodule(parent)._modules[leaf] = LoRALinear(modules[name], rank, alpha)
    actual = {name for name, p in model.named_parameters() if p.requires_grad}
    if actual != {name + suffix for name in expected for suffix in (".lora_A", ".lora_B")}:
        raise ValueError("Unexpected trainable parameters")
    return trainable_manifest(model)


def trainable_manifest(model):
    return [{"name": canonical_name(n), "shape": list(p.shape), "dtype": str(p.dtype), "parameters": p.numel()}
            for n, p in model.named_parameters() if p.requires_grad]


def canonical_name(name):
    return name.replace("._checkpoint_wrapped_module", "")


def adapter_state(model):
    return {canonical_name(n): p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_adapter(model, state):
    expected = {canonical_name(n): p for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != set(expected):
        raise ValueError("Adapter key inventory mismatch")
    for n, p in expected.items():
        if state[n].shape != p.shape or not torch.isfinite(state[n]).all():
            raise ValueError("Adapter shape/nonfinite mismatch: " + n)
    with torch.no_grad():
        for n, p in expected.items():
            p.copy_(state[n].to(p))


@dataclass
class Roles:
    """The frozen teacher and the LoRA student."""
    teacher: nn.Module
    student: nn.Module
    # M-OPD: wam_policy.mopd.experts.ExpertBank on the teacher (task-routed specialist teachers), else None.
    experts: object = None

    def validate(self):
        teacher = {id(p) for p in self.teacher.parameters()}
        # A frozen base may be one set of tensors under both module trees (tied_copy); what a role
        # trains must be its own.
        if any(id(p) in teacher for p in self.student.parameters() if p.requires_grad):
            raise ValueError("Role models share mutable parameters")
        if any(p.requires_grad for p in self.teacher.parameters()):
            raise ValueError("Teacher must remain frozen")
        if not trainable_manifest(self.student):
            raise ValueError("The student requires an independent LoRA")
        return self


def tied_copy(model):
    """A second module tree over the same frozen parameter and buffer tensors: its own modules
    (adapters, KV caches) at no extra weight memory (config.shared_base)."""
    import copy
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Only a frozen model can be tied")
    memo = {id(t): t for t in (*model.parameters(), *model.buffers())}
    return copy.deepcopy(model, memo)


def build_roles(factory, config):
    teacher = factory().eval().requires_grad_(False)
    student = factory()
    # Same effective base; reproducible A initialization across ranks (B starts at zero, so the
    # student is functionally the teacher at update 0).
    devices = [next(student.parameters()).device] if next(student.parameters()).is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        inject_lora(student, config.lora_rank, config.lora_alpha)
    return Roles(teacher, student).validate()
