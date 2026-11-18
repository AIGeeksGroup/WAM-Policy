"""Task-routed specialist teachers (Stage 2.2 M-OPD, routed teacher pi_M).

A specialist teacher is the frozen Stage 1.1 teacher plus one task's Stage 2.1 low-rank adapter on
ten Linear layers of every DiT block (``wam_policy.models.metadata.SPECIALIST_TARGETS``). The teacher
holds the adapters of all expert tasks side by side; every context uses the adapter of its own task,
or none, for all of its forwards (history replay, plan, action), so a batch and the teacher actor may
mix tasks. A task without a specialist is taught by the plain teacher, as in OPSD.

``resources['experts'] = {task: <specialist dir>}``; a specialist directory holds ``wam_policy.json``
(kind specialist) and ``adapter.pt`` = {'lora': {'<i>.old_A', '<i>.old_B'}}, i = adapted Linear in
module order. The student LoRA (six targets, no attn2) is independent of these adapters.
"""
import torch
from torch import nn
from wam_policy.common.tasks import TASKS
from wam_policy.models import metadata
from wam_policy.models.metadata import SPECIALIST_TARGETS as EXPERT_TARGETS
from wam_policy.opsd.config import digest

# 'old': the adapters after a round's EMA (what Stage 2.1 evaluates); 'current': that round's trained ones.
KINDS = {"old": ("old_A", "old_B"), "current": ("lora_A", "lora_B")}


class ExpertLinear(nn.Module):
    """A frozen Linear plus frozen per-task low-rank updates; the bank's switch names the active task."""

    def __init__(self, base, switch):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("Expert target must be a Linear")
        self.base = base.requires_grad_(False)
        self.switch, self.scales = switch, {}

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def put(self, task, a, b, scale):
        w = self.base.weight
        if tuple(a.shape) != (a.shape[0], w.shape[1]) or tuple(b.shape) != (w.shape[0], a.shape[0]):
            raise ValueError("Specialist adapter shape does not match its Linear")
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise FloatingPointError("Nonfinite specialist adapter")
        # FP32 masters; the surrounding BF16 autocast sets the activation dtype.
        self.register_buffer("A_" + task, a.detach().to(w.device, torch.float32).clone(), persistent=False)
        self.register_buffer("B_" + task, b.detach().to(w.device, torch.float32).clone(), persistent=False)
        self.scales[task] = float(scale)

    def forward(self, x):
        y = self.base(x)
        task = self.switch["task"]
        if task is None:
            return y
        a, b = getattr(self, "A_" + task), getattr(self, "B_" + task)
        update = nn.functional.linear(nn.functional.linear(x.to(a.dtype), a), b)
        return y + (update * self.scales[task]).to(y.dtype)


class ExpertBank:
    """The teacher's specialist adapters. `select(task)` activates that task's adapter, or none."""

    def __init__(self, model):
        if any(isinstance(m, ExpertLinear) for m in model.modules()):
            raise ValueError("Specialist adapters already installed")
        for block in model.blocks:
            for name in EXPERT_TARGETS:
                if not isinstance(block.get_submodule(name), nn.Linear):
                    raise ValueError("Teacher is missing a specialist target: " + name)
        self.switch = {"task": None}
        for block in model.blocks:
            for name in EXPERT_TARGETS:
                parent, leaf = name.rsplit(".", 1)
                module = block.get_submodule(parent)
                # In-place replacement keeps the module order, which indexes the adapter file.
                module._modules[leaf] = ExpertLinear(module._modules[leaf], self.switch)
        self.layers = [m for m in model.modules() if isinstance(m, ExpertLinear)]
        self.meta = {}

    @property
    def tasks(self):
        return tuple(sorted(self.meta))

    def load(self, task, directory):
        meta = read_specialist(task, directory)
        if task in self.meta:
            raise ValueError("Duplicate expert task: " + task)
        state = torch.load(metadata.weights(directory, meta), map_location="cpu", weights_only=True)
        state = state.get("lora", state)
        ka, kb = KINDS[meta["adapter_keys"]]
        rank, alpha = meta["lora"]["rank"], meta["lora"]["alpha"]
        if len([k for k in state if k.endswith("." + ka)]) != len(self.layers):
            raise ValueError(f"Specialist {directory} must hold {ka}/{kb} adapters of {len(self.layers)} layers")
        for i, layer in enumerate(self.layers):
            a, b = state[f"{i}.{ka}"], state[f"{i}.{kb}"]
            if a.shape[0] != rank:
                raise ValueError(f"Specialist {directory} adapter rank differs from its metadata")
            layer.put(task, a, b, alpha / rank)
        self.meta[task] = meta

    def select(self, task):
        self.switch["task"] = task if task in self.meta else None

    @property
    def active(self):
        return self.switch["task"]


def read_specialist(task, directory):
    """Specialist metadata of `directory`, which must belong to `task` and adapt SPECIALIST_TARGETS."""
    if task not in TASKS:
        raise ValueError("Unknown expert task: " + str(task))
    meta = metadata.read(directory, "specialist")
    if meta["task"] != task:
        raise ValueError(f"--expert {task}={directory}: that specialist was trained on {meta['task']}")
    if tuple(meta["lora"]["targets"]) != EXPERT_TARGETS or meta["adapter_keys"] not in KINDS:
        raise ValueError(f"Specialist {directory} must adapt {EXPERT_TARGETS} with old/current adapter keys")
    return meta


def expert_key(config):
    """Path-free key of the configured specialist teachers ({} for OPSD); part of the target cache
    reuse key, since an expert task's targets depend on its adapter."""
    experts = config.resources.get("experts") or {}
    return {task: digest(read_specialist(task, path)) for task, path in sorted(experts.items())}


def routing_table(config):
    """{task: {'specialist': dir, 'round': ...}} of the configured specialist teachers (validated)."""
    return {task: {"specialist": path, "round": read_specialist(task, path).get("round")}
            for task, path in sorted((config.resources.get("experts") or {}).items())}


def install_experts(model, config):
    """ExpertBank with every configured specialist on this (teacher) model, or None without experts."""
    experts = config.resources.get("experts") or {}
    if not experts:
        return None
    bank = ExpertBank(model)
    for task in sorted(experts):
        bank.load(task, experts[task])
    return bank


def expert_arguments(values):
    """['TASK=DIR', ...] -> {task: dir}."""
    experts = {}
    for value in values or ():
        task, sep, directory = value.partition("=")
        if not (sep and task and directory) or task in experts:
            raise ValueError("--expert takes TASK=<specialist dir>, once per task: " + value)
        experts[task] = directory
    return experts
