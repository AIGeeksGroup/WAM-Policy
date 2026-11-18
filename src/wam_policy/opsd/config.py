"""Run configuration of the distillation engine (Stage 1.2 OPSD and Stage 2.2 M-OPD).

A config file holds the algorithm fields below plus ``resources`` (paths). Resource values may be
``${ENV_VAR}`` placeholders, expanded from the environment when the file is loaded; a resource the
config lists whose variable is unset or empty is an error. Optional resources (qlm_src, osmesa_lib,
model_python) are not listed in the paper configs: pass them with ``--resource KEY=VALUE``, which,
like ``--set FIELD=VALUE``, overrides the file. Keys starting with ``_`` are comments.

Resources:
  teacher              Stage 1.1 teacher directory (``wam_policy.json`` + ``transformer/``)
  base_model           LingBot-VA base snapshot (``vae/``) used for the video codec
  prompt_root          cached prompt embeddings ``<prompt_root>/<task>/embedding.pt``
  train_pool           QLM-Bench train reset pool ``manifest.json``
  qlm_python           interpreter of the QLM-Bench CPU environment (simulation workers)
  qlm_src              optional ``<QLM-Bench>/src`` added to PYTHONPATH of both sides (default: $QLM_BENCH_SRC)
  osmesa_lib           optional directory with libOSMesa.so for the workers' LD_LIBRARY_PATH (default: $QLM_OSMESA_LIB)
  model_python         optional learner interpreter (default: the current one)
  offline_cache        teacher target cache directory (``manifest.json`` + ``entries/``)
  initialized_student  optional student directory the LoRA starts from (M-OPD: the OPSD student)
  experts              M-OPD only: {task: specialist directory}
"""
from dataclasses import asdict, dataclass, field, fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re

DATA_MODES = ("streaming", "offline_only")
RENDER_MODES = ("full", "sampled")
PATH_RESOURCES = ("teacher", "base_model", "prompt_root", "train_pool", "qlm_src", "offline_cache",
                  "initialized_student", "osmesa_lib", "model_python")
# Fields that may change between a checkpoint and its resumed run (exact or bookkeeping only).
RESUMABLE = ("world_size", "checkpoint_every", "collection_timeout_s", "rpc_timeout_s", "collective_timeout_s",
             "protected_checkpoints", "visualization_every", "adapter_metrics_every", "history_reuse",
             "render_mode", "port_base", "worker_wall_timeout_s", "service_start_timeout_s", "cache_contexts",
             "shared_base", "resources")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@dataclass(frozen=True)
class Config:
    seed: int = 42
    world_size: int = 8
    # On-policy contexts per rank and update (one per stream chunk, three chunks per round).
    chunks_per_clip: int = 3
    student_updates: int = 300
    # Student (deployment) sampler: 1 video / 2 action Euler steps, CFG 1.
    video_steps: int = 1
    action_steps: int = 2
    window_units: int = 8
    lora_rank: int = 256
    lora_alpha: int = 256
    learning_rate: float = 3e-5
    weight_decay: float = .01
    grad_clip: float = 1.0
    lr_warmup_steps: int = 20
    ema_decay: float = .99
    ema_start_step: int = 50
    # Teacher target sampler: the multi-step ODE endpoint the student regresses onto.
    teacher_video_steps: int = 25
    teacher_action_steps: int = 50
    huber_c: float = 1e-3
    action_loss_weight: float = 1.0
    # Regression covers action dims [0, action_valid_dims): 6 masks the zeroed force dims.
    action_valid_dims: int = 6
    # streaming: chunks_per_clip on-policy + offline_contexts_per_rank cached contexts per rank;
    # offline_only: cached contexts only. The paper's context ratio eta = offline / (on-policy + offline).
    data_mode: str = "streaming"
    offline_contexts_per_rank: int = 3
    heldout_contexts: int = 64
    heldout_every: int = 50
    activation_checkpointing: bool = False
    checkpoint_every: int = 100
    protected_checkpoints: tuple = ()
    collection_timeout_s: float = 1800.
    rpc_timeout_s: float = 1800.
    collective_timeout_s: float = 1800.
    # Optional closed-loop prediction video on rank 0 every N updates (0 disables; needs imageio-ffmpeg).
    visualization_every: int = 0
    adapter_metrics_every: int = 10
    # Acceleration switches (off by default). history_reuse: restore the stored history KV of the
    # longest known prefix instead of replaying the whole prefix (value-identical). render_mode:
    # QLM worker render cadence, "sampled" renders only the frames the policy receives.
    history_reuse: bool = False
    render_mode: str = "full"
    # One frozen weight copy shared by teacher and student (each keeps its own modules and KV caches).
    shared_base: bool = False
    # Teacher target cache build size.
    cache_contexts: int = 5000
    # Launcher: RPC ports port_base + rank, worker wall-time, learner start-up deadline.
    port_base: int = 19380
    worker_wall_timeout_s: float = 86400.
    service_start_timeout_s: float = 3600.
    # Tiny synthetic model (smoke): prompt-width checks are skipped.
    tiny: bool = False
    resources: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.protected_checkpoints, list):
            object.__setattr__(self, "protected_checkpoints", tuple(self.protected_checkpoints))

    @property
    def streams(self):
        """Whether training collects on-policy contexts from QLM streams."""
        return self.data_mode == "streaming"

    @property
    def method(self):
        return "mopd" if self.resources.get("experts") else "opsd"

    def validate(self):
        integers = (self.seed, self.world_size, self.chunks_per_clip, self.student_updates, self.video_steps,
                    self.action_steps, self.window_units, self.lora_rank, self.lora_alpha, self.checkpoint_every,
                    self.teacher_video_steps, self.teacher_action_steps, self.heldout_every, self.cache_contexts,
                    self.port_base)
        if any(type(x) is not int for x in integers) or self.seed < 0 or any(x < 1 for x in integers[1:]):
            raise ValueError("Positive integer budgets and a nonnegative seed required")
        if self.world_size not in (1, 2, 4, 8) or self.window_units % 2 or not 4 <= self.window_units <= 64:
            raise ValueError("world_size must be 1/2/4/8 and window_units even in [4, 64]")
        if self.data_mode not in DATA_MODES or self.render_mode not in RENDER_MODES:
            raise ValueError(f"data_mode must be one of {DATA_MODES}, render_mode one of {RENDER_MODES}")
        counts = (self.offline_contexts_per_rank, self.heldout_contexts)
        if any(type(x) is not int or x < 0 for x in counts):
            raise ValueError("Nonnegative integer offline/held-out counts required")
        if self.data_mode == "offline_only" and self.offline_contexts_per_rank < 1:
            raise ValueError("offline_only trains on offline cache contexts only")
        floats = (self.learning_rate, self.weight_decay, self.grad_clip, self.huber_c, self.action_loss_weight,
                  self.collection_timeout_s, self.rpc_timeout_s, self.collective_timeout_s,
                  self.worker_wall_timeout_s, self.service_start_timeout_s)
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in floats):
            raise ValueError("Finite positive optimization, loss and timeout settings required")
        if not self.huber_c < 1:
            raise ValueError("Pseudo-Huber c must lie in (0, 1)")
        schedule = (self.lr_warmup_steps, self.ema_start_step, self.action_valid_dims,
                    self.visualization_every, self.adapter_metrics_every)
        if (any(type(x) is not int for x in schedule) or self.lr_warmup_steps < 0 or self.ema_start_step < 0 or
                self.action_valid_dims not in (6, 9) or self.visualization_every < 0 or self.adapter_metrics_every < 1):
            raise ValueError("Invalid warm-up/EMA/action-mask/diagnostic cadence")
        if not isinstance(self.ema_decay, float) or not 0 < self.ema_decay < 1:
            raise ValueError("EMA decay must lie in (0, 1)")
        if any(type(getattr(self, k)) is not bool for k in ("history_reuse", "shared_base", "tiny",
                                                            "activation_checkpointing")):
            raise ValueError("history_reuse/shared_base/tiny/activation_checkpointing are booleans")
        if any(type(x) is not int or x < 1 for x in self.protected_checkpoints):
            raise ValueError("protected_checkpoints are positive update counts")
        return self

    def record(self):
        value = asdict(self)
        value["protected_checkpoints"] = list(self.protected_checkpoints)
        return value

    def training_record(self):
        """What a resumed run must share with its checkpoint: the algorithm fields and the teacher target
        cache reuse key (teacher, train pool, samplers, specialist teachers; contents, not paths)."""
        from wam_policy.opsd.targets import cache_key
        value = {k: v for k, v in self.record().items() if k not in RESUMABLE}
        value["cache_key"] = cache_key(self)
        return value

    def need(self, *keys):
        """The named resources; a clear error for one that is unset."""
        missing = [k for k in keys if not self.resources.get(k)]
        if missing:
            raise ValueError("Unset resource(s) " + ", ".join(missing) + ": pass --resource KEY=VALUE or set the "
                             "environment variable named in the config file")
        values = [self.resources[k] for k in keys]
        return values[0] if len(values) == 1 else values

    @classmethod
    def load(cls, path, *, overrides=(), resources=(), experts=None, unused=()):
        """`unused`: resource keys the command does not read (dropped before placeholder expansion)."""
        value = {k: v for k, v in json.loads(Path(path).read_text()).items() if not k.startswith("_")}
        return cls.from_dict(value, overrides=overrides, resources=resources, experts=experts, unused=unused)

    @classmethod
    def from_dict(cls, value, *, overrides=(), resources=(), experts=None, unused=()):
        value = dict(value)
        known = {f.name for f in fields(cls)}
        for item in overrides:
            key, sep, text = item.partition("=")
            if not sep or key not in known or key == "resources":
                raise ValueError("--set takes FIELD=VALUE with a config field name: " + item)
            try:
                value[key] = json.loads(text)
            except json.JSONDecodeError:
                value[key] = text
        res = {k: v for k, v in (value.get("resources") or {}).items() if not k.startswith("_") and k not in unused}
        for item in resources:
            key, sep, text = item.partition("=")
            if not sep or not key:
                raise ValueError("--resource takes KEY=VALUE: " + item)
            res[key] = text
        if experts:
            res["experts"] = {**(res.get("experts") or {}), **experts}
        value["resources"] = resolve_resources(res)
        unknown = set(value) - known
        if unknown:
            raise ValueError("Unknown config fields: " + ", ".join(sorted(unknown)))
        return cls(**value).validate()

    def with_resources(self, **changes):
        return replace(self, resources={**self.resources, **changes})


_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand(value, name):
    """'${VAR}' -> environment value; an unset or empty variable is an error naming the resource."""
    if not isinstance(value, str):
        return value

    def sub(match):
        found = os.environ.get(match.group(1))
        if not found:
            flag = (f"--expert {name[8:]}=DIR" if name.startswith("experts.") else f"--resource {name}=VALUE")
            raise ValueError(f"resources.{name} needs ${{{match.group(1)}}}, which is unset: export it or pass {flag}")
        return found
    return _PLACEHOLDER.sub(sub, value)


def _absolute(value):
    # abspath keeps a venv's bin/python symlink (resolve() would select the system interpreter).
    return None if value in (None, "") else os.path.abspath(Path(value).expanduser())


def resolve_resources(resources):
    result = {}
    for key, value in resources.items():
        if key == "experts":
            result[key] = {task: _absolute(expand(path, "experts." + task)) for task, path in (value or {}).items()}
        elif key in PATH_RESOURCES or key == "qlm_python":
            result[key] = _absolute(expand(value, key))
        else:
            result[key] = expand(value, key)
    return {k: v for k, v in result.items() if v not in (None, {}, "")}
