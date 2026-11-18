"""Configuration of OPSD and its variants on the arm WAM (LingBot-VA, LIBERO-Long).

Four layers:
  ArmContract        the fixed LingBot-VA / LIBERO-Long observation, latent and action shapes;
  MethodConfig       what the student is trained on and how (the five students differ only here);
  ExperimentConfig   seed, optimizer, schedule, EMA, snapshots, held-out set;
  MachineConfig      paths and devices of one site (never part of a student).

A student config file (configs/students/*.json) holds the `method` and `experiment` sections; the
`machine` section comes from a site file (configs/site.example.json). ``ArmConfig.load(path, site=...)``
merges the two. Paper terms -> fields:
  OPSD (Ours)            data_mode=streaming,    offline_contexts_per_rank=3 (24 streaming + 24 cached, eta=0.5)
  Teacher cache only     data_mode=offline_only, offline_contexts_per_rank=6 (eta=1)
  Streaming only         data_mode=streaming,    offline_contexts_per_rank=0 (eta=0)
  Collected once         data_mode=collect_once, collect_once_rounds=209, replay_contexts_per_rank=6
  SFT only (no teacher)  objective=sft_fm on the official demonstrations
"""
from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from arm_opsd.sim.libero.rollout_contract import TRAIN_CHUNKS

# The official LingBot-VA LIBERO-Long weights (theta_0).
BASE_REPO = "robbyant/lingbot-va-posttrain-libero-long"
BASE_REVISION = "0e89d1e753019988aba484e8da2dc0810e264d9f"
BASE_COMPONENTS = ("transformer", "vae", "tokenizer", "text_encoder")
# Official LIBERO action normalization quantiles of the 7 active channels.
Q01 = (-.6589285731315613, -.84375, -.9375, -.12107142806053162,
       -.15964286029338837, -.26571428775787354, -1.)
Q99 = (.8999999761581421, .8544642925262451, .9375, .17142857611179352,
       .1842857152223587, .34392857551574707, 1.)

OBJECTIVES = ("regression", "sft_fm")
DATA_MODES = ("streaming", "offline_only", "collect_once")
PROFILES = ("full", "tiny")


def canonical_hash(value):
    """sha256 of canonical JSON; the seeding rules (episode order, generation seeds) are built on it."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ArmContract:
    """Tensor shapes of the official LIBERO-Long LingBot-VA; checked, never configured."""
    suite: str = "libero_10"
    task_order_index: int = 0
    control_hz: int = 20
    image_height: int = 128
    image_width: int = 128
    cameras: tuple = ("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb")
    latent_channels: int = 48
    latent_height: int = 8
    latent_width: int = 16
    frame_groups: int = 4
    action_dim: int = 30
    active_channels: tuple = tuple(range(7))
    actions_per_group: int = 4
    patch_size: tuple = (1, 2, 2)
    attention_window: int = 30
    video_shift: float = 5.0
    action_shift: float = .05
    video_cfg: float = 5.0
    action_cfg: float = 1.0
    horizon_protocol: str = "native_chunk800_v1"
    q01: tuple = Q01
    q99: tuple = Q99

    @property
    def state_shape(self):
        return (1, self.latent_channels, self.frame_groups, self.latent_height, self.latent_width)

    @property
    def action_shape(self):
        return (1, self.action_dim, self.frame_groups, self.actions_per_group, 1)

    @property
    def video_tokens(self):
        return self.frame_groups * self.latent_height * self.latent_width // 4

    @property
    def action_tokens(self):
        return self.frame_groups * self.actions_per_group

    @property
    def capacity(self):
        return (self.attention_window // 2) * (self.video_tokens + self.action_tokens)


@dataclass(frozen=True)
class MethodConfig:
    objective: str = "regression"
    # Student sampler (1 video / 2 action Euler steps) and the teacher's target sampler (20 / 50).
    state_steps: int = 1
    action_steps: int = 2
    teacher_video_steps: int = 20
    teacher_action_steps: int = 50
    lora_rank: int = 256
    lora_alpha: int = 256
    lora_dropout: float = 0.
    # Per rank and student update: fresh streaming contexts and teacher-cache draws.
    onpolicy_contexts_per_rank: int = 3
    offline_contexts_per_rank: int = 3
    # Collected once only: rounds of the one-time collection, and contexts drawn per rank and update from it.
    collect_once_rounds: int = 100
    replay_contexts_per_rank: int = 0
    data_mode: str = "streaming"
    student_updates: int = 300
    world_size: int = 8
    student_lr: float = 3e-5
    weight_decay: float = .01
    grad_clip: float = 1.
    huber_c: float = 1e-3
    action_loss_weight: float = 1.

    @property
    def collects(self):
        return self.data_mode != "offline_only"

    @property
    def collect_once_replay_per_rank(self):
        """Pooled contexts per rank and update after collection: defaults to the on-policy slot count."""
        return self.replay_contexts_per_rank or self.onpolicy_contexts_per_rank

    def batch_per_rank(self, *, collecting):
        """(onpolicy, replay, offline) loss units of one student update on one rank."""
        if self.data_mode == "streaming":
            return (self.onpolicy_contexts_per_rank, 0, self.offline_contexts_per_rank)
        if self.data_mode == "offline_only":
            return (0, 0, self.offline_contexts_per_rank)
        return (0, 0 if collecting else self.collect_once_replay_per_rank, self.offline_contexts_per_rank)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    adam_betas: tuple = (.9, .999)
    adam_epsilon: float = 1e-8
    # Linear warm-up over this many updates, then constant.
    student_lr_warmup_steps: Optional[int] = 20
    ema_decay: Optional[float] = .99
    ema_start_update: Optional[int] = 50
    checkpoint_every: int = 100
    retain_checkpoints: int = 2
    protected_checkpoints: tuple = ()
    # Raw (and, from ema_start_update, EMA) student exports at these updates; the last update is always exported.
    student_snapshots: tuple = tuple(range(100, 2001, 100))
    # Held-out regression contexts: drawn from the teacher cache, stratified by task and episode position.
    heldout_contexts: int = 64
    heldout_every: int = 50
    # Longest journal prefix a teacher-cache entry may carry (the whole training episode).
    offline_max_prefix: int = max(TRAIN_CHUNKS) - 1


@dataclass(frozen=True)
class MachineConfig:
    model_root: str = ""
    cpu_python: str = ""
    worker_config: str = ""
    reset_manifest: str = ""
    output_dir: str = ""
    # Teacher cache directory (manifest.json + entries/), from build-teacher-cache.
    offline_cache: str = ""
    # Root of robbyant/libero-long-lerobot (SFT only).
    sft_dataset: str = ""
    device: str = "cuda"
    network_dtype: str = "bfloat16"
    worker_timeout_s: float = 180.
    process_group_timeout_s: float = 1800.
    cpu_affinity: tuple = ()
    activation_checkpointing: bool = False
    # Rank-0 closed-loop prediction video at update 0 and every `visualization_every` updates (0: never).
    visualization_every: int = 400


MACHINE_PATHS = ("model_root", "cpu_python", "worker_config", "reset_manifest", "output_dir",
                 "offline_cache", "sft_dataset")
SECTIONS = {"native": ArmContract, "method": MethodConfig, "experiment": ExperimentConfig, "machine": MachineConfig}
def teacher_sampler_digest(native, video_steps, action_steps):
    """Teacher-cache reuse key: the target-generating sampler (steps, shifts, guidance, scheduler)."""
    return canonical_hash({"state_steps": int(video_steps), "action_steps": int(action_steps),
                           "video_shift": native.video_shift, "action_shift": native.action_shift,
                           "video_cfg": native.video_cfg, "action_cfg": native.action_cfg,
                           "scheduler": "FlowMatchScheduler(sigma_min=0,extra_one_step=True)+terminal_zero",
                           "integrator": "euler"})


@dataclass(frozen=True)
class ArmConfig:
    native: ArmContract = field(default_factory=ArmContract)
    method: MethodConfig = field(default_factory=MethodConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    machine: MachineConfig = field(default_factory=MachineConfig)
    # tiny (smoke runs only): the first SFT update draws the longest episode, and a checkpoint may omit
    # the scheduler state.
    profile: str = "full"

    @property
    def teacher_sampler_digest(self):
        return teacher_sampler_digest(self.native, self.method.teacher_video_steps, self.method.teacher_action_steps)

    def training_sections(self):
        """What a resumed run must share with the run that wrote the checkpoint."""
        return {"native": asdict(self.native), "method": asdict(self.method), "experiment": asdict(self.experiment)}

    def validate(self):
        if self.native != ArmContract():
            raise ValueError("The LingBot-VA / LIBERO-Long tensor contract is fixed")
        m, e = self.method, self.experiment
        if self.profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}")
        if m.objective not in OBJECTIVES or m.data_mode not in DATA_MODES:
            raise ValueError(f"objective in {OBJECTIVES}, data_mode in {DATA_MODES}")
        if min(m.world_size, m.student_updates, m.collect_once_rounds, m.state_steps, m.action_steps,
               m.teacher_video_steps, m.teacher_action_steps) < 1:
            raise ValueError("World size, update budget, collection rounds and sampler steps must be positive")
        if min(m.onpolicy_contexts_per_rank, m.offline_contexts_per_rank, m.replay_contexts_per_rank) < 0:
            raise ValueError("Context counts must be non-negative")
        if m.replay_contexts_per_rank and m.data_mode != "collect_once":
            raise ValueError("replay_contexts_per_rank applies only to data_mode collect_once (Collected once)")
        if m.objective == "regression" and sum(m.batch_per_rank(collecting=False)) == 0 and m.data_mode != "streaming":
            raise ValueError("A student update needs at least one loss unit")
        if m.objective == "sft_fm" and (m.data_mode != "offline_only" or m.offline_contexts_per_rank < 1):
            raise ValueError("SFT only draws offline_contexts_per_rank demonstrations per rank (data_mode offline_only)")
        if min(e.checkpoint_every, e.retain_checkpoints, e.heldout_every) < 1 or e.heldout_contexts < 0:
            raise ValueError("Positive checkpoint/held-out schedule required")
        if e.ema_start_update is not None and not 0 < (e.ema_decay or 0) < 1:
            raise ValueError("EMA decay must lie in (0, 1)")
        if self.machine.worker_timeout_s <= 0 or self.machine.process_group_timeout_s <= 0 or self.machine.visualization_every < 0:
            raise ValueError("Invalid machine resources")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value, *, strict=True):
        """strict=False ignores fields this code does not know (training configs inside previously
        published export.json files carry such fields)."""
        allowed = {f.name for f in fields(cls)}
        unknown = set(value) - allowed
        if unknown and strict:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        out = {}
        for name, typ in SECTIONS.items():
            data = dict(value.get(name, {}))
            known = {f.name for f in fields(typ)}
            if not strict:
                data = {k: v for k, v in data.items() if k in known}
            if set(data) - known:
                raise ValueError(f"Unknown {name} fields: {sorted(set(data) - known)}")
            for f in fields(typ):
                if isinstance(data.get(f.name), list):
                    data[f.name] = tuple(data[f.name])
            out[name] = typ(**data)
        profile = value.get("profile", "full")
        if not strict and profile not in PROFILES:
            profile = "tiny" if profile == "smoke" else "full"
        return cls(**out, profile=profile).validate()

    @classmethod
    def load(cls, path, *, site=None):
        """A config file, plus the `machine` section of a site file when given. Relative machine paths
        are resolved against the file that holds them."""
        path = Path(path).resolve()
        value = json.loads(path.read_text())
        value.pop("description", None)
        machine = _resolve_machine(value.get("machine", {}), path.parent)
        if site:
            site = Path(site).resolve()
            machine = {**machine, **_resolve_machine(json.loads(site.read_text()).get("machine", {}), site.parent)}
        value["machine"] = machine
        return cls.from_dict(value)


def _resolve_machine(machine, base):
    machine = dict(machine)
    for name in MACHINE_PATHS:
        if machine.get(name):
            # No symlink resolution: a venv's bin/python must stay the venv's interpreter.
            machine[name] = os.path.abspath(base / os.path.expanduser(os.path.expandvars(machine[name])))
    return machine
