"""Student export in the release ``student`` format and its loader.

A student directory holds ``wam_policy.json`` (kind student, method opsd|mopd) and
``student_adapter.pt`` ({"blocks.<i>.<target>.lora_A|lora_B": FP32 tensor}); its base weights are
the Stage 1.1 teacher. ``load_student`` attaches such an adapter to a model that already carries
the injected student LoRA (``models.inject_lora``).
"""
from pathlib import Path
import shutil
import tempfile
import torch
from wam_policy.models import metadata
from wam_policy.opsd.models import TARGETS, load_adapter

VARIANTS = ("raw", "ema")


def publish_dir(target, writer):
    """Write a directory atomically: staging next to the target, then one rename."""
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=target.parent))
    try:
        writer(staging)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def student_metadata(config, *, variant, updates):
    return {"kind": "student", "method": config.method,
            "lora": {"rank": config.lora_rank, "alpha": config.lora_alpha, "targets": list(TARGETS)},
            "sampling": {"video_steps": config.video_steps, "action_steps": config.action_steps,
                         "window_units": config.window_units},
            "variant": variant, "updates": updates,
            "ema": {"decay": config.ema_decay, "start_step": config.ema_start_step},
            "notes": {"data_mode": config.data_mode, "offline_contexts_per_rank": config.offline_contexts_per_rank,
                      "teacher_sampling": [config.teacher_video_steps, config.teacher_action_steps],
                      "learning_rate": config.learning_rate, "world_size": config.world_size,
                      "experts": sorted(config.resources.get("experts") or {})}}


def export_student(adapter, destination, *, config, variant, updates):
    """adapter: an adapter_state() / Learner.ema_state() mapping of canonical names to tensors."""
    if variant not in VARIANTS or not adapter:
        raise ValueError("Raw or EMA student adapter required")

    def writer(staging):
        torch.save({k: v.detach().cpu().float().clone() for k, v in adapter.items()}, staging / "student_adapter.pt")
        metadata.write(staging, student_metadata(config, variant=variant, updates=updates))
    return {"path": str(publish_dir(destination, writer)), "variant": variant, "updates": updates}


def read_student(source):
    """(metadata, adapter state) of a student directory, with its LoRA geometry checked."""
    meta = metadata.read(source, "student")
    if tuple(meta["lora"]["targets"]) != TARGETS:
        raise ValueError(f"Student LoRA targets must be {TARGETS}")
    state = torch.load(metadata.weights(source, meta), map_location="cpu", weights_only=True)
    return meta, state


def load_student(model, source, *, config=None):
    """Load a student adapter into `model` (LoRA already injected). With `config`, its LoRA rank/alpha
    and student sampler must match the export."""
    meta, state = read_student(source)
    if config is not None:
        lora, sampling = meta["lora"], meta["sampling"]
        if (lora["rank"], lora["alpha"]) != (config.lora_rank, config.lora_alpha):
            raise ValueError(f"Student {source} has LoRA rank/alpha {lora['rank']}/{lora['alpha']}, "
                             f"the config {config.lora_rank}/{config.lora_alpha}")
        if (sampling["video_steps"], sampling["action_steps"]) != (config.video_steps, config.action_steps):
            raise ValueError("Student export and config use different student samplers")
    load_adapter(model, state)
    return meta
