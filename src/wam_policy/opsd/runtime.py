"""Model loading (teacher, student, codec, prompts) and streaming-service composition."""
from functools import lru_cache
from pathlib import Path
import torch
from wam_policy.opsd.config import digest, file_sha256
from wam_policy.opsd.models import build_roles

RGB_PROFILE = "dual-rgb-360p-v1"


def device():
    """The rank's accelerator (CUDA or ROCm builds of PyTorch)."""
    if not torch.cuda.is_available():
        raise RuntimeError("Full-model training needs a GPU (CUDA or ROCm); the CPU smoke uses the tiny model")
    return torch.device("cuda", torch.cuda.current_device())


def teacher_key(config):
    """Path-free key of the configured teacher (its wam_policy.json), None without one (smoke)."""
    path = config.resources.get("teacher")
    if not path:
        return None
    from wam_policy.models import metadata
    return digest(metadata.read(path, "teacher"))


@lru_cache(maxsize=8)
def _sha(path):
    return file_sha256(path)


def pool_sha256(config):
    """sha256 of the train reset pool manifest (the QLM worker's --pool-sha256), None without one."""
    path = config.resources.get("train_pool")
    return None if not path else _sha(str(path))


def load_roles_and_codec(config):
    """Teacher + LoRA student (+ M-OPD specialist teachers), video codec, prompt encoder, normalizer."""
    from wam_policy.inference.loader import load_codec, load_prompts, load_teacher
    from wam_policy.common.tasks import TASKS
    teacher_dir, base_model, prompt_root = config.need("teacher", "base_model", "prompt_root")
    dev = str(device())
    model, normalizer, meta = load_teacher(teacher_dir, device=dev, dtype="bfloat16", attn_mode="torch")
    if meta.get("rgb_profile", RGB_PROFILE) != RGB_PROFILE:
        raise ValueError(f"The QLM training environment renders {RGB_PROFILE}; teacher uses {meta['rgb_profile']}")
    codec = load_codec(base_model, device=dev, dtype="bfloat16", rgb_profile=RGB_PROFILE)
    prompt = load_prompts(prompt_root=prompt_root, tasks=TASKS)
    pending = [model]

    def factory():
        if pending:
            return pending.pop()
        if config.shared_base:
            from wam_policy.opsd.models import tied_copy
            return tied_copy(model)
        return load_teacher(teacher_dir, device=dev, dtype="bfloat16", attn_mode="torch")[0]
    roles = build_roles(factory, config)
    attach_experts_and_student(roles, config)
    return roles, codec, prompt, normalizer


def attach_experts_and_student(roles, config):
    """M-OPD specialist teachers on the teacher; the initialized student LoRA, when configured."""
    if config.resources.get("experts"):
        from wam_policy.mopd.experts import install_experts
        roles.experts = install_experts(roles.teacher, config)
    roles.validate()
    if config.resources.get("initialized_student"):
        from wam_policy.opsd.export import load_student
        load_student(roles.student, config.resources["initialized_student"], config=config)
    return roles


def actor_contract(config):
    """Journal contract of every training/cache context (tensor geometry the learner relies on)."""
    return {"rgb_profile": RGB_PROFILE, "window_units": config.window_units}


def build_coordinator(learner, codec, prompt, normalizer, *, actor=None):
    """Streaming coordinator; the actor defaults to the learner's student (shared history). A
    teacher actor at the teacher sampler steps collects the teacher target cache."""
    from wam_policy.opsd.streaming import ActorSampler, Coordinator
    config = learner.config
    if actor is None:
        sampler = ActorSampler(learner.student, normalizer, contract=actor_contract(config), share_history=True)
    else:
        sampler = ActorSampler(actor, normalizer, contract=actor_contract(config),
                               steps=(config.teacher_video_steps, config.teacher_action_steps))
    sampler.round_id = learner.student_steps
    return Coordinator(sampler, codec, prompt)
