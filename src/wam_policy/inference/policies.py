"""Build a qlm-live-v1 policy session for each released WAM-Policy model.

=================  ===================================================  =============
``--policy``       model                                                default NFE
=================  ===================================================  =============
``teacher``        Stage 1.1 teacher                                    25/50
``specialist``     teacher + Stage 2.1 task-specific RL LoRA (frozen)   25/50
``opsd_student``   teacher + Stage 1.2 OPSD student LoRA (unmerged)     1/2
``mopd_student``   teacher + Stage 2.2 M-OPD student LoRA (unmerged)    1/2
=================  ===================================================  =============

NFE is written ``video_steps/action_steps``. Teacher and specialist accept any NFE
(the paper also reports the teacher at 2/4 and 1/2 without distillation); students run the
sampler they were distilled for (recorded in their ``wam_policy.json``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from wam_policy.inference.loader import load_inference_bundle
from wam_policy.inference.sampler import SamplingConfig, VideoActionSampler
from wam_policy.inference.session import OnlineSession
from wam_policy.models import metadata

POLICIES = ("teacher", "specialist", "opsd_student", "mopd_student")
STUDENT_METHOD = {"opsd_student": "opsd", "mopd_student": "mopd"}


@dataclass
class PolicyHandle:
    """An OnlineSession plus the per-reset hook the student deployment sampler needs."""
    session: OnlineSession
    on_reset: object = None

    def __call__(self, request):
        if self.on_reset is not None and request.get("method") == "reset":
            self.on_reset(request["payload"]["reset"])
        return self.session.handle(request)


def offload_blocks(model, count, device):
    """Keep the first ``count`` transformer blocks in CPU memory (executed on ``device``)."""
    if type(count) is not int or not 0 <= count <= len(model.blocks):
        raise ValueError(f"offload block count must be in [0, {len(model.blocks)}]")
    if count:
        import torch
        from accelerate import cpu_offload
        for block in list(model.blocks)[:count]:
            cpu_offload(block, execution_device=device, preload_module_classes=["WanTransformerBlock"])
        torch.cuda.empty_cache()


def attach_adapter(policy, model, normalizer, adapter):
    """Install the LoRA of ``adapter`` on the teacher model -> (sampler or None, metadata, on_reset)."""
    if policy == "specialist":
        from wam_policy.task_rl.adapters import attach_specialist
        meta = attach_specialist(model, adapter)
        return None, meta, None
    meta = metadata.read(adapter, "student")
    if meta["method"] != STUDENT_METHOD[policy]:
        raise ValueError(f"{adapter} is a {meta['method']} student, not {STUDENT_METHOD[policy]}")
    from wam_policy.opsd.deploy import student_actor
    actor, meta = student_actor(model, normalizer, adapter)
    return actor, meta, actor.begin_episode


def build_policy(policy, *, teacher, base_model, adapter=None, prompt_root=None, tasks=None,
                 video_steps=None, action_steps=None, device="cuda", vae_device=None, dtype="bfloat16",
                 policy_seed=42, offload=0):
    """-> PolicyHandle serving ``policy``; see the module docstring for the policy names."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    if (policy == "teacher") != (adapter is None):
        raise ValueError("teacher takes no --adapter; specialist and students require one")
    from wam_policy.common.tasks import TASKS
    bundle = load_inference_bundle(checkpoint=teacher, base_model=base_model, prompt_root=prompt_root,
                                   tasks=tasks or TASKS, device=device, vae_device=vae_device, dtype=dtype)
    info = {"policy": policy, **bundle.info}
    sampler, on_reset = None, None
    if adapter is not None:
        sampler, info["adapter"], on_reset = attach_adapter(policy, bundle.model, bundle.normalizer, adapter)
        info["adapter_dir"] = str(adapter)
    if sampler is None:
        default = (info.get("adapter") or info["teacher"])["sampling"]
        sampling = SamplingConfig(video_steps=video_steps or default["video_steps"],
                                  action_steps=action_steps or default["action_steps"]).validate()
        sampler = VideoActionSampler(bundle.model, bundle.normalizer, sampling)
        info["sampling"] = asdict(sampling)
    else:
        if video_steps or action_steps:
            raise ValueError("A student runs the 1/2 sampler it was distilled for; do not override the NFE")
        info["sampling"] = info["adapter"]["sampling"]
    offload_blocks(bundle.model, offload, device)
    info["offloaded_blocks"] = offload
    session = OnlineSession(sampler, bundle.codec, bundle.prompt_encoder, policy_seed=policy_seed, info=info)
    return PolicyHandle(session, on_reset)
