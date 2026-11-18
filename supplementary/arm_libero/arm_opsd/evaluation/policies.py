"""The evaluated policies and how they are loaded.

  base_1_2     Student w/o distillation: the official LingBot-VA LIBERO-Long weights (theta_0) sampled
               with 1 video / 2 action steps. Training starts every student from a zero-initialized
               lora_B, so this is exactly the untrained student.
  student_1_2  A trained student (arm_libero.json + student_adapter.pt) on theta_0, same sampler.
Both use the guidance scales of LingBot-VA's LIBERO client (video 5.0, action 1.0).
"""
from dataclasses import dataclass, replace
import json
from pathlib import Path

import torch

from arm_opsd.config import ArmContract
from arm_opsd.model.kv_views import NativeRole
from arm_opsd.model.sampler import NativeSampler

# Published checkpoint names, locations and training configs.
SNAPSHOTS_FILE = Path(__file__).resolve().parents[2] / "configs/evaluation/released_students.json"


@dataclass(frozen=True)
class EvalArm:
    label: str
    state_steps: int
    action_steps: int
    video_cfg: float
    action_cfg: float


ARMS = {
    "base_1_2": EvalArm("Student w/o distillation (1/2 NFE)", 1, 2, 5., 1.),
    "student_1_2": EvalArm("Student (1/2 NFE)", 1, 2, 5., 1.),
}
NFE = {policy: {"state_steps": arm.state_steps, "action_steps": arm.action_steps} for policy, arm in ARMS.items()}


def arm_sampler(policy, contract=None):
    arm = ARMS[policy]
    contract = contract or ArmContract()
    return NativeSampler(replace(contract, video_cfg=arm.video_cfg, action_cfg=arm.action_cfg),
                         state_steps=arm.state_steps, action_steps=arm.action_steps)


def load_policy(policy, model_root, student=None, *, device="cuda"):
    """The frozen role and a short description of the student (None for base_1_2)."""
    from arm_opsd.checkpoint import load_student
    from arm_opsd.model.loading import load_native_transformer
    if policy not in ARMS:
        raise ValueError(f"policy must be one of {tuple(ARMS)}")
    if (policy == "student_1_2") != bool(student):
        raise ValueError("student_1_2 needs --student (a student export directory); base_1_2 takes none")
    model = load_native_transformer(model_root, device=device, dtype=torch.bfloat16)
    role = NativeRole(model, "student")
    if policy == "base_1_2":
        return role, None
    metadata = load_student(student, role)
    if (metadata["sampling"]["video_steps"], metadata["sampling"]["action_steps"]) != (ARMS[policy].state_steps,
                                                                                 ARMS[policy].action_steps):
        raise ValueError(f"Student sampler {metadata['sampling']} differs from {policy}")
    return role, {"path": str(Path(student).resolve()), "method": metadata["method"], "update": metadata["updates"],
                  "variant": metadata["variant"]}


def released_students():
    return json.loads(SNAPSHOTS_FILE.read_text())["students"]


def fetch_student(name, root, *, repo=None, path=None, revision=None):
    """Download one released student (arm_libero.json + student_adapter.pt); returns its local directory.
    repo/path/revision override configs/evaluation/released_students.json."""
    from huggingface_hub import hf_hub_download
    entry = dict(released_students().get(name, {}))
    repo, path, revision = repo or entry.get("repo"), (path or entry.get("path", "")).strip("/"), revision or entry.get("revision")
    if not (repo and path):
        raise ValueError(f"Unknown released student {name!r}; give --repo and --path")
    for file in ("arm_libero.json", "student_adapter.pt"):
        hf_hub_download(repo, f"{path}/{file}", revision=revision, local_dir=str(root))
    return Path(root) / path
