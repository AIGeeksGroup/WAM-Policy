"""Training checkpoints (complete resume state) and the released teacher export.

Training checkpoint ``<output_dir>/checkpoints/step_XXXXXXXX/``::

    transformer/          save_pretrained of the FP32 training weights
    training_state.pt     optimizer, scheduler, step, per-rank sampler + RNG states
    checkpoint.json       step, world size, resume-relevant config, action normalizer, RGB profile

``checkpoint.json`` is written last and the directory is renamed into place, so a directory with
``checkpoint.json`` is complete. ``export_teacher`` turns one into the release teacher format
(``wam_policy.json`` + ``transformer/``) read by ``wam_policy.inference.loader.load_teacher``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

import torch

from wam_policy.common import site
from wam_policy.models import metadata
from wam_policy.models.checkpoint_io import export_model
from wam_policy.transfer.config import write_json

FORMAT = "wam-policy-sft-checkpoint-v1"
TEACHER_SAMPLING = {"video_steps": 25, "action_steps": 50}


def checkpoint_name(step):
    return f"step_{step:08d}"


def save_checkpoint(model, optimizer, scheduler, target, *, step, ranks, info):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=target.parent))
    try:
        export_model(model, staging / "transformer")
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "step": step, "ranks": ranks}, staging / "training_state.pt")
        write_json(staging / "checkpoint.json", {"format": FORMAT, "step": step, "world_size": len(ranks), **info})
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def read_checkpoint(path):
    path = Path(path)
    if not (path / "checkpoint.json").is_file():
        raise ValueError(f"{path} is not a complete training checkpoint (checkpoint.json missing)")
    info = json.loads((path / "checkpoint.json").read_text())
    if info.get("format") != FORMAT:
        raise ValueError(f"{path}: unsupported checkpoint format {info.get('format')!r}")
    return info


def load_training_state(path, *, config, world_size, normalizer):
    """Training state of a checkpoint, after checking that it belongs to the same recipe."""
    info = read_checkpoint(path)
    if info["world_size"] != world_size:
        raise ValueError(f"Checkpoint was written by {info['world_size']} ranks, this run has {world_size}")
    expected = config.resume_values()
    differences = {k: (info["config"].get(k), v) for k, v in expected.items() if info["config"].get(k) != v}
    if differences:
        raise ValueError(f"Run config differs from the checkpoint (checkpoint, run): {differences}")
    norm = {"offset": list(normalizer.offset), "scale": list(normalizer.scale)}
    if info["action_normalizer"] != norm:
        raise ValueError("The cache action normalizer differs from the one the checkpoint was trained with")
    # Our own training state (optimizer tensors, numpy/python RNG states); not an untrusted file.
    state = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=False, mmap=True)
    if state["step"] != info["step"] or len(state["ranks"]) != world_size:
        raise ValueError(f"{path}: training_state.pt is inconsistent with checkpoint.json")
    return state


def retain_checkpoints(directory, keep):
    complete = sorted(p for p in Path(directory).glob("step_*") if (p / "checkpoint.json").is_file())
    for path in complete[:-keep]:
        shutil.rmtree(path)


def latest_checkpoint(directory):
    complete = sorted(p for p in Path(directory).glob("step_*") if (p / "checkpoint.json").is_file())
    return complete[-1] if complete else None


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def export_teacher(checkpoint, output, *, notes=None):
    """Write ``output/{wam_policy.json, transformer/}`` from a training checkpoint."""
    checkpoint, output = Path(checkpoint), Path(output)
    info = read_checkpoint(checkpoint)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent))
    try:
        shutil.copytree(checkpoint / "transformer", staging / "transformer", copy_function=_link_or_copy)
        metadata.write(staging, {
            "kind": "teacher", "action_normalizer": info["action_normalizer"],
            "rgb_profile": info["rgb_profile"], "sampling": dict(TEACHER_SAMPLING),
            "base_model": {"repo_id": site.get("WAM_BASE_MODEL_REPO"),
                           "revision": site.get("WAM_BASE_MODEL_REVISION")},
            "weights_sha256": metadata.weights_sha256(staging, "transformer/*.safetensors"),
            "notes": {"paper_name": "Stage 1.1 teacher (cross-morphology transfer, SFT)",
                      "training_steps": info["step"], **(notes or {})}})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def upload(directory, repo_id, path_in_repo=""):
    """Upload a directory to a Hugging Face model repository (needs a write token)."""
    from huggingface_hub import HfApi
    commit = HfApi().upload_folder(folder_path=str(directory), repo_id=repo_id, repo_type="model",
                                   path_in_repo=path_in_repo or None)
    return {"repo_id": repo_id, "path_in_repo": path_in_repo, "commit": getattr(commit, "oid", str(commit))}
