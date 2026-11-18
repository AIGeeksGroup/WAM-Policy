"""Training checkpoint -> released teacher directory (``wam_policy.json`` + ``transformer/``).

The metadata is ``kind: "teacher"`` (see :mod:`wam_policy.models.metadata`) with two extra fields
read by :func:`wam_policy.ablations.proprioception.serving.build_session`:
``proprioception_dim: 10`` and ``proprioception_normalizer``, plus ``weights_sha256``
({shard file name: sha256} of ``transformer/*.safetensors``, computed once at export).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from wam_policy.common import site
from wam_policy.models import metadata

from .engine import CHECKPOINT_FILE

RGB_PROFILE = "dual-rgb-360p-v1"
SAMPLING = {"video_steps": 25, "action_steps": 50}


def teacher_metadata(record):
    inputs, step = record["inputs"], record["step"]
    return {"kind": "teacher", "action_normalizer": inputs["action_normalizer"], "rgb_profile": RGB_PROFILE,
            "sampling": dict(SAMPLING),
            "base_model": {"repo_id": site.get("WAM_BASE_MODEL_REPO"),
                           "revision": site.get("WAM_BASE_MODEL_REVISION")},
            "proprioception_dim": 10, "proprioception_normalizer": inputs["proprioception_normalizer"],
            "notes": {"paper_name": "Stage 1.1 ablation: RGB + proprioception teacher",
                      "training_steps": step,
                      "global_batch_size": record["world_size"] * record["config"]["accumulation"]}}


def weights_sha256(transformer):
    result = {}
    for path in sorted(Path(transformer).glob("*.safetensors")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        result[path.name] = digest.hexdigest()
    return result


def export_teacher(checkpoint, output):
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError(output)
    record = json.loads((checkpoint / CHECKPOINT_FILE).read_text())
    config = json.loads((checkpoint / "transformer/config.json").read_text())
    if config.get("proprioception_dim") != 10 or config.get("action_dim") != 9:
        raise ValueError("Checkpoint transformer is not a native9 + 10D proprioception model")
    staging = output.with_name("." + output.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(checkpoint / "transformer", staging / "transformer")
    metadata.write(staging, {**teacher_metadata(record), "weights_sha256": weights_sha256(staging / "transformer")})
    staging.rename(output)
    return {"output": str(output), "step": record["step"]}
