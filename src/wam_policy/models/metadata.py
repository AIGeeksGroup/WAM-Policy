"""Minimal metadata of released WAM-Policy checkpoints (``wam_policy.json``).

Every released artifact directory carries one ``wam_policy.json`` next to its weights:

* ``kind: "teacher"`` -- Stage 1.1 cross-morphology transfer (SFT) model::

      <dir>/wam_policy.json
      <dir>/transformer/{config.json, diffusion_pytorch_model*.safetensors[.index.json]}

  fields: ``action_normalizer`` ({offset[9], scale[9]}), ``rgb_profile``, ``sampling``
  (default NFE, 25/50), ``base_model`` (LingBot-VA repo/revision whose VAE/T5 the model uses).

* ``kind: "student"`` -- Stage 1.2 OPSD or Stage 2.2 M-OPD low-NFE student::

      <dir>/wam_policy.json
      <dir>/student_adapter.pt      # {"blocks.<i>.<target>.lora_A|lora_B": tensor}

  fields: ``method`` (opsd|mopd), ``lora`` ({rank, alpha, targets}), ``sampling`` (1/2),
  ``variant`` (raw|ema), ``updates``. The base weights are the Stage 1.1 teacher.

* ``kind: "specialist"`` -- Stage 2.1 task-specific RL adapter on the teacher::

      <dir>/wam_policy.json
      <dir>/adapter.pt              # {"lora": {"<i>.old_A|old_B": tensor}}, i = module order

  fields: ``task``, ``lora`` ({rank, alpha, targets}), ``adapter_keys`` ("old"|"current"),
  ``sampling`` (25/50), ``round``.

Every kind may also carry ``weights_sha256`` ({weight file relative to the directory: sha256}),
computed once when the artifact is written. It is not verified at load time; it makes the
teacher-cache reuse key (which digests this file) change whenever the weights change.
Only the fields above are read; anything else (``notes``) is informational.
"""
from __future__ import annotations

import json
from pathlib import Path

FILENAME = "wam_policy.json"
FORMAT = "wam-policy-checkpoint-v1"
KINDS = ("teacher", "student", "specialist")
WEIGHTS = {"student": "student_adapter.pt", "specialist": "adapter.pt"}

STUDENT_TARGETS = ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                   "ffn.net.0.proj", "ffn.net.2")
SPECIALIST_TARGETS = ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                      "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
                      "ffn.net.0.proj", "ffn.net.2")


def _sampling(value):
    if not isinstance(value, dict) or any(type(value.get(k)) is not int or value[k] < 1
                                          for k in ("video_steps", "action_steps")):
        raise ValueError("sampling needs positive integer video_steps/action_steps")
    return value


def _lora(value):
    if (not isinstance(value, dict) or type(value.get("rank")) is not int or value["rank"] < 1
            or not isinstance(value.get("alpha"), (int, float)) or not value.get("targets")):
        raise ValueError("lora needs rank, alpha and targets")
    return value


def validate(meta):
    if meta.get("format") != FORMAT or meta.get("kind") not in KINDS:
        raise ValueError(f"Not a {FORMAT} metadata file (kind must be one of {KINDS})")
    _sampling(meta.get("sampling"))
    kind = meta["kind"]
    if kind == "teacher":
        norm = meta.get("action_normalizer") or {}
        if len(norm.get("offset", ())) != 9 or len(norm.get("scale", ())) != 9:
            raise ValueError("teacher metadata needs a 9D action_normalizer")
        if not meta.get("rgb_profile"):
            raise ValueError("teacher metadata needs rgb_profile")
    elif kind == "student":
        _lora(meta.get("lora"))
        if meta.get("method") not in ("opsd", "mopd"):
            raise ValueError("student method must be opsd or mopd")
    else:
        _lora(meta.get("lora"))
        if not meta.get("task") or meta.get("adapter_keys") not in ("old", "current"):
            raise ValueError("specialist metadata needs task and adapter_keys")
    return meta


def read(directory, kind=None):
    path = Path(directory) / FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; convert the checkpoint with "
                                "scripts/release/convert_checkpoint_metadata.py")
    meta = validate(json.loads(path.read_text()))
    if kind is not None and meta["kind"] != kind:
        raise ValueError(f"{directory} holds a {meta['kind']} checkpoint, expected {kind}")
    return meta


def write(directory, meta):
    meta = validate({"format": FORMAT, **meta})
    path = Path(directory) / FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


def weights_sha256(directory, pattern):
    """{relative file: sha256} of the weight files matching ``pattern`` below ``directory``."""
    import hashlib
    root, out = Path(directory), {}
    for path in sorted(root.glob(pattern)):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(16 * 1024 ** 2), b""):
                h.update(block)
        out[path.relative_to(root).as_posix()] = h.hexdigest()
    return out


def weights(directory, meta):
    """Path of the adapter weights of a student/specialist directory."""
    return Path(directory) / meta.get("weights", WEIGHTS[meta["kind"]])
