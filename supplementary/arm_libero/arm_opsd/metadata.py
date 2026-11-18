"""Minimal metadata of arm students (``arm_libero.json``), the arm counterpart of WAM-Policy's ``wam_policy.json``.

Every student directory carries one ``arm_libero.json`` next to its LoRA weights::

    <dir>/arm_libero.json
    <dir>/student_adapter.pt      # {"blocks.<i>.<target>.lora_A|lora_B": tensor}

    {
      "format": "arm-libero-student-v1",
      "kind": "student",
      "method": "opsd" | "teacher_cache_only" | "streaming_only" | "collected_once" | "sft_only",
      "variant": "ema" | "raw",
      "updates": 300,
      "sampling": {"video_steps": 1, "action_steps": 2},
      "lora": {"rank": 256, "alpha": 256, "dropout": 0.0, "targets": [...six shared-block Linear layers...]},
      "base_model": {"repo": "robbyant/lingbot-va-posttrain-libero-long", "revision": "<40-hex>"},
      "weights": "student_adapter.pt",
      "ema": {"decay": 0.99, "start_update": 50},            # null for a raw student
      "training": {"method": {...}, "experiment": {...}}      # MethodConfig / ExperimentConfig, informational
    }

Only format, kind, variant, updates, sampling, lora, base_model and weights are read. A directory with the
``export.json`` of a student published before this release is read through ``from_legacy_export``.
"""
from dataclasses import asdict
import json
from pathlib import Path

from arm_opsd.config import BASE_REPO, BASE_REVISION
from arm_opsd.model.lora import TARGETS

FILENAME = "arm_libero.json"
FORMAT = "arm-libero-student-v1"
WEIGHTS = "student_adapter.pt"
LEGACY_EXPORT = "export.json"
LEGACY_WEIGHTS = "adapter.pt"


def method_name(method):
    """Paper name of a student from its MethodConfig fields (a dict)."""
    if method.get("objective") == "sft_fm":
        return "sft_only"
    if method.get("data_mode") == "collect_once":
        return "collected_once"
    if method.get("data_mode") == "offline_only":
        return "teacher_cache_only"
    return "opsd" if method.get("offline_contexts_per_rank", 0) > 0 else "streaming_only"


def validate(meta):
    if meta.get("format") != FORMAT or meta.get("kind") != "student":
        raise ValueError(f"Not an {FORMAT} student metadata file")
    sampling, lora = meta.get("sampling") or {}, meta.get("lora") or {}
    if any(type(sampling.get(k)) is not int or sampling[k] < 1 for k in ("video_steps", "action_steps")):
        raise ValueError("sampling needs positive integer video_steps/action_steps")
    if type(lora.get("rank")) is not int or lora["rank"] < 1 or not isinstance(lora.get("alpha"), (int, float)):
        raise ValueError("lora needs an integer rank and an alpha")
    if meta.get("variant") not in ("raw", "ema") or type(meta.get("updates")) is not int:
        raise ValueError("variant must be raw or ema, updates an integer")
    if not {"repo", "revision"} <= set(meta.get("base_model") or {}):
        raise ValueError("base_model needs repo and revision")
    return meta


def build(config, *, update, variant, weights=WEIGHTS):
    m, e = config.method, config.experiment
    return validate({"format": FORMAT, "kind": "student", "method": method_name(asdict(m)), "variant": variant,
                     "updates": int(update), "sampling": {"video_steps": m.state_steps, "action_steps": m.action_steps},
                     "lora": {"rank": m.lora_rank, "alpha": m.lora_alpha, "dropout": m.lora_dropout, "targets": list(TARGETS)},
                     "base_model": {"repo": BASE_REPO, "revision": BASE_REVISION}, "weights": weights,
                     "ema": {"decay": e.ema_decay, "start_update": e.ema_start_update} if variant == "ema" else None,
                     "training": {"method": asdict(m), "experiment": asdict(e)}})


def from_legacy_export(export, *, weights=LEGACY_WEIGHTS):
    """Metadata of a previously published export.json; drops machine paths, resource identities and hashes."""
    from arm_opsd.config import ArmConfig
    config = ArmConfig.from_dict({k: export["config"][k] for k in ("native", "method", "experiment", "profile")
                                  if k in export["config"]}, strict=False)
    lora = (export.get("lora") or [{}])[0]
    meta = build(config, update=export["student_update"], variant=export.get("variant", "raw"), weights=weights)
    meta["lora"]["alpha"] = lora.get("alpha", meta["lora"]["alpha"])
    meta["base_model"] = {"repo": export.get("base_repo", BASE_REPO), "revision": export.get("base_revision", BASE_REVISION)}
    if meta["ema"] is not None:
        meta["ema"] = {"decay": export.get("ema_decay", meta["ema"]["decay"]),
                       "start_update": export.get("ema_start_update", meta["ema"]["start_update"])}
    return validate(meta)


def read(directory):
    directory = Path(directory)
    if (directory / FILENAME).is_file():
        return validate(json.loads((directory / FILENAME).read_text()))
    if (directory / LEGACY_EXPORT).is_file():
        return from_legacy_export(json.loads((directory / LEGACY_EXPORT).read_text()))
    raise FileNotFoundError(f"No {FILENAME} (or previously published {LEGACY_EXPORT}) in {directory}")


def write(directory, meta):
    path = Path(directory) / FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validate(meta), indent=2) + "\n")
    return path


def weights(directory, meta):
    return Path(directory) / meta.get("weights", WEIGHTS)
