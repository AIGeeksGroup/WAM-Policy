#!/usr/bin/env python3
"""Maintainers only: write the release ``wam_policy.json`` of the paper checkpoints.

Reproducers do not need this script: the released weights already carry ``wam_policy.json``, and
the training code of this repository writes it directly. It converts the metadata files the paper
checkpoints were first stored with (``export_manifest.json``, ``binding.json``, run ``manifest.json``)
into the fields ``wam_policy.models.metadata`` reads, plus ``weights_sha256`` of the weight files
(computed once; the teacher's ~20 GB of shards take a few minutes). The weight files themselves are
reused byte for byte (``--link`` symlinks them under their release names for local use).

  teacher     SRC = .../step_00005000            (resume_manifest.json, transformer/export_manifest.json)
  student     SRC = .../students/step_00000300/ema  (binding.json, student_adapter.pt)   --method opsd|mopd
  specialist  SRC = <run dir> (manifest.json)  --round r14 --task push_box   (weights: <run>/<round>/old-lora.pt)

Examples:
  python scripts/release/convert_checkpoint_metadata.py teacher <src> --output stage1_1_teacher [--link]
  python scripts/release/convert_checkpoint_metadata.py student <src> --method opsd --output stage1_2_opsd_student
  python scripts/release/convert_checkpoint_metadata.py specialist <run> --round r14 --task push_box \
      --output stage2_1_specialists/push_box

Release layout (weights renamed, bytes unchanged):
  teacher:    wam_policy.json + transformer/{config.json, *.safetensors, *.safetensors.index.json}
              (export_manifest.json is not needed)
  student:    wam_policy.json + student_adapter.pt
  specialist: wam_policy.json + adapter.pt   (= <round>/old-lora.pt)
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wam_policy.models import metadata  # noqa: E402

BASE_REPO = "robbyant/lingbot-va-base"
BASE_REVISION = "68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c"


WEIGHT_SHAS = None   # --weights-sha256 override: {file name: sha256} (e.g. the LFS sha256 listed by the Hub)


def WEIGHTS(directory, pattern):
    """sha256 of local weight files, or the override for weights that are not present locally."""
    found = metadata.weights_sha256(directory, pattern)
    if WEIGHT_SHAS is not None:
        import fnmatch
        given = {k: v for k, v in WEIGHT_SHAS.items() if fnmatch.fnmatch(k, pattern) or fnmatch.fnmatch(k, pattern.split("/")[-1])}
        return {(k if "/" in k or "/" not in pattern else pattern.rsplit("/", 1)[0] + "/" + k): v for k, v in given.items()}
    if not found:
        raise SystemExit(f"No weight files {pattern} under {directory}; pass --weights-sha256 <json>")
    return found


def read(path):
    return json.loads(Path(path).read_text())


def link(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    os.symlink(Path(source).resolve(), target)


def teacher(src, output, do_link):
    export = read(src / "transformer" / "export_manifest.json")
    data = export["provenance"]["data"]
    norm = data["action_normalizer"]
    config = export["provenance"].get("config", {})
    meta = {"kind": "teacher",
            "action_normalizer": {"offset": list(norm["offset"]), "scale": list(norm["scale"])},
            "rgb_profile": "dual-rgb-360p-v1",
            "sampling": {"video_steps": 25, "action_steps": 50},
            "base_model": {"repo_id": BASE_REPO, "revision": BASE_REVISION},
            "weights_sha256": WEIGHTS(src, "transformer/*.safetensors"),
            "notes": {"paper_name": "Stage 1.1 teacher (cross-morphology transfer, SFT)",
                      "training_steps": config.get("max_steps"), "learning_rate": config.get("learning_rate"),
                      "global_batch_size": 8}}
    path = metadata.write(output, meta)
    if do_link:
        for item in (src / "transformer").iterdir():
            if item.name != "export_manifest.json":
                link(item, output / "transformer" / item.name)
    return path


def student(src, output, method, do_link):
    binding = read(src / "binding.json")
    algorithm = binding["base_identity"]["algorithm"]
    sampler = binding["sampler"]
    meta = {"kind": "student", "method": method,
            "lora": {"rank": algorithm["lora_rank"], "alpha": algorithm["lora_alpha"],
                     "targets": list(metadata.STUDENT_TARGETS)},
            "sampling": {"video_steps": sampler["state"]["steps"], "action_steps": sampler["action"]["steps"],
                         "window_units": sampler["state"]["window_units"]},
            "variant": binding["variant"], "updates": binding["successful_updates"],
            "weights_sha256": WEIGHTS(src, "student_adapter.pt"),
            "ema": {"decay": binding.get("ema_decay"), "start_step": binding.get("ema_start_step")},
            "notes": {"paper_name": {"opsd": "Stage 1.2 OPSD student", "mopd": "Stage 2.2 M-OPD student"}[method],
                      "data_mode": algorithm.get("data_mode"),
                      "offline_contexts_per_rank": algorithm.get("offline_contexts_per_rank"),
                      "teacher_sampling": [algorithm.get("teacher_video_steps"), algorithm.get("teacher_action_steps")],
                      "learning_rate": algorithm.get("learning_rate"), "world_size": algorithm.get("world_size")}}
    path = metadata.write(output, meta)
    if do_link:
        link(src / "student_adapter.pt", output / "student_adapter.pt")
    return path


def specialist(src, output, round_name, task, do_link):
    manifest = read(src / "manifest.json")
    adapters = manifest["adapters"]
    if f"{round_name}/old-lora.pt" not in manifest.get("files", {round_name + "/old-lora.pt": None}):
        raise SystemExit(f"{round_name}/old-lora.pt is not listed in {src}/manifest.json")
    meta = {"kind": "specialist", "task": task,
            "lora": {"rank": adapters["rank"], "alpha": adapters["alpha"],
                     "targets": list(metadata.SPECIALIST_TARGETS)},
            "adapter_keys": "old", "sampling": {"video_steps": 25, "action_steps": 50},
            "round": round_name,
            "weights_sha256": {"adapter.pt": v for v in WEIGHTS(src / round_name, "old-lora.pt").values()},
            "notes": {"paper_name": f"Stage 2.1 task-specific RL specialist ({task})",
                      "adapter_layout": "{'lora': {'<i>.old_A', '<i>.old_B'}}, i = adapted Linear in module order "
                                        "(30 blocks x targets)"}}
    path = metadata.write(output, meta)
    if do_link:
        link(src / round_name / "old-lora.pt", output / "adapter.pt")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=("teacher", "student", "specialist"))
    p.add_argument("source", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--method", choices=("opsd", "mopd"))
    p.add_argument("--round")
    p.add_argument("--task")
    p.add_argument("--link", action="store_true", help="symlink the weights under their release names")
    p.add_argument("--weights-sha256", type=Path,
                   help="JSON {weight file name: sha256} used instead of hashing local files (e.g. Hub LFS sha256)")
    a = p.parse_args(argv)
    global WEIGHT_SHAS
    if a.weights_sha256:
        WEIGHT_SHAS = read(a.weights_sha256)
    if a.kind == "teacher":
        path = teacher(a.source, a.output, a.link)
    elif a.kind == "student":
        if not a.method:
            p.error("student requires --method")
        path = student(a.source, a.output, a.method, a.link)
    else:
        if not (a.round and a.task):
            p.error("specialist requires --round and --task")
        path = specialist(a.source, a.output, a.round, a.task, a.link)
    print(path)


if __name__ == "__main__":
    main()
