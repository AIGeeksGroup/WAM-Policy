"""Complete-round training checkpoints and LoRA student exports.

A student export is a directory with ``student_adapter.pt`` (the LoRA tensors) and ``arm_libero.json``
(format arm-libero-student-v1, defined in arm_opsd/metadata.py). ``load_student`` also reads the
``export.json`` + ``adapter.pt`` of student snapshots published before this release
(``scripts/convert_student_export.py`` writes their arm_libero.json).
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import shutil
from uuid import uuid4

import numpy as np
import torch

from arm_opsd import metadata as student_metadata
from arm_opsd.config import BASE_REPO, BASE_REVISION
from arm_opsd.model.lora import adapter_state_dict, inject_lora, load_adapter_state_dict
from arm_opsd.opsd.learner import TrainingCounters

TRAINING_FORMAT = "arm-opsd-training-v1"
VARIANTS = ("raw", "ema")


def check_counters(counters, config, world_size):
    """Committed-round invariants by data mode: contexts per source follow the batch layout."""
    m = config.method
    updates = counters.student_updates
    if m.data_mode == "streaming":
        collected = updates * world_size * m.onpolicy_contexts_per_rank
        if counters.collect_once_rounds_done:
            raise ValueError("Streaming checkpoint carries collect_once rounds")
    elif m.data_mode == "offline_only":
        collected = 0
        if counters.collect_once_rounds_done or counters.replay_contexts:
            raise ValueError("Offline-only checkpoint carries collected/replayed contexts")
    else:
        done = counters.collect_once_rounds_done
        if done > m.collect_once_rounds or (updates and done != m.collect_once_rounds):
            raise ValueError("Checkpoint collect_once progress differs")
        collected = done * world_size * m.onpolicy_contexts_per_rank
        if counters.replay_contexts != updates * world_size * m.collect_once_replay_per_rank:
            raise ValueError("Checkpoint collect_once replay budget differs")
    if m.objective == "regression" and counters.collection_contexts != collected:
        raise ValueError("Checkpoint collection budget differs")
    if counters.offline_contexts != updates * world_size * m.offline_contexts_per_rank:
        raise ValueError("Checkpoint offline budget differs")


def ema_expected(config, student_updates):
    start = config.experiment.ema_start_update
    return start is not None and student_updates >= start


def capture_rng(learner):
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state(),
             "torch_gpu": ([torch.cuda.get_rng_state(learner.roles.student.device)]
                           if learner.roles.student.device.type == "cuda" else []),
             "generation": learner.generation_rng.get_state(), "offline": learner.offline_rng.get_state(),
             "flow_matching": learner.flow_matching_rng.get_state()}
    return state


def restore_rng(state, learner):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_gpu"]:
        if learner.roles.student.device.type != "cuda" or len(state["torch_gpu"]) != 1:
            raise ValueError("Checkpoint requires one GPU RNG state for this rank")
        torch.cuda.set_rng_state(state["torch_gpu"][0], learner.roles.student.device)
    learner.generation_rng.set_state(state["generation"])
    learner.offline_rng.set_state(state["offline"])
    learner.flow_matching_rng.set_state(state["flow_matching"])


def _publish_directory(destination, write):
    """Write into a temporary sibling and rename, so a directory either exists complete or not at all."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".incomplete-" + uuid4().hex)
    temporary.mkdir()
    try:
        write(temporary)
        for p in temporary.iterdir():
            if p.is_file():
                with p.open("rb") as f:
                    os.fsync(f.fileno())
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def check_resume_config(recorded, config):
    """Resume consistency: the checkpoint's native/method/experiment sections must equal the current ones."""
    current = json.loads(json.dumps(config.training_sections()))
    recorded = json.loads(json.dumps({k: recorded.get(k) for k in current}))
    differing = [f"{section}.{key}" for section in current for key in current[section]
                 if (recorded[section] or {}).get(key) != current[section][key]]
    if differing:
        raise ValueError(f"Resume config differs from the checkpoint's: {differing}")


def save_training_checkpoint(destination, learner, *, runtime_state=None, control_state=None):
    ctx = learner.distributed
    if not learner.round_committed:
        raise ValueError("Only committed rounds may be saved")
    if learner.config.profile == "full" and control_state is None:
        raise ValueError("A checkpoint requires the global scheduling state")
    check_counters(learner.counters, learner.config, ctx.world_size)
    if (learner.student_ema is not None) != ema_expected(learner.config, learner.counters.student_updates):
        raise ValueError("Student EMA presence differs from its start update")
    if learner.roles.student.version != learner.counters.student_updates:
        raise ValueError("Parameter versions and optimizer counters differ")
    per_rank = ctx.gather({"rng": capture_rng(learner), "runtime": runtime_state or {},
                           "pool": learner.pool.state()})
    ctx.agree(asdict(learner.counters), "checkpoint counters")
    ctx.agree(control_state, "checkpoint scheduler")
    error = None
    if ctx.rank == 0:
        def write(directory):
            state = {"format": TRAINING_FORMAT, "config": learner.config.to_dict(),
                     "student": adapter_state_dict(learner.roles.student.model),
                     "student_ema": learner.ema_state(),
                     "student_optimizer": learner.student_optimizer.state_dict(),
                     "counters": asdict(learner.counters), "actor_version": learner.roles.student.version,
                     "world_size": ctx.world_size, "per_rank": per_rank, "control_state": control_state}
            torch.save(state, directory / "training.pt")
            (directory / "completion.json").write_text(json.dumps(
                {"format": TRAINING_FORMAT, "counters": state["counters"], "complete_round": True}, indent=2) + "\n")
        try:
            _publish_directory(destination, write)
        except Exception as exc:
            error = exc
    ctx.consensus(error, "checkpoint write")
    ctx.barrier()
    return Path(destination)


def read_training_checkpoint(directory, config):
    directory = Path(directory)
    meta = json.loads((directory / "completion.json").read_text())
    if not meta.get("complete_round"):
        raise ValueError(f"Incomplete checkpoint: {directory}")
    state = torch.load(directory / "training.pt", map_location="cpu", weights_only=False)
    check_resume_config(state["config"], config)
    return state


def load_training_checkpoint(directory, learner):
    state = read_training_checkpoint(directory, learner.config)
    if state["world_size"] != learner.distributed.world_size:
        raise ValueError("Resume requires the same number of ranks")
    counters = TrainingCounters(**state["counters"])
    check_counters(counters, learner.config, learner.distributed.world_size)
    if state["actor_version"] != counters.student_updates:
        raise ValueError("Checkpoint parameter versions differ from optimizer counts")
    if (state.get("student_ema") is not None) != ema_expected(learner.config, counters.student_updates):
        raise ValueError("Checkpoint EMA presence differs from its start update")
    load_adapter_state_dict(learner.roles.student.model, state["student"])
    learner.load_ema_state(state.get("student_ema"))
    learner.student_optimizer.load_state_dict(state["student_optimizer"])
    learner.counters = counters
    learner.roles.student.version = state["actor_version"]
    learner.history.invalidate("student")
    rank_state = state["per_rank"][learner.distributed.rank]
    restore_rng(rank_state["rng"], learner)
    learner.pool.load_state(rank_state["pool"], learner.history)
    learner.round_committed = True
    return {"runtime": rank_state["runtime"], "control_state": state.get("control_state"),
            "continuity_break": True, "new_segment_id": uuid4().hex}


def retain_checkpoints(directory, keep=2, protected=()):
    """Keep round zero, the protected rounds and the `keep` latest; incomplete directories survive."""
    periodic = []
    for path in Path(directory).glob("round-*"):
        if not (path / "completion.json").is_file():
            continue
        meta = json.loads((path / "completion.json").read_text())
        update = meta["counters"]["student_updates"]
        if meta.get("complete_round") and update != 0:
            periodic.append((update, path))
    for update, path in sorted(periodic)[:-keep]:
        if update not in protected:
            shutil.rmtree(path)


def export_student(destination, role, config, *, update, adapter=None, variant="raw"):
    """adapter: an adapter_state_dict()/StreamingLearner.ema_state() mapping; default the role's own."""
    if variant not in VARIANTS:
        raise ValueError("Raw or EMA student export required")
    adapter = adapter_state_dict(role.model) if adapter is None else adapter
    if set(adapter) != set(adapter_state_dict(role.model)):
        raise ValueError("Export adapter tensors differ from the student's LoRA modules")

    def write(directory):
        torch.save({k: v.detach().cpu().clone() for k, v in adapter.items()}, directory / student_metadata.WEIGHTS)
        student_metadata.write(directory, student_metadata.build(config, update=update, variant=variant))
    return _publish_directory(destination, write)


def load_student(directory, role, *, inject=True):
    """Install the student LoRA on `role` (official base weights) and load its adapter; returns the metadata."""
    directory = Path(directory)
    metadata = student_metadata.read(directory)
    if (metadata["base_model"]["repo"], metadata["base_model"]["revision"]) != (BASE_REPO, BASE_REVISION):
        raise ValueError(f"Student was trained on {metadata['base_model']}, not {BASE_REPO}@{BASE_REVISION}")
    if inject:
        lora = metadata["lora"]
        inject_lora(role.model, rank=lora["rank"], alpha=lora["alpha"], dropout=lora.get("dropout", 0.))
    load_adapter_state_dict(role.model, torch.load(student_metadata.weights(directory, metadata), map_location="cpu",
                                                   weights_only=True))
    role.version = metadata["updates"]
    return metadata
