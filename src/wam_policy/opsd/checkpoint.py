"""Round-boundary resume checkpoints. Physical simulator state is not restored: a resumed run starts
new QLM episodes (the episode schedule continues from its saved snapshot)."""
from pathlib import Path
import json
import shutil
import torch
from wam_policy.common.distributed import rng_state, restore_rng
from wam_policy.opsd.config import write_json
from wam_policy.opsd.distributed import collective_call
from wam_policy.opsd.export import publish_dir
from wam_policy.opsd.models import adapter_state, load_adapter

SCHEMA = "wam-policy-distillation-resume-v1"


def save(learner, target, *, actor_state=None, control_state=None):
    def boundary():
        if not learner.round_committed or any(r.active for r in (learner.student, learner.teacher)):
            raise ValueError("Checkpoint requires a committed round and suspended cache leases")
        return learner.counters()
    counters = collective_call(learner.world, boundary)
    if len({json.dumps(c, sort_keys=True) for c in learner.world.gather(counters)}) != 1:
        raise ValueError("Checkpoint rank counters disagree")
    ranks = learner.world.gather({"rank": learner.world.rank, "rng": rng_state(learner.world.device),
                                  "streams": learner.rng.state_dict(), "actor_rng": actor_state})
    config = learner.config

    def writer(staging):
        torch.save({"student": adapter_state(learner.roles.student), "student_ema": learner.ema_state(),
                    "student_optimizer": learner.student_optimizer.state_dict(),
                    "student_scheduler": learner.student_scheduler.state_dict(),
                    "counters": counters, "ranks": ranks, "control_state": control_state}, staging / "state.pt")
        write_json(staging / "manifest.json", {"schema": SCHEMA, "student_step": learner.student_steps,
                                               "world_size": learner.world.size, "config": config.record(),
                                               "training": config.training_record()})
    return learner.world.leader(lambda: {"path": str(publish_dir(target, writer))})


def read_manifest(path):
    manifest = json.loads((Path(path) / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a resume checkpoint ({SCHEMA})")
    return manifest


def check_resume_config(manifest, config):
    """The run config must match the checkpoint's (timeouts, cadences and acceleration switches may differ)."""
    saved, current = manifest["training"], config.training_record()
    differing = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
    if differing:
        raise ValueError("Resume config differs from the checkpoint's: " + ", ".join(differing))


def restore(learner, path):
    manifest = read_manifest(path)
    check_resume_config(manifest, learner.config)
    if manifest["world_size"] != learner.world.size:
        raise ValueError("Resume needs the checkpoint's rank count")
    # Local reads and copies may fail on one rank only: agree before continuing.
    return collective_call(learner.world, lambda: _restore_local(learner, path, manifest))


def _restore_local(learner, path, manifest):
    if any(r.active for r in (learner.student, learner.teacher)):
        raise ValueError("Restore requires suspended cache leases")
    state = torch.load(Path(path) / "state.pt", map_location="cpu", weights_only=False)
    if state["counters"]["student_steps"] != manifest["student_step"]:
        raise ValueError("Checkpoint counter mismatch")
    if [r["rank"] for r in state["ranks"]] != list(range(learner.world.size)):
        raise ValueError("Missing/duplicate per-rank recovery state")
    load_adapter(learner.roles.student, state["student"])
    learner.load_ema_state(state["student_ema"])
    learner.student_optimizer.load_state_dict(state["student_optimizer"])
    learner.student_scheduler.load_state_dict(state["student_scheduler"])
    learner.restore_counters(state["counters"])
    rank = state["ranks"][learner.world.rank]
    learner.rng.load_state_dict(rank["streams"])
    restore_rng(rank["rng"], learner.world.device)
    return {"student_step": manifest["student_step"], "control_state": state["control_state"]}


def retain(directory, *, protected, keep=2):
    """Keep the newest complete checkpoints plus the configured protected steps."""
    complete = []
    for p in Path(directory).glob("step_*/manifest.json"):
        m = json.loads(p.read_text())
        if m.get("schema") == SCHEMA:
            complete.append((m["student_step"], p.parent))
    for step, p in sorted(complete)[:-keep]:
        if step not in protected:
            shutil.rmtree(p)
