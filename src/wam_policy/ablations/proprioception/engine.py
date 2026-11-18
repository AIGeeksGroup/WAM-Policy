"""Whole-episode single-node (DDP) SFT of the proprioception model with step-boundary resume."""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import signal
import tempfile
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from wam_policy.common import site
from wam_policy.common.distributed import World, environment, isolated_rng, restore_rng, rng_state, seed_all
from wam_policy.models.checkpoint_io import load_model
from wam_policy.transfer.data import EpisodeSampler
from wam_policy.transfer.policy import parameter_group as base_parameter_group

from .data import ProprioceptionEpisodes
from .flow import ProprioceptionFlow

CHECKPOINT_FILE = "checkpoint.json"


def parameter_group(name):
    if name.startswith("proprioception_embedder."):
        return "proprioception_input"
    if name.startswith("proprioception_proj_out."):
        return "proprioception_output"
    return base_parameter_group(name)


def apply_trainability(model):
    """Full fine-tuning; the unused Conv3d patch embedding stays frozen (as in Stage 1.1)."""
    groups = defaultdict(lambda: {"parameters": 0, "trainable_parameters": 0})
    for name, p in model.named_parameters():
        group = parameter_group(name)
        p.requires_grad_(group != "checkpoint_only_conv3d")
        groups[group]["parameters"] += p.numel()
        groups[group]["trainable_parameters"] += p.numel() if p.requires_grad else 0
    return dict(groups)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def autocast(world):
    return torch.autocast("cuda", dtype=torch.bfloat16) if world.device.type == "cuda" else nullcontext()


def build_model(config, world, path=None):
    model = load_model(path or config.transformer, dtype=torch.float32, device=world.device,
                       attn_mode=config.attention)
    if getattr(model, "proprioception_dim", 0) != 10:
        raise ValueError("Expected a transformer with proprioception_dim=10 (run the convert command)")
    groups = apply_trainability(model)
    if config.activation_checkpointing:
        from wan_va.distributed.fsdp import apply_ac
        apply_ac(model)
    return model, groups


def make_optimizer(model, config):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.learning_rate, betas=config.betas, eps=1e-8,
                                  weight_decay=config.weight_decay, fused=False, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / config.warmup_steps, 1.0))
    return optimizer, scheduler


class Tracker:
    """Local ``metrics.jsonl`` journal; optional W&B run (WANDB_MODE disabled by default)."""

    def __init__(self, directory, config, run_name, payload):
        self.path = Path(directory) / "metrics.jsonl"
        mode, entity, project = site.wandb_settings(config.wandb_mode, config.wandb_entity, config.wandb_project)
        self.run = None
        if mode != "disabled":
            import wandb
            self.run = wandb.init(entity=entity, project=project, name=run_name, group=run_name,
                                  job_type="ablation-proprioception-sft", config=payload, mode=mode,
                                  dir=str(directory), settings=wandb.Settings(disable_git=True, save_code=False))
            self.run.define_metric("optimizer_step")
            for prefix in ("train/*", "validation/*", "system/*"):
                self.run.define_metric(prefix, step_metric="optimizer_step")

    def log(self, values, step):
        with self.path.open("a") as stream:
            stream.write(json.dumps({"optimizer_step": step, **values}) + "\n")
        if self.run:
            self.run.log({"optimizer_step": step, **values})

    def finish(self, status):
        if self.run:
            self.run.finish(exit_code=0 if status in ("completed", "stopped") else 1)


def train_step(model, raw_model, optimizer, scheduler, flow, batches, world, config):
    optimizer.zero_grad(set_to_none=True)
    loss_video = loss_state = loss_action = valid_actions = valid_states = 0.0
    for index, batch in enumerate(batches):
        sync = nullcontext() if index + 1 == len(batches) or not isinstance(model, DDP) else model.no_sync()
        with sync:
            prepared = flow.prepare(batch)
            with autocast(world):
                losses = flow.compute_loss(prepared, model(prepared, train_mode=True))
            if not world.all(all(bool(torch.isfinite(x)) for x in losses)):
                raise FloatingPointError("Nonfinite loss on at least one rank; no optimizer update")
            sum(losses).backward()
        loss_video += float(losses[0].detach())
        loss_state += float(losses[1].detach())
        loss_action += float(losses[2].detach())
        valid_actions += int(batch["actions_mask"][:, 0].sum())
        valid_states += int(batch["proprioception_mask"].sum())
    norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.grad_clip_norm)
    if not world.all(bool(torch.isfinite(norm))):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Nonfinite gradients on at least one rank; no optimizer update")
    lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    video, state, action, grad, actions, states = world.reduce(
        [loss_video, loss_state, loss_action, float(norm), valid_actions, valid_states])
    return {"train/video_loss": video, "train/proprioception_loss": state, "train/action_loss": action,
            "train/total_loss": video + state + action, "train/grad_norm": grad, "train/lr": lr,
            "train/valid_actions_global": int(actions * world.size),
            "train/valid_states_global": int(states * world.size)}


def validation(model, data, flow, world, *, seed=1729):
    """Rank-0 forward over the validation split with fixed chunk/window and a fixed seed."""
    def evaluate():
        was_training = model.training
        try:
            model.eval()
            values = []
            with isolated_rng(world.device, seed), torch.no_grad():
                for uid in data.split["validation"]:
                    prepared = flow.prepare(data.batch(uid, split="validation"), chunk_size=2,
                                            window_size=8, noisy_video_probability=0)
                    with autocast(world):
                        values.append([float(x) for x in flow.compute_loss(prepared, model(prepared, train_mode=True))])
            video, state, action = (sum(v[i] for v in values) / len(values) for i in range(3))
            return {"validation/total_loss": video + state + action, "validation/video_loss": video,
                    "validation/proprioception_loss": state, "validation/action_loss": action,
                    "validation/episodes": len(values)}
        finally:
            model.train(was_training)
    return world.leader(evaluate)


def save_checkpoint(model, optimizer, scheduler, target, *, step, ranks, record):
    target = Path(target)
    staging = Path(tempfile.mkdtemp(prefix="." + target.name + "-", dir=target.parent))
    try:
        model.save_pretrained(staging / "transformer", max_shard_size="2GB", safe_serialization=True)
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "step": step, "ranks": ranks}, staging / "training_state.pt")
        write_json(staging / CHECKPOINT_FILE, {**record, "step": step})
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"path": str(target), "step": step}


def check_resume(path, config, world_size):
    """Resumed runs must use the checkpoint's training configuration and world size."""
    record = json.loads((Path(path) / CHECKPOINT_FILE).read_text())
    if not (Path(path) / "training_state.pt").is_file():
        raise ValueError(f"{path} is not a training checkpoint (training_state.pt missing)")
    if record["config"] != config.training_values():
        changed = sorted(k for k in set(record["config"]) | set(config.training_values())
                         if record["config"].get(k) != config.training_values().get(k))
        raise ValueError(f"Resume config differs from the checkpoint's config: {changed}")
    if record["world_size"] != world_size:
        raise ValueError(f"Checkpoint was written by {record['world_size']} ranks, now {world_size}")
    return record


def retain_checkpoints(directory, keep):
    found = sorted(p for p in Path(directory).glob("step_*") if (p / CHECKPOINT_FILE).is_file())
    for path in found[:-keep]:
        shutil.rmtree(path)


def run_training(config, *, run_name, resume=None, max_steps=None):
    world = World(config.device)
    target = config.max_steps if max_steps is None else min(max_steps, config.max_steps)
    out = Path(config.output_root) / run_name
    tracker = None
    stopping = [False]
    handlers = {sig: signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
    started = time.monotonic()
    status = {"run_name": run_name, "training_status": "initializing", "step": 0}
    try:
        record = check_resume(resume, config, world.size) if resume else None
        world.leader(lambda: out.mkdir(parents=True, exist_ok=resume is not None))
        data = ProprioceptionEpisodes(config)
        inputs = data.inputs()
        if record is not None and record["inputs"] != inputs:
            raise ValueError("Resume data normalization differs from the checkpoint's")
        payload = {"config": asdict(config), "inputs": inputs, "environment": environment(world),
                   "resume": str(resume) if resume else None}

        def start_tracking():
            nonlocal tracker
            write_json(out / "run.json", payload)
            tracker = Tracker(out, config, run_name, payload)
        world.leader(start_tracking)
        seed_all(config.seed)
        model, groups = build_model(config, world, Path(resume) / "transformer" if resume else None)
        model.train()
        optimizer, scheduler = make_optimizer(model, config)
        wrapped = DDP(model, device_ids=[world.local_rank] if world.device.type == "cuda" else None,
                      broadcast_buffers=False, find_unused_parameters=True,
                      gradient_as_bucket_view=True) if world.size > 1 else model
        sampler = EpisodeSampler(data.split["train"], seed=config.seed, rank=world.rank, world_size=world.size)
        seed_all(config.seed + world.rank)
        step = 0
        if resume:
            state = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            sampler.load_state_dict(state["ranks"][world.rank]["sampler"])
            restore_rng(state["ranks"][world.rank]["rng"], world.device)
            step = state["step"]
            del state
        if not step < target:
            raise ValueError(f"Nothing to do: checkpoint step {step} >= target {target}")
        flow = ProprioceptionFlow(world.device, gradient_accumulation_steps=config.accumulation)
        eval_flow = ProprioceptionFlow(world.device)
        checkpoint_record = {"config": config.training_values(), "world_size": world.size, "inputs": inputs}
        checkpoint = None

        def evaluate():
            result = validation(model, data, eval_flow, world)
            world.leader(lambda: tracker.log(result, step))

        def save():
            nonlocal checkpoint
            ranks = world.gather({"rng": rng_state(world.device), "sampler": sampler.state_dict()})
            checkpoint = world.leader(lambda: save_checkpoint(
                model, optimizer, scheduler, out / "checkpoints" / f"step_{step:08d}", step=step,
                ranks=ranks, record=checkpoint_record))
            world.leader(lambda: retain_checkpoints(out / "checkpoints", config.keep_checkpoints))

        world.leader(lambda: (out / "checkpoints").mkdir(parents=True, exist_ok=True))
        evaluate()
        status["training_status"] = "running"
        while step < target:
            tick = time.monotonic()
            batches = error = None
            try:
                batches = [data.batch(sampler.next()) for _ in range(config.accumulation)]
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            if not world.all(error is None):
                raise RuntimeError("Data read failure: " + str(world.gather(error)))
            metrics = train_step(wrapped, model, optimizer, scheduler, flow, batches, world, config)
            world.synchronize()
            elapsed = max(world.reduce([time.monotonic() - tick], maximum=True))
            step += 1
            status["step"] = step
            metrics["train/seconds_per_step"] = elapsed
            if world.device.type == "cuda":
                metrics["system/peak_allocated_bytes"] = max(world.reduce(
                    [torch.cuda.max_memory_allocated(world.device)], maximum=True))
            world.leader(lambda: tracker.log(metrics, step))
            over_time = config.walltime_seconds and time.monotonic() - started > config.walltime_seconds - 300
            stop = not world.all(not stopping[0] and not over_time)
            if step % config.validation_every == 0 or step == target or stop:
                evaluate()
            if step % config.checkpoint_every == 0 or step == target or stop:
                save()
            if stop:
                break
        status.update(training_status="completed" if step == config.max_steps else "stopped", step=step,
                      target_steps=target, checkpoint=checkpoint, groups=groups,
                      elapsed_seconds=time.monotonic() - started)
        world.leader(lambda: write_json(out / "result.json", status))
        world.leader(lambda: tracker.finish(status["training_status"]))
        return status
    except BaseException as exc:
        status.update(training_status="failed", error=f"{type(exc).__name__}: {exc}")
        if world.rank == 0 and out.is_dir():
            write_json(out / "result.json", status)
            if tracker:
                tracker.finish("failed")
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        world.close()
