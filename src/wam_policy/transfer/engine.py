"""Whole-episode single-process / DDP SFT with step-exact resume.

Each optimizer step, every rank draws ``accumulation`` whole episodes, runs the JointFlow
video + action loss with the full model trainable, all-reduces gradients (DDP), clips and takes
an AdamW step with linear warm-up. Validation loss over the validation split runs at step 0 and
every ``validation_every`` steps; complete checkpoints every ``checkpoint_every`` steps.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import signal
import sys
import time
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from wam_policy.common.distributed import World, environment, isolated_rng, restore_rng, rng_state, seed_all
from wam_policy.common.tracking import Tracker
from wam_policy.models.checkpoint_io import load_model
from wam_policy.transfer.checkpoint import (checkpoint_name, export_teacher, load_training_state,
                                            retain_checkpoints, save_checkpoint)
from wam_policy.transfer.config import write_json
from wam_policy.transfer.data import EpisodeSampler, FrozenEpisodes
from wam_policy.transfer.flow import JointFlow
from wam_policy.transfer.policy import apply_trainability, parameter_group
from wam_policy.transfer.visualization import maybe_visualize

# The text embedder of the action-conditioning branch is never used in forward.
KNOWN_UNUSED = {f"condition_embedder_action.text_embedder.linear_{n}.{kind}"
                for n in (1, 2) for kind in ("weight", "bias")}


def autocast(world):
    return torch.autocast("cuda", dtype=torch.bfloat16) if world.device.type == "cuda" else nullcontext()


def build_model(config, world, path=None):
    model = load_model(path or config.transformer, dtype=torch.float32, device=world.device,
                       attn_mode=config.attention)
    groups = apply_trainability(model)
    if config.activation_checkpointing:
        from wan_va.distributed.fsdp import apply_ac
        apply_ac(model)  # non-reentrant activation checkpointing per block (no sharding)
    return model, groups


def make_optimizer(model, config):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.learning_rate, betas=tuple(config.betas), eps=1e-8,
                                  weight_decay=config.weight_decay, fused=False, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / config.warmup_steps, 1.0))
    return optimizer, scheduler


def train_step(model, raw_model, optimizer, scheduler, core, batches, world, config):
    optimizer.zero_grad(set_to_none=True)
    loss_video = loss_action = targets = 0.0
    for index, batch in enumerate(batches):
        sync = nullcontext() if index + 1 == len(batches) or not isinstance(model, DDP) else model.no_sync()
        with sync:
            prepared = core.prepare(batch)
            with autocast(world):
                output = model(prepared, train_mode=True)
                losses = core.compute_loss(prepared, output)
            if not world.all(all(bool(torch.isfinite(x)) for x in losses)):
                raise FloatingPointError("Nonfinite loss on at least one rank; no optimizer update")
            sum(losses).backward()
        loss_video += float(losses[0].detach())
        loss_action += float(losses[1].detach())
        targets += int(batch["actions_mask"][:, 0].sum())
    norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.grad_clip_norm)
    if not world.all(bool(torch.isfinite(norm))):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Nonfinite gradients on at least one rank; no optimizer update")
    unused = {name for name, p in raw_model.named_parameters() if p.requires_grad and p.grad is None}
    if not world.all(unused <= KNOWN_UNUSED):
        raise RuntimeError(f"Unexpected parameters without gradients: {sorted(unused - KNOWN_UNUSED)[:10]}")
    lr = optimizer.param_groups[0]["lr"]
    probes = {}
    for name, p in raw_model.named_parameters():
        group = parameter_group(name)
        if p.grad is not None and group not in probes:
            probes[group] = (p, p.detach().flatten()[:256].clone())
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    video, action, grad, valid = world.reduce([loss_video, loss_action, float(norm), targets])
    changes = {"train/parameter_delta/" + group: float((p.detach().flatten()[:256] - before).abs().max())
               for group, (p, before) in probes.items()}
    return {**changes, "train/video_loss": video, "train/action_loss": action, "train/total_loss": video + action,
            "train/grad_norm": grad, "train/lr": lr, "train/valid_actions_global": int(valid * world.size)}


def validation(model, data, core, world, *, seed=1729):
    """Mean validation loss on rank 0 (unwrapped model, fixed RNG); other ranks wait."""
    def evaluate():
        was_training = model.training
        try:
            model.eval()
            values = []
            with isolated_rng(world.device, seed), torch.no_grad():
                for uid in data.split["validation"]:
                    batch = data.batch(uid, split="validation")
                    prepared = core.prepare(batch, chunk_size=2, window_size=8, noisy_video_probability=0)
                    with autocast(world):
                        losses = core.compute_loss(prepared, model(prepared, train_mode=True))
                    values.append([float(x) for x in losses])
            if not values:
                return {}
            if not all(torch.isfinite(torch.tensor(v)).all() for v in values):
                raise FloatingPointError("Nonfinite validation loss")
            video_loss = sum(v[0] for v in values) / len(values)
            action_loss = sum(v[1] for v in values) / len(values)
            return {"validation/total_loss": video_loss + action_loss, "validation/video_loss": video_loss,
                    "validation/action_loss": action_loss, "validation/episodes": len(values)}
        finally:
            model.train(was_training)
    return world.leader(evaluate)


def run_training(config, *, resume=None, stop_after=None, world=None):
    owned_world = world is None
    world = world or World(config.device)
    out = Path(config.output_dir)
    checkpoints = out / "checkpoints"
    if resume is None and world.leader(lambda: any(checkpoints.glob("step_*"))):
        raise FileExistsError(f"{checkpoints} already has checkpoints; pass --resume or use a new output_dir")
    world.leader(lambda: out.mkdir(parents=True, exist_ok=True))
    started = time.monotonic()
    stopping = [False]
    previous = {sig: signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    tracker = None
    try:
        data = FrozenEpisodes(config.cache_root)
        info = {"config": config.resume_values(),
                "action_normalizer": {"offset": list(data.normalizer.offset), "scale": list(data.normalizer.scale)},
                "rgb_profile": data.rgb_profile}
        state = None
        if resume:
            state = load_training_state(resume, config=config, world_size=world.size, normalizer=data.normalizer)

        def start_tracking():
            nonlocal tracker
            record = {"config": asdict(config), "world_size": world.size, "resume": str(resume) if resume else None,
                      "episodes": data.counts(), "environment": environment(world), "command": sys.argv,
                      "started_at_utc": datetime.now(timezone.utc).isoformat()}
            write_json(out / ("run-resume-%s.json" % (state["step"] if state else 0)), record)
            tracker = Tracker(out, name=config.run_name or out.name, group=config.run_name or out.name,
                              job_type="stage1.1-sft", config=record, mode=config.wandb_mode or None,
                              entity=config.wandb_entity or None, project=config.wandb_project or None)
        world.leader(start_tracking)

        seed_all(config.seed)
        if world.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(world.device)
        model, _ = build_model(config, world, Path(resume) / "transformer" if resume else None)
        model.train()
        optimizer, scheduler = make_optimizer(model, config)
        wrapped = DDP(model, device_ids=[world.local_rank] if world.device.type == "cuda" else None,
                      broadcast_buffers=False, find_unused_parameters=True,
                      gradient_as_bucket_view=True) if world.size > 1 else model
        sampler = EpisodeSampler(data.split["train"], seed=config.seed, rank=world.rank, world_size=world.size)
        seed_all(config.seed + world.rank)
        step = 0
        if state:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            sampler.load_state_dict(state["ranks"][world.rank]["sampler"])
            restore_rng(state["ranks"][world.rank]["rng"], world.device)
            step = state["step"]
            del state
        target = config.max_steps if stop_after is None else min(stop_after, config.max_steps)
        if step >= target:
            raise ValueError(f"Checkpoint step {step} has already reached the target {target}")
        core = JointFlow(world.device, gradient_accumulation_steps=config.accumulation)
        eval_core = JointFlow(world.device)

        def evaluate():
            result = validation(model, data, eval_core, world)
            if result:
                world.leader(lambda: tracker.log(result, step))

        def save():
            ranks = world.gather({"rng": rng_state(world.device), "sampler": sampler.state_dict()})

            def write():
                path = save_checkpoint(model, optimizer, scheduler, checkpoints / checkpoint_name(step),
                                       step=step, ranks=ranks, info=info)
                retain_checkpoints(checkpoints, config.keep_checkpoints)
                return str(path)
            return world.leader(write)

        evaluate()
        checkpoint = None
        while step < target:
            tick = time.monotonic()
            error, batches = None, []
            try:
                batches = [data.batch(sampler.next()) for _ in range(config.accumulation)]
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            if not world.all(error is None):
                raise RuntimeError("Data read failure: " + str(world.gather(error)))
            metrics = train_step(wrapped, model, optimizer, scheduler, core, batches, world, config)
            world.synchronize()
            elapsed = max(world.reduce([time.monotonic() - tick], maximum=True))
            step += 1
            metrics.update({"train/seconds_per_step": elapsed,
                            "train/episode_exposures": step * world.size * config.accumulation})
            if world.device.type == "cuda":
                allocated, reserved = world.reduce([torch.cuda.max_memory_allocated(world.device),
                                                    torch.cuda.max_memory_reserved(world.device)], maximum=True)
                metrics.update({"system/peak_allocated_bytes": allocated, "system/peak_reserved_bytes": reserved})
            world.leader(lambda: tracker.log(metrics, step))
            out_of_time = config.walltime_seconds and time.monotonic() - started > max(1, config.walltime_seconds - 180)
            stop = not world.all(not stopping[0] and not out_of_time)
            if step % config.validation_every == 0 or step == target or stop:
                evaluate()
            if config.visualize_every and step % config.visualize_every == 0:
                maybe_visualize(model, data, config, world, tracker, out, step=step)
            if step % config.checkpoint_every == 0 or step == target or stop:
                checkpoint = save()
            if stop:
                break
        status = "completed" if step == config.max_steps else "stopped"
        if status == "completed":
            world.leader(lambda: export_teacher(checkpoint, out / "teacher"))
        world.leader(lambda: tracker.finish(status, summary={"final/step": step}))
        return {"status": status, "step": step, "checkpoint": checkpoint,
                "teacher": str(out / "teacher") if status == "completed" else None}
    except BaseException as exc:
        if tracker is not None:
            try:
                tracker.finish("failed", summary={"failure": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if owned_world:
            world.close()
