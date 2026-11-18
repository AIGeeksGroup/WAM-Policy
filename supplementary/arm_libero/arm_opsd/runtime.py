"""Training orchestration on one node: build the teacher cache, train a student, export it.

``train`` runs under torchrun with ``method.world_size`` ranks. Each rank owns one CPU LIBERO stream
(streaming / collected-once variants), a frozen teacher copy and the LoRA student; rank 0 writes the
checkpoints, the student exports and the run's events. The OPSD student draws, per rank and update,
3 streaming contexts labelled by the teacher and 3 teacher-cache entries (24 + 24 over 8 ranks).
"""
from contextlib import ExitStack
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import traceback
from uuid import uuid4

import numpy as np
import torch

from arm_opsd.assets.base_model import PromptBank, check_base_model
from arm_opsd.assets.resets import load_manifest
from arm_opsd.checkpoint import (export_student, load_training_checkpoint, read_training_checkpoint,
                                 retain_checkpoints, save_training_checkpoint, check_resume_config)
from arm_opsd.config import ArmConfig, canonical_hash
from arm_opsd.distributed import DistributedContext, RoundAborted
from arm_opsd.model.codec import ObservationEncoder
from arm_opsd.model.history import HistoryStore
from arm_opsd.model.kv_views import NativeRole
from arm_opsd.model.loading import load_native_transformer, load_roles
from arm_opsd.model.lora import inject_lora, load_adapter_state_dict
from arm_opsd.model.sampler import NativeSampler
from arm_opsd.monitoring.failure_report import write_rank_failure
from arm_opsd.opsd.learner import StreamingLearner
from arm_opsd.opsd.targets import HeldoutSet, TargetCache, TargetCacheWriter, stratified_heldout
from arm_opsd.streaming.collector import StreamCollector, TaskQueue
from arm_opsd.streaming.scheduler import EpisodeSchedule


def json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


class EventLog:
    """Append-only events.jsonl of one rank; rank 0 also mirrors the curves to W&B when configured."""

    def __init__(self, path, logger=None):
        self.path, self.logger = Path(path), logger

    def __call__(self, record):
        with self.path.open("a") as stream:
            stream.write(json.dumps(json_value(record), allow_nan=False, sort_keys=True) + "\n")
        if self.logger is not None:
            self.logger.log(json_value(record))


def read_site(site):
    """The optional `wandb` / `hub` sections of a site file (machine paths are merged by ArmConfig.load)."""
    return json.loads(Path(site).read_text()) if site else {}


def export_snapshot(output, learner, config, update):
    """students/update-N (raw) and, once the EMA exists, students/update-N-ema."""
    export_student(output / f"students/update-{update:06d}", learner.roles.student, config, update=update)
    ema = learner.ema_state()
    if ema is not None:
        export_student(output / f"students/update-{update:06d}-ema", learner.roles.student, config,
                       update=update, adapter=ema, variant="ema")


def point_latest(output, destination, update):
    temporary = Path(output) / "latest.json.incomplete"
    temporary.write_text(json.dumps({"checkpoint": str(destination), "update": update}) + "\n")
    temporary.replace(Path(output) / "latest.json")


def agreed(distributed, callback, phase, *, rank0=False):
    error, result = None, None
    try:
        if not rank0 or distributed.rank == 0:
            result = callback()
    except Exception as exc:
        error = exc
    distributed.consensus(error, phase)
    return result


def preflight_output(output, config, resume=None):
    """A new run needs a new output directory; a resumed one must keep its configuration and must not
    leave later checkpoints or snapshots behind."""
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(f"A new run requires a new output directory: {output}")
    if not output.exists():
        return
    original = output / "resolved-config.json"
    if original.is_file():
        check_resume_config(json.loads(original.read_text()), config)
    update = json.loads((Path(resume) / "completion.json").read_text())["counters"]["student_updates"]
    for path in (output / "checkpoints").glob("round-*/completion.json"):
        if json.loads(path.read_text())["counters"]["student_updates"] > update:
            raise ValueError("Later checkpoints exist; resume into a new output directory")
    for path in (output / "students").glob("update-*/arm_libero.json"):
        if json.loads(path.read_text())["updates"] > update:
            raise ValueError("Later student exports exist; resume into a new output directory")


def initialize_run_metadata(output, config, resume=None):
    output = Path(output)
    preflight_output(output, config, resume)
    output.mkdir(parents=True, exist_ok=bool(resume))
    path = output / "resolved-config.json"
    if not path.exists():
        path.write_text(json.dumps(config.to_dict(), indent=2) + "\n")


def require_runtime(config, *, training=False):
    m = config.machine
    missing = [name for name in ("model_root", "output_dir", "reset_manifest") if not getattr(m, name)]
    if config.method.collects or training:
        missing += [name for name in ("cpu_python", "worker_config") if not getattr(m, name)]
    if missing:
        raise ValueError(f"Site machine paths required: {missing}")
    if m.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Training needs CUDA or ROCm GPUs (machine.device=cuda)")
    if m.network_dtype != "bfloat16":
        raise ValueError("The LingBot-VA networks run in BF16")
    if training and ((config.method.offline_contexts_per_rank > 0 and config.method.objective == "regression")
                     or config.experiment.heldout_contexts) and not m.offline_cache:
        raise ValueError("Training needs machine.offline_cache (the build-teacher-cache output)")
    if training and config.method.objective == "sft_fm" and not m.sft_dataset:
        raise ValueError("SFT only needs machine.sft_dataset")


def _inputs(config, prompt_path):
    check_base_model(config.machine.model_root)
    manifest = load_manifest(config.machine.reset_manifest, require_full_splits=True)
    return manifest, PromptBank(prompt_path)


def _seed(config, rank):
    random.seed(config.experiment.seed + rank)
    np.random.seed(config.experiment.seed + rank)
    torch.manual_seed(config.experiment.seed)


def _vae_encoder(config, device):
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(str(Path(config.machine.model_root) / "vae"),
        torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
    return ObservationEncoder(vae)


def _stream(config, distributed, manifest, prompts, encoder, role, sampler, history, *, run_id,
            rank_dir, stack, output, target_sink=None):
    """One CPU LIBERO worker, the shared episode schedule and this rank's stream collector."""
    env = json.loads(Path(config.machine.worker_config).read_text())
    if config.machine.cpu_affinity:
        if len(config.machine.cpu_affinity) < distributed.world_size:
            raise ValueError("CPU affinity pool needs at least one core per rank")
        assigned = config.machine.cpu_affinity[distributed.rank::distributed.world_size]
        env["ARM_OPSD_CPU_AFFINITY"] = ",".join(map(str, assigned))
    from arm_opsd.sim.libero.worker_client import WorkerClient
    stderr = stack.enter_context((rank_dir / f"worker-{run_id}.stderr.log").open("wb"))
    worker = WorkerClient(python=config.machine.cpu_python, env=env,
                          timeout=config.machine.worker_timeout_s, stderr=stderr)
    schedule_path = output / "schedules" / f"{run_id}.sqlite"
    agreed(distributed, lambda: EpisodeSchedule.create(schedule_path, manifest, run_id,
            config.experiment.seed), "global schedule creation", rank0=True)
    schedule = EpisodeSchedule(schedule_path, manifest["content_key"], config.experiment.seed)
    queue = TaskQueue(manifest, Path(config.machine.reset_manifest).parent, schedule,
                      rank=distributed.rank, segment=run_id)
    collector = StreamCollector(worker, encoder, history, role, sampler, prompts, queue, run_id,
        rank=distributed.rank, artifact_dir=rank_dir / "episodes" / run_id, training=True, target_sink=target_sink)
    return worker, schedule, collector


def train(config_path, prompt_path, *, site=None, resume=None):
    config = ArmConfig.load(config_path, site=site)
    require_runtime(config, training=True)
    if int(os.environ.get("RANK", "0")) == 0:
        # Read-only check before any model loads; torchrun stops the other ranks if it fails.
        preflight_output(config.machine.output_dir, config, resume)
    extras = read_site(site)
    method = config.method
    distributed = worker = events = learner = None
    uploader, logger = None, None
    try:
        distributed = DistributedContext(device="cuda", initialize=True, timeout_s=config.machine.process_group_timeout_s)
        if distributed.world_size != method.world_size:
            raise ValueError(f"torchrun started {distributed.world_size} ranks; method.world_size is {method.world_size}")
        _seed(config, distributed.rank)
        manifest, prompts = _inputs(config, prompt_path)
        roles = load_roles(config.machine.model_root, device=distributed.device, dtype=torch.bfloat16,
                           rank=method.lora_rank, alpha=method.lora_alpha)
        if config.machine.activation_checkpointing:
            roles.student.model.enable_gradient_checkpointing()
        learner = StreamingLearner(roles, HistoryStore(), prompts, config, distributed)
        encoder = _vae_encoder(config, distributed.device)
        heldout_ids = []
        if method.objective == "sft_fm":
            from arm_opsd.sft_only.dataset import OfficialLiberoDataset
            dataset = agreed(distributed, lambda: OfficialLiberoDataset(config.machine.sft_dataset, q01=config.native.q01,
                                                                         q99=config.native.q99), "demonstration dataset")
            learner.flow_matching_data, learner.flow_matching_empty_text = dataset, dataset.empty_text()
        # Held-out contexts come from the cache also when a variant draws no training context from it.
        if (method.offline_contexts_per_rank > 0 and method.objective == "regression") or config.experiment.heldout_contexts:
            cache = agreed(distributed, lambda: TargetCache(config.machine.offline_cache, config=config), "teacher cache")
            if config.experiment.heldout_contexts:
                heldout_ids = stratified_heldout(cache.manifest["entries"], config.experiment.heldout_contexts,
                                                 config.experiment.seed)
                cache = TargetCache(config.machine.offline_cache, config=config, exclude=heldout_ids)
            learner.offline_cache = cache
        adapters = None
        if distributed.rank == 0:
            # Built before any restore: A0 is the seeded LoRA initialization, also when resuming.
            try:
                from arm_opsd.monitoring.adapter_metrics import AdapterMetrics
                adapters = AdapterMetrics(learner.roles.student.model)
            except Exception:
                traceback.print_exc()
        output = Path(config.machine.output_dir)
        run_id = distributed.gather(uuid4().hex)[0]
        agreed(distributed, lambda: initialize_run_metadata(output, config, resume), "output initialization", rank0=True)
        rank_dir = output / f"rank-{distributed.rank:02d}"
        rank_dir.mkdir(exist_ok=True)
        if distributed.rank == 0:
            from arm_opsd.monitoring.hub_upload import SnapshotUploader
            from arm_opsd.monitoring.wandb_logger import WandbLogger
            logger = WandbLogger.from_site(extras.get("wandb"), name=output.name, group=output.name,
                                           config=config.training_sections(), directory=rank_dir)
            uploader = SnapshotUploader.from_site(extras.get("hub"))
        events = EventLog(rank_dir / "events.jsonl", logger)

        def journal_uploads():
            if uploader is not None:
                for record in uploader.poll():
                    events(record)
        with ExitStack() as stack:
            schedule = collector = None
            if method.collects:
                worker, schedule, collector = _stream(config, distributed, manifest, prompts, encoder,
                    learner.roles.student, learner.sampler, learner.history, run_id=run_id,
                    rank_dir=rank_dir, stack=stack, output=output)
            stream_closed = collector is None
            visual_updates, visual_case = set(), None
            if distributed.rank == 0:
                from arm_opsd.monitoring.visualization import schedule as visual_schedule, select_case
                try:
                    visual_updates = visual_schedule(method.student_updates, every=config.machine.visualization_every)
                    visual_case = select_case(manifest, Path(config.machine.reset_manifest).parent) if visual_updates else None
                except Exception as exc:
                    events({"event": "prediction_video_disabled", "error": repr(exc)})
                    visual_updates = set()
            visual_updates = set(distributed.gather(sorted(visual_updates))[0])

            def visualize(update):
                """Closed-loop prediction video on rank 0; a failure is logged, never raised."""
                def call():
                    from arm_opsd.monitoring.visualization import rollout
                    try:
                        record = rollout(learner.roles.student, encoder, prompts, visual_case, config=config,
                            output=rank_dir / "visualization", update=update,
                            seed=int(canonical_hash([config.experiment.seed, "visualization", visual_case["case_id"]])[:15], 16))
                    except Exception as exc:
                        traceback.print_exc()
                        record = {"event": "prediction_video_failed", "optimizer_step": update, "error": repr(exc)}
                    events(record)
                agreed(distributed, call, "prediction video", rank0=True)

            def evaluate_heldout(update):
                """Rank 0: held-out regression loss of the current raw student; logged, never raised."""
                def call():
                    if learner.heldout is None:
                        return
                    started = time.perf_counter()
                    try:
                        metrics = learner.evaluate(learner.heldout.pairs)
                        learner.history.drop_views(learner.heldout.journal_ids())
                        events({"event": "heldout", "optimizer_step": update, "metrics": metrics,
                                "elapsed_s": time.perf_counter() - started})
                    except Exception as exc:
                        traceback.print_exc()
                        events({"event": "heldout_failed", "optimizer_step": update, "error": repr(exc)})
                agreed(distributed, call, "held-out evaluation", rank0=True)

            def close_stream():
                nonlocal stream_closed, worker
                if stream_closed:
                    return
                agreed(distributed, collector.finish_session, "final episode budget stop")
                agreed(distributed, schedule.stop, "stop scheduler", rank0=True)
                worker.close()
                worker = None
                stream_closed = True

            heldout_dir = output / "heldout"
            if resume:
                restored = agreed(distributed, lambda: load_training_checkpoint(resume, learner), "checkpoint recovery")
                if collector is not None:
                    if restored["control_state"] is None:
                        raise ValueError("Resuming a streaming run requires the global scheduler state")
                    agreed(distributed, lambda: schedule.restore(restored["control_state"], run_id),
                           "global scheduler recovery", rank0=True)
                    collector.resume(restored["runtime"], run_id)
                if config.experiment.heldout_contexts:
                    def reload_heldout():
                        if not heldout_dir.exists():
                            # Resume into a new output directory: <run>/checkpoints/round-N -> <run>/heldout.
                            shutil.copytree(Path(resume).resolve().parents[1] / "heldout", heldout_dir)
                        heldout = HeldoutSet.load(heldout_dir, config=config)
                        heldout.attach(learner.history)
                        learner.heldout = heldout
                    agreed(distributed, reload_heldout, "held-out recovery", rank0=True)
                events({"event": "resume", "segment_id": collector.segment_id if collector else None,
                        "checkpoint": str(resume), "counters": asdict(learner.counters)})
            else:
                save_training_checkpoint(output / "checkpoints/round-000000", learner,
                    runtime_state=collector.runtime_state() if collector else {},
                    control_state=schedule.snapshot() if schedule else {"offline_only": True})
                agreed(distributed, lambda: point_latest(output, output / "checkpoints/round-000000", 0),
                       "initial recovery pointer", rank0=True)
                agreed(distributed, lambda: export_student(output / "students/update-000000",
                    learner.roles.student, config, update=0), "initial student export", rank0=True)
                if config.experiment.heldout_contexts:
                    started = time.perf_counter()
                    def from_cache():
                        full = TargetCache(config.machine.offline_cache, config=config)
                        heldout = HeldoutSet([full.load(entry, source="heldout") for entry in full.manifest["entries"]
                                              if entry["id"] in set(heldout_ids)])
                        heldout.save(heldout_dir, config=config)
                        heldout.attach(learner.history)
                        learner.heldout = heldout
                    agreed(distributed, from_cache, "held-out from cache", rank0=True)
                    events({"event": "heldout_collected", "path": str(heldout_dir),
                            "count": config.experiment.heldout_contexts, "elapsed_s": time.perf_counter() - started})
                    evaluate_heldout(0)
                if 0 in visual_updates:
                    visualize(0)
            while learner.counters.student_updates < method.student_updates:
                round_id = learner.counters.attempted_rounds + 1  # Monotonic over collection-only and student rounds.
                start = time.perf_counter()
                collected = []
                if learner.collecting:
                    error = None
                    try:
                        collected = collector.collect(round_id)
                    except Exception as exc:
                        error = exc
                    distributed.consensus(error, "environment collection")
                    events({"event": "collection", "segment_id": collector.segment_id,
                            "round": round_id, "collection_ms": (time.perf_counter() - start) * 1000,
                            "timings": collector.last_timings,
                            "contexts": [asdict(row.snapshot) for row in collected],
                            "execution": [asdict(row) for row in collected]})
                collection_s = time.perf_counter() - start
                metrics = learner.train_round([row.snapshot for row in collected], round_id=round_id)
                metrics["time"]["collection_s"] = collection_s
                collect_round = metrics["collect_round"]
                if collector is not None:
                    collector.prune(keep=learner.retained_journals())
                    def rebuild_student():
                        """Rebuild the new student's KV view of the live stream before the next prediction."""
                        if not collector.terminal and learner.collecting:
                            context = collector.journal.snapshot(actor_version=learner.roles.student.version,
                                round_id=round_id + 1, slot_id=0)
                            positive, negative = prompts(context.task_id, context.instruction)
                            learner.history.view(learner.roles.student, context, text_emb=positive,
                                                 negative_text_emb=negative)
                    agreed(distributed, rebuild_student, "post-update student replay")
                else:
                    learner.release()
                if learner.roles.student.device.type == "cuda":
                    torch.cuda.synchronize(learner.roles.student.device)
                    metrics["hbm"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(learner.roles.student.device),
                                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(learner.roles.student.device)}
                if collector is not None and (not learner.collecting
                                              or learner.counters.student_updates == method.student_updates):
                    close_stream()  # Collected once after its collection rounds; every mode at the budget.
                if collector is not None:
                    task_counts = distributed.gather(collector.stats)
                    metrics["task_totals"] = {str(t): {key: sum(row[str(t)][key] for row in task_counts)
                                              for key in collector.stats[str(t)]} for t in range(10)}
                rank_losses = distributed.gather(metrics["losses"])
                metrics["global_losses"] = {key: sum(row[key] for row in rank_losses if key in row)
                                                 / sum(key in row for row in rank_losses)
                                            for key in metrics["losses"]}
                metrics["schedule"] = schedule.snapshot() if schedule is not None else None
                update = learner.counters.student_updates
                if adapters is not None and not collect_round and update % 10 == 0:
                    metrics.update(adapters.log_values(learner.roles.student.model, learner.student_ema, config, update))
                events({"event": "round_complete", "segment_id": collector.segment_id if collector else None,
                        "round_ms": (time.perf_counter() - start) * 1000, **metrics})
                # A collection-only round keeps update 0; the last one is saved too (the stream is closed by
                # then), so a later failure does not collect again.
                if not collect_round and (update % config.experiment.heldout_every == 0 or update == method.student_updates):
                    evaluate_heldout(update)
                collection_finished = collect_round and not learner.collecting
                if collection_finished or (not collect_round and (update % config.experiment.checkpoint_every == 0
                                                                  or update == method.student_updates)):
                    destination = output / ("checkpoints/round-000000-collected" if collect_round else f"checkpoints/round-{update:06d}")
                    checkpoint_started = time.perf_counter()
                    save_training_checkpoint(destination, learner,
                                             runtime_state=collector.runtime_state() if collector else {},
                                             control_state=schedule.snapshot() if schedule else {"offline_only": True})
                    protected = (*config.experiment.protected_checkpoints, method.student_updates)
                    def retain_latest():
                        point_latest(output, destination, update)
                        retain_checkpoints(output / "checkpoints", config.experiment.retain_checkpoints, protected=protected)
                    agreed(distributed, retain_latest, "checkpoint retention", rank0=True)
                    events({"event": "checkpoint_complete", "update": update,
                            "elapsed_s": time.perf_counter() - checkpoint_started, "path": str(destination)})
                if not collect_round and (update in config.experiment.student_snapshots or update == method.student_updates):
                    agreed(distributed, lambda: export_snapshot(output, learner, config, update),
                           "student snapshot export", rank0=True)
                    if uploader is not None:
                        snapshots = {"raw": output / f"students/update-{update:06d}"}
                        if (output / f"students/update-{update:06d}-ema").is_dir():
                            snapshots["ema"] = output / f"students/update-{update:06d}-ema"
                        events({"event": "hub_upload_queued", "update": update, "keys": uploader.submit(update, snapshots)})
                if not collect_round and update in visual_updates:
                    visualize(update)
                journal_uploads()
                distributed.barrier()  # All logging/checkpoint snapshots precede any next reset.

            def complete():
                (output / "completion.json").write_text(json.dumps({"status": "training_complete",
                    "counters": asdict(learner.counters), "data_mode": method.data_mode,
                    "objective": method.objective}, indent=2) + "\n")
            agreed(distributed, complete, "training completion", rank0=True)
            if uploader is not None:
                pending = uploader.close(wait=True)
                journal_uploads()
                events({"event": "hub_upload_closed", "pending": pending})
                uploader = None
    except Exception as exc:
        if events is not None:
            events({"event": "run_aborted", "error": repr(exc), "time": time.time()})
            write_rank_failure(events.path.parent, exc, traceback.format_exc(), rank=distributed.rank,
                               counters=asdict(learner.counters) if learner is not None else None)
        raise
    finally:
        if uploader is not None:
            uploader.close(wait=False)
        if logger is not None:
            logger.finish()
        if worker is not None:
            worker.close()
        if distributed is not None:
            distributed.close()


def build_target_cache(config_path, prompt_path, output, *, contexts, site=None):
    """The teacher cache: the frozen teacher acts in every rank's stream with its 20/50-step sampler, and
    each chunk's (noise, ODE endpoint) pair is written as a regression target.

    Every control-flow decision after a collective is taken on gathered values, so all ranks run the
    same sequence of collectives and leave the loop after the same round, once the global entry count
    reaches `contexts`."""
    config = ArmConfig.load(config_path, site=site)
    require_runtime(config)
    if contexts < 1:
        raise ValueError("A positive context budget is required")
    output = Path(output)
    distributed = worker = writer = None
    owned = False  # A failure writes into the output only if this build created it.
    round_id = 0
    try:
        distributed = DistributedContext(device="cuda", initialize=True, timeout_s=config.machine.process_group_timeout_s)
        _seed(config, distributed.rank)
        manifest, prompts = _inputs(config, prompt_path)
        # Rank 0 alone decides: another rank checking existence could see the directory rank 0 just made.
        agreed(distributed, lambda: output.mkdir(parents=True, exist_ok=False), "cache output", rank0=True)
        owned = True
        teacher = NativeRole(load_native_transformer(config.machine.model_root, device=distributed.device,
                                                     dtype=torch.bfloat16), "teacher")
        sampler = NativeSampler(config.native, state_steps=config.method.teacher_video_steps,
                                action_steps=config.method.teacher_action_steps)
        history = HistoryStore()
        encoder = _vae_encoder(config, distributed.device)
        run_id = distributed.gather(uuid4().hex)[0]
        rank_dir = output / f"rank-{distributed.rank:02d}"
        rank_dir.mkdir(exist_ok=True)
        events = EventLog(rank_dir / "events.jsonl")
        writer = TargetCacheWriter(output, config=config, source="teacher_rollout",
                                   prefix=f"{distributed.rank:02d}-", teacher_version=0)
        # Every collected context is kept (offline_max_prefix covers the training horizon), so the budget
        # needs ceil(contexts / (3 * world)) rounds; the bound stops only a build whose streams stopped
        # producing entries, which then fails below instead of writing a short cache.
        round_limit = 4 * math.ceil(contexts / (3 * distributed.world_size))
        started = time.perf_counter()
        counts = [0] * distributed.world_size
        with ExitStack() as stack:
            worker, schedule, collector = _stream(config, distributed, manifest, prompts, encoder, teacher, sampler,
                history, run_id=run_id, rank_dir=rank_dir, stack=stack, output=output, target_sink=writer.add)
            while sum(counts) < contexts and round_id < round_limit:
                round_id += 1
                round_started = time.perf_counter()
                error = None
                try:
                    collector.collect(round_id)
                    collector.prune()
                    events({"event": "cache_round", "round": round_id, "entries": len(writer.entries),
                            "skipped": writer.skipped, "round_ms": (time.perf_counter() - round_started) * 1000,
                            "timings": collector.last_timings})
                except Exception as exc:
                    error = exc
                distributed.consensus(error, "teacher collection")
                rows = distributed.gather((round_id, len(writer.entries), writer.skipped))
                if len({row[0] for row in rows}) != 1:
                    raise RoundAborted("cache round: ranks are in different rounds")
                counts = [row[1] for row in rows]
                if distributed.rank == 0:
                    events({"event": "cache_progress", "round": round_id, "global_entries": sum(counts),
                            "rank_entries": counts, "global_skipped": sum(row[2] for row in rows),
                            "requested": contexts, "round_ms": (time.perf_counter() - round_started) * 1000})
            agreed(distributed, collector.finish_session, "cache episode stop")
            agreed(distributed, schedule.stop, "stop scheduler", rank0=True)
        entries = [e for rows in distributed.gather(writer.entries) for e in rows]
        skipped = sum(distributed.gather(writer.skipped))
        if len(entries) < contexts:
            # A short cache gets no manifest, so training refuses the directory.
            raise RoundAborted(f"cache build delivered {len(entries)} of {contexts} contexts in {round_id} rounds")
        agreed(distributed, lambda: writer.finish(entries=entries, extra={"requested_contexts": contexts,
               "world_size": distributed.world_size, "skipped": skipped, "rank_entries": counts, "rounds": round_id}),
               "cache manifest", rank0=True)
        summary = {"output": str(output), "count": len(entries), "requested": contexts, "skipped": skipped,
                   "rank_entries": counts, "rounds": round_id, "elapsed_s": time.perf_counter() - started}
        events({"event": "cache_complete", **summary})
        return summary
    except BaseException as exc:
        if owned:
            write_rank_failure(output / f"rank-{distributed.rank:02d}", exc, traceback.format_exc(),
                               rank=distributed.rank, entries=len(writer.entries) if writer is not None else None,
                               round=round_id)
        raise
    finally:
        if worker is not None:
            worker.close()
        if distributed is not None:
            distributed.close()


def export_checkpoint(config_path, checkpoint, destination, *, variant="ema", site=None, device="cuda"):
    """A student export (arm_libero.json + student_adapter.pt) from a complete training checkpoint."""
    config = ArmConfig.load(config_path, site=site)
    state = read_training_checkpoint(checkpoint, config)
    adapter = state["student"] if variant == "raw" else state.get("student_ema")
    if adapter is None:
        raise ValueError("Checkpoint has no EMA student (before ema_start_update)")
    model = load_native_transformer(config.machine.model_root, device=device, dtype=torch.bfloat16)
    inject_lora(model, rank=config.method.lora_rank, alpha=config.method.lora_alpha, dropout=config.method.lora_dropout)
    load_adapter_state_dict(model, adapter)  # Checks the adapter tensors against the real base.
    role = NativeRole(model, "student", state["actor_version"])
    update = state["counters"]["student_updates"]
    export_student(destination, role, config, update=update, adapter=adapter, variant=variant)
    return {"export": str(destination), "student_update": update, "variant": variant}
