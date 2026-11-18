"""Streaming contexts: one persistent CPU LIBERO stream per rank, driven by the current student.

A student update changes the derived KV views, never the physical facts of the stream."""
import time
from pathlib import Path
from uuid import uuid4
import numpy as np
import torch
from arm_opsd.records import CollectedContext
from arm_opsd.model.history import EpisodeJournal
from arm_opsd.model.codec import ActionCodec
from arm_opsd.assets.resets import materialize_case


class TaskQueue:
    """Per-rank handle to the shared ledger; contains no independent task deck."""
    split = "train"

    def __init__(self, manifest, directory, schedule, *, rank=0, segment):
        self.directory, self.schedule = Path(directory), schedule
        self.rank, self.segment, self.episode_index = rank, segment, 0
        self.cases = {c["case_id"]: c for c in manifest["cases"] if c["split"] == "train"}
        self.ticket = None

    def next(self):
        self.ticket = self.schedule.reserve(self.segment, self.rank, self.episode_index)
        return materialize_case(self.cases[self.ticket["case_id"]], self.directory)

    def confirm(self):
        self.schedule.confirm(self.ticket["id"])
        self.episode_index += 1

    def cancel(self):
        if self.ticket is not None:
            self.schedule.cancel(self.ticket["id"])

    def finish(self, reason):
        self.schedule.finish(self.ticket["id"], reason)
        self.ticket = None

    def state_dict(self):
        return {"episode_index": self.episode_index, "rank": self.rank, "split": self.split}

    def load_state_dict(self, state):
        if state["rank"] != self.rank or state["split"] != self.split:
            raise ValueError("Queue state belongs to another rank or split")
        self.episode_index, self.ticket = state["episode_index"], None


class StreamCollector:
    def __init__(self, worker, encoder, history, role, sampler, prompt_lookup, queue,
                 stream_tag, *, rank=0, artifact_dir=None, generation_rng=None,
                 training=False, target_sink=None):
        if queue.split != "train":
            raise ValueError("Streaming collection accepts only train reset cases")
        self.worker, self.encoder, self.history = worker, encoder, history
        self.role, self.sampler, self.prompt_lookup, self.queue = role, sampler, prompt_lookup, queue
        self.stream_tag, self.rank = stream_tag, rank
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.generation_rng = generation_rng or torch.Generator(device=role.device).manual_seed(42 + rank)
        self.journal = None
        self.terminal = True
        self.segment_id = uuid4().hex
        self.case = None
        self.training = training
        # target_sink(TargetSample, journal): a teacher-driven stream records its own (noise, endpoint)
        # pairs as offline targets (build-target-cache; TargetCacheWriter.add).
        self.target_sink = target_sink
        self.stats = {str(t): {"contexts": 0, "actions": 0, "successes": 0, "training_timeouts": 0, "partial_terminals": 0} for t in range(10)}
        self.last_timings = {}

    def _save(self, name, value):
        if self.artifact_dir:
            directory = self.artifact_dir / self.journal.episode_id
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / name
            if destination.exists():
                raise FileExistsError(destination)
            torch.save(value, destination)

    def reset(self):
        case = self.queue.next()
        episode_id = uuid4().hex
        try:
            if self.training:
                from arm_opsd.sim.libero.rollout_contract import training_profile
                profile = training_profile(case["task_id"])
                result = self.worker.reset(case, episode_id, rollout_profile=profile)
                if result.get("rollout_profile") != profile:
                    raise ValueError("Worker did not bind the training horizon")
            else:
                result = self.worker.reset(case, episode_id)
        except Exception:
            self.queue.cancel()
            raise
        self.queue.confirm()
        self.encoder.reset()
        initial = self.encoder.encode(result["observation"])
        self.case = case
        self.journal = EpisodeJournal(stream_id=f"rank-{self.rank}", session_id=self.worker.session_id,
            episode_id=episode_id, task_id=case["task_id"], instruction=case["instruction"],
            initial_latent=initial, resource_hash=self.stream_tag,
            reset_receipt={k: v for k, v in result.items() if k != "observation"})
        self.history.add(self.journal)
        self.terminal = False
        self._save("initial.pt", {"initial_latent": self.journal.initial_latent,
            "case_id": case["case_id"], "reset": self.journal.reset_receipt,
            "journal_id": self.journal.journal_id, "instruction": case["instruction"], "segment_id": self.segment_id})

    @torch.no_grad()
    def collect(self, round_id):
        contexts = []
        self.last_timings = {"generation_s": 0., "feedback_encode_s": 0., "worker_roundtrip_s": 0.,
                             "cpu_step_render_s": 0., "cpu_physics_s": 0., "cpu_render_s": 0.}
        for slot in range(3):
            if self.terminal:
                self.reset()
            snapshot = self.journal.snapshot(actor_version=self.role.version, round_id=round_id, slot_id=slot)
            positive, negative = self.prompt_lookup(snapshot.task_id, snapshot.instruction)
            view = self.history.view(self.role, snapshot, text_emb=positive, negative_text_emb=negative)
            if self.role.device.type == "cuda":
                torch.cuda.synchronize(self.role.device)
            generation_started = time.perf_counter()
            sample = self.sampler.joint(self.role, view, first=snapshot.first,
                initial_latent=self.journal.initial_latent.to(self.role.device, self.role.dtype),
                generator=self.generation_rng)
            if self.role.device.type == "cuda":
                torch.cuda.synchronize(self.role.device)
            self.last_timings["generation_s"] += time.perf_counter() - generation_started
            commands = ActionCodec().decode(sample.action, first=snapshot.first)
            worker_started = time.perf_counter()
            result = self.worker.step_chunk(commands, episode_id=snapshot.episode_id,
                chunk_id=snapshot.prefix, actor_version=self.role.version)
            self.last_timings["worker_roundtrip_s"] += time.perf_counter() - worker_started
            self.last_timings["cpu_step_render_s"] += result["cpu_step_render_seconds"]
            self.last_timings["cpu_physics_s"] += result.get("cpu_physics_seconds", 0.)
            self.last_timings["cpu_render_s"] += result.get("cpu_render_seconds", 0.)
            applied = np.asarray(result["applied_actions"])
            if (not 0 < len(applied) <= len(commands) or result.get("episode_id") != snapshot.episode_id
                    or result.get("chunk_id") != snapshot.prefix or result.get("actor_version") != self.role.version):
                raise ValueError("Execution receipt does not match the requested chunk")
            if not np.array_equal(applied, commands[:len(applied)]):
                raise ValueError("Requested/applied policy command mismatch")
            if len(result["observations"]) != len(applied):
                raise ValueError("Every executed action requires its actual observation")
            receipt_id = result.get("receipt_id") or result.get("request_id")
            if receipt_id is None:
                raise ValueError("Execution receipt requires a receipt id")
            self.terminal = bool(result["terminal"])
            if self.target_sink is not None:
                from arm_opsd.opsd.targets import TargetSample
                self.target_sink(TargetSample.from_joint(snapshot, sample, teacher_version=self.role.version,
                                                         source="offline"), self.journal)
            stats = self.stats[str(snapshot.task_id)]
            stats["contexts"] += 1
            stats["actions"] += len(applied)
            stats["successes"] += int(result.get("success", False))
            stats["training_timeouts"] += int(result.get("terminal_reason") == "training_timeout")
            stats["partial_terminals"] += int(self.terminal and len(applied) < len(commands))
            terminal_observation = ({k: np.asarray(v).copy() for k, v in result["observations"][-1].items()}
                                    if self.terminal and result["observations"] else None)
            self._save(f"chunk-{snapshot.prefix:04d}.pt", {"snapshot": snapshot.__dict__,
                "receipt": {k: v for k, v in result.items() if k != "observations"},
                "terminal_observation": terminal_observation})
            if self.terminal:
                self.queue.finish(result["terminal_reason"])
                self.journal.finish({**{k: v for k, v in result.items() if k != "observations"},
                                     "terminal_observation": terminal_observation})
            else:
                encode_started = time.perf_counter()
                latents = self.encoder.encode(result["observations"])
                if self.role.device.type == "cuda":
                    torch.cuda.synchronize(self.role.device)
                self.last_timings["feedback_encode_s"] += time.perf_counter() - encode_started
                normalized = ActionCodec().encode_executed(applied, first=snapshot.first)
                record = self.journal.append(receipt_id=receipt_id, latents=latents,
                    normalized_actions=normalized, applied_actions=applied, actor_version=self.role.version)
                self._save(f"feedback-{snapshot.prefix:04d}.pt", record.__dict__)
            contexts.append(CollectedContext(snapshot, receipt_id, len(applied), self.terminal))
        return contexts

    def runtime_state(self):
        return {"segment_id": self.segment_id, "stats": self.stats, "task_queue": self.queue.state_dict(),
                "collection_rng": self.generation_rng.get_state(),
                "episode_id": self.journal.episode_id if self.journal else None,
                "prefix": len(self.journal.records) if self.journal else 0, "terminal": self.terminal}

    def resume(self, state, new_segment_id):
        self.queue.load_state_dict(state["task_queue"])
        self.generation_rng.set_state(state["collection_rng"])
        self.segment_id = new_segment_id
        self.stats = state.get("stats", self.stats)
        self.journal, self.terminal = None, True

    def finish_session(self):
        if self.journal is not None and not self.terminal:
            self.queue.finish("budget_stop")
            self.journal.finish({"terminal_reason": "budget_stop", "actor_version": self.role.version})
            self._save("budget-stop.pt", self.journal.terminal_receipt)
            self.terminal = True

    def prune(self, keep=()):
        """Release every journal except the live stream and `keep` (collected-once/held-out facts)."""
        live = [self.journal.journal_id] if self.journal is not None and not self.terminal else []
        self.history.release(set(live) | set(keep))
        self.history.prune_views({(self.journal.journal_id, len(self.journal.records))} if live else set())
