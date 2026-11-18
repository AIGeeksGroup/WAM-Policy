"""Model-side streaming actor, per-round feedback barrier and thread-safe request queue.

Each rank serves one QLM training-environment worker (``python -m qlm_bench.training_env worker``)
over the QLM live RPC protocol. A round holds chunks_per_clip chunks: the actor predicts, the worker
executes and reports feedback, and the last feedback of the round is held until the learner has
updated the student and the actor cache is rebuilt for the new version."""
from dataclasses import dataclass, replace
from collections import Counter
from concurrent.futures import Future
import copy
import math
import queue
import threading
import time
import numpy as np
import torch
from wam_policy.inference.session import OnlineSession, packet_digest
from wam_policy.opsd.config import digest
from wam_policy.opsd.history import EpisodeJournal, ContextRecord, cpu_copy, tensor_hash
from wam_policy.opsd.sampling import generate, draw_noise


# The stop reasons the QLM worker accepts (SynchronousDriver.run_episode refuses any other).
STOP_REASONS = ('training_budget_end', 'smoke_budget_end', 'safe_stop')
BUDGET_END = 'training_budget_end'


def plan_error(plan, latent):
    """State plan vs the real feedback latent of the same group (same model latent space).

    Diagnostic only: None instead of a non-finite value or an exception."""
    try:
        diff = plan.detach().double().cpu() - latent.detach().double().cpu()
        result = {"mae": float(diff.abs().mean()), "rmse": float(diff.square().mean().sqrt())}
        return result if all(math.isfinite(v) for v in result.values()) else None
    except Exception:
        return None


def terminal_outcome(task, terminal, committed_groups):
    """Mutually exclusive outcome by QLM termination reason; length from the terminal tick."""
    reason = terminal.get("reason", terminal.get("termination_reason"))
    flags = {"success": reason == "task_success", "fallen": reason == "robot_fall", "timed_out": reason == "timeout"}
    if reason not in ("task_success", "robot_fall", "timeout"):
        flags = {k: bool(terminal.get(k)) for k in flags}
    tick = terminal.get("tick")
    seconds = tick * .002 if isinstance(tick, int) and not isinstance(tick, bool) else committed_groups * .32
    return {"task": task, "reason": reason, "groups": committed_groups, "seconds": seconds, **flags}


class ActorSampler:
    def __init__(self, runtime, normalizer, *, contract=None, share_history=False, steps=None):
        self.runtime, self.model, self.normalizer = runtime, runtime.model, normalizer
        # Only the training actor shares its cache with the learner's student role.
        self.share_history = share_history
        # (video, action) Euler steps; None is the student/deployment sampler of the config.
        # The teacher acting at its deployment steps builds the offline target cache.
        self.steps = None if steps is None else tuple(steps)
        self.last_plan_error = None
        self.config, self.device, self.dtype = runtime.config, runtime.device, runtime.dtype
        self.contract = copy.deepcopy(contract or {})
        self.cache, self.journal, self.last_record = None, None, None
        self.ready = self.pending = False
        self.generator = None
        self.round_id = 0
        self.episode_request = None
        self.last_diagnostics = {}

    def spawn(self):
        """A new, independent sampler on the same model/runtime (own KV cache leases, journal, RNG and
        episode state; same normalizer, contract and steps): one per concurrent episode."""
        return type(self)(self.runtime, self.normalizer, contract=self.contract, steps=self.steps)

    def begin_episode(self, request):
        self.episode_request = copy.deepcopy(request)

    def clear(self):
        if self.cache is not None:
            self.cache.close()
        self.cache = None
        self.ready = self.pending = False

    @torch.no_grad()
    def reset(self, latent, text_emb, *, seed, empty_text_emb=None):
        if self.episode_request is None or type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Bound episode identity and policy seed required")
        self.clear()
        request = self.episode_request
        episode = tuple(request[k] for k in ("session_id", "episode_id", "reset_epoch"))
        if not self.config.tiny and tuple(text_emb.shape) != (1, 512, 4096):
            raise ValueError("Prompt embedding must be [1,512,4096]")
        task = request["body"]["task"]
        self.journal = EpisodeJournal(episode, latent, text_emb,
            contract={**self.contract, "task": task["id"], "instruction_sha256": digest(task["instruction"])})
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.ready = True
        self.refresh()

    def snapshot(self, observation_id="actor"):
        return self.journal.snapshot(actor_version=self.runtime.version, round_id=self.round_id,
                    observation_id=observation_id, tick=self.journal.length*160)

    @torch.no_grad()
    def predict(self):
        if not self.ready or self.pending:
            raise ValueError("Confirmed real history required before generation")
        snap = self.snapshot(getattr(self, "observation_id", "actor"))
        if self.cache is None:
            self.cache = self.runtime.prepare(ContextRecord(snap, self.journal))
        if self.share_history:
            # Pure history at this prefix: plans and queries below write only after this copy.
            self.runtime.remember(self.cache, snap.prefix_digest)
        rng_before = cpu_copy(self.generator.get_state())
        video_steps, action_steps = self.steps or (None, None)
        # The noise is drawn exactly as generate() would draw it, then kept with the record.
        eps_z = draw_noise(self.generator, self.runtime, action=False)
        z = generate(self.cache, None, initial_noise=eps_z, steps=video_steps)
        self.cache.condition(z)
        eps_a = draw_noise(self.generator, self.runtime, action=True)
        a = generate(self.cache, None, action=True, initial_noise=eps_a, steps=action_steps)
        self.cache.forward(a, 0., action=True, update=1, count="plan_cache")
        commands = self.normalizer.decode(a[0,:,0,:,0].float().cpu().numpy().T)
        commands[:,6:] = 0.
        if commands.shape != (16,9) or not np.isfinite(commands).all():
            raise ValueError("Finite physical native9 group required")
        self.last_model_action = cpu_copy(a)
        self.last_plan = cpu_copy(z)
        self.last_record = ContextRecord(snap, self.journal, requested=commands.copy(), rng_before=rng_before,
                                        model_normalized=cpu_copy(a), noise=(cpu_copy(eps_z), cpu_copy(eps_a)),
                                        plan=cpu_copy(z))
        self.pending = True
        self.last_diagnostics = {"actor_version": self.runtime.version, "round_id": self.round_id,
            "prefix_digest": snap.prefix_digest, "state_plan_sha256": tensor_hash(z),
            "model_action_sha256": tensor_hash(a), "requested_physical_sha256": digest(commands.tolist())}
        return commands

    @torch.no_grad()
    def commit(self, latent, executed_commands):
        if not self.pending:
            raise ValueError("No pending actor chunk")
        normalized = self.normalizer.encode(np.asarray(executed_commands, np.float32))
        action = torch.from_numpy(normalized.T.copy()).reshape(1,9,1,16,1)
        group = self.journal.length + 1
        self.journal.append(latent, action, tick=group*160, receipt=getattr(self,"receipt_id",str(group)))
        self.last_plan_error = plan_error(self.last_plan, latent)
        self.cache.commit(latent, action, group)
        self.cache.group = group + 1
        self.cache.plan_set = False
        self.pending = False

    def suspend(self):
        if self.pending:
            raise ValueError("Cannot train before actual feedback")
        if self.cache is not None:
            self.cache.close()
            self.cache = None

    @torch.no_grad()
    def refresh(self):
        self.suspend()
        if self.ready and not self.journal.terminal:
            self.cache = self.runtime.prepare(ContextRecord(self.snapshot(), self.journal))

    def cache_summary(self):
        return self.cache.summary() if self.cache is not None else {"observed_units":[],"predicted_units":[]}


@dataclass
class DeferredFeedback:
    key: tuple
    response: dict


class Coordinator:
    def __init__(self, sampler, codec, prompt_encoder):
        self.sampler = sampler
        self.session = OnlineSession(sampler, codec, prompt_encoder, info={"role": "training"})
        self.expected_segment = None
        self.expected_pool_sha256 = None
        self.records, self.waiting = [], None
        self.seen = {}
        self.aliases = {}
        self.feedback_ids = {}
        self.stopped = False
        self.physical_chunks = self.completed_commands = 0
        self.events = Counter()
        # Per-round diagnostics, drained by the engine after each round.
        self.outcomes, self.plan_errors = [], []

    def check_reset(self, reset):
        """The worker must belong to this run: same train pool and episode-schedule segment (the
        other fields of the reset extension are informational)."""
        binding = reset.get("extensions", {}).get("training", {})
        problems = [name for name, expected in (("train_pool_sha256", self.expected_pool_sha256),
                                                ("segment", self.expected_segment))
                    if expected is not None and binding.get(name) != expected]
        if problems:
            raise ValueError("Training reset from a QLM worker of another run: " + ", ".join(problems))

    def handle(self, request):
        key = tuple(request[k] for k in ("session_id","episode_id","reset_epoch","request_id"))
        signature = packet_digest(request)
        if key in self.seen:
            old, response = self.seen[key]
            if old != signature:
                self.events['conflicts'] += 1
                raise ValueError("Conflicting duplicate training request")
            self.events['duplicates'] += 1
            return self.seen[self.aliases.get(key, key)][1]
        method = request["method"]
        # RPC IDs are transport identities; ExecutionAck IDs identify physical
        # effects. OnlineSession also deduplicates acks, so mirror that boundary
        # here before incrementing slots (including delayed aliases after release).
        feedback_id = None
        if method == "feedback":
            feedback_id = key[:3] + (request['payload']['ack']['message_id'],)
            feedback_signature = packet_digest(request['payload'])
            if feedback_id in self.feedback_ids:
                old, original_key = self.feedback_ids[feedback_id]
                if old != feedback_signature:
                    self.events['conflicts'] += 1
                    raise ValueError('Conflicting duplicate execution feedback')
                self.events['duplicates'] += 1
                response = self.seen[original_key][1]
                # Keep aliases pointed at the original result: a DeferredFeedback
                # must become the final release response for every retry.
                self.aliases[key] = original_key
                self.seen[key] = (signature, response)
                return response
        if self.waiting is not None:
            raise ValueError("Feedback barrier has not released this stream")
        if method == "predict" and self.stopped:
            raise ValueError("Training budget exhausted; no further predict")
        if method == "reset":
            if not self.sampler.config.tiny:
                self.check_reset(request["payload"]["reset"])
            self.sampler.begin_episode(request["payload"]["reset"])
        if method == "predict":
            self.sampler.observation_id = request["payload"]["observation_id"]
        record = self.sampler.last_record if method == "feedback" else None
        if method == "feedback":
            self.sampler.receipt_id = request["payload"]["ack"]["message_id"]
            self.sampler.last_plan_error = None
        response = self.session.handle(request)
        if method == 'reset':self.events['resets'] += int(not response.get('duplicate',False))
        if method == "feedback":
            if record is None or len(self.records) >= self.sampler.config.chunks_per_clip:
                raise ValueError("Unpaired or excess round feedback")
            payload = request["payload"]
            ack = payload["ack"]["body"]
            terminal = payload.get("terminal")
            task = record.journal.contract.get("task", "unspecified")
            if terminal is not None:
                record.journal.terminal = True
                self.events['terminals'] += 1
                self.outcomes.append(terminal_outcome(task, terminal, record.journal.length))
            if self.sampler.last_plan_error is not None:
                self.plan_errors.append({"task": task, **self.sampler.last_plan_error})
            executed = [r["executed"] for r in ack["commands"] if r["status"] == "completed"]
            self.records.append(replace(record, executed=copy.deepcopy(executed), terminal=copy.deepcopy(terminal)))
            self.physical_chunks += 1
            self.completed_commands += len(executed)
            if len(self.records) == self.sampler.config.chunks_per_clip:
                self.sampler.suspend()
                self.waiting = DeferredFeedback(key, response)
                response = self.waiting
        self.seen[key] = (signature, response)
        if feedback_id is not None:
            self.feedback_ids[feedback_id] = (feedback_signature, key)
        return response

    def release(self, *, student_steps, stop=False, stop_reason='training_budget_end'):
        if stop_reason not in STOP_REASONS:  # Checked at every release, not only the stopping one.
            raise ValueError(f"The QLM worker refuses stop reason {stop_reason!r}")
        if self.waiting is None or len(self.records) != self.sampler.config.chunks_per_clip:
            raise ValueError("Only a complete round may release feedback")
        self.sampler.round_id = student_steps
        start=time.monotonic()
        self.sampler.refresh()
        if self.sampler.device.type=='cuda':torch.cuda.synchronize(self.sampler.device)
        self.cache_rebuild_s=time.monotonic()-start
        response = {**self.waiting.response, "stop_after_feedback": bool(stop),
                    "training": {"student_version": self.sampler.runtime.version,
                                 "stop_reason":stop_reason,
                                 "student_updates_successful": student_steps}}
        key = self.waiting.key
        self.seen[key] = (self.seen[key][0], response)
        self.waiting, self.records, self.stopped = None, [], bool(stop)
        return response

    def metadata(self):
        j = self.sampler.journal
        return {"physical_chunks":self.physical_chunks,"completed_commands":self.completed_commands,
                "stream_events":dict(self.events),
                "episode":None if j is None else j.episode,"prefix_end":None if j is None else j.length,
                "pending_feedback":None if self.waiting is None else list(self.waiting.key)}


class QueuedService:
    """RPC threads never run Torch, mutate codecs or enter distributed collectives."""
    def __init__(self, coordinator, timeout):
        self.coordinator, self.timeout = coordinator, timeout
        self.queue = queue.Queue()
        self.lock, self.requests = threading.Lock(), {}
        self.feedback_requests = {}
        self.pending = None
        self.pending_since = None
        self.last_barrier_wait_s = 0.

    def __call__(self, request):
        key = tuple(request[k] for k in ("session_id","episode_id","reset_epoch","request_id"))
        sha = packet_digest(request)
        with self.lock:
            if key in self.requests:
                signature, future = self.requests[key]
                if signature != sha:
                    raise ValueError("Conflicting queued request")
            else:
                feedback_id = None
                if request.get('method') == 'feedback':
                    feedback_id = key[:3] + (request['payload']['ack']['message_id'],)
                    feedback_sha = packet_digest(request['payload'])
                if feedback_id is not None and feedback_id in self.feedback_requests:
                    signature, future = self.feedback_requests[feedback_id]
                    if signature != feedback_sha:
                        raise ValueError('Conflicting queued execution feedback')
                else:
                    future = Future()
                    if feedback_id is not None:
                        self.feedback_requests[feedback_id] = (feedback_sha, future)
                    self.queue.put((copy.deepcopy(request), future))
                self.requests[key] = (sha, future)
        return future.result(timeout=self.timeout)

    def collect(self, timeout):
        deadline = time.monotonic() + timeout
        while self.coordinator.waiting is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Stream collection timeout")
            try:request, future = self.queue.get(timeout=remaining)
            except queue.Empty as exc:raise TimeoutError('Stream collection timeout') from exc
            try:
                response = self.coordinator.handle(request)
                if isinstance(response, DeferredFeedback):
                    self.pending = future
                    self.pending_since = time.monotonic()
                else:
                    future.set_result(response)
            except BaseException as exc:
                future.set_exception(exc)
                raise
        return tuple(self.coordinator.records)

    def prepare_release(self, **kwargs):
        # Caller reaches collective consensus after all caches are actually ready.
        self.prepared_response=self.coordinator.release(**kwargs)

    def publish_release(self):
        self.pending.set_result(self.prepared_response)
        self.last_barrier_wait_s=time.monotonic()-self.pending_since
        self.pending = None

    def release(self, **kwargs):
        self.prepare_release(**kwargs)
        self.publish_release()

    def fail(self, exc):
        with self.lock:
            for _, future in self.requests.values():
                if not future.done():
                    future.set_exception(exc)

    def drain_end(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            request, future = self.queue.get(timeout=max(.001,deadline-time.monotonic()))
            try:
                result = self.coordinator.handle(request)
                future.set_result(result)
                if request["method"] == "end":
                    return
            except Exception as exc:
                future.set_exception(exc)
                raise
        raise TimeoutError("Final QLM end was not acknowledged")
