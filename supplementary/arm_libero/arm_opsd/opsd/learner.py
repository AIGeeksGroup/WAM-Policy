"""OPSD learner: one student optimizer step per round.

Every loss unit is one labelled context. The student integrates its 1-step video and 2-step
action Euler chains from the stored initial noises and regresses the frozen teacher's multi-step
ODE endpoints with a pseudo-Huber distance; the action chain is conditioned on the student's own
stop-gradient plan. Targets are fixed labels, so one update mixes the streaming contexts collected
this round (labelled by the teacher before the update), teacher-cache draws and, for the Collected-once
variant, contexts of its one-time collection. SFT only replaces the regression with the upstream
flow-matching loss on demonstrations."""
from dataclasses import dataclass, asdict
import math
import time
import torch
from torch import nn
from arm_opsd.config import ArmConfig
from arm_opsd.distributed import DistributedContext
from arm_opsd.model.lora import lora_parameters
from arm_opsd.opsd.losses import regression_loss
from arm_opsd.model.sampler import NativeSampler
from arm_opsd.opsd.targets import SamplePool, TargetGenerator

MODALITIES = ("video", "action")


@dataclass
class TrainingCounters:
    student_updates: int = 0
    collection_contexts: int = 0
    offline_contexts: int = 0
    replay_contexts: int = 0
    attempted_rounds: int = 0
    aborted_rounds: int = 0
    collect_once_rounds_done: int = 0


class GradientAttribution:
    """Rank-local split of the accumulated student LoRA gradient into its video and action parts.

    Diagnostic only: reads .grad after each per-unit backward. The last action backward is the
    DDP-synchronized one (already the cross-rank mean), so it is excluded."""

    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.previous = None
        self.sums = {"video": None, "action": None}
        self.norms = {"video": [], "action": []}

    @torch.no_grad()
    def record(self, modality):
        grads = [torch.zeros_like(p) if p.grad is None else p.grad.detach() for p in self.parameters]
        increments = grads if self.previous is None else [g - q for g, q in zip(grads, self.previous)]
        self.norms[modality].append(math.sqrt(sum(float(x.float().square().sum()) for x in increments)))
        if self.sums[modality] is None:
            self.sums[modality] = [x.clone() for x in increments]
        else:
            for total, x in zip(self.sums[modality], increments):
                total.add_(x)
        self.previous = [g.clone() for g in grads]

    def summary(self):
        result = {}
        for modality, norms in self.norms.items():
            if norms:
                result[f"{modality}_grad_norm"] = sum(norms) / len(norms)
        v, a = self.sums["video"], self.sums["action"]
        if v is not None and a is not None:
            dot = sum(float((x.float() * y.float()).sum()) for x, y in zip(v, a))
            nv = math.sqrt(sum(float(x.float().square().sum()) for x in v))
            na = math.sqrt(sum(float(y.float().square().sum()) for y in a))
            if nv > 0 and na > 0:
                result["cosine_video_action"] = dot / (nv * na)
        if result.get("video_grad_norm", 0.) > 0 and "action_grad_norm" in result:
            result["action_over_video_grad"] = result["action_grad_norm"] / result["video_grad_norm"]
        return result


class StudentLossUnit(nn.Module):
    """One regression loss unit; DDP-wrapped around the student model only."""

    def __init__(self, roles, sampler, config):
        super().__init__()
        self.model = roles.student.model
        self.roles, self.sampler, self.config = roles, sampler, config

    def forward(self, modality, sample, views, initial_latent, plan=None):
        if modality not in MODALITIES:
            raise ValueError("modality must be video or action")
        student, context = self.roles.student, sample.context
        first, action = context.first, modality == "action"
        if not action:
            noise = sample.eps_video.to(student.device, student.dtype)
            prediction = self.sampler.state(student, views["student"], first=first,
                                            initial_latent=initial_latent, noise=noise).endpoint
            target, mask = sample.target_video, sample.mask_video
        else:
            if plan is None or plan.requires_grad:
                raise ValueError("Action loss requires the detached student plan")
            view = self.sampler.plan(student, views["student"], plan)
            noise = sample.eps_action.to(student.device, student.dtype)
            prediction = self.sampler.action(student, view, first=first, noise=noise).endpoint
            target, mask = sample.target_action, sample.mask_action
        loss, diagnostics = regression_loss(prediction, target.to(prediction.device), mask.to(prediction.device),
                                            self.config.method.huber_c)
        unweighted = loss.detach()
        weight = self.config.method.action_loss_weight if action else 1.
        if weight != 1.:
            loss = loss * weight  # Only the action term entering backward is weighted.
        return {"loss": loss, "unweighted_loss": unweighted, "plan": prediction.detach() if not action else None,
                "residual_l1": diagnostics["residual_l1"], "valid_count": diagnostics["valid_count"]}


class FlowMatchingLossUnit(nn.Module):
    """SFT only (objective sft_fm): one whole-episode joint FM forward of the student with the upstream
    video + action loss (arm_opsd.sft_only.flow_matching); DDP-wrapped."""

    def __init__(self, roles):
        super().__init__()
        from arm_opsd.sft_only.flow_matching import JointModel, joint_fm_loss
        self.joint, self.loss = JointModel(roles.student.model), joint_fm_loss

    def forward(self, inputs):
        video, action = self.loss(inputs, self.joint(inputs), self.joint.model.patch_size)
        return {"loss": video + action, "video": video, "action": action}


class StreamingLearner:
    def __init__(self, roles, history, prompt_lookup, config=None, distributed=None,
                 generation_rng=None, offline_rng=None, offline_cache=None):
        self.config = (config or ArmConfig()).validate()
        self.roles, self.history, self.prompt_lookup = roles, history, prompt_lookup
        method, exp, native = self.config.method, self.config.experiment, self.config.native
        self.sampler = NativeSampler(native, state_steps=method.state_steps, action_steps=method.action_steps)
        self.teacher_sampler = NativeSampler(native, state_steps=method.teacher_video_steps,
                                             action_steps=method.teacher_action_steps)
        self.distributed = distributed or DistributedContext(device=roles.student.device)
        device = roles.student.device
        seed = exp.seed
        self.generation_rng = generation_rng or torch.Generator(device=device).manual_seed(seed + 1000 + self.distributed.rank)
        self.offline_rng = offline_rng or torch.Generator(device=device).manual_seed(seed + 3000 + self.distributed.rank)
        self.targets = TargetGenerator(roles, self.teacher_sampler, history, prompt_lookup, generator=self.generation_rng)
        self.offline_cache = offline_cache
        # The Collected-once set (unbounded); empty for every other data mode.
        self.pool = SamplePool(rounds=None)
        self.heldout = None
        self.student_optimizer = torch.optim.AdamW(list(lora_parameters(roles.student.model)),
            lr=method.student_lr, betas=exp.adam_betas, eps=exp.adam_epsilon,
            weight_decay=method.weight_decay)
        # Held-out evaluation always uses the regression unit, also under the no-teacher objective.
        self.regression_unit = StudentLossUnit(roles, self.sampler, self.config)
        self.flow_matching = method.objective == "sft_fm"
        self.flow_matching_rng = torch.Generator().manual_seed(seed + 5000 + self.distributed.rank)
        self.flow_matching_data = self.flow_matching_empty_text = None
        self.flow_matching_preparer = None
        if self.flow_matching:
            from arm_opsd.sft_only.flow_matching import JointBatchPreparer
            self.flow_matching_preparer = JointBatchPreparer(roles.student.model.patch_size)
        self.student_unit = self.distributed.wrap(FlowMatchingLossUnit(roles) if self.flow_matching else self.regression_unit)
        self.counters = TrainingCounters()
        self.last_metrics = {}
        self.round_committed = True
        # FP32 EMA of the student adapter (adapter_state_dict names); never used by training or the actor.
        self.student_ema = None

    # ----- data sources -----
    @property
    def collecting(self):
        """Whether the next round takes fresh on-policy contexts from the stream."""
        method = self.config.method
        if method.data_mode == "streaming":
            return True
        if method.data_mode == "offline_only":
            return False
        return self.counters.collect_once_rounds_done < method.collect_once_rounds

    @property
    def expected_onpolicy(self):
        return self.config.method.onpolicy_contexts_per_rank if self.collecting else 0

    def retained_journals(self):
        """Journals the history store must keep across releases (collected-once pool and held-out facts)."""
        keep = self.pool.journal_ids()
        if self.heldout is not None:
            keep |= self.heldout.journal_ids()
        return keep

    def release(self, live_prefixes=()):
        """Drop every journal/view except live streams and retained facts (offline-only rounds)."""
        self.history.release({journal for journal, _ in live_prefixes} | self.retained_journals())
        self.history.prune_views(set(live_prefixes))

    def _ensure_journal(self, journal):
        return self.history.merge(journal)

    def label(self, contexts, source="onpolicy"):
        """Teacher targets for contexts whose journals are already in the history store."""
        return [(self.history.journals[c.journal_id], self.targets.label(c, source=source)) for c in contexts]

    def _draw_offline(self, count):
        if count <= 0:
            return []
        if self.offline_cache is None:
            raise ValueError("Offline cache draws require a target cache (machine.offline_cache)")
        return self.offline_cache.draw(count, self.offline_rng)

    # ----- EMA -----
    def _student_adapter(self):
        return [(name, parameter) for name, parameter in self.roles.student.model.named_parameters()
                if name.endswith((".lora_A", ".lora_B"))]

    @torch.no_grad()
    def _update_ema(self):
        start, decay = self.config.experiment.ema_start_update, self.config.experiment.ema_decay
        if start is None:
            return
        step = self.counters.student_updates
        if step == start:
            self.student_ema = {name: p.detach().float().clone() for name, p in self._student_adapter()}
        elif step > start:
            for name, p in self._student_adapter():
                self.student_ema[name].mul_(decay).add_(p.detach().float(), alpha=1 - decay)

    def ema_state(self):
        """CPU copy for checkpoint/export; None before ema_start_update."""
        if self.student_ema is None:
            return None
        return {name: value.detach().cpu().clone() for name, value in self.student_ema.items()}

    def load_ema_state(self, state):
        if state is None:
            self.student_ema = None
            return
        expected = dict(self._student_adapter())
        if set(state) != set(expected) or any(state[n].shape != p.shape or not torch.isfinite(state[n]).all()
                                              for n, p in expected.items()):
            raise ValueError("Student EMA inventory/shape mismatch")
        self.student_ema = {n: state[n].detach().to(device=p.device, dtype=torch.float32).clone()
                            for n, p in expected.items()}

    def student_lr(self):
        """lr of the next student update: linear warm-up over student_lr_warmup_steps, then constant.
        sft_fm uses the upstream `warmup_constant_lambda` as LambdaLR applies it: lr(k) = base * min(1, k / steps)."""
        steps = self.config.experiment.student_lr_warmup_steps
        base = self.config.method.student_lr
        if not steps:
            return base
        position = self.counters.student_updates if self.flow_matching else self.counters.student_updates + 1
        return base * min(1., position / steps)

    # ----- views and consensus helpers -----
    def _views(self, context, names=("student",)):
        positive, negative = self.prompt_lookup(context.task_id, context.instruction)
        return {name: self.history.view(getattr(self.roles, name), context, text_emb=positive,
                                        negative_text_emb=negative) for name in names}

    def _initial(self, context):
        return self.history.journals[context.journal_id].initial_latent.to(
            self.roles.student.device, self.roles.student.dtype)

    def _forward(self, wrapper, *args, **kwargs):
        error, output = None, None
        try:
            output = wrapper(*args, **kwargs)
            if not torch.isfinite(output["loss"]).all():
                raise FloatingPointError("nonfinite loss")
        except Exception as exc:
            error = exc
        self.distributed.consensus(error, "loss forward")
        return output

    def _prepare(self, callback, phase):
        """Agree before any rank enters the next fixed DDP loss unit."""
        error, result = None, None
        try:
            result = callback()
        except Exception as exc:
            error = exc
        self.distributed.consensus(error, phase)
        return result

    def _step(self, role, optimizer):
        self.distributed.finite_gradients(role.model, role.role)
        norm = torch.nn.utils.clip_grad_norm_(list(lora_parameters(role.model)), self.config.method.grad_clip)
        self.distributed.consensus(None if torch.isfinite(norm) else FloatingPointError("nonfinite norm"), "clip")
        error = None
        try:
            optimizer.step()
        except Exception as exc:
            error = exc
        self.distributed.consensus(error, "optimizer step")
        role.version += 1
        self.history.invalidate(role)
        return float(norm.detach().cpu())

    def _unit_outputs(self, unit, journal, sample, *, backward=None):
        """Video then action forward of one labelled context. With `backward` (training) every
        preparation is a collective consensus; without it (rank-local evaluation) nothing is."""
        journal = self._ensure_journal(journal)
        context = sample.context
        agreed = (lambda callback, phase: self._prepare(callback, phase)) if backward is not None else (lambda callback, phase: callback())
        views, initial = agreed(lambda: (self._views(context), self._initial(context)), "student history preparation")
        plan, outputs = None, {}
        for modality in MODALITIES:
            if backward is None:
                out = unit(modality, sample, views, initial, plan)
            else:
                out = backward(modality, sample, views, initial, plan)
            if modality == "video":
                plan = out["plan"]
            outputs[modality] = out
        return outputs

    # ----- one round -----
    def _flow_matching_round(self, round_id):
        """One optimizer step on whole-episode demonstration samples, no teacher."""
        count, world = self.config.method.offline_contexts_per_rank, self.distributed.world_size
        metrics = {"loss_units": [], "collect_round": False, "time": {}, "data": {}}
        history_before = dict(self.history.stats)
        device = self.roles.student.device
        try:
            self.distributed.agree((self.roles.student.version, self.counters.student_updates, round_id), "round version")
            self.round_committed = False
            student_started = time.perf_counter()
            # A smoke run draws the longest episode first, so the worst-case memory shows up at once.
            longest = self.config.profile == "tiny" and self.counters.student_updates == 0
            samples = self._prepare(lambda: self.flow_matching_data.draw(count, self.offline_rng, longest=longest), "demonstration draw")
            metrics["sft_latent_frames"] = [int(sample["latents"].shape[1]) for sample in samples]
            self.student_optimizer.zero_grad(set_to_none=True)
            for index, sample in enumerate(samples):
                def batch():
                    return self.flow_matching_preparer({key: value[None].to(device) for key, value in sample.items()},
                                                       generator=self.flow_matching_rng,
                                                       empty_text=self.flow_matching_empty_text.to(device))
                inputs = self._prepare(batch, "demonstration batch")
                final = index == len(samples) - 1
                with self.distributed.accumulation(self.student_unit, final):
                    out = self._forward(self.student_unit, inputs)
                    if final:
                        (out["loss"] / count).backward()
                    else:
                        self._prepare(lambda: (out["loss"] / count).backward(), "local student backward")
                for modality in MODALITIES:
                    value = float(out[modality].detach().cpu())
                    metrics["loss_units"].append({"source": "sft", "task_id": None, "slot": index, "modality": modality,
                                                  "loss": value, "weighted_loss": value})
            lr = self.student_lr()
            for group in self.student_optimizer.param_groups:
                group["lr"] = lr
            metrics["student_lr"] = lr
            metrics["student_grad_norm"] = self._step(self.roles.student, self.student_optimizer)
            self._synchronize()
            metrics["time"]["student_s"] = time.perf_counter() - student_started
            metrics["student_grad_clipped"] = metrics["student_grad_norm"] > self.config.method.grad_clip
            self.counters.student_updates += 1
            # One unit is one whole-episode demonstration here, not one context.
            self.counters.offline_contexts += count * world
            metrics["data"] = {"onpolicy": 0, "offline": 0, "replay": 0, "sft_episodes": count * world}
            self._update_ema()
            self.distributed.agree(asdict(self.counters), "committed counters")
            self.round_committed = True
        except Exception:
            self.counters.aborted_rounds += 1
            raise
        return self._finish_round(metrics, [], history_before)

    def train_round(self, contexts, round_id=None):
        self.counters.attempted_rounds += 1
        method, world = self.config.method, self.distributed.world_size
        contexts = list(contexts)
        if self.flow_matching:
            return self._flow_matching_round(self.counters.attempted_rounds if round_id is None else round_id)
        expected = self.expected_onpolicy
        if round_id is None:
            round_id = contexts[0].round_id if contexts else self.counters.attempted_rounds
        error = None
        if len(contexts) != expected:
            error = ValueError(f"Round requires exactly {expected} fresh on-policy contexts per rank")
        elif any(c.actor_version != self.roles.student.version for c in contexts):
            error = ValueError("All collection contexts must use the current student version")
        elif contexts and (len({(c.journal_id, c.prefix) for c in contexts}) != expected
                           or [c.slot_id for c in contexts] != list(range(expected))
                           or {c.round_id for c in contexts} != {round_id}):
            error = ValueError("Contexts require distinct ordered slots from one round")
        metrics = {"loss_units": [], "collect_round": False, "time": {}, "data": {}}
        history_before = dict(self.history.stats)
        try:
            self.distributed.consensus(error, "collection barrier")
            self.distributed.agree((self.roles.student.version, self.counters.student_updates, round_id), "round version")
            if contexts:
                ids = self.distributed.gather([c.key for c in contexts])
                flat = [tuple(i) for rows in ids for i in rows]
                if len(set(flat)) != expected * world:
                    raise ValueError("Duplicate global contexts")
            self.round_committed = False
            target_started = time.perf_counter()
            labelled = self._prepare(lambda: self.label(contexts, "onpolicy"), "teacher target generation")
            self._synchronize()
            metrics["time"]["target_s"] = time.perf_counter() - target_started
            if method.data_mode == "collect_once" and self.collecting:
                # Collection-only round: label and pool; the student, its version and lr position stay put.
                self.pool.add(round_id, labelled)
                self.counters.collect_once_rounds_done += 1
                self.counters.collection_contexts += len(labelled) * world
                metrics.update(collect_round=True, student_grad_norm=None)
                metrics["data"] = {"onpolicy": len(labelled) * world, "offline": 0, "replay": 0}
                self.distributed.agree(asdict(self.counters), "committed counters")
                self.round_committed = True
                return self._finish_round(metrics, contexts, history_before)
            n_on, n_replay, n_off = method.batch_per_rank(collecting=self.collecting)
            if len(labelled) != n_on:
                raise ValueError("Labelled on-policy count differs from the batch layout")
            replay = self._prepare(lambda: self.pool.draw(n_replay, self.offline_rng), "replay draw")
            offline = self._prepare(lambda: self._draw_offline(n_off), "offline cache draw")
            if n_replay and len(replay) != n_replay:
                raise ValueError("Replay pool cannot serve the batch layout")
            units = labelled + replay + offline
            if not units:
                raise ValueError("Empty update batch")
            student_started = time.perf_counter()
            self.student_optimizer.zero_grad(set_to_none=True)
            attribution = GradientAttribution(lora_parameters(self.roles.student.model))
            for index, (journal, sample) in enumerate(units):
                def backward(modality, sample, views, initial, plan, index=index):
                    final = index == len(units) - 1 and modality == "action"
                    with self.distributed.accumulation(self.student_unit, final):
                        out = self._forward(self.student_unit, modality, sample, views, initial, plan)
                        if final:
                            (out["loss"] / len(units)).backward()
                        else:
                            self._prepare(lambda: (out["loss"] / len(units)).backward(), "local student backward")
                    if not final:
                        attribution.record(modality)
                    return out
                outputs = self._unit_outputs(self.student_unit, journal, sample, backward=backward)
                for modality, out in outputs.items():
                    metrics["loss_units"].append({"source": sample.source, "task_id": sample.context.task_id,
                        "slot": index, "modality": modality, "loss": float(out["unweighted_loss"].cpu()),
                        "weighted_loss": float(out["loss"].detach().cpu()),
                        "residual_l1": float(out["residual_l1"].mean().cpu()),
                        "valid_count": int(out["valid_count"].sum().cpu())})
            lr = self.student_lr()
            for group in self.student_optimizer.param_groups:
                group["lr"] = lr
            metrics["student_lr"] = lr
            metrics["grad"] = attribution.summary()
            metrics["student_grad_norm"] = self._step(self.roles.student, self.student_optimizer)
            self._synchronize()
            metrics["time"]["student_s"] = time.perf_counter() - student_started
            metrics["student_grad_clipped"] = metrics["student_grad_norm"] > method.grad_clip
            self.counters.student_updates += 1
            self.counters.collection_contexts += len(labelled) * world
            self.counters.offline_contexts += len(offline) * world
            self.counters.replay_contexts += len(replay) * world
            metrics["data"] = {"onpolicy": len(labelled) * world, "offline": len(offline) * world,
                               "replay": len(replay) * world}
            self._update_ema()
            self.distributed.agree(asdict(self.counters), "committed counters")
            self.round_committed = True
        except Exception:
            self.counters.aborted_rounds += 1
            raise
        return self._finish_round(metrics, contexts, history_before)

    def _synchronize(self):
        if self.roles.student.device.type == "cuda":
            torch.cuda.synchronize(self.roles.student.device)

    def _finish_round(self, metrics, contexts, history_before):
        units = metrics["loss_units"]
        losses = {}
        for modality in MODALITIES:
            values = [u["loss"] for u in units if u["modality"] == modality]
            if values:
                losses[f"loss/{modality}"] = sum(values) / len(values)
        weighted = [u["weighted_loss"] for u in units if u["modality"] == "action"]
        if weighted:
            losses["loss/action_weighted"] = sum(weighted) / len(weighted)
        if "loss/video" in losses:
            losses["loss/total"] = losses["loss/video"] + losses.get("loss/action_weighted", 0.)
        for source in ("onpolicy", "replay", "offline", "sft"):
            rows = [u for u in units if u["source"] == source]
            if rows:
                video = [u["loss"] for u in rows if u["modality"] == "video"]
                action = [u["weighted_loss"] for u in rows if u["modality"] == "action"]
                losses[f"loss/total_{source}"] = sum(video) / len(video) + sum(action) / len(action)
        metrics["losses"] = losses
        metrics["counters"] = asdict(self.counters)
        metrics["actor_version"] = self.roles.student.version
        metrics["context_ids"] = [c.key for c in contexts]
        metrics["task_ids"] = [c.task_id for c in contexts]
        metrics["history"] = {key: self.history.stats[key] - history_before[key] for key in history_before}
        self.last_metrics = metrics
        return metrics

    # ----- evaluation (no update) -----
    def _unwrapped_unit(self):
        return self.regression_unit if self.flow_matching else getattr(self.student_unit, "module", self.student_unit)

    @torch.no_grad()
    def evaluate(self, pairs, *, unit=None):
        """Mean regression loss of labelled (journal, sample) pairs with the current raw student;
        no optimizer or EMA change, no collective. Rank-local."""
        unit = unit or self._unwrapped_unit()
        weight = self.config.method.action_loss_weight
        totals = {"video": [], "action": [], "video_l1": [], "action_l1": []}
        for journal, sample in pairs:
            outputs = self._unit_outputs(unit, journal, sample)
            totals["video"].append(float(outputs["video"]["unweighted_loss"]))
            totals["action"].append(float(outputs["action"]["unweighted_loss"]))
            totals["video_l1"].append(float(outputs["video"]["residual_l1"].mean()))
            totals["action_l1"].append(float(outputs["action"]["residual_l1"].mean()))
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        video, action = mean(totals["video"]), mean(totals["action"])
        return {"heldout/video": video, "heldout/action": action, "heldout/total": video + weight * action,
                "residual/video_l1": mean(totals["video_l1"]), "residual/action_l1": mean(totals["action_l1"]),
                "heldout/count": len(totals["video"])}

    @torch.no_grad()
    def self_check(self, pairs):
        """S0 self-check: with the student integrated at the teacher's steps the loss must be 0
        (the zero-initialised LoRA makes the student the teacher); at 1/2 steps it must be > 0."""
        teacher_steps = StudentLossUnit(self.roles, self.teacher_sampler, self.config)
        at_teacher = self.evaluate(pairs, unit=teacher_steps)
        at_student = self.evaluate(pairs)
        return {"teacher_steps_total": at_teacher["heldout/total"], "student_steps_total": at_student["heldout/total"],
                "student_steps_video": at_student["heldout/video"], "student_steps_action": at_student["heldout/action"]}
