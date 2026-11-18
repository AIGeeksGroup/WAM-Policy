"""One round: on-policy contexts -> same-noise teacher targets -> one student update.

Every context gets a fixed regression target, the frozen teacher's multi-step ODE endpoint from the
same initial noise (M-OPD: the teacher routed to the context's task). The student runs its 1/2-step
Euler from that noise and regresses onto the target with pseudo-Huber; the action branch is
conditioned on the student's own detached plan. The per-rank batch is the on-policy contexts plus
offline_contexts_per_rank teacher-cache samples (offline_only: cache samples only)."""
from collections import Counter
import math
import time
import torch
from wam_policy.opsd.cache import RoleRuntime
from wam_policy.opsd.sampling import RNGStreams, generate, draw_noise
from wam_policy.opsd.losses import regression_loss
from wam_policy.opsd.distributed import wrap, collective_call, backward_unit, optimizer_step
from wam_policy.opsd.models import canonical_name
from wam_policy.opsd.targets import TargetGenerator


def delta(now, before):
    return {k: v - before.get(k, 0) for k, v in now.items()}


class GradientAttribution:
    """Rank-local split of the student's accumulated LoRA gradient into its video and action parts.

    Diagnostic only: it reads .grad after each per-sample backward and never changes training.
    The last action backward is the DDP-synchronized one, so its increment is already the
    cross-rank mean and is not attributed; action statistics cover the first len(batch)-1
    samples, video statistics all of them."""

    def __init__(self, model):
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.previous = None
        self.sums = {'video': None, 'action': None}
        self.norms = {'video': [], 'action': []}

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
                result[modality + '_grad_norm'] = sum(norms) / len(norms)
        v, a = self.sums['video'], self.sums['action']
        if v is not None and a is not None:
            dot = sum(float((x.float() * y.float()).sum()) for x, y in zip(v, a))
            nv = math.sqrt(sum(float(x.float().square().sum()) for x in v))
            na = math.sqrt(sum(float(y.float().square().sum()) for y in a))
            if nv > 0 and na > 0:
                result['cosine_video_action'] = dot / (nv * na)
        video = result.get('video_grad_norm', 0.)
        if video > 0 and 'action_grad_norm' in result:
            result['action_over_video_grad'] = result['action_grad_norm'] / video
        return result


class Learner:
    def __init__(self, roles, config, world, *, offline=None):
        self.config, self.world, self.roles = config.validate(), world, roles.validate()
        if world.size != config.world_size:
            raise ValueError("Configured/actual rank count mismatch")
        # Optional exact history KV reuse (config.history_reuse); the frozen teacher's never expires.
        self.student = RoleRuntime(roles.student, "student", config, history=True)
        self.teacher = RoleRuntime(roles.teacher, "teacher", config, history=True, experts=roles.experts)
        if config.activation_checkpointing:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl
            for i, block in enumerate(roles.student.blocks):
                roles.student.blocks[i] = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                                                             preserve_rng_state=True)
        self.student_optimizer = torch.optim.AdamW([p for p in roles.student.parameters() if p.requires_grad],
                lr=config.learning_rate, betas=(.9, .999), eps=1e-8,
                weight_decay=config.weight_decay, fused=False, foreach=False)
        warmup = config.lr_warmup_steps
        self.student_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.student_optimizer, lambda step: min(1., (step + 1) / max(1, warmup)))
        self.student_unit = wrap(roles.student, self._student_loss, world)
        self.rng = RNGStreams(config.seed, world.rank, self.student.device)
        self.targets = TargetGenerator(self.teacher, config)
        # Offline target cache (targets.TargetCache) or None. The held-out contexts are task/position-
        # stratified cache entries (never drawn for training); each rank evaluates its share.
        self.offline = offline
        if offline is not None and config.heldout_contexts:
            self.heldout_ids = offline.reserve(config.heldout_contexts, config.seed)[world.rank::world.size]
        else:
            self.heldout_ids = []
        self.student_steps, self.attempted_rounds = 0, 0
        self.heldout = []
        # FP32 EMA of the student adapter (canonical names); never used by training or the actor.
        self.student_ema = self._adapter_copy() if config.ema_start_step == 0 else None
        self.round_committed = True
        self.last_metrics = {}
        self.plan_digests = []
        self.location = {'phase': 'initialized'}

    def _adapter_copy(self):
        return {canonical_name(n): p.detach().clone().float()
                for n, p in self.roles.student.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def _update_ema(self):
        if self.student_steps == self.config.ema_start_step:
            self.student_ema = self._adapter_copy()
        elif self.student_steps > self.config.ema_start_step:
            d = self.config.ema_decay
            for n, p in self.roles.student.named_parameters():
                if p.requires_grad:
                    self.student_ema[canonical_name(n)].mul_(d).add_(p.detach().float(), alpha=1 - d)

    def ema_state(self):
        """CPU copy for checkpoint/export; None before ema_start_step."""
        if self.student_ema is None:
            return None
        return {k: v.detach().cpu().clone() for k, v in self.student_ema.items()}

    def load_ema_state(self, state):
        expected = {canonical_name(n): p for n, p in self.roles.student.named_parameters() if p.requires_grad}
        if state is None:
            self.student_ema = None
            return
        if set(state) != set(expected) or any(state[k].shape != p.shape or not torch.isfinite(state[k]).all()
                                              for k, p in expected.items()):
            raise ValueError("Student EMA inventory/shape mismatch")
        self.student_ema = {k: state[k].detach().to(device=p.device, dtype=torch.float32).clone()
                            for k, p in expected.items()}

    # ----------------------------------------------------------------- targets

    def _target(self, record, key, *, source):
        """Same-noise teacher endpoint for one on-policy context; the noise is drawn once here,
        stored with the sample and reused by the student."""
        self.location.update(role='teacher')
        eps_video = draw_noise(self.rng.generator("target-video", *key), self.student, action=False)
        eps_action = draw_noise(self.rng.generator("target-action", *key), self.student, action=True)
        id = f"r{self.world.rank}-s{self.student_steps}-c{self.attempted_rounds}-{record.snapshot.key[:16]}"
        return self.targets.generate(record, eps_video, eps_action, source=source, id=id)

    def expects_records(self):
        """Whether the next round consumes a complete on-policy slot from the streams."""
        return self.config.streams

    # ------------------------------------------------------------------ student

    def _student_loss(self, model, context, sample, *, action, plan=None):
        if model is not self.student.model:
            raise RuntimeError("Generator escaped its DDP-owned model")
        config = self.config
        self.location.update(role='student')
        if action:
            if plan is None or not context.plan_set:
                raise RuntimeError("Action regression requires the student's detached plan in the context")
            pred = generate(context, None, action=True, initial_noise=sample.eps_action)
            target = sample.target_action.to(pred.device)
            loss, m = regression_loss(pred, target, sample.mask_action.to(pred.device), c=config.huber_c)
            diff = (pred.detach().float() - target.float())
            metrics = {'loss/action': float(loss.detach()), 'residual/action_l1': m['residual_l1'],
                       'action/valid': m['valid'],
                       'residual/action_pose_l1': float(diff[:, :6].abs().mean()),
                       'residual/action_force_l1': float(diff[:, 6:].abs().mean()),
                       'action/raw_force_mean_abs': float(pred.detach()[:, 6:].float().abs().mean()),
                       'action/endpoint_min': float(pred.detach().min()), 'action/endpoint_max': float(pred.detach().max())}
            # lambda_A scales what enters the update; loss/action stays the unweighted value.
            loss = loss * config.action_loss_weight
            metrics['loss/action_weighted'] = float(loss.detach())
            self.plan_digests.append(context.plan_digest)
            return loss, metrics, pred.detach()
        pred = generate(context, None, initial_noise=sample.eps_video)
        target = sample.target_video.to(pred.device)
        loss, m = regression_loss(pred, target, sample.mask_video.to(pred.device), c=config.huber_c)
        metrics = {'loss/video': float(loss.detach()), 'residual/video_l1': m['residual_l1'], 'video/valid': m['valid'],
                   'video/endpoint_min': float(pred.detach().min()), 'video/endpoint_max': float(pred.detach().max())}
        return loss, metrics, pred.detach()

    def train_round(self, records):
        config = self.config
        expected = config.chunks_per_clip if self.expects_records() else 0

        def validate():
            if len(records) != expected or self.student_steps >= config.student_updates:
                raise ValueError("Complete on-policy round (or no records off-stream) and remaining budget required")
            for r in records:
                r.validate()
                if (r.snapshot.actor_version, r.snapshot.round_id) != (self.student.version, self.student_steps):
                    raise ValueError("Mixed/stale actor versions in collection")
            if any(role.active for role in (self.student, self.teacher)):
                raise RuntimeError("Suspend actor cache before learner phase")
        collective_call(self.world, validate)
        versions = self.world.gather((self.student.version, self.student_steps))
        if len(set(versions)) != 1:
            raise ValueError("Rank versions disagree")
        self.attempted_rounds += 1
        self.round_committed = False
        roles = (self.student, self.teacher)
        # The actor and the rank-0 video share the student runtime; report learner-phase deltas only.
        before = {r.role: (Counter(r.counts), Counter(r.history.stats) if r.history else None) for r in roles}
        started = time.monotonic()
        metrics = []
        target_started = time.monotonic()
        onpolicy = []
        for j, record in enumerate(records):
            key = (self.student_steps, record.snapshot.key, j)
            self.location = {'phase': 'target', 'round': self.student_steps, 'context': record.snapshot.key,
                             'prefix_end': record.snapshot.prefix_end, 'role': 'teacher'}
            onpolicy.append(collective_call(self.world, lambda: self._target(record, key, source='onpolicy')))
        target_elapsed = time.monotonic() - target_started
        offline = [] if self.offline is None else self.offline.sample(config.offline_contexts_per_rank,
                                                                       self.rng.generator('draw-offline', self.student_steps))
        batch = [*onpolicy, *offline]
        if not batch:
            raise ValueError("Empty training batch: no on-policy or offline contexts")
        self.student_optimizer.zero_grad(set_to_none=True)
        student_started = time.monotonic()
        attribution = GradientAttribution(self.roles.student)
        for j, sample in enumerate(batch):
            record = sample.record
            self.location = {'phase': 'student', 'round': self.student_steps, 'context': sample.key, 'source': sample.source,
                             'modality': 'video', 'prefix_end': record.snapshot.prefix_end, 'role': 'student'}
            with collective_call(self.world, lambda: self.student.prepare(record)) as c:
                m_video, plan = backward_unit(self.student_unit, self.world, synchronize=False,
                        scale=1/len(batch), args=(c, sample), kwargs={"action": False})
            attribution.record('video')
            # Video graph is already backwarded/released. The action is conditioned on sg(plan).
            with collective_call(self.world, lambda: self.student.prepare(record)) as c:
                self.location.update(modality='action')
                collective_call(self.world, lambda: c.condition(plan))
                m_action, _ = backward_unit(self.student_unit, self.world, synchronize=j == len(batch)-1,
                        scale=1/len(batch), args=(c, sample), kwargs={"action": True, "plan": plan})
            if j < len(batch) - 1:
                attribution.record('action')
            metrics.append({**m_video, **m_action, 'loss/total': m_video['loss/video'] + m_action['loss/action_weighted'],
                            'task': sample.task_id, 'context': sample.key, 'source': sample.source})
        lr = self.student_optimizer.param_groups[0]['lr']
        grad = optimizer_step(self.roles.student, self.student_optimizer, self.world, self.config.grad_clip)
        self.student_scheduler.step()
        self.student_steps += 1
        self.student.advance()
        self._update_ema()
        return self._commit_round(records, metrics, roles, before, started, grad=grad, lr=lr,
                                  attribution=attribution.summary(),
                                  phases={'target': target_elapsed, 'student': time.monotonic()-student_started},
                                  data={'onpolicy': len(onpolicy), 'offline': len(offline)})

    @torch.no_grad()
    def evaluate_heldout(self):
        """Regression loss of the current raw student on the held-out contexts (no update).

        Every rank evaluates its share of the stratified cache entries; the result is the global mean."""
        config = self.config
        started = time.monotonic()
        if any(role.active for role in (self.student, self.teacher)):
            raise RuntimeError("Suspend actor cache before the held-out evaluation")
        if not self.heldout and self.heldout_ids:
            self.heldout = [self.offline.read(i, source='heldout') for i in self.heldout_ids]
        sums = Counter()
        for sample in self.heldout:
            self.location = {'phase': 'heldout', 'round': self.student_steps, 'context': sample.key, 'role': 'student'}
            with self.student.prepare(sample.record) as c:
                z = generate(c, None, initial_noise=sample.eps_video)
                video, _ = regression_loss(z, sample.target_video.to(z.device), sample.mask_video.to(z.device), c=config.huber_c)
            with self.student.prepare(sample.record) as c:
                c.condition(z)
                a = generate(c, None, action=True, initial_noise=sample.eps_action)
                action, _ = regression_loss(a, sample.target_action.to(a.device), sample.mask_action.to(a.device), c=config.huber_c)
            sums['video'] += float(video); sums['action'] += float(action)
            sums['total'] += float(video) + config.action_loss_weight * float(action); sums['count'] += 1
        total = Counter()
        for row in self.world.gather(dict(sums)):
            total.update(row)
        n = total['count']
        result = {k: total[k] / n for k in ('video', 'action', 'total')} if n else {}
        result.update(count=n, elapsed_s=time.monotonic() - started, student_step=self.student_steps)
        return result

    def _commit_round(self, records, metrics, roles, before, started, *, phases, data, grad=None,
                      lr=None, attribution=None):
        final_versions = self.world.gather(self.student.version)
        if len(set(final_versions)) != 1:
            raise RuntimeError("Rank parameters/versions diverged")
        self.round_committed = True
        self.last_metrics = {"student_updates_successful": self.student_steps,
                             "contexts_global": len(records)*self.world.size,
                             "student_lr_applied": lr,
                             # Rank-local video/action split of the accumulated gradient.
                             "student_grad_attribution": attribution,
                             "student_grad_norm": grad, "elapsed_s": time.monotonic()-started,
                             "units": metrics, "calls": {r.role: dict(r.counts) for r in roles},
                             "history_reuse": {r.role: None if r.history is None else dict(r.history.stats) for r in roles},
                             "learner_calls": {r.role: delta(r.counts, before[r.role][0]) for r in roles},
                             "learner_history": {r.role: None if r.history is None else delta(r.history.stats, before[r.role][1])
                                                 for r in roles},
                             "data": dict(data)}
        self.last_metrics['context_ids']=[r.snapshot.key for r in records]
        self.last_metrics['context_prefix_end']=[r.snapshot.prefix_end for r in records]
        self.last_metrics.update(
            task_contexts=dict(Counter(r.journal.contract.get('task','unspecified') for r in records)),
            student_version=self.student.version,
            learning_rates={'student':self.student_optimizer.param_groups[0]['lr']},
            phase_wall_s=phases)
        counts=Counter()
        for local in self.world.gather(self.last_metrics["task_contexts"]):counts.update(local)
        if sum(counts.values())!=len(records)*self.world.size:raise RuntimeError("Global context count mismatch")
        self.last_metrics["task_contexts_global"]=dict(counts)
        return self.last_metrics

    def counters(self):
        return {"student_steps": self.student_steps, "attempted_rounds": self.attempted_rounds,
                "student_version": self.student.version, "round_committed": self.round_committed}

    def restore_counters(self, counters):
        if (not counters["round_committed"] or counters["student_steps"] != counters["student_version"] or
                not 0 <= counters["student_steps"] <= self.config.student_updates):
            raise ValueError("Only committed round checkpoints may resume")
        self.student_steps, self.attempted_rounds = counters["student_steps"], counters["attempted_rounds"]
        self.student.version = counters["student_version"]
        for role in (self.student, self.teacher):
            role.forget()
        self.round_committed = True
