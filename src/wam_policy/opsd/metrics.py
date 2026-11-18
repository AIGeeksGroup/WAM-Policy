"""Per-round training metrics (one gather on every rank, aggregation and logging on rank 0) and
rank-0 LoRA displacement diagnostics. Logging never changes the computation: local summaries return
an error entry instead of raising, so every rank always reaches the same gather. Rates are
cumulative within the run. Metrics go to ``metrics.jsonl`` and, when enabled, to W&B
(``wam_policy.common.tracking``; disabled unless WANDB_MODE is online/offline).
"""
from collections import Counter, defaultdict
import math
import statistics
import traceback
import torch
from wam_policy.opsd.models import LoRALinear, canonical_name


def host_rss_gib():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 2**20
    except (OSError, ValueError, IndexError):
        pass
    return None


def _number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value))


def _kinds(counts):
    """history_state + history_action -> history, etc."""
    result = Counter()
    for key, value in counts.items():
        result[key.rsplit('_', 1)[0]] += value
    return result


# Per-task student diagnostics: unit key -> task/<t>/student/<name>.
TASK_KEYS = {'loss/video': 'video_loss', 'loss/action': 'action_loss', 'loss/total': 'total_loss',
             'residual/video_l1': 'video_residual_l1', 'residual/action_l1': 'action_residual_l1'}
# Learner round metrics the upload requires; absence is an error, never a silently empty step.
REQUIRED = ('student_grad_norm', 'elapsed_s', 'task_contexts_global', 'data', 'phase_wall_s')


class RoundTracker:
    def __init__(self):
        self.events = Counter()
        self.terminals, self.contexts = defaultdict(Counter), Counter()
        self.pending_sum = Counter()

    def local(self, learner, records, metrics, coordinator):
        """Rank-local summary of one committed round. Drains the coordinator's diagnostics."""
        outcomes, plan_errors = getattr(coordinator, 'outcomes', []), getattr(coordinator, 'plan_errors', [])
        coordinator.outcomes, coordinator.plan_errors = [], []
        try:
            mean, high, low, total = defaultdict(list), defaultdict(list), defaultdict(list), Counter()
            for unit in metrics['units']:
                for key, value in unit.items():
                    if key in ('task', 'context', 'source'):
                        continue
                    if key in TASK_KEYS:
                        mean[f'task/{unit.get("task", "unspecified")}/student/{TASK_KEYS[key]}'].append(value)
                    if '/' in key and not key.endswith('/valid'):
                        name = 'student/' + key
                        if key.endswith('/endpoint_min'):
                            low[name].append(value)
                        elif key.endswith('/endpoint_max'):
                            high[name].append(value)
                        else:
                            mean[name].append(value)
            phases = metrics.get('phase_wall_s', {})
            for name, value in (('learner_s', metrics.get('elapsed_s')),
                                ('collection_s', metrics.get('collection_wall_s')),
                                ('target_s', phases.get('target')),
                                ('student_s', phases.get('student')),
                                ('heldout_s', (metrics.get('heldout') or {}).get('elapsed_s'))):
                high['time/' + name].append(value)
            # Per-modality student gradient split (rank-local; averaged over ranks on upload).
            for name, value in (metrics.get('student_grad_attribution') or {}).items():
                mean['student/loss/' + name].append(value)
            # Batch composition of this round, summed over ranks on upload.
            for name, value in (metrics.get('data') or {}).items():
                total['data/' + name] += int(value)
            prefixes = [r.snapshot.prefix_end for r in records]
            mean['stream/prefix_end'].extend(prefixes)
            high['stream/prefix_end_max'].extend(prefixes)
            # Learner phase only: actor collection, release refresh and the video are excluded.
            for role, counts in metrics.get('learner_calls', {}).items():
                for kind, value in _kinds(counts).items():
                    mean[f'calls/{role}/{kind}'].append(value)
            for role, stats in metrics.get('learner_history', {}).items():
                for name, value in (stats or {}).items():
                    mean[f'history/{role}/{name}'].append(value)
            events = Counter(getattr(coordinator, 'events', {}))
            for name in ('resets', 'terminals', 'duplicates'):
                total['stream/' + name] += events[name] - self.events[name]
            self.events = events
            device = learner.student.device
            if device.type == 'cuda':
                capacity = torch.cuda.get_device_properties(device).total_memory
                high['memory/hbm_peak_fraction'].append(torch.cuda.max_memory_allocated(device) / capacity)
                high['memory/hbm_allocated_fraction'].append(torch.cuda.memory_allocated(device) / capacity)
            high['memory/host_rss_gib'].append(host_rss_gib())
            clean = lambda bucket: {k: [float(v) for v in values if _number(v)] for k, values in bucket.items()}
            return {'mean': clean(mean), 'max': clean(high), 'min': clean(low), 'sum': dict(total),
                    'outcomes': outcomes, 'plan_errors': plan_errors}
        except Exception as exc:
            traceback.print_exc()
            return {'error': f'{type(exc).__name__}: {exc}', 'outcomes': outcomes, 'plan_errors': plan_errors}

    def log(self, tracking, rows, learner, metrics, control_state):
        """Rank 0: aggregate the gathered rows and upload one W&B step.

        Raises KeyError when the learner round lacks a required metric; returns None only when
        the upload itself fails."""
        missing = [k for k in REQUIRED if k not in metrics]
        if missing:
            raise KeyError('Learner round metrics lack required keys: ' + ', '.join(missing))
        step = learner.student_steps
        data = {'student_updates': step, 'student_grad_norm': metrics['student_grad_norm'],
                'round_seconds': metrics['elapsed_s']}
        merged = {name: defaultdict(list) for name in ('mean', 'max', 'min')}
        total = Counter()
        for row in rows:
            for name, bucket in merged.items():
                for key, values in row.get(name, {}).items():
                    bucket[key].extend(values)
            total.update(row.get('sum', {}))
        data['diagnostics/rank_errors'] = sum(1 for row in rows if row.get('error'))
        data.update({k: sum(v) / len(v) for k, v in merged['mean'].items() if v})
        data.update({k: max(v) for k, v in merged['max'].items() if v})
        data.update({k: min(v) for k, v in merged['min'].items() if v})
        total.update(self.pending_sum)  # Keeps zero counts, unlike Counter addition.
        data.update(total)
        self.pending_sum = Counter()
        # Rank-0 adapter displacement (computed by the engine every adapter_metrics_every steps).
        data.update({k: v for k, v in metrics.items() if k.startswith(('adapter/', 'adapter_ema/')) and _number(v)})
        if _number(metrics.get('student_lr_applied')):
            data['optim/student_lr'] = metrics['student_lr_applied']
        data.update(self.heldout_values(metrics.get('heldout')))
        for task, count in metrics['task_contexts_global'].items():
            self.contexts[task] += count
            data[f'task/{task}/contexts'] = count
        for task, count in self.contexts.items():
            data[f'task/{task}/contexts_total'] = count
        for field, name in (('confirmed', 'episode_starts'), ('completed', 'episodes_completed')):
            for task, count in (control_state or {}).get(field, {}).items():
                data[f'task/{task}/{name}'] = count
        outcomes = [o for row in rows for o in row.get('outcomes', [])]
        data['episode/terminals'] = len(outcomes)
        self._terminals(outcomes)
        overall = Counter()
        for c in self.terminals.values():
            overall.update(c)
        for prefix, c in [*((f'task/{t}/', c) for t, c in self.terminals.items()), ('episode/', overall)]:
            if c['terminals']:
                n = c['terminals']
                data.update({prefix + 'terminals_total': n, prefix + 'success_rate': c['success'] / n,
                             prefix + 'fall_rate': c['fallen'] / n, prefix + 'timeout_rate': c['timed_out'] / n,
                             prefix + 'episode_seconds_mean': c['seconds'] / n})
        errors = [e for row in rows for e in row.get('plan_errors', [])]
        by_task = defaultdict(list)
        for e in errors:
            by_task[e['task']].append(e['mae'])
        if errors:
            data['world_model/plan_mae'] = sum(e['mae'] for e in errors) / len(errors)
            data['world_model/plan_rmse'] = sum(e['rmse'] for e in errors) / len(errors)
        for task, values in by_task.items():
            data[f'task/{task}/plan_mae'] = sum(values) / len(values)
        try:
            tracking.log(data, step=step)
        except Exception:
            traceback.print_exc()
            return None
        return data

    @staticmethod
    def heldout_values(heldout):
        """student/heldout/{video,action,total} from a learner held-out evaluation record."""
        if not heldout or not heldout.get('count'):
            return {}
        return {f'student/heldout/{k}': float(heldout[k]) for k in ('video', 'action', 'total') if _number(heldout.get(k))}

    def _terminals(self, outcomes):
        for o in outcomes:
            c = self.terminals[o['task']]
            c['terminals'] += 1; c['seconds'] += o['seconds']
            c['success'] += o['success']; c['fallen'] += o['fallen']; c['timed_out'] += o['timed_out']

    @staticmethod
    def release_local(service, coordinator):
        try:
            return {'barrier_wait_s': float(service.last_barrier_wait_s),
                    'actor_cache_rebuild_s': float(coordinator.cache_rebuild_s)}
        except Exception as exc:
            return {'error': f'{type(exc).__name__}: {exc}'}

    @staticmethod
    def release_log(tracking, rows, step):
        """Rank 0: feedback-release barrier and actor KV rebuild, merged into the same step."""
        try:
            data = {}
            for key, name in (('barrier_wait_s', 'time/release_barrier_wait_s'),
                              ('actor_cache_rebuild_s', 'time/actor_cache_rebuild_s')):
                values = [row[key] for row in rows if _number(row.get(key))]
                if values:
                    data[name] = max(values)
            if data:
                tracking.log(data, step=step)
            return data
        except Exception:
            traceback.print_exc()
            return None


# ---------------------------------------------------------------- LoRA displacement (rank 0)
# For each target layer W (out x in) with adapter s*B@A, QR of A^T = Q R gives the nonzero squared
# singular values of B@A as the eigenvalues of R B^T B R^T; Frobenius/spectral norms and the top-8
# energy share follow without materializing B@A.


def lr_sum(config, steps):
    """Sum of the student lrs applied in the first `steps` updates (linear warm-up, then constant)."""
    warmup = max(1, config.lr_warmup_steps)
    return sum(config.learning_rate * min(1., (s + 1) / warmup) for s in range(steps))


@torch.no_grad()
def spectral_norm(weight, iterations=100):
    w = weight.float()
    v = torch.ones(w.shape[1], device=w.device) / math.sqrt(w.shape[1])
    for _ in range(iterations):
        v = w.T @ (w @ v)
        v = v / v.norm().clamp_min(1e-30)
    return float((w @ v).norm())


class AdapterMetrics:
    def __init__(self, model):
        self.layers = {}
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, LoRALinear):
                    self.layers[canonical_name(name)] = {
                        'a0': module.lora_A.detach().cpu().clone(), 'scale': module.scale,
                        'w_fro': float(module.base.weight.float().norm()), 'w_spec': spectral_norm(module.base.weight)}

    @torch.no_grad()
    def summary(self, adapters, applied_lr_sum):
        """adapters: canonical layer name -> (A, B). Returns flat finite metrics (no prefix)."""
        fro, spec, top8 = [], [], []
        b_square = b_count = drift = a0_square = 0.
        for name, (a, b) in adapters.items():
            layer = self.layers[name]
            a, b = a.float(), b.float()
            b_square += float(b.square().sum()); b_count += b.numel()
            a0 = layer['a0'].to(a.device)
            drift += float((a - a0).square().sum()); a0_square += float(a0.square().sum())
            r = torch.linalg.qr(a.T, mode='r').R
            gram = (r @ (b.T @ b) @ r.T).double()
            eig = torch.linalg.eigvalsh((gram + gram.T) / 2).clamp_min(0).flip(0) * layer['scale'] ** 2
            energy = float(eig.sum())
            fro.append(math.sqrt(energy) / layer['w_fro'])
            spec.append(math.sqrt(float(eig[0])) / layer['w_spec'])
            if energy > 0:
                top8.append(float(eig[:8].sum()) / energy)
        result = {'b_rms': math.sqrt(b_square / b_count), 'a_drift_rms': math.sqrt(drift / a0_square),
                  'rel_fro_median': statistics.median(fro), 'rel_fro_max': max(fro),
                  'rel_spec_median': statistics.median(spec), 'rel_spec_max': max(spec)}
        if applied_lr_sum > 0:
            result['b_rms_over_lr_sum'] = result['b_rms'] / applied_lr_sum
        if top8:
            result['top8_energy_mean'] = sum(top8) / len(top8)
        return {k: v for k, v in result.items() if math.isfinite(v)}

    def log_values(self, model, ema, config, steps):
        """W&B keys adapter/* (student) and adapter_ema/* (EMA, when present); {} on any failure."""
        try:
            applied = lr_sum(config, steps)
            modules = {canonical_name(n): m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
            data = {'adapter/' + k: v for k, v in self.summary(
                {n: (m.lora_A, m.lora_B) for n, m in modules.items()}, applied).items()}
            data['adapter/lr_sum'] = applied
            if ema is not None:
                pairs = {n: (ema[n + '.lora_A'], ema[n + '.lora_B']) for n in modules}
                data.update({'adapter_ema/' + k: v for k, v in self.summary(pairs, applied).items()})
            return data
        except Exception:
            traceback.print_exc()
            return {}
