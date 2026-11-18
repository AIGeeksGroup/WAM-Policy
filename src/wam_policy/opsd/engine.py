"""Rank processes started by the launcher: ``train`` (student updates) and ``build_target_cache``
(the frozen teacher acts in the streams and its own chunks become regression targets)."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import signal
import sys
import threading
import time
import traceback
import torch
from wam_policy.opsd.config import write_json
from wam_policy.opsd.runtime import load_roles_and_codec, build_coordinator, pool_sha256
from wam_policy.opsd.distributed import World, collective_call
from wam_policy.opsd.learner import Learner
from wam_policy.opsd.streaming import QueuedService, BUDGET_END
from wam_policy.opsd.checkpoint import save, restore, retain
from wam_policy.opsd.export import export_student
from wam_policy.opsd.models import adapter_state
from wam_policy.opsd.control import TrainingControl
from wam_policy.opsd.metrics import RoundTracker, AdapterMetrics
from wam_policy.opsd.targets import open_cache, write_entries, write_manifest, cache_header, append_rows
from wam_policy.opsd.launch import cache_rounds

OFFLINE_CONTROL = {'confirmed': {}, 'completed': {}}


def _append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, allow_nan=False, default=str) + '\n')


def start_tracker(output, config, *, name, job_type):
    """Optional W&B mirror of the metrics journal (disabled unless WANDB_MODE is online/offline)."""
    from wam_policy.common.tracking import Tracker
    try:
        return Tracker(output, name=name, group=config.method, job_type=job_type, config=config.record())
    except Exception:  # Logging never stops training.
        traceback.print_exc()
        return None


def export_students(learner, output, config):
    """students/step_n/{raw,ema} in the release student format at every saved point."""
    step = learner.student_steps
    root = Path(output) / 'students' / f'step_{step:08d}'
    variants = {'raw': adapter_state(learner.roles.student), 'ema': learner.ema_state()}
    exported = []
    for variant, adapter in variants.items():
        if adapter is not None:  # No EMA before ema_start_step.
            export_student(adapter, root / variant, config=config, variant=variant, updates=step)
            exported.append(variant)
    return {'event': 'student_export', 'student_step': step, 'variants': exported}


def start_service(config, world, coordinator, control):
    from qlm_bench.live import RPCServer
    coordinator.expected_segment = control.segment
    coordinator.expected_pool_sha256 = pool_sha256(config)
    service = QueuedService(coordinator, config.rpc_timeout_s)
    port = config.port_base + world.rank
    server = RPCServer(service, port=port, timeout_s=config.rpc_timeout_s)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return service, server, thread, port


def train(config, output, *, resume=None, stop_after=None):
    world = World('cuda', timeout=config.collective_timeout_s)
    if world.size != config.world_size:
        raise ValueError('torchrun world size differs from config.world_size')
    output = Path(output)
    world.leader(lambda: output.mkdir(parents=True, exist_ok=False))
    metrics_file = output / f'metrics-rank-{world.rank}.jsonl'
    streams = config.streams
    service = server = thread = learner = tracking = coordinator = control = failure = None
    stopping = [False]
    handlers = {}
    started = time.monotonic()
    for sig in (signal.SIGTERM, signal.SIGINT):
        handlers[sig] = signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    try:
        roles, codec, prompt, normalizer = collective_call(world, lambda: load_roles_and_codec(config))
        offline = collective_call(world, lambda: open_cache(config))
        if offline is None and config.offline_contexts_per_rank:
            raise ValueError('offline_contexts_per_rank > 0 needs resources.offline_cache (build it with `cache`)')
        learner = Learner(roles, config, world, offline=offline)
        # Before restore: the initial adapter is the reference of the displacement metrics.
        adapters = None
        if world.rank == 0:
            try:
                adapters = AdapterMetrics(roles.student)
            except Exception:  # Diagnostics never stop training.
                traceback.print_exc()
        recovery = restore(learner, resume) if resume else None
        if streams:
            control = TrainingControl(config, output.parent, config.need('training_profile'))
            world.leader(lambda: control.initialize(None if recovery is None else recovery['control_state']))
        if learner.student_steps >= config.student_updates:
            raise ValueError('The update budget is already complete; use `export`')
        port = None
        if streams:
            coordinator = build_coordinator(learner, codec, prompt, normalizer)
            service, server, thread, port = start_service(config, world, coordinator, control)

        def start():
            nonlocal tracking
            write_json(output / 'resolved.json', config.record())
            tracking = start_tracker(output, config, name=output.parent.name, job_type='train')
        world.leader(start)
        write_json(output / f'services/rank_{world.rank}.ready.json',
                   {'port': port, 'rank': world.rank, 'pid': os.getpid(), 'resumed': recovery is not None})
        limit = config.student_updates if stop_after is None else min(config.student_updates,
                                                                       learner.student_steps + stop_after)

        def visual_plan():
            try:
                from wam_policy.opsd.visualization import encoder_available, rollout_timeout, schedule
                steps = schedule(config) if streams else set()
                if steps and rollout_timeout(config) <= 60:
                    print('WARNING: closed-loop video disabled; timeouts leave no rollout time', file=sys.stderr)
                    return set()
                return steps if steps and encoder_available() else set()
            except Exception:
                traceback.print_exc()
                return set()
        visual_steps = world.leader(visual_plan)  # Ranks that disagreed would issue mismatched collectives.

        def visualize(step):
            def call():
                from wam_policy.opsd.visualization import caption, rollout
                try:
                    record = rollout(learner, codec, prompt, normalizer, config, output / 'visualization', step)
                    if tracking is not None:
                        tracking.log_videos({'visualization/closed_loop': record['path']}, step, fps=record['fps'],
                                            caption=caption(record))
                except Exception:  # A diagnostic video never stops training.
                    traceback.print_exc()
            world.leader(call)
        if recovery is None and 0 in visual_steps:
            visualize(0)

        def heldout_s0():
            value = learner.evaluate_heldout()

            def upload():
                if tracking is not None and value.get('count'):
                    tracking.log(RoundTracker.heldout_values(value), step=0)
                _append(metrics_file, {'event': 'heldout', 'student_step': 0, **value})
            world.leader(upload)
        if recovery is None:
            heldout_s0()
        if world.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(world.device)
        round_tracker = RoundTracker()
        while learner.student_steps < limit:
            collection_started = time.monotonic()
            records = collective_call(world, lambda: service.collect(config.collection_timeout_s)) if streams else ()
            collection_elapsed = time.monotonic() - collection_started
            metrics = learner.train_round(records)
            metrics['collection_wall_s'] = collection_elapsed
            if learner.student_steps % config.heldout_every == 0:
                metrics['heldout'] = learner.evaluate_heldout()
            stop_requested = not world.all(not stopping[0])
            stop = learner.student_steps >= limit or stop_requested
            reason = 'safe_stop' if stop_requested else BUDGET_END
            if streams:
                if stop:
                    world.leader(control.stop)
                control_state = world.leader(control.snapshot)
            else:
                control_state = dict(OFFLINE_CONTROL)
            metrics['episode_starts_global'] = control_state.get('confirmed')
            if learner.student_steps % config.checkpoint_every == 0 or stop:
                target = output / 'checkpoints' / f'step_{learner.student_steps:08d}'
                save(learner, target, actor_state=coordinator.sampler.generator.get_state() if streams else None,
                     control_state=control_state)
                exported = world.leader(lambda: export_students(learner, output, config))
                _append(metrics_file, exported)
                world.leader(lambda: retain(output / 'checkpoints',
                                            protected={*config.protected_checkpoints, config.student_updates}))
            if adapters is not None and learner.student_steps % config.adapter_metrics_every == 0:
                metrics.update(adapters.log_values(roles.student, learner.student_ema, config, learner.student_steps))
            empty = SimpleNamespace(outcomes=[], plan_errors=[], events={})
            local = round_tracker.local(learner, records, metrics, coordinator if streams else empty)
            rows = world.gather(local)  # Every rank reaches this gather; RoundTracker.local never raises.
            _append(metrics_file, {'rank': world.rank, **metrics, **(coordinator.metadata() if streams else {}),
                                   'episode_outcomes': local['outcomes'], 'plan_errors': local['plan_errors']})
            if tracking is not None:
                round_tracker.log(tracking, rows, learner, metrics, control_state)
            if learner.student_steps in visual_steps and not stop_requested:
                visualize(learner.student_steps)
            if streams:
                collective_call(world, lambda: service.prepare_release(student_steps=learner.student_steps, stop=stop,
                                                                       stop_reason=reason))
                service.publish_release()
                released = world.gather(RoundTracker.release_local(service, coordinator))
                if tracking is not None:
                    RoundTracker.release_log(tracking, released, learner.student_steps)
            if stop:
                break
        if streams:
            collective_call(world, lambda: service.drain_end(config.rpc_timeout_s))
            coordinator.sampler.clear()
        if learner.student_steps == config.student_updates:
            def final():
                # final_student is the EMA when it exists (its metadata says which); raw is always kept.
                ema = learner.ema_state()
                raw = adapter_state(roles.student)
                export_student(raw, output / 'final_student_raw', config=config, variant='raw',
                               updates=learner.student_steps)
                export_student(raw if ema is None else ema, output / 'final_student', config=config,
                               variant='raw' if ema is None else 'ema', updates=learner.student_steps)
            world.leader(final)
        result = {'student_updates_successful': learner.student_steps, 'data_mode': config.data_mode,
                  'elapsed_s': time.monotonic() - started, 'safe_stop': stopping[0], 'resumed': recovery is not None,
                  'completed': learner.student_steps == config.student_updates}
        world.leader(lambda: write_json(output / 'completion.json', result))
        return result
    except BaseException as exc:
        failure = exc
        if service is not None:
            service.fail(exc)
        write_json(output / f'failure-rank-{world.rank}.json',
                   {'error': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()[-8000:],
                    'counters': None if learner is None else learner.counters(),
                    'stream': None if coordinator is None else coordinator.metadata(),
                    'location': None if learner is None else learner.location})
        raise
    finally:
        if tracking is not None:
            try:
                tracking.finish('failed' if failure is not None else 'stopped' if stopping[0] else 'completed')
            except Exception:
                pass
        if server is not None:
            server.shutdown()
            server.close()
        if thread is not None:
            thread.join(timeout=5)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        world.close()


def build_target_cache(config, output, *, contexts):
    """Teacher-as-actor collection: the frozen teacher (M-OPD: routed to each context's specialist)
    runs the streams at its own sampler steps and every chunk's (noise, plan, action) becomes an offline
    regression target. Writes <output>/cache; no student update happens.

    Every round each rank writes its entries and journals their rows (rows-rank-XX.jsonl), so a build
    that fails later can be finalized (`finalize-cache`) instead of collected again."""
    if type(contexts) is not int or contexts < 1:
        raise ValueError('Positive target context count required')
    world = World('cuda', timeout=config.collective_timeout_s)
    if world.size != config.world_size:
        raise ValueError('torchrun world size differs from config.world_size')
    output = Path(output)
    cache = output / 'cache'
    world.leader(lambda: output.mkdir(parents=True, exist_ok=False))
    metrics_file = output / f'metrics-rank-{world.rank}.jsonl'
    service = server = thread = coordinator = tracking = failure = None
    started = time.monotonic()
    try:
        roles, codec, prompt, normalizer = collective_call(world, lambda: load_roles_and_codec(config))
        learner = Learner(roles, config, world)
        control = TrainingControl(config, output.parent, config.need('training_profile'))
        world.leader(lambda: control.initialize(None))
        coordinator = build_coordinator(learner, codec, prompt, normalizer, actor=learner.teacher)
        service, server, thread, port = start_service(config, world, coordinator, control)
        header = cache_header(config, source='teacher_rollout')

        def start():
            nonlocal tracking
            cache.mkdir(parents=True, exist_ok=False)
            write_json(cache / 'header.json', header)  # finalize_cache reads it if the build stops early.
            tracking = start_tracker(output, config, name=output.parent.name, job_type='teacher-cache')
        world.leader(start)
        write_json(output / f'services/rank_{world.rank}.ready.json',
                   {'port': port, 'rank': world.rank, 'pid': os.getpid(), 'teacher_cache': True})
        rounds = cache_rounds(config, contexts)
        rows = []
        for r in range(rounds):
            round_started = time.monotonic()
            records = collective_call(world, lambda: service.collect(config.collection_timeout_s))

            def write_round():
                samples = [learner.targets.from_actor_record(record.validate(), source='offline',
                                                             id=f'r{world.rank}-{r:05d}-{j}')
                           for j, record in enumerate(records)]
                written = write_entries(cache, samples)
                append_rows(cache, world.rank, written)
                return written
            rows += collective_call(world, write_round)
            stop = r == rounds - 1
            if stop:
                world.leader(control.stop)
            collective_call(world, lambda: service.prepare_release(student_steps=r + 1, stop=stop, stop_reason=BUDGET_END))
            service.publish_release()
            total = sum(world.gather(len(rows)))
            event = {'event': 'cache_round', 'round': r + 1, 'rounds': rounds, 'contexts': len(rows),
                     'global_contexts': total, 'round_s': time.monotonic() - round_started,
                     'outcomes': coordinator.outcomes}
            _append(metrics_file, event)
            if tracking is not None:
                world.leader(lambda: tracking.log({'cache/contexts': total, 'cache/round_s': event['round_s'],
                                                   'cache/elapsed_s': time.monotonic() - started}, step=r + 1))
            coordinator.outcomes, coordinator.plan_errors = [], []
        collective_call(world, lambda: service.drain_end(config.rpc_timeout_s))
        coordinator.sampler.clear()
        per_rank = world.gather(len(rows))
        all_rows = [row for local in world.gather(rows) for row in local]
        world.leader(lambda: write_manifest(cache, header, all_rows))
        result = {'contexts': len(all_rows), 'requested': contexts, 'rounds': rounds, 'world_size': world.size,
                  'rank_contexts': per_rank, 'cache': str(cache)}
        world.leader(lambda: write_json(output / 'completion.json', result))
        return result
    except BaseException as exc:
        failure = exc
        if service is not None:
            service.fail(exc)
        write_json(output / f'failure-rank-{world.rank}.json',
                   {'error': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()[-8000:],
                    'stream': None if coordinator is None else coordinator.metadata()})
        raise
    finally:
        if tracking is not None:
            try:
                tracking.finish('failed' if failure is not None else 'completed')
            except Exception:
                pass
        if server is not None:
            server.shutdown()
            server.close()
        if thread is not None:
            thread.join(timeout=5)
        world.close()
