"""Single-node launcher: a torchrun learner (one rank per GPU) plus one CPU QLM-Bench
training-environment worker per rank (``python -m qlm_bench.training_env worker``).

``launch(...)`` writes the resolved run config and the QLM training profile into the output
directory, starts the learner, waits until every rank's RPC endpoint is ready, then starts the
workers. SIGTERM/SIGINT/SIGUSR1 ask the learner to save at the next round boundary (safe stop).
"""
from pathlib import Path
from contextlib import contextmanager
import json
import os
import signal
import socket
import subprocess
import sys
import time
from wam_policy.opsd.config import write_json

CPU_ENV = {'MUJOCO_GL': 'osmesa', 'PYOPENGL_PLATFORM': 'osmesa', 'LIBGL_ALWAYS_SOFTWARE': 'true',
           'GALLIUM_DRIVER': 'llvmpipe', 'CUDA_VISIBLE_DEVICES': '', 'HIP_VISIBLE_DEVICES': '',
           'ROCR_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'LP_NUM_THREADS': '1'}
WAM_SRC = str(Path(__file__).resolve().parents[2])


def child_environment(config, *, cpu):
    """Environment of a worker (cpu=True: OSMesa, no accelerators) or of the learner."""
    env = os.environ.copy()
    paths = [] if cpu else [WAM_SRC]
    qlm_src = config.resources.get('qlm_src') or env.get('QLM_BENCH_SRC')  # optional; site.env default
    if qlm_src:
        paths.append(qlm_src)
    if paths:
        env['PYTHONPATH'] = os.pathsep.join(paths + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    if cpu:
        env.update(CPU_ENV)
        osmesa = config.resources.get('osmesa_lib') or env.get('QLM_OSMESA_LIB')  # dir with libOSMesa.so
        if osmesa:
            env['LD_LIBRARY_PATH'] = os.pathsep.join([osmesa] +
                                                     ([env['LD_LIBRARY_PATH']] if env.get('LD_LIBRARY_PATH') else []))
    return env


def check_loopback_ports(port, world_size):
    """Every rank's RPC port must be bindable on 127.0.0.1 (SO_REUSEADDR like the QLM RPC server)."""
    ports = list(range(port, port + world_size))
    for number in ports:
        with socket.socket() as endpoint:
            endpoint.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                endpoint.bind(('127.0.0.1', number))
            except OSError as exc:
                raise RuntimeError(f'Port {number} is in use; set port_base to a free range') from exc
    return ports


@contextmanager
def forward_stop_signals(model_ref, stopping, learner_pids=None):
    """A stop request reaches the learner ranks, which save at a round boundary. CPU workers are not
    terminated here: their outstanding feedback stays alive until the learner releases the stop."""
    def request_stop(signum=None, *_):
        stopping[0] = True
        child = model_ref[0]
        if child is not None and child.poll() is None:
            if learner_pids:
                for pid in learner_pids:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            else:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    handlers = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            handlers[sig] = signal.signal(sig, request_stop)
        yield
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def cache_rounds(config, contexts):
    return -(-contexts // (config.chunks_per_clip * config.world_size))


def stream_rounds(config, *, stop_after=None, cache_contexts=None):
    """Rounds the CPU streams serve: cache-build rounds, or the student updates of a training run."""
    if cache_contexts is not None:
        return cache_rounds(config, cache_contexts)
    return config.student_updates if stop_after is None else min(config.student_updates, stop_after)


def process_plan(config, output, *, stop_after=None, resume=None, cache_contexts=None):
    """Learner/cache-builder torchrun command plus one CPU worker per rank (offline_only training: none)."""
    from wam_policy.opsd.control import QLM_ENTRY, TrainingControl
    from wam_policy.opsd.runtime import pool_sha256
    output = Path(output).absolute()
    command = 'rank-cache' if cache_contexts is not None else 'rank-train'
    model = [config.resources.get('model_python', sys.executable), '-m', 'torch.distributed.run', '--standalone',
             '--nproc_per_node=' + str(config.world_size), '-m', 'wam_policy.opsd', command,
             '--config', str(output / 'config.json'), '--output', str(output / 'learner')]
    if cache_contexts is not None:
        model += ['--contexts', str(cache_contexts)]
    if stop_after is not None:
        model += ['--stop-after', str(stop_after)]
    if resume:
        model += ['--resume', str(Path(resume).absolute())]
    workers = []
    if config.streams or cache_contexts is not None:
        control = TrainingControl(config, output, output / 'training-profile.json')
        pool = config.need('train_pool')
        episodes = config.chunks_per_clip * stream_rounds(config, stop_after=stop_after, cache_contexts=cache_contexts) + 1
        for rank in range(config.world_size):
            workers.append([config.need('qlm_python'), '-m', QLM_ENTRY, 'worker',
                            '--pool', pool, '--pool-sha256', pool_sha256(config), '--rank', str(rank),
                            '--world-size', str(config.world_size), '--port', str(config.port_base + rank),
                            '--output', str(output / 'qlm' / f'rank_{rank}'), '--rpc-timeout-s', str(config.rpc_timeout_s),
                            '--wall-timeout-s', str(config.worker_wall_timeout_s),
                            '--training-profile', control.profile, '--schedule-db', str(control.db),
                            '--segment', control.segment, '--max-episodes', str(episodes),
                            '--render-mode', config.render_mode])
    return {'model': model, 'cpu_workers': workers, 'cpu_environment': CPU_ENV, 'data_mode': config.data_mode}


def wait_children(children, timeout):
    """Bound the whole attempt, including learner shutdown after workers exit."""
    deadline = time.monotonic() + timeout
    while True:
        codes = [child.poll() for child in children]
        if any(code not in (None, 0) for code in codes):
            exits = ', '.join(f'{child.name}={code}' for child, code in zip(children, codes) if code is not None)
            raise RuntimeError(f'Child process failed ({exits}); see the logs in the output directory')
        if all(code is not None for code in codes):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('Run/shutdown deadline exceeded')
        time.sleep(.2)


def launch(config, output, *, execute=True, stop_after=None, resume=None, cache_contexts=None, profile_tasks=None):
    """Run training (cache_contexts None) or a teacher target cache build on this node.
    profile_tasks: (tasks, max_active_per_task) of an M-OPD expert collection profile."""
    output = Path(output).absolute()
    plan = process_plan(config, output, stop_after=stop_after, resume=resume, cache_contexts=cache_contexts)
    if cache_contexts is None and config.world_size != 8:
        print(f'WARNING: world_size={config.world_size}: each update uses {config.world_size}x the per-rank batch; '
              'the paper recipe is 8 ranks (24 on-policy + 24 cached contexts) and has no fewer-GPU equivalent',
              file=sys.stderr, flush=True)
    if config.resources.get('experts'):
        from wam_policy.mopd.experts import routing_table
        plan['routing'] = routing_table(config)
    if not execute:
        return {**plan, 'execute': False}
    output.mkdir(parents=True, exist_ok=False)
    if plan['cpu_workers']:
        from wam_policy.opsd.control import make_profile
        tasks, cap = profile_tasks or (None, None)
        profile = make_profile(config, output / 'training-profile.json', tasks=tasks, max_active_per_task=cap)
        config = config.with_resources(training_profile=profile)
        check_loopback_ports(config.port_base, config.world_size)
    write_json(output / 'config.json', config.record())
    write_json(output / 'process-plan.json', plan)
    children, logs, model_ref, stopping, learner_pids = [], [], [None], [False], []

    def spawn(command, env, name):
        f = (output / (name + '.log')).open('wb')
        logs.append(f)
        child = subprocess.Popen(command, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        child.name = name  # A negative exit code is the killing signal (-9: OOM/SIGKILL).
        children.append(child)
        return child
    with forward_stop_signals(model_ref, stopping, learner_pids):
        try:
            model = spawn(plan['model'], child_environment(config, cpu=False), 'learner')
            model_ref[0] = model
            deadline = time.monotonic() + config.service_start_timeout_s
            while plan['cpu_workers'] and not learner_pids:
                if stopping[0]:
                    raise InterruptedError('Stopped before collection')
                if model.poll() is not None:
                    raise RuntimeError('Learner exited before all endpoints were ready; see learner.log')
                if time.monotonic() > deadline:
                    raise TimeoutError('Learner start-up deadline exceeded')
                try:
                    ready = [json.loads((output / 'learner/services' / f'rank_{rank}.ready.json').read_text())
                             for rank in range(config.world_size)]
                    learner_pids.extend(row['pid'] for row in ready)
                except (FileNotFoundError, json.JSONDecodeError):
                    time.sleep(.2)
            worker_env = child_environment(config, cpu=True)
            for rank, cmd in enumerate(plan['cpu_workers']):
                spawn(cmd, worker_env, f'qlm-rank-{rank}')
            wait_children(children, config.worker_wall_timeout_s + config.rpc_timeout_s + config.collective_timeout_s)
            return {'passed': not stopping[0], 'children': len(children), 'stop_requested': stopping[0],
                    'output': str(output)}
        finally:
            for c in children:
                if c.poll() is None:
                    try:
                        os.killpg(c.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + 30
            for c in children:
                try:
                    c.wait(timeout=max(.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(c.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    c.wait()
            for f in logs:
                f.close()
