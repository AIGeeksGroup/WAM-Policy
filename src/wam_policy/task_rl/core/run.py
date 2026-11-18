"""The run directory of one Task RL run and the small helpers every stage shares.

A run directory holds ``run.json`` (written once by ``python -m wam_policy.task_rl init``): the task, the training
configuration and the machine-specific inputs (teacher, base model, prompt root, prepared cases, QLM-Bench interpreter
and source, GL backend). Every stage takes ``--run <dir>`` and writes only below it:

  pool/ states/ probe/ rounds/rNN/ eval/<name>/ results/ logs/ scale.json

Relative paths given to a stage are resolved against the run directory.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

RUN_FILE = 'run.json'
SCHEMA = 'wam-policy-task-rl-run-v1'
NFE = '25/50'                  # teacher / specialist sampling (no CFG)
CH = 16                        # native9 commands per chunk
CHUNK_TICKS = 160              # physics ticks per chunk (0.32 s at 500 Hz)
TICK_S = 0.002

# Defaults of every configuration field that is not task specific (the task profiles add theirs; the paper configs
# in configs/stage2_1_task_rl/ spell every value out).
CONFIG_DEFAULTS = {
    'seed': 20261114,
    'candidates': 8,
    'starts_per_round': 64,
    'alpha_shared_noise': 1.0,
    'beta': 0.1,
    'nft_epsilon': 1e-05,
    'reference_weight': 1.0,
    'learning_rate': 0.0001,
    'warmup_steps': 12,
    'steps_per_round': 24,
    'grad_clip': 1.0,
    'ema': 0.5,
    'probe_samples': 4,
    'lora_rank': 32,
    'lora_alpha': 32,
    'pool_cases': 80,
    'pool_reps': 1,
    'probe_k': 2,
    'eval_every': 0,
    'eval_scenes': 'probe',
    'eval_reps': 2,
    'max_rounds': 15,
    'smoke_starts': 4,
    # acceleration switches (default off; the paper configs set rows 8, render_mode sampled)
    'rows': 1,                 # rows sampled in lock step per GPU process (= scene workers per process)
    'learn_batch': 1,          # update: samples of equal history length per training forward
    'render_mode': 'full',     # QLM scene worker rendering: full | sampled (same policy frames)
}


# Pure throughput switches: they change how fast a round runs, not what it computes (starts, candidates, seeds,
# samples per round and gradient batches are independent of them), so a run may resume with other values.
THROUGHPUT_KEYS = ('rows', 'learn_batch', 'render_mode')


def algorithm_fields(config):
    """The configuration without the throughput switches: what resume compares."""
    return {k: v for k, v in (config or {}).items() if k not in THROUGHPUT_KEYS}


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=1, default=float) + '\n')
    os.replace(tmp, path)


def jsonl(path):
    path = Path(path)
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def jsonl_many(paths):
    return [r for p in sorted(paths) for r in jsonl(p)]


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(row, default=float) + '\n')


def seed_of(*parts):
    """Deterministic 63-bit seed from a tuple of labels."""
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:8], 'big') % (2 ** 63)


def batches_of(items, size):
    """Consecutive groups of ``size`` items: a function of the list and the size only (never of the shard count)."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def shard_of(items, shard, nshards):
    """Round-robin split: item i goes to shard i % nshards (any shard count gives the same records)."""
    return [x for i, x in enumerate(items) if i % nshards == shard]


def isolate_gpu(index):
    """Make exactly one GPU visible to this process (call before torch is imported). ``index`` is the logical index
    among the devices the job was given; HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES are both set (ROCm and CUDA)."""
    base = os.environ.get('HIP_VISIBLE_DEVICES') or os.environ.get('CUDA_VISIBLE_DEVICES')
    ids = [x for x in base.split(',') if x != ''] if base else None
    chosen = ids[index] if ids else str(index)
    os.environ['HIP_VISIBLE_DEVICES'] = chosen
    os.environ['CUDA_VISIBLE_DEVICES'] = chosen
    return chosen


def peak_gib():
    import torch
    return {'max_memory_allocated_GiB': torch.cuda.max_memory_allocated() / 2 ** 30,
            'max_memory_reserved_GiB': torch.cuda.max_memory_reserved() / 2 ** 30}


class Run:
    """``run.json`` of a run directory: task, config, inputs; and path helpers."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        spec = load_json(self.root / RUN_FILE)
        if spec.get('schema') != SCHEMA:
            raise ValueError(f'{self.root / RUN_FILE} is not a Task RL run file')
        self.spec = spec
        self.task = spec['task']
        self.config = {**CONFIG_DEFAULTS, **spec['config']}
        self._profile = None

    # ---- inputs
    def path(self, x):
        """A path argument: absolute as given, relative to the run directory otherwise."""
        x = Path(x)
        return x if x.is_absolute() else self.root / x

    @property
    def teacher(self):
        return Path(self.spec['teacher'])

    @property
    def base_model(self):
        return Path(self.spec['base_model'])

    @property
    def prompt_root(self):
        return None if not self.spec.get('prompt_root') else Path(self.spec['prompt_root'])

    @property
    def cases(self):
        return Path(self.spec['cases'])

    @property
    def qlm_python(self):
        return self.spec['qlm_python']

    @property
    def qlm_src(self):
        return Path(self.spec['qlm_src'])

    @property
    def gl(self):
        return self.spec.get('gl', 'osmesa')

    @property
    def rows(self):
        return max(1, int(self.config['rows']))

    @property
    def learn_batch(self):
        return max(1, int(self.config['learn_batch']))

    @property
    def profile(self):
        if self._profile is None:
            from wam_policy.task_rl.tasks import get
            self._profile = get(self.task)(self)
        return self._profile

    # ---- scene workers
    def lp_threads(self):
        v = self.spec.get('lp_threads') or 'auto'
        if v != 'auto':
            return int(v)
        cores = int(os.environ.get('TASK_RL_SHARD_CORES') or len(os.sched_getaffinity(0)))   # set by the launcher
        return max(1, min(4, cores // self.rows))

    def scene_environment(self):
        """Extra environment of every scene worker (CPU process). The scene client itself sets MUJOCO_GL /
        PYOPENGL_PLATFORM, hides the GPUs and uses one OpenMP/BLAS thread."""
        env = {'HIP_VISIBLE_DEVICES': '', 'ROCR_VISIBLE_DEVICES': ''}
        if self.gl == 'osmesa':
            env.update(LIBGL_ALWAYS_SOFTWARE='1', LP_NUM_THREADS=str(self.lp_threads()))
        lib = self.spec.get('osmesa_lib')
        if lib:
            env['LD_LIBRARY_PATH'] = lib + (':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
        return env


def init_run(root, *, task, config, teacher, base_model, prompt_root, cases, qlm_python, qlm_src, gl='osmesa',
             osmesa_lib=None, lp_threads=None):
    """Create (or reopen) a run directory. Reopening with a different algorithm configuration is refused: a run
    continues only with the configuration its rounds were trained with (the throughput switches may change; run.json
    records their current values)."""
    from wam_policy.task_rl.tasks import get
    root = Path(root).resolve()
    profile_cls = get(task)
    config = {**profile_cls.CONFIG_DEFAULTS, **config}
    config = {k: v for k, v in config.items() if k != 'note'}
    spec = {'schema': SCHEMA, 'task': task, 'config': config, 'teacher': str(Path(teacher).resolve()),
            'base_model': str(Path(base_model).resolve()),
            'prompt_root': None if prompt_root is None else str(Path(prompt_root).resolve()),
            'cases': str(Path(cases).resolve()), 'qlm_python': str(qlm_python),
            'qlm_src': str(Path(qlm_src).resolve()), 'gl': gl, 'osmesa_lib': osmesa_lib, 'lp_threads': lp_threads}
    path = root / RUN_FILE
    if path.exists():
        old = load_json(path)
        if old.get('task') != task or algorithm_fields(old.get('config')) != algorithm_fields(config):
            raise SystemExit(f'{path} exists with another task/config; a run resumes only with its own configuration')
        old.update({k: v for k, v in spec.items() if k != 'task'})   # machine inputs and throughput switches may change
        spec = old
    for d in ('logs', 'results'):
        (root / d).mkdir(parents=True, exist_ok=True)
    write_json(path, spec)
    return Run(root)
