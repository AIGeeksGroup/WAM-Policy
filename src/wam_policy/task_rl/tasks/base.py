"""Task profile: everything of Stage 2.1 that differs between tasks.

The core (sampling, candidate windows, DiffusionNFT update, round driver) is task independent. A profile supplies:

* the training scenes (prepared cases index) and their seeded scene order;
* window length and which chunks of a candidate train (``window_chunks``, ``train_chunks``), episode caps;
* start capture during the teacher's episodes from reset (``episode_tracker``) and the start split / per-round start
  lists (``select``, ``round_starts``);
* the window reward of a candidate, the group statistics and weights rho (``candidate_rewards``, ``score``);
* the probe comparison and the round table with its stop rules.

Profiles are selected by task name (``wam_policy.task_rl.tasks.get``) and read their settings from the run config.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from wam_policy.task_rl.core import run as R


class EpisodeTracker:
    """Per-episode start capture during a rollout from reset (default: none)."""

    def __init__(self, unit, capture_dir, capture):
        self.unit, self.capture_dir, self.capture = unit, Path(capture_dir), capture
        self.boundaries = []
        self.last_distance = None

    def execute(self, env, commands):
        """Called on the worker thread of the row."""
        return env.execute(commands)

    def observe(self, rows, cols):
        """Privileged rows of the chunk just executed (main thread)."""

    def capture_now(self, end_tick, contact_seen):
        """Persist the live state at this chunk boundary?"""
        return False

    def summary(self, allrows, cols):
        return {}


class TaskProfile:
    name = None
    CONFIG_DEFAULTS = {}
    SCENE_ORDER_SEED = 0
    TRAIN_ORDER_SEED = 1
    reward_unit = 'm'

    def __init__(self, run):
        self.run = run
        self.cfg = run.config
        self._cases = None

    # ---- geometry of the window
    @property
    def window_chunks(self):
        return int(self.cfg['window_chunks'])

    def train_chunks(self, chunks):
        """Training samples of a candidate that sampled ``chunks`` chunks."""
        return chunks

    @property
    def pool_max_chunks(self):
        raise NotImplementedError

    @property
    def eval_max_chunks(self):
        raise NotImplementedError

    # ---- scenes
    def _index(self):
        if self._cases is None:
            ix = R.load_json(self.run.cases / 'cases-index.json')
            if ix.get('task') != self.name:
                raise ValueError(f'{self.run.cases} holds cases of {ix.get("task")}, not {self.name}')
            self._cases = ix
        return self._cases

    def cases(self):
        """Training cases: dicts case_id, split, case (file), model (scene.xml), horizon_s."""
        return [dict(c) for c in self._index()['cases'] if c['split'] == 'train']

    def controller(self):
        return self._index()['controller']

    def scene_order(self):
        cases = sorted(self.cases(), key=lambda c: c['case_id'])
        random.Random(self.SCENE_ORDER_SEED).shuffle(cases)
        return cases

    def episode_tracker(self, unit, capture_dir, capture):
        return EpisodeTracker(unit, capture_dir, capture)

    # ---- starts
    def select(self):
        raise NotImplementedError

    def smoke_starts(self, sel, rnd):
        """Plumbing rounds (names starting with ``smoke``): one start per scene first, then filled from the same scenes."""
        n_smoke = self.cfg.get('smoke_starts', 4)
        seen, out = set(), []
        for b in sel['train']:
            if b['case_id'] not in seen and len(out) < n_smoke:
                seen.add(b['case_id']); out.append(b)
        for b in sel['train']:
            if len(out) >= n_smoke:
                break
            if b not in out:
                out.append(b)
        return out

    def round_starts(self, sel, rnd):
        raise NotImplementedError

    def train_order(self, n):
        order = list(range(n))
        random.Random(self.TRAIN_ORDER_SEED).shuffle(order)
        return order

    # ---- reward
    def candidate_rewards(self, collect_dir, starts):
        raise NotImplementedError

    def score(self, cands, scale):
        """-> (aggregate, per-group, rho) for a collection at the given scale (None: no weights)."""
        raise NotImplementedError

    def pooled_sd(self, cands):
        raise NotImplementedError

    @property
    def scale_key(self):
        return f'scale_{self.reward_unit}'

    def base_rows(self, unit):
        z = np.load(self.run.root / 'pool/episodes' / f'{unit}.npz')
        return z['privileged'], [str(c) for c in z['columns']]

    # ---- probe and round table
    def probe_compare(self, ref, run_cands):
        raise NotImplementedError

    def round_table(self, fail_on_stop=False):
        raise NotImplementedError


def group_weights(rewards, scale, skip_range):
    """rho in [0,1] per candidate: (r - group mean)/scale clipped to +-1, mapped to [0,1]; None if the group is skipped
    (range below ``skip_range``). A zero scale (only possible when every group is constant) gives rho = 0.5."""
    r = np.asarray(rewards, np.float64)
    if np.ptp(r) < skip_range:
        return None
    if not scale:
        return np.full(len(r), 0.5)
    a = np.clip((r - r.mean()) / scale, -1.0, 1.0)
    return 0.5 + 0.5 * a


def splice(base_priv, branch_rows):
    """Base-episode rows up to (and including) the restore tick, then the branch rows after it."""
    t0 = branch_rows[0, 0]
    i0 = int(np.flatnonzero(base_priv[:, 0] == t0)[0])
    return np.concatenate([base_priv[:i0 + 1], branch_rows[1:]]), i0


def cluster_se(d, keys):
    """SE of the mean of d with clusters (sum within cluster, CR0 with small-sample factor G/(G-1))."""
    d = np.asarray(d, float); n = len(d)
    groups = {}
    for x, k in zip(d - d.mean(), keys):
        groups[k] = groups.get(k, 0.) + x
    G = len(groups)
    if G < 2:
        return None
    return float(np.sqrt(G / (G - 1) * sum(v ** 2 for v in groups.values())) / n)


# The normalizer scale of the paper teacher (rounded), the unit of the probe's first-chunk command shift.
SHIFT_SCALE = [0.0841, 0.0484, 0.0868, 0.1965, 0.0894, 0.1623]
DIMS = ['vx', 'vy', 'yaw_rate', 'fl_x', 'fl_y', 'fl_z']


def command_shift(run_cands, ref, ids):
    scale = np.array(SHIFT_SCALE)
    f = np.stack([(np.array(run_cands[i]['first_chunk']) - np.array(ref[i]['first_chunk'])) / scale for i in ids])
    w = np.stack([(np.array(run_cands[i]['window_mean']) - np.array(ref[i]['window_mean'])) / scale for i in ids])
    return {'first_chunk_shift_mean': dict(zip(DIMS, f.mean((0, 1)).round(6).tolist())),
            'first_chunk_shift_rms': dict(zip(DIMS, np.sqrt((f ** 2).mean((0, 1))).round(6).tolist())),
            'first_chunk_identical_fraction': float((f == 0).all((1, 2)).mean()),
            'window_mean_shift_mean': dict(zip(DIMS, w.mean(0).round(6).tolist())),
            'window_mean_shift_rms': dict(zip(DIMS, np.sqrt((w ** 2).mean(0)).round(6).tolist()))}


def load_or_none(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, nd=2):
    return '' if v is None else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


def stage_seconds(rd):
    t = {r['stage']: r['seconds'] for r in R.jsonl(rd / 'stage-times.jsonl')}
    return t, sum(t.values())


def learn_columns(lm):
    st = lm['steps']
    return dict(samples=lm['samples'], steps=lm['steps_this_round'], batch=lm['batch_size'], world=lm.get('world_size', 1),
                grad_norm_mean=float(np.mean([s['grad_norm'] for s in st])) if st else None,
                reg_mean=float(np.mean([s['reg'] for s in st])) if st else 0.,
                grad_probe_video=lm['grad_probe_mean']['video'], grad_probe_action=lm['grad_probe_mean']['action'],
                disp_current=lm['displacement_current'], disp_old=lm['displacement_old_after_ema'],
                update_seconds=lm['update_seconds'])
