"""Model-side history at a rollout-state restore point.

The physics rollout state (QLM ``capture_state`` / ``persist_ring``) holds no WAM history, VAE causal-encoder cache or
policy RNG. Per teacher episode we keep what is needed to rebuild them exactly from real data:

* every feedback frame actually observed since t=0 (tick 0, then four 80 ms ticks per executed 0.32 s chunk; both
  views, uint8, lossless), and
* every executed native9 command group (what the worker confirmed).

``rebuild`` re-encodes the frames from tick 0 (reproducing the latents and the streaming encoder cache) and commits the
groups in order through the ordinary ``commit`` path (reproducing the observed K/V, whose deeper-layer values depend
on the whole preceding window chain, so a window-only replay is not exact).
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

SCHEMA = 'wam-policy-task-rl-history-v1'
VIEWS = ('ego', 'task_centric')
PHYSICAL = {'go2_ego': 'ego', 'd435i_rgb_task': 'task_centric'}   # codec camera key -> scene-worker RGB name


def chunk_views(frames, ticks):
    """Codec input for the given ticks from a mapping tick -> {'ego','task_centric'} uint8 RGB."""
    return {k: np.stack([frames[t][physical] for t in ticks]) for k, physical in PHYSICAL.items()}


def feedback_ticks(groups):
    """[0], then four 80 ms feedback ticks per 0.32 s chunk."""
    return [[0]] + [list(range(g * 160 + 40, (g + 1) * 160 + 1, 40)) for g in range(groups)]


@dataclass
class HistoryRecord:
    """Feedback frames {tick: {'ego','task_centric'}} and executed (16,9) command groups since t=0."""
    frames: dict = field(default_factory=dict)
    executed: list = field(default_factory=list)

    @property
    def groups(self):
        return len(self.executed)

    def initial(self, rgb):
        if self.frames or self.executed:
            raise ValueError('Initial frame only at an empty history')
        self.frames[0] = {v: np.asarray(rgb[v], np.uint8).copy() for v in VIEWS}

    def chunk(self, samples, executed):
        """One executed chunk: the worker's four feedback samples and its executed commands."""
        ticks = feedback_ticks(self.groups + 1)[-1]
        if [s['tick'] for s in samples] != ticks:
            raise ValueError(f'Feedback ticks {[s["tick"] for s in samples]} != expected {ticks}')
        commands = np.asarray(executed, np.float32)
        if commands.shape != (16, 9) or not np.isfinite(commands).all():
            raise ValueError('Complete finite executed command group required')
        for s in samples:
            self.frames[s['tick']] = {v: np.asarray(s['rgb'][v], np.uint8).copy() for v in VIEWS}
        self.executed.append(commands.copy())

    def prefix(self, groups):
        """Record truncated to the first ``groups`` executed chunks (an earlier restore point)."""
        keep = {t for ticks in feedback_ticks(groups) for t in ticks}
        return HistoryRecord({t: self.frames[t] for t in keep}, [x.copy() for x in self.executed[:groups]])


def save(path, record):
    """Lossless compressed .npz plus a small JSON manifest next to it."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    ticks = sorted(record.frames)
    frames = np.stack([np.stack([record.frames[t][v] for v in VIEWS]) for t in ticks])
    executed = np.stack(record.executed) if record.executed else np.zeros((0, 16, 9), np.float32)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, ticks=np.asarray(ticks, np.int64), frames=frames, executed=executed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_bytes(buffer.getvalue()); tmp.replace(path)
    manifest = {'schema': SCHEMA, 'payload': path.name, 'groups': record.groups, 'frames': len(ticks),
                'bytes': path.stat().st_size, 'frame_shape': list(frames.shape[1:])}
    path.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load(path):
    path = Path(path)
    manifest = json.loads(path.with_suffix('.json').read_text())
    if manifest.get('schema') != SCHEMA:
        raise ValueError(f'{path} is not a Task RL history record')
    with np.load(path) as data:
        ticks, frames, executed = data['ticks'], data['frames'], data['executed']
    record = HistoryRecord({int(t): dict(zip(VIEWS, f)) for t, f in zip(ticks, frames)}, list(executed))
    expected = [t for g in feedback_ticks(record.groups) for t in g]
    if sorted(record.frames) != expected:
        raise ValueError('History frame ticks do not match its executed groups')
    return record


@torch.inference_mode()
def rebuild(sampler, codec, record, text, *, seed):
    """Exact sampler history/K-V and encoder cache at the end of ``record``; returns the latents."""
    codec.reset()
    ticks = feedback_ticks(record.groups)
    latents = [codec.push(chunk_views(record.frames, ticks[0]))]
    sampler.reset(latents[0], text, seed=seed)
    for g in range(record.groups):
        latents.append(codec.push(chunk_views(record.frames, ticks[g + 1])))
        sampler.pending = True  # Same live commit path as after an executed prediction.
        sampler.commit(latents[-1], record.executed[g])
    return latents
