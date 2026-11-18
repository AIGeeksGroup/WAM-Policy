"""Whole-episode training data = the Stage 1.1 frozen cache + one 10D state per sampled RGB frame.

Video latents, text embeddings and native9 actions come unchanged from the Stage 1.1 frozen cache
(:class:`wam_policy.data.frozen_cache.FrozenEpisodes`; build it with ``python -m wam_policy.data
cache``). The ``data`` command reads the root-link state columns of the QLM-Bench releases in
``snapshot`` and writes ``<state_root>/normalization.json`` (fit on the training split only) and
``<state_root>/episodes/<uid>.npz`` with ``state [T,4,10]``, ``mask [T,4]`` and ``timestamp_ns [T,4]``:
group 0 holds the initial state, every later group the four states at the four RGB frames of that
latent group (the last one may be the pre-reset terminal state).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from wam_policy.data.frozen_cache import FrozenEpisodes
from wam_policy.data.lerobot import LeRobotRelease
from wam_policy.data.qlm_bench import releases

from .state import FIELDS, GROUND_Z, STATE_DIM, StateNormalizer, select_state


def episode_states(release, index, *, ground_z=GROUND_Z):
    """Physical 10D states at the RGB frames selected for the VAE, grouped like the latents."""
    episode = release.episodes[index]
    alignment = release.alignment(index)
    table = pq.read_table(release.parquet_path(index), columns=["simulation_time_ns", *FIELDS])
    times = np.asarray(table["simulation_time_ns"].to_pylist(), dtype=np.int64)
    if len(times) != episode["length"] or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid row timestamps")
    rows = {field: np.asarray(table[field].to_pylist(), dtype=np.float64) for field in FIELDS}
    selected = np.asarray(alignment["selected_frame_ids"], dtype=np.int64)
    if selected[-1] == len(times):
        # The last selected RGB frame is the pre-reset terminal observation.
        terminal = episode["terminal"]
        for field in FIELDS:
            rows[field] = np.concatenate([rows[field], np.asarray(terminal["state"][field], np.float64)[None]])
        times = np.append(times, int(terminal["simulation_time_ns"]))
    physical = select_state({key: value[selected] for key, value in rows.items()}, ground_z=ground_z)
    groups = int(alignment["latent_frames"])
    lengths = [1] + [4] * (groups - 1)
    if len(selected) != sum(lengths):
        raise ValueError("Selected RGB frames do not match the VAE groups")
    state = np.zeros((groups, 4, STATE_DIM), np.float32)
    mask = np.zeros((groups, 4), bool)
    timestamp = np.full((groups, 4), -1, np.int64)
    cursor = 0
    for k, count in enumerate(lengths):
        state[k, :count] = physical[cursor:cursor + count]
        mask[k, :count] = True
        timestamp[k, :count] = times[selected[cursor:cursor + count]]
        cursor += count
    return {"state": state, "mask": mask, "timestamp_ns": timestamp, "selected_frame_ids": selected}


def prepare_states(config):
    """Write the 10D state directory for every cached episode (needs the frozen cache and the snapshot)."""
    episodes = FrozenEpisodes(config.cache_root)
    roots = {entry["release"]: Path(config.snapshot) / entry["canonical_root"] for entry in releases()}
    opened = {}
    root = Path(config.state_root)
    (root / "episodes").mkdir(parents=True, exist_ok=True)
    train_rows = []
    for uid, record in episodes.records.items():
        if record["release"] not in opened:
            opened[record["release"]] = LeRobotRelease(roots[record["release"]])
        packed = episode_states(opened[record["release"]], record["episode_index"])
        if packed.pop("selected_frame_ids").tolist() != record["sampled_frame_ids"]:
            raise ValueError(f"{uid}: release RGB sampling differs from the frozen cache")
        if record["split"] == "train":
            train_rows.append(packed["state"][packed["mask"]])
        temporary = root / "episodes" / (uid + ".npz.partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **packed)
        temporary.replace(root / "episodes" / (uid + ".npz"))
    normalizer = StateNormalizer.fit(np.concatenate(train_rows))
    stats = {**normalizer.record(), "train_episode_uids": episodes.split["train"],
             "fit_rows": int(sum(len(rows) for rows in train_rows)), "ground_z": GROUND_Z}
    (root / "normalization.json").write_text(json.dumps(stats, indent=2) + "\n")
    return {"state_root": str(root), "episodes": len(episodes.records), "fit_rows": stats["fit_rows"]}


class ProprioceptionEpisodes:
    """Stage 1.1 whole-episode batches plus ``proprioception[_mask|_timestamps_ns]``."""

    def __init__(self, config):
        self.base = FrozenEpisodes(config.cache_root)
        self.split, self.frames = self.base.split, self.base.frames
        self.longest_train = self.base.longest_train
        self.action_normalizer = self.base.normalizer
        self.root = Path(config.state_root)
        stats = json.loads((self.root / "normalization.json").read_text())
        if stats["train_episode_uids"] != self.split["train"]:
            raise ValueError("State normalization was fit on a different training split")
        self.state_normalizer = StateNormalizer.from_record(stats)

    def states(self, uid):
        with np.load(self.root / "episodes" / (uid + ".npz"), allow_pickle=False) as data:
            packed = {key: data[key] for key in ("state", "mask", "timestamp_ns")}
        groups = self.frames[uid]
        if packed["state"].shape != (groups, 4, STATE_DIM) or packed["mask"].shape != (groups, 4) \
                or packed["timestamp_ns"].shape != (groups, 4) or packed["mask"][0].sum() != 1 \
                or not packed["mask"][1:].all():
            raise ValueError(f"State file of {uid} does not match its {groups} latent groups")
        return packed

    def batch(self, uid, *, split="train"):
        result = self.base.batch(uid, split=split)
        packed = self.states(uid)
        normalized = self.state_normalizer.encode(packed["state"])
        normalized[~packed["mask"]] = 0.0
        result["proprioception"] = torch.from_numpy(normalized.transpose(2, 0, 1)[None, ..., None].copy())
        result["proprioception_mask"] = torch.from_numpy(packed["mask"][None, None, ..., None].copy())
        result["proprioception_timestamps_ns"] = torch.from_numpy(packed["timestamp_ns"][None].copy())
        return result

    def inputs(self):
        """Normalization statistics a checkpoint carries for inference/export."""
        norm = self.action_normalizer
        return {"action_normalizer": {"offset": list(norm.offset), "scale": list(norm.scale)},
                "proprioception_normalizer": self.state_normalizer.record()}
