"""Read one QLM-Bench LeRobot v2.1 release (50 Hz native9 actions, two policy RGB videos).

Layout of a release directory::

    meta/info.json        LeRobot info (fps 50, data_path/video_path templates, features)
    meta/contract.json    episodes: episode_uid, length, task_text, terminal RGB snapshots
    meta/split.json       {"episodes": {"train": [uid...], "validation": [...], "test": [...]}}
    data/chunk-*/episode_*.parquet              one row per executed 50 Hz action
    videos/chunk-*/<camera>/episode_*.mp4       ego and task-centric RGB (one frame per row)
    terminal/episode_*/<camera>.png             the observation after the last action

The model samples RGB every 4th row (12.5 Hz) and groups 16 actions per latent frame; the tail
that does not fill a complete group is left out of training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wam_policy.common.rgb_profiles import RGB_HW

# LeRobot feature name -> model camera key (the model input order is CAMERA_KEYS).
CAMERAS = {"observation.images.ego": "go2_ego", "observation.images.task_centric": "d435i_rgb_task"}
FPS = 50
FRAME_STRIDE = 4
ACTIONS_PER_LATENT = 16
ACTION_DIM = 9


def _json(path):
    return json.loads(Path(path).read_text())


def read_png(path, hw):
    from PIL import Image
    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB")).copy()
    if value.shape != (*hw, 3):
        raise ValueError(f"Terminal image {path} has shape {value.shape}, expected {(*hw, 3)}")
    return value


class LeRobotRelease:
    def __init__(self, root):
        self.root = Path(root)
        self.info = _json(self.root / "meta/info.json")
        self.episodes = _json(self.root / "meta/contract.json")["episodes"]
        self.split = _json(self.root / "meta/split.json")["episodes"]
        if self.info.get("fps") != FPS:
            raise ValueError(f"{self.root}: expected a {FPS} Hz release")
        shapes = {tuple(self.info["features"][k]["shape"][1:]) for k in CAMERAS}
        profiles = [name for name, hw in RGB_HW.items() if {hw} == shapes]
        if len(profiles) != 1:
            raise ValueError(f"{self.root}: unsupported camera resolution {shapes}")
        self.rgb_profile = profiles[0]
        self.hw = RGB_HW[self.rgb_profile]
        self.split_of = {uid: name for name, uids in self.split.items() for uid in uids}
        self.starts = np.cumsum([0] + [e["length"] for e in self.episodes]).tolist()
        self._dataset = None

    def __len__(self):
        return len(self.episodes)

    def parquet_path(self, index):
        chunk = index // self.info["chunks_size"]
        return self.root / self.info["data_path"].format(episode_chunk=chunk, episode_index=index)

    def actions(self, index):
        """Executed native9 actions [N,9] (force dimensions are zero)."""
        import pyarrow.parquet as pq
        table = pq.read_table(self.parquet_path(index), columns=["action"])
        actions = np.asarray(table["action"].to_pylist(), np.float32)
        if actions.shape != (self.episodes[index]["length"], ACTION_DIM):
            raise ValueError(f"{self.root} episode {index}: action shape {actions.shape}")
        return actions

    def alignment(self, index):
        """Sampled RGB rows and the grouped action targets of one whole episode.

        Latent frame 0 is the initial observation (no action); latent frame t>0 covers the 16
        actions executed between sampled RGB rows 16(t-1) and 16t.
        """
        n = self.episodes[index]["length"]
        eligible = list(range(0, n + 1, FRAME_STRIDE))
        selected = eligible[:1 + 4 * ((len(eligible) - 1) // 4)]
        endpoint = selected[-1]
        frames = 1 + endpoint // ACTIONS_PER_LATENT
        actions = np.zeros((ACTION_DIM, frames, ACTIONS_PER_LATENT, 1), np.float32)
        mask = np.zeros_like(actions, dtype=bool)
        if endpoint:
            executed = self.actions(index)[:endpoint]
            actions[:, 1:, :, 0] = executed.reshape(frames - 1, ACTIONS_PER_LATENT, ACTION_DIM).transpose(2, 0, 1)
            mask[:, 1:] = True
        return {"actions": actions, "actions_mask": mask, "selected_frame_ids": selected,
                "latent_frames": frames, "source_actions": n, "used_actions": endpoint}

    @property
    def dataset(self):
        if self._dataset is None:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            self._dataset = LeRobotDataset(repo_id=str(self.root), root=self.root,
                                           video_backend="pyav", download_videos=False)
        return self._dataset

    def sampled_rgb(self, index):
        """({camera key: uint8 [N,H,W,3]}, alignment) for the sampled RGB rows of one episode."""
        aligned = self.alignment(index)
        episode = self.episodes[index]
        frames = {key: [] for key in CAMERAS.values()}
        for row_index in aligned["selected_frame_ids"]:
            row = None if row_index == episode["length"] else self.dataset[self.starts[index] + row_index]
            for feature, key in CAMERAS.items():
                if row is None:
                    value = read_png(self.root / episode["terminal"]["images"][feature]["path"], self.hw)
                else:
                    # LeRobot decodes RGB8 as CHW float32 / 255; restore the pixel values.
                    value = np.rint(row[feature].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                if value.shape != (*self.hw, 3):
                    raise ValueError(f"{self.root} episode {index}: frame shape {value.shape}")
                frames[key].append(value)
        return {k: np.stack(v) for k, v in frames.items()}, aligned
