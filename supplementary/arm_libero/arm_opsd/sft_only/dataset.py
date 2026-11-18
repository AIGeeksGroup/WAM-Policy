"""Whole-episode demonstrations of robbyant/libero-long-lerobot for SFT only (objective sft_fm).

The samples are the ones the upstream `LatentLeRobotDataset` returns for LIBERO: one action segment
(a whole episode here) with its two camera latent files, packed along the width, and the parquet actions
through the upstream offline action codec (`arm_opsd.sft_only.episodes.offline_action_postprocess`).
Episodes are never cut. `empty_emb.pt`, the upstream CFG drop text, is neither in the dataset nor documented
upstream; scripts/prepare/sft_empty_embedding.py generates it next to the dataset (the official text
encoder on ""). Dataset license: CC BY-NC-SA 4.0 (not redistributed here)."""
import json
from pathlib import Path
import numpy as np
import torch
from arm_opsd.sft_only.episodes import native_camera_latents, offline_action_postprocess, validate_sample

REPO = "robbyant/libero-long-lerobot"
REVISION = "8c0313b1c7cd9fa3798798479cbf59b11af8979d"
CAMERAS = ("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb")
ACTION_CHANNELS = 30
EMPTY_EMB = "empty_emb.pt"


def _load_latent(path):
    # The files hold plain tensors plus a numpy `frame_ids`; nothing else may be unpickled.
    from numpy.core.multiarray import _reconstruct, scalar
    with torch.serialization.safe_globals([_reconstruct, scalar, np.ndarray, np.dtype, np.dtypes.Int64DType]):
        return torch.load(path, map_location="cpu", weights_only=True)


def _camera_latent(payload):
    frames, height, width = (int(payload[f"latent_{name}"]) for name in ("num_frames", "height", "width"))
    return payload["latent"].reshape(frames, height, width, -1).permute(3, 0, 1, 2).float()


def _read_actions(path):
    import pyarrow.parquet as parquet
    return torch.from_numpy(np.asarray(parquet.read_table(path, columns=["action"])["action"].to_pylist(), dtype=np.float64))


class OfficialLiberoDataset:
    """Reads the dataset files directly; `draw` is deterministic under the rank's generator."""

    def __init__(self, root, *, q01, q99, native_geometry=True):
        self.root = Path(root)
        self.empty_path = self.root / EMPTY_EMB
        if not self.empty_path.is_file():
            raise FileNotFoundError(f"{self.empty_path} is missing; run scripts/prepare/sft_empty_embedding.py")
        self.q01 = torch.tensor([*q01, *[0.] * (ACTION_CHANNELS - len(q01))], dtype=torch.float64)
        self.q99 = torch.tensor([*q99, *[0.] * (ACTION_CHANNELS - len(q99))], dtype=torch.float64)
        self.native_geometry = native_geometry
        self.episodes = []
        for line in self._checked("meta/episodes.jsonl").read_text().splitlines():
            row = json.loads(line)
            for segment in row["action_config"]:
                self.episodes.append((row["episode_index"], segment["start_frame"], segment["end_frame"]))

    def __len__(self):
        return len(self.episodes)

    def empty_text(self):
        """The CFG drop text the upstream dataset loads from empty_emb.pt."""
        value = torch.load(self.empty_path, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor) or (self.native_geometry and value.shape != (512, 4096)):
            raise ValueError("empty_emb.pt must be one [512, 4096] text embedding")
        return value.float()

    def _checked(self, name):
        path = self.root / name
        if not path.is_file():
            raise FileNotFoundError(f"Demonstration file missing: {path} ({REPO}@{REVISION})")
        return path

    def longest_index(self):
        return max(range(len(self)), key=lambda index: self.episodes[index][2] - self.episodes[index][1])

    def load(self, index):
        episode, start, end = self.episodes[index]
        cameras = [_load_latent(self._checked(f"latents/chunk-000/{camera}/episode_{episode:06d}_{start}_{end}.pth"))
                   for camera in CAMERAS]
        actions = _read_actions(self._checked(f"data/chunk-000/episode_{episode:06d}.parquet"))[start:end]
        action, mask = offline_action_postprocess(actions, cameras[0]["frame_ids"], start, self.q01, self.q99)
        sample = {"latents": native_camera_latents(*(_camera_latent(c) for c in cameras)), "actions": action,
                  "actions_mask": mask, "text_emb": cameras[0]["text_emb"].float()}
        validate_sample(sample, native_geometry=self.native_geometry)
        return sample

    def draw(self, count, generator, longest=False):
        """`longest` draws the longest episode on every slot: the worst case the smokes must expose."""
        indices = torch.randint(len(self), (count,), generator=generator, device=generator.device).tolist()
        if longest:
            indices = [self.longest_index()] * count
        return [self.load(index) for index in indices]
