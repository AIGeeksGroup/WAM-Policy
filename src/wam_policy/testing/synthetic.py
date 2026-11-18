"""Synthetic inputs for the Stage 1.1 smoke (``scripts/smoke/stage1_1_sft.sh``).

    python -m wam_policy.testing.synthetic --output DIR

writes ``DIR/source/transformer`` (a tiny LingBot-VA-shaped transformer with the 30D action
head, to be converted by ``wam_policy.models.native9``) and ``DIR/cache`` (a frozen cache in the
real format with random latents/text and small random native9 actions).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wam_policy.common.action_space import CAMERA_KEYS
from wam_policy.common.tasks import TASKS
from wam_policy.data.frozen_cache import write_cache_info, write_episode, write_text
from wam_policy.data.prepare import NORMALIZER, write_json
from wam_policy.testing.fixtures import TINY_CONFIG

SPLITS = {"train": 4, "validation": 2, "test": 1}


def write_source(directory, seed=0):
    from wan_va.modules.model import WanTransformer3DModel
    torch.manual_seed(seed)
    model = WanTransformer3DModel(**{**TINY_CONFIG, "action_dim": 30})
    model.save_pretrained(directory, safe_serialization=True)
    return Path(directory)


def write_cache(cache, *, frames=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    tasks = TASKS[:2]
    rank = 0
    for t, task in enumerate(tasks):
        release = f"synthetic-{task}"
        index = 0
        for split, count in SPLITS.items():
            for _ in range(count):
                latents = {k: torch.randn(1, 48, frames, 12, 20, generator=generator).to(torch.bfloat16)
                           for k in CAMERA_KEYS}
                actions = torch.zeros(9, frames, 16, 1)
                actions[:6, 1:] = torch.randn(6, frames - 1, 16, 1, generator=generator) * 0.1
                mask = torch.zeros_like(actions, dtype=torch.bool)
                mask[:, 1:] = True
                meta = {"uid": f"{task}-{index:03d}", "release": release, "task": task, "split": split,
                        "release_rank": rank, "episode_index": index, "length": 16 * (frames - 1),
                        "latent_frames": frames, "text": f"text/{task}.pt"}
                write_episode(cache, meta, latents=latents, actions=actions, actions_mask=mask)
                index += 1
        rank += 1
        write_text(cache, task, (torch.randn(1, 512, 4096, generator=generator) * 0.02).to(torch.bfloat16))
    write_json(Path(cache) / NORMALIZER, {"offset": [0.0] * 9, "scale": [1.0] * 9})
    write_cache_info(cache, "dual-rgb-360p-v1")
    return Path(cache)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = write_source(args.output / "source" / "transformer")
    cache = write_cache(args.output / "cache")
    print(json.dumps({"source_transformer": str(source), "cache": str(cache)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
