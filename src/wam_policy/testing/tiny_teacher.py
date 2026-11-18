"""Write a randomly initialised tiny teacher in the release format (smoke runs only).

    python -m wam_policy.testing.tiny_teacher --output <dir> [--seed 0]
"""
import argparse
from pathlib import Path

import torch

from wam_policy.models import metadata
from wam_policy.testing.fixtures import TINY_CONFIG


def write(output, seed=0):
    from wan_va.modules.model import WanTransformer3DModel
    torch.manual_seed(seed)
    model = WanTransformer3DModel(**TINY_CONFIG, attn_mode="torch")
    model.save_pretrained(Path(output) / "transformer", safe_serialization=True)
    metadata.write(output, {"kind": "teacher", "action_normalizer": {"offset": [0.0] * 9, "scale": [1.0] * 9},
                            "rgb_profile": "dual-rgb-360p-v1", "sampling": {"video_steps": 1, "action_steps": 2},
                            "notes": {"purpose": "random tiny model for smoke runs; not a policy"}})
    return Path(output)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    print(write(a.output, a.seed))


if __name__ == "__main__":
    main()
