#!/usr/bin/env python3
"""Write <dataset>/empty_emb.pt for SFT only: the official text encoder on the empty prompt.

The upstream LIBERO dataset class loads this file as the CFG drop text in place of an episode's text embedding.
Neither the dataset nor the model repository ships it and the upstream README does not describe how it is made,
so it is generated here in the shape the dataset class needs ([512, 4096], the encoding of "" with padding zeroed,
the same as the prompt bank's negative embedding). An existing file is kept and only checked."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="configs/students/sft_only.json")
    parser.add_argument("--site", help="site file with machine.model_root and machine.sft_dataset")
    args = parser.parse_args(argv)
    import torch
    from arm_opsd.assets.base_model import check_base_model, text_embedder
    from arm_opsd.config import ArmConfig
    from arm_opsd.sft_only.dataset import EMPTY_EMB
    config = ArmConfig.load(args.config, site=args.site)
    if not config.machine.sft_dataset:
        raise ValueError("machine.sft_dataset is required")
    output = Path(config.machine.sft_dataset) / EMPTY_EMB
    if not output.is_file():
        check_base_model(config.machine.model_root)
        embedding = text_embedder(config.machine.model_root, device="cuda")("")[0]
        temporary = output.with_suffix(".pt.incomplete")
        torch.save(embedding, temporary)
        temporary.replace(output)
    value = torch.load(output, map_location="cpu", weights_only=True)
    if value.shape != (512, 4096):
        raise ValueError(f"{output} must be one [512, 4096] text embedding, found {tuple(value.shape)}")
    print({"empty_emb": str(output), "shape": list(value.shape)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
