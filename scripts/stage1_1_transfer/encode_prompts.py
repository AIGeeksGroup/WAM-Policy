"""Encode the ten QLM-Bench task instructions with the frozen LingBot-VA T5 encoder.

    python scripts/stage1_1_transfer/encode_prompts.py --base-model $WAM_BASE_MODEL --output $WAM_PROMPTS

Writes ``<output>/<task>/embedding.pt`` (bf16 [1,512,4096]; 512-token padding, padded positions
zeroed) and ``<output>/<task>/reference.json``. Inference can then use cached prompts
(``wam_policy.inference.loader.load_prompts(prompt_root=...)``) without loading T5.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from wam_policy.common.tasks import INSTRUCTIONS, TASKS  # noqa: E402
from wam_policy.common.text import encode_prompts  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", type=Path, required=True, help="LingBot-VA base snapshot (tokenizer/, text_encoder/)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true", help="overwrite existing embedding.pt files")
    args = parser.parse_args(argv)
    existing = [t for t in args.tasks if (args.output / t / "embedding.pt").exists()]
    if existing and not args.force:
        raise SystemExit(f"{args.output} already holds embeddings for {existing}; re-encoding on other hardware does "
                         "not reproduce the released bytes. Use a new --output, or --force to overwrite.")

    from transformers import T5TokenizerFast, UMT5EncoderModel
    tokenizer = T5TokenizerFast.from_pretrained(str(args.base_model / "tokenizer"), local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(str(args.base_model / "text_encoder"), torch_dtype=torch.bfloat16,
                                              local_files_only=True).eval().to(args.device)
    with torch.inference_mode():
        for task in args.tasks:
            text = INSTRUCTIONS[task]
            value = encode_prompts([text], tokenizer, encoder).cpu()
            directory = args.output / task
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(value, directory / "embedding.pt")
            (directory / "reference.json").write_text(json.dumps(
                {"task": task, "instruction": text, "shape": list(value.shape), "dtype": "bfloat16",
                 "padding": "max_length_512_attention_mask_zeroed"}, indent=2) + "\n")
            print(f"{task}: {list(value.shape)} -> {directory / 'embedding.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
