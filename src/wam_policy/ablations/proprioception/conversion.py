"""Native 9D transformer -> native 9D + 10D proprioception transformer (initial SFT weights).

The input is the Stage 1.1 native9 conversion of LingBot-VA (``python -m wam_policy.models.native9``).
The two proprioception layers are added with Xavier-uniform weights (gain 1, CPU generator seed 0,
embedder first) and zero bias, exactly what ``WanTransformer3DModel(proprioception_dim=10)`` does
at construction. All other tensors are the native9 tensors, unchanged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

import torch

SHARD = "diffusion_pytorch_model-proprioception.safetensors"
INDEX = "diffusion_pytorch_model.safetensors.index.json"
NAMES = ("proprioception_embedder.weight", "proprioception_embedder.bias",
         "proprioception_proj_out.weight", "proprioception_proj_out.bias")


def initial_tensors(hidden: int, state_dim: int = 10) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(0)
    first = torch.empty(hidden, state_dim, dtype=torch.float32)
    second = torch.empty(state_dim, hidden, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(first, gain=1.0, generator=generator)
    torch.nn.init.xavier_uniform_(second, gain=1.0, generator=generator)
    return {"proprioception_embedder.weight": first.to(torch.bfloat16).contiguous(),
            "proprioception_embedder.bias": torch.zeros(hidden, dtype=torch.bfloat16),
            "proprioception_proj_out.weight": second.to(torch.bfloat16).contiguous(),
            "proprioception_proj_out.bias": torch.zeros(state_dim, dtype=torch.bfloat16)}


def convert(native9: str | Path, target: str | Path, *, symlink: bool = False) -> dict:
    """Write ``target`` (new directory). ``symlink`` links the unchanged native9 shards instead of copying."""
    from safetensors.torch import save_file
    source, target = Path(native9).resolve(), Path(target).resolve()
    if target.exists():
        raise FileExistsError(target)
    config = json.loads((source / "config.json").read_text())
    index = json.loads((source / INDEX).read_text())
    if config.get("action_dim") != 9 or config.get("proprioception_dim", 0) != 0:
        raise ValueError("Expected a native 9D transformer without proprioception layers")
    if set(NAMES) & set(index["weight_map"]):
        raise ValueError("Source already contains proprioception tensors")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + target.name + "-", dir=target.parent))
    try:
        for path in source.iterdir():
            if path.is_file() and path.name not in ("config.json", INDEX):
                if symlink and path.suffix == ".safetensors":
                    (staging / path.name).symlink_to(path)
                else:
                    shutil.copy2(path, staging / path.name)
        hidden = config["num_attention_heads"] * config["attention_head_dim"]
        extra = initial_tensors(hidden)
        save_file(extra, staging / SHARD)
        index["weight_map"].update({name: SHARD for name in extra})
        index.setdefault("metadata", {})["total_size"] = index.get("metadata", {}).get("total_size", 0) + sum(
            t.numel() * t.element_size() for t in extra.values())
        (staging / INDEX).write_text(json.dumps(index, sort_keys=True, indent=2) + "\n")
        config["proprioception_dim"] = 10
        (staging / "config.json").write_text(json.dumps(config, sort_keys=True, indent=2) + "\n")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"output": str(target), "added": list(NAMES), "hidden": hidden, "symlinked_shards": symlink}
