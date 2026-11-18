"""Stage 1.1 initialization: convert the LingBot-VA base transformer to the native 9D action space.

    python -m wam_policy.models.native9 --source <lingbot-va-base>/transformer --output <dir> --seed 0

Every tensor is copied unchanged except the action head: ``action_embedder`` and
``action_proj_out`` change from 30D to 9D and are re-initialized (Xavier-uniform, gain 1, drawn
from a CPU generator seeded with ``--seed`` in the order embedder weight, projection weight;
biases zero). Source shards are streamed one at a time, so the 5B model is never materialized.

The paper used ``robbyant/lingbot-va-base`` at revision ``SOURCE_REVISION`` (configurable through
``WAM_BASE_MODEL_REPO`` / ``WAM_BASE_MODEL_REVISION``; see ``scripts/stage1_1_transfer/convert_base.sh``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

SOURCE_REVISION = "68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c"  # robbyant/lingbot-va-base
UPSTREAM_COMMIT = "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"  # LingBot-VA code
SOURCE_ACTION_DIM = 30
ACTION_DIM = 9
REINITIALIZED = ("action_embedder.weight", "action_embedder.bias",
                 "action_proj_out.weight", "action_proj_out.bias")


def _shards(directory):
    index = sorted(directory.glob("*.safetensors.index.json"))
    if index:
        weight_map = json.loads(index[0].read_text())["weight_map"]
        return index[0].name, sorted(set(weight_map.values())), weight_map
    files = sorted(p.name for p in directory.glob("*.safetensors"))
    if len(files) != 1:
        raise ValueError(f"Expected one safetensors file or an index in {directory}")
    return None, files, None


def convert(source, output, *, seed=0):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from wan_va.modules.model import WanTransformer3DModel

    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((source / "config.json").read_text())
    if config.get("action_dim") != SOURCE_ACTION_DIM:
        raise ValueError(f"Source must have action_dim={SOURCE_ACTION_DIM}, got {config.get('action_dim')}")
    target_config = {**config, "action_dim": ACTION_DIM}
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        target_state = WanTransformer3DModel.from_config(dict(target_config)).state_dict()
    index_name, shard_names, weight_map = _shards(source)

    dtypes, shapes = {}, {}
    for name in shard_names:
        with safe_open(str(source / name), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_slice(key)
                dtypes[key], shapes[key] = tensor.get_dtype(), tuple(tensor.get_shape())
    if set(shapes) != set(target_state):
        raise ValueError("Source tensors do not match the model layout: "
                         f"{sorted(set(shapes) ^ set(target_state))[:10]}")
    changed = sorted(k for k in shapes if shapes[k] != tuple(target_state[k].shape))
    if changed != ["action_embedder.weight", "action_proj_out.bias", "action_proj_out.weight"]:
        raise ValueError(f"Only the action head may change shape, got {changed}")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    reset = {}
    for name in ("action_embedder.weight", "action_proj_out.weight"):
        value = torch.empty(tuple(target_state[name].shape), dtype=torch.float32)
        torch.nn.init.xavier_uniform_(value, gain=1.0, generator=generator)
        reset[name] = value
    for name in ("action_embedder.bias", "action_proj_out.bias"):
        reset[name] = torch.zeros(tuple(target_state[name].shape), dtype=torch.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        (staging / "config.json").write_text(json.dumps(target_config, indent=2, sort_keys=True) + "\n")
        total = 0
        for name in shard_names:
            with safe_open(str(source / name), framework="pt", device="cpu") as handle:
                tensors = {}
                for key in handle.keys():
                    tensors[key] = (reset[key].to(_torch_dtype(torch, dtypes[key])) if key in reset
                                    else handle.get_tensor(key))
                    total += tensors[key].numel() * tensors[key].element_size()
                save_file(tensors, str(staging / name), metadata=handle.metadata() or None)
                del tensors
        if index_name is not None:
            (staging / index_name).write_text(json.dumps(
                {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))},
                indent=2, sort_keys=True) + "\n")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"source": str(source), "output": str(output), "seed": seed, "action_dim": ACTION_DIM,
            "reinitialized": list(REINITIALIZED), "copied_tensors": len(shapes) - len(REINITIALIZED)}


def _torch_dtype(torch, name):
    return {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}[name]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="LingBot-VA base transformer/ directory")
    parser.add_argument("--output", type=Path, required=True, help="new native9 transformer directory")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(convert(args.source, args.output, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
