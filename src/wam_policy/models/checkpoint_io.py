"""Load and save native9 WAM transformers (Diffusers ``save_pretrained`` directories)."""

from pathlib import Path
import json
import shutil
import tempfile

import torch
from wan_va.modules.model import WanTransformer3DModel


def load_model(path, *, dtype=torch.bfloat16, device="cpu", attn_mode="flex"):
    """Strictly load a native 9D transformer directory (missing/unexpected keys are errors)."""
    path = Path(path)
    if not (path / "config.json").is_file():
        raise ValueError(f"Missing model config: {path / 'config.json'}")
    if json.loads((path / "config.json").read_text()).get("action_dim") != 9:
        raise ValueError("Expected a native 9D checkpoint")
    model, info = WanTransformer3DModel.from_pretrained(
        str(path), torch_dtype=dtype, local_files_only=True, use_safetensors=True,
        ignore_mismatched_sizes=False, output_loading_info=True, attn_mode=attn_mode)
    failures = {k: v for k, v in info.items()
                if k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") and v}
    if failures:
        raise ValueError(f"Strict model load failed: {failures}")
    return model.to(device=device, dtype=dtype)


def export_model(model, path):
    """``save_pretrained`` into a new directory (written to a staging directory, then renamed)."""
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"Refusing existing output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + target.name + "-", dir=target.parent))
    try:
        model.save_pretrained(staging, max_shard_size="2GB", safe_serialization=True)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target
