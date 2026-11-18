"""Strict local loading of the LingBot-VA transformer (no network fetch, no FSDP) and the teacher/student roles."""
from copy import deepcopy
from dataclasses import dataclass
import inspect
import json
from pathlib import Path

import torch

from arm_opsd.model.kv_views import NativeRole
from arm_opsd.model.lora import inject_lora


def resolve_native_config(source):
    """Resolve the official LIBERO-Long config through the actual native constructor.

    The published LIBERO config omits action_dim (native default 30) and retains
    three legacy diffusers fields. They are accepted only for the exact native
    behavior, rather than forwarded as unsupported constructor kwargs. The sole
    execution override is selecting torch SDPA instead of native training flex.
    """
    from wan_va.modules.model import WanTransformer3DModel

    legacy = {"added_kv_proj_dim": None, "image_dim": None,
              "qk_norm": "rms_norm_across_heads"}
    parameters = {name: parameter for name, parameter in
                  inspect.signature(WanTransformer3DModel.__init__).parameters.items()
                  if name != "self"}
    unexpected = {name for name in source
                  if not name.startswith("_") and name not in parameters and name not in legacy}
    if unexpected:
        raise ValueError(f"unsupported native config fields: {sorted(unexpected)}")
    for name, expected in legacy.items():
        if name in source and source[name] != expected:
            raise ValueError(f"legacy field conflicts with native structure: {name}")
    if source.get("_class_name", "WanTransformer3DModel") != "WanTransformer3DModel":
        raise ValueError("config is not a native WanTransformer3DModel")
    resolved = {}
    for name, parameter in parameters.items():
        value = source[name] if name in source else parameter.default
        if value is inspect.Parameter.empty:
            raise ValueError(f"native constructor requires config field: {name}")
        resolved[name] = deepcopy(value)
    geometry = {"patch_size": [1, 2, 2], "num_attention_heads": 24,
                "attention_head_dim": 128, "num_layers": 30, "in_channels": 48,
                "out_channels": 48, "action_dim": 30, "text_dim": 4096,
                "ffn_dim": 14336}
    for name, expected in geometry.items():
        if resolved[name] != expected:
            raise ValueError(f"official native geometry mismatch: {name}")
    resolved["attn_mode"] = "torch"
    return resolved


def load_native_transformer(model_root, *, device="cpu", dtype=torch.bfloat16):
    """Load <model_root>/transformer with every key strict and the native geometry checked.

    ``from_pretrained`` retains native diffusers FP32-module handling. Its ignored-key rules are not
    allowed to hide unexpected checkpoint tensors: safetensors headers are compared to a meta model first.
    """
    directory = Path(model_root) / "transformer"
    shards = sorted(directory.glob("*.safetensors"))
    if not shards or not (directory / "config.json").is_file():
        raise FileNotFoundError(f"No LingBot-VA transformer (config.json + safetensors) under {directory}")
    return _load_strict(directory, shards, device=device, dtype=dtype)


def _load_strict(directory, shards, *, device, dtype):
    from safetensors import safe_open
    from wan_va.modules.model import WanTransformer3DModel

    config = json.loads((directory / "config.json").read_text())
    init_config = resolve_native_config(config)
    with torch.device("meta"):
        shape_model = WanTransformer3DModel(**init_config)
    expected_shapes = {name: tuple(value.shape) for name, value in shape_model.state_dict().items()}
    del shape_model
    checkpoint_shapes = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key in checkpoint_shapes:
                    raise ValueError(f"duplicate sharded checkpoint key: {key}")
                checkpoint_shapes[key] = tuple(reader.get_slice(key).get_shape())
    if checkpoint_shapes != expected_shapes:
        missing = set(expected_shapes) - set(checkpoint_shapes)
        extra = set(checkpoint_shapes) - set(expected_shapes)
        wrong = [name for name in expected_shapes.keys() & checkpoint_shapes.keys()
                 if expected_shapes[name] != checkpoint_shapes[name]]
        raise ValueError(f"strict base load failed: missing={sorted(missing)}, unexpected={sorted(extra)}, shapes={wrong}")
    model, info = WanTransformer3DModel.from_pretrained(
        str(directory), torch_dtype=dtype, attn_mode="torch", local_files_only=True,
        output_loading_info=True, low_cpu_mem_usage=True)
    if any(info.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"strict diffusers load failed: {info}")
    if resolve_native_config(dict(model.config)) != init_config:
        raise ValueError("loaded model config differs from the resolved LingBot-VA constructor config")
    model.source_config = deepcopy(config)
    model.resolved_config = deepcopy(init_config)
    model.to(device).eval().requires_grad_(False)
    return model


@dataclass(frozen=True)
class Roles:
    """The frozen teacher (target generator) and the LoRA student: two full base copies per rank."""
    teacher: NativeRole
    student: NativeRole


def _finish_roles(teacher, student, rank, alpha):
    teacher.eval().requires_grad_(False)
    inject_lora(student, rank=rank, alpha=alpha)
    teacher_parameters = {id(parameter) for parameter in teacher.parameters()}
    if any(id(parameter) in teacher_parameters for parameter in student.parameters()):
        raise RuntimeError("teacher and student parameters must not alias")
    return Roles(NativeRole(teacher, "teacher"), NativeRole(student, "student"))


def roles_from_model(base, *, rank=256, alpha=256):
    """Independent role copies for small-model tests (not a full-base loader)."""
    return _finish_roles(deepcopy(base), deepcopy(base), rank, alpha)


def load_roles(model_root, *, device="cpu", dtype=torch.bfloat16, rank=256, alpha=256):
    models = [load_native_transformer(model_root, device=device, dtype=dtype)
              for _ in range(2)]
    return _finish_roles(*models, rank, alpha)
