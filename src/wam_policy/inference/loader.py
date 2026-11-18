"""Load released WAM-Policy checkpoints for inference.

``load_teacher`` reads a Stage 1.1 teacher directory (``wam_policy.json`` + ``transformer/``);
``load_codec`` loads the frozen LingBot-VA VAE; ``load_prompts`` returns a prompt encoder backed
by cached T5 embeddings (``<prompt_root>/<task>/embedding.pt``) or by the frozen T5 encoder.
``load_inference_bundle`` composes the three. Students and specialists are LoRA adapters on the
teacher and are attached by :mod:`wam_policy.inference.policies`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from wam_policy.common.action_space import ActionNormalizer
from wam_policy.common.tasks import INSTRUCTIONS, TASKS
from wam_policy.models import metadata
from wam_policy.models.checkpoint_io import load_model

DEFAULT_RGB_PROFILE = "dual-rgb-360p-v1"


@dataclass
class InferenceBundle:
    model: object
    codec: object
    prompt_encoder: object
    normalizer: ActionNormalizer
    info: dict = field(default_factory=dict)


def _dtype(name):
    if name not in ("bfloat16", "float32", "float16"):
        raise ValueError("dtype must be bfloat16, float32 or float16")
    return getattr(torch, name)


def load_teacher(checkpoint, *, device="cuda", dtype="bfloat16", attn_mode="torch"):
    """-> (model, ActionNormalizer, metadata) of a Stage 1.1 teacher directory."""
    meta = metadata.read(checkpoint, "teacher")
    model = load_model(Path(checkpoint) / "transformer", dtype=_dtype(dtype), device=device, attn_mode=attn_mode)
    norm = meta["action_normalizer"]
    return model, ActionNormalizer(offset=tuple(norm["offset"]), scale=tuple(norm["scale"])), meta


def load_codec(base_model, *, device="cuda", dtype="bfloat16", rgb_profile=DEFAULT_RGB_PROFILE):
    """Frozen dual-view VAE codec from a LingBot-VA base snapshot (``<base_model>/vae``)."""
    from diffusers import AutoencoderKLWan
    from wam_policy.common.video import DualViewVAE
    vae = AutoencoderKLWan.from_pretrained(str(Path(base_model) / "vae"), torch_dtype=_dtype(dtype),
                                           local_files_only=True).to(device)
    return DualViewVAE(vae, rgb_profile=rgb_profile)


def load_prompt_tensor(path):
    """A cached prompt embedding: tensor [1,512,4096] or {'text_emb', 'empty_text_emb'}."""
    value = torch.load(path, map_location="cpu", weights_only=True)
    pair = dict(value) if isinstance(value, dict) else {"text_emb": value}
    for name, tensor in list(pair.items()):
        if name not in ("text_emb", "empty_text_emb") or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Unexpected cached prompt content in {path}")
        if tensor.shape == (512, 4096):
            tensor = pair[name] = tensor[None]
        if tensor.shape != (1, 512, 4096) or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"Prompt embedding must be a finite [1,512,4096] tensor: {path}")
    if "text_emb" not in pair:
        raise ValueError(f"Cached prompt is missing text_emb: {path}")
    return pair


def prompt_catalogue(prompt_root, tasks=TASKS):
    """{instruction: embedding path} for the given tasks under ``<prompt_root>/<task>/embedding.pt``."""
    root = Path(prompt_root)
    catalogue = {}
    for task in tasks:
        path = root / task / "embedding.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing prompt embedding for {task}: {path}")
        catalogue[INSTRUCTIONS[task]] = path
    return catalogue


def load_prompts(*, prompt_root=None, tasks=TASKS, base_model=None, device="cuda", dtype="bfloat16"):
    """Prompt encoder ``instruction -> {'text_emb': [1,512,4096], ...}``.

    With ``prompt_root`` the cached embeddings of ``tasks`` are used (no T5 in memory); otherwise
    the frozen T5 encoder of ``base_model`` encodes instructions on first use.
    """
    if prompt_root is not None:
        cached = {instruction: load_prompt_tensor(path)
                  for instruction, path in prompt_catalogue(prompt_root, tasks).items()}

        def encode(instruction):
            if instruction not in cached:
                raise ValueError(f"No cached prompt embedding for instruction {instruction!r}")
            return {k: v.clone() for k, v in cached[instruction].items()}
        return encode
    if base_model is None:
        raise ValueError("Either prompt_root or base_model is required")
    from transformers import T5TokenizerFast, UMT5EncoderModel
    from wam_policy.common.text import encode_prompt_pair
    root = Path(base_model)
    tokenizer = T5TokenizerFast.from_pretrained(str(root / "tokenizer"), local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(str(root / "text_encoder"), torch_dtype=_dtype(dtype),
                                              local_files_only=True).to(device)
    memo = {}

    def encode(instruction):
        if instruction not in memo:
            memo[instruction] = encode_prompt_pair([instruction], tokenizer, encoder)
        return memo[instruction]
    return encode


def load_inference_bundle(*, checkpoint, base_model, prompt_root=None, tasks=TASKS, device="cuda",
                          vae_device=None, dtype="bfloat16", rgb_profile=None, attn_mode="torch"):
    """Teacher model + VAE codec + prompt encoder + action normalizer."""
    model, normalizer, meta = load_teacher(checkpoint, device=device, dtype=dtype, attn_mode=attn_mode)
    rgb_profile = rgb_profile or meta.get("rgb_profile", DEFAULT_RGB_PROFILE)
    vae_device = vae_device or device
    codec = load_codec(base_model, device=vae_device, dtype=dtype, rgb_profile=rgb_profile)
    prompts = load_prompts(prompt_root=prompt_root, tasks=tasks, base_model=base_model, device=device, dtype=dtype)
    info = {"checkpoint": str(checkpoint), "teacher": meta, "device": str(device), "vae_device": str(vae_device),
            "dtype": dtype, "rgb_profile": rgb_profile, "cached_prompts": prompt_root is not None}
    return InferenceBundle(model, codec, prompts, normalizer, info)
