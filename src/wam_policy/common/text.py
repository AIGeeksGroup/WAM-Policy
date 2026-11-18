"""Frozen local T5 prompt encoding with explicit 512-token padding."""

import torch
from diffusers.pipelines.wan.pipeline_wan import prompt_clean


@torch.no_grad()
def encode_prompts(prompts, tokenizer, encoder):
    if (
        not isinstance(prompts, (list, tuple))
        or not prompts
        or not all(isinstance(p, str) for p in prompts)
    ):
        raise ValueError("Expected a nonempty list of prompts")
    encoder.eval().requires_grad_(False)
    cleaned = [prompt_clean(p) for p in prompts]
    inputs = tokenizer(
        cleaned,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    device = next(encoder.parameters()).device
    mask = inputs.attention_mask.to(device)
    hidden = encoder(inputs.input_ids.to(device), attention_mask=mask).last_hidden_state
    if hidden.shape != (len(prompts), 512, 4096) or not torch.isfinite(hidden).all():
        raise ValueError("Expected finite T5 embedding [B,512,4096]")
    return (hidden * mask[:, :, None]).to(torch.bfloat16).contiguous()


def encode_prompt_pair(prompts, tokenizer, encoder):
    if not isinstance(prompts, (list, tuple)) or not prompts:
        raise ValueError("Expected a nonempty prompt sequence")
    both = encode_prompts(list(prompts) + [""] * len(prompts), tokenizer, encoder)
    return {"text_emb": both[: len(prompts)], "empty_text_emb": both[len(prompts) :]}
