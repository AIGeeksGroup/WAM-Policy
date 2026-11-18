"""The official LingBot-VA LIBERO-Long weights (theta_0) and the T5 prompt embeddings built from them."""
from pathlib import Path

import torch

from arm_opsd.config import BASE_COMPONENTS, BASE_REPO, BASE_REVISION

PROMPT_SCHEMA = "arm-opsd-prompts-v1"
TEXT_SHAPE = (1, 512, 4096)


def download_base_model(root, *, repo=BASE_REPO, revision=BASE_REVISION):
    """robbyant/lingbot-va-posttrain-libero-long at the revision the paper used (about 40 GB)."""
    from huggingface_hub import snapshot_download
    snapshot_download(repo, revision=revision, local_dir=str(root),
                      allow_patterns=[f"{component}/*" for component in BASE_COMPONENTS])
    return check_base_model(root)


def check_base_model(root):
    root = Path(root)
    missing = [component for component in BASE_COMPONENTS if not (root / component).is_dir()]
    if missing:
        raise FileNotFoundError(f"{root} lacks {missing}; download {BASE_REPO}@{BASE_REVISION} "
                                "(python -m arm_opsd.cli download-base-model --output <dir>)")
    return {"model_root": str(root), "repo": BASE_REPO, "revision": BASE_REVISION}


class PromptBank:
    """T5 embeddings of the ten LIBERO-Long instructions and of the empty negative prompt."""

    def __init__(self, path):
        self.payload = torch.load(path, map_location="cpu", weights_only=True)
        if self.payload.get("schema") != PROMPT_SCHEMA:
            raise ValueError(f"{path} is not an {PROMPT_SCHEMA} prompt bank (scripts: prepare-prompts)")
        if tuple(self.payload["negative"].shape) != TEXT_SHAPE:
            raise ValueError("Negative prompt embedding must be [1, 512, 4096]")

    def __call__(self, task_id, instruction):
        row = self.payload["tasks"][str(task_id)]
        if row["instruction"] != instruction or tuple(row["embedding"].shape) != TEXT_SHAPE:
            raise ValueError(f"Prompt bank has another instruction or shape for task {task_id}")
        return row["embedding"], self.payload["negative"]


def text_embedder(model_root, *, device="cuda"):
    """The official prompt encoding: [1, 512, 4096] BF16 on the CPU, padding set to zero."""
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import T5TokenizerFast, UMT5EncoderModel
    tokenizer = T5TokenizerFast.from_pretrained(str(Path(model_root) / "tokenizer"), local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(str(Path(model_root) / "text_encoder"),
        torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def embed(text):
        tokens = tokenizer([prompt_clean(text)], padding="max_length", max_length=512,
                           truncation=True, add_special_tokens=True, return_tensors="pt")
        value = encoder(tokens.input_ids.to(device), tokens.attention_mask.to(device)).last_hidden_state
        length = int(tokens.attention_mask.sum())
        value[:, length:] = 0
        return value.to(torch.bfloat16).cpu()
    return embed


def prepare_prompt_bank(model_root, tasks, output, *, device="cuda"):
    check_base_model(model_root)
    embed = text_embedder(model_root, device=device)
    payload = {"schema": PROMPT_SCHEMA, "negative": embed(""), "tasks": {}}
    for task in tasks:
        key = str(task["task_id"])
        if key in payload["tasks"] and payload["tasks"][key]["instruction"] != task["instruction"]:
            raise ValueError("Conflicting task instruction")
        if key not in payload["tasks"]:
            payload["tasks"][key] = {"instruction": task["instruction"], "embedding": embed(task["instruction"])}
    if len(payload["tasks"]) != 10:
        raise ValueError("The prompt bank needs the ten LIBERO-Long task instructions")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {"output": str(output), "tasks": len(payload["tasks"])}
