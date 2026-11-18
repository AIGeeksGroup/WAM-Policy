"""Frozen VAE/T5 cache of the demonstrations and the whole-episode training dataset built on it.

Cache layout (``<cache>``)::

    action_normalizer.json            train-split normalizer (``python -m wam_policy.data fit``)
    cache.json                        {"format", "rgb_profile"}
    text/<task>.pt                    frozen T5 instruction embedding, bf16 [1,512,4096]
    episodes/<release>/<uid>/
        episode.json                  uid, task, split, ordering and length metadata
        episode.pt                    {"latents": {camera: [1,48,T,12,20]},
                                       "actions": float32 [9,T,16,1] (raw native9), "actions_mask": bool}

Every episode is committed atomically (written to a hidden staging directory, then renamed), so
an interrupted ``cache`` run resumes where it stopped. The normalizer is applied when batches
are loaded, so the cache does not depend on it.
"""
from __future__ import annotations

from functools import lru_cache
import gc
import json
from pathlib import Path
import shutil

import torch

from wam_policy.common.action_space import ActionNormalizer, CAMERA_KEYS
from wam_policy.common.video import DualViewVAE, compose_latents, split_latents
from wam_policy.data.prepare import NORMALIZER, open_releases, write_json
from wam_policy.data.qlm_bench import releases

FORMAT = "wam-policy-frozen-cache-v1"
LATENT_SHAPE = (48, 12, 20)  # per camera, per latent frame


def _check_episode(latents, actions, mask, frames):
    if set(latents) != set(CAMERA_KEYS):
        raise ValueError(f"Expected latents for cameras {CAMERA_KEYS}")
    for value in latents.values():
        if tuple(value.shape) != (1, LATENT_SHAPE[0], frames, *LATENT_SHAPE[1:]):
            raise ValueError(f"Latent shape {tuple(value.shape)} != [1,48,{frames},12,20]")
    if tuple(actions.shape) != (9, frames, 16, 1) or tuple(mask.shape) != (9, frames, 16, 1):
        raise ValueError(f"Actions must be [9,{frames},16,1]")
    if bool(mask[:, 0].any()):
        raise ValueError("Latent frame 0 carries no action targets")
    if bool((actions[6:] != 0).any()):
        raise ValueError("Force dimensions must be zero")


def write_episode(cache_root, meta, *, latents, actions, actions_mask):
    """Atomically write one episode; ``meta`` needs uid, release, task, split, release_rank,
    episode_index and latent_frames."""
    actions = torch.as_tensor(actions, dtype=torch.float32)
    actions_mask = torch.as_tensor(actions_mask, dtype=torch.bool)
    latents = {k: v.detach().cpu().contiguous() for k, v in latents.items()}
    _check_episode(latents, actions, actions_mask, meta["latent_frames"])
    parent = Path(cache_root) / "episodes" / meta["release"]
    final, staging = parent / meta["uid"], parent / ("." + meta["uid"] + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    torch.save({"latents": latents, "actions": actions, "actions_mask": actions_mask}, staging / "episode.pt")
    (staging / "episode.json").write_text(json.dumps(meta, indent=2) + "\n")
    staging.rename(final)
    return final


def write_text(cache_root, task, embedding):
    embedding = embedding.detach().cpu().contiguous()
    if tuple(embedding.shape) != (1, 512, 4096):
        raise ValueError("Text embedding must be [1,512,4096]")
    path = Path(cache_root) / "text" / (task + ".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    torch.save(embedding, partial)
    partial.replace(path)
    return path


def write_cache_info(cache_root, rgb_profile):
    write_json(Path(cache_root) / "cache.json", {"format": FORMAT, "rgb_profile": rgb_profile})


def _build(snapshot, cache_root, base_model, *, device, names):
    cache_root, base_model = Path(cache_root), Path(base_model)
    opened = open_releases(snapshot, names)
    profiles = {release.rgb_profile for _, release in opened}
    if len(profiles) != 1:
        raise ValueError(f"Releases mix RGB profiles {profiles}")
    rgb_profile = profiles.pop()
    write_cache_info(cache_root, rgb_profile)
    rank = {e["release"]: i for i, e in enumerate(releases())}
    codec = vae = None
    encoded = reused = 0
    try:
        for entry, release in opened:
            for index, episode in enumerate(release.episodes):
                uid = episode["episode_uid"]
                if (cache_root / "episodes" / entry["release"] / uid / "episode.json").is_file():
                    reused += 1
                    continue
                if episode["task_text"] != entry["instruction"]:
                    raise ValueError(f"{entry['release']}: unexpected instruction {episode['task_text']!r}")
                if codec is None:
                    from diffusers import AutoencoderKLWan
                    vae = AutoencoderKLWan.from_pretrained(str(base_model / "vae"), torch_dtype=torch.bfloat16,
                                                          local_files_only=True).to(device)
                    codec = DualViewVAE(vae, rgb_profile=rgb_profile)
                views, aligned = release.sampled_rgb(index)
                with torch.no_grad():
                    latent = codec.encode_clip(views).cpu()
                meta = {"uid": uid, "release": entry["release"], "task": entry["task"],
                        "split": release.split_of[uid], "release_rank": rank[entry["release"]],
                        "episode_index": index, "length": episode["length"],
                        "latent_frames": aligned["latent_frames"],
                        "sampled_frame_ids": aligned["selected_frame_ids"], "text": "text/" + entry["task"] + ".pt"}
                write_episode(cache_root, meta, latents=split_latents(latent),
                              actions=aligned["actions"], actions_mask=aligned["actions_mask"])
                encoded += 1
                print(f"cached {entry['release']}/{uid} ({aligned['latent_frames']} latent frames)", flush=True)
                del views, latent
                gc.collect()
    finally:
        del codec, vae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    missing = [e for e, _ in opened if not (cache_root / "text" / (e["task"] + ".pt")).is_file()]
    if missing:
        from transformers import T5TokenizerFast, UMT5EncoderModel
        from wam_policy.common.text import encode_prompt_pair
        tokenizer = T5TokenizerFast.from_pretrained(str(base_model / "tokenizer"), local_files_only=True)
        encoder = UMT5EncoderModel.from_pretrained(str(base_model / "text_encoder"), torch_dtype=torch.bfloat16,
                                                  local_files_only=True).to(device)
        text = encode_prompt_pair([e["instruction"] for e in missing], tokenizer, encoder)["text_emb"]
        for i, entry in enumerate(missing):
            write_text(cache_root, entry["task"], text[i:i + 1])
        del encoder, tokenizer
    return {"cache_root": str(cache_root), "rgb_profile": rgb_profile, "encoded_episodes": encoded,
            "reused_episodes": reused, "encoded_instructions": len(missing)}


def build_cache(snapshot, cache_root, base_model, *, device="cuda", names=None):
    """Encode every episode with the frozen VAE and every instruction with the frozen T5."""
    import fcntl
    Path(cache_root).mkdir(parents=True, exist_ok=True)
    with (Path(cache_root) / ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is writing {cache_root}") from error
        return _build(snapshot, cache_root, base_model, device=device, names=names)


def load_normalizer(cache_root):
    value = json.loads((Path(cache_root) / NORMALIZER).read_text())
    return ActionNormalizer(offset=tuple(value["offset"]), scale=tuple(value["scale"]))


class FrozenEpisodes:
    """Whole-episode batches from a frozen cache. Test episodes are listed but never served."""

    def __init__(self, cache_root):
        self.root = Path(cache_root)
        if not (self.root / NORMALIZER).is_file():
            raise FileNotFoundError(f"{self.root / NORMALIZER} is missing; run `python -m wam_policy.data fit`")
        self.normalizer = load_normalizer(self.root)
        self.rgb_profile = json.loads((self.root / "cache.json").read_text())["rgb_profile"]
        records = [json.loads(p.read_text()) for p in self.root.glob("episodes/*/*/episode.json")]
        records.sort(key=lambda r: (r["release_rank"], r["episode_index"]))
        self.records = {r["uid"]: r for r in records}
        if len(self.records) != len(records):
            raise ValueError("Duplicate episode uids in the cache")
        self.split = {s: [r["uid"] for r in records if r["split"] == s] for s in ("train", "validation", "test")}
        if not self.split["train"]:
            raise ValueError(f"No cached train episodes under {self.root}")
        self.frames = {uid: r["latent_frames"] for uid, r in self.records.items()}
        self.longest_train = max(self.split["train"], key=lambda u: self.frames[u])

    def counts(self):
        return {s: len(v) for s, v in self.split.items()}

    @lru_cache(maxsize=2)
    def _batch(self, uid):
        from wam_policy.interface.alignment import to_model_batch
        record = self.records[uid]
        stored = torch.load(self.root / "episodes" / record["release"] / uid / "episode.pt",
                            map_location="cpu", weights_only=True)
        _check_episode(stored["latents"], stored["actions"], stored["actions_mask"], record["latent_frames"])
        text = torch.load(self.root / record["text"], map_location="cpu", weights_only=True)
        aligned = {"actions": stored["actions"].numpy(), "actions_mask": stored["actions_mask"].numpy()}
        return to_model_batch(aligned, latents=compose_latents(stored["latents"]), text_emb=text,
                              normalizer=self.normalizer)

    def batch(self, uid, *, split="train"):
        if split not in ("train", "validation") or uid not in self.split[split]:
            raise ValueError("Only train/validation episodes are served")
        # JointFlow augmentation mutates tensors in place; never hand out the cached copy.
        return {k: v.clone() for k, v in self._batch(uid).items()}
