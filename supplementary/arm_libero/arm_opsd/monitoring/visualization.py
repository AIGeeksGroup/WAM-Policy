"""Periodic closed-loop prediction video on one fixed LIBERO-Long validation case.

Rank 0 rolls out the current student for one complete episode with its own CPU worker and
streaming encoder (the training stream and its encoder state are untouched). The video latents
the student actually sampled for each chunk are decoded with the observed causal history:
top-left shows the predicted ego (eye-in-hand) and task (agentview) views, bottom-left the
simulator frames at the same control steps, right the diagnostic `frontview` observer camera
(never a model input). The MP4 and an events.jsonl record are written for the W&B tracker.
Every failure is returned as a message and never interrupts training.
"""
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

EGO, TASK = "observation.images.eye_in_hand_rgb", "observation.images.agentview_rgb"
OBSERVER = "observer"
FPS = 20  # One frame per control step.


def schedule(total_updates, every=None):
    """Student-update counts to visualize: the initial policy and every `every` updates
    (MachineConfig.visualization_every; a fifth of the run when None).
    ARM_OPSD_VISUALIZATION_EVERY overrides the interval (0 disables)."""
    default = max(1, total_updates // 5) if every is None else every
    every = int(os.environ.get("ARM_OPSD_VISUALIZATION_EVERY", default))
    return set() if every <= 0 else {0, *range(every, total_updates + 1, every)}


def select_case(manifest, manifest_directory):
    """The first validation case (task, case id), or ARM_OPSD_VISUALIZATION_CASE."""
    from arm_opsd.assets.resets import materialize_case
    rows = [row for row in manifest["cases"] if row["split"] == "validation"]
    wanted = os.environ.get("ARM_OPSD_VISUALIZATION_CASE")
    row = next(r for r in rows if r["case_id"] == wanted) if wanted else min(rows, key=lambda r: (r["task_id"], r["case_id"]))
    case = materialize_case(row, manifest_directory)
    case["observer_camera"] = {"name": os.environ.get("ARM_OPSD_OBSERVER_CAMERA", "frontview"), "height": 256, "width": 256}
    return case


@torch.no_grad()
def decode_predictions(vae, initial, observed, predicted, contract_cameras):
    """Decode each sampled chunk from a copy of the decoder state warmed on the observed history.

    initial [1,48,1,8,16]; observed[k] the encoded feedback of chunk k (3 groups first, then 4);
    predicted[k] the sampled [1,48,4,8,16] state (group 0 of chunk 0 is the conditioning frame).
    Returns {camera: uint8 [frames,128,128,3]} aligned with the executed control steps."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
    weight = next(vae.parameters())
    mean = torch.tensor(vae.config.latents_mean, device=weight.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=weight.device).view(1, -1, 1, 1, 1)
    def features(latent, index):
        value = latent[..., 8 * index:8 * (index + 1)].to(weight.device, torch.float32)
        return vae.post_quant_conv((value * std + mean).to(weight.dtype))
    def run(x, cache, first=False):
        frames = []
        for i in range(x.shape[2]):
            out = vae.decoder(x[:, :, i:i + 1], feat_cache=cache, feat_idx=[0], first_chunk=first and i == 0)
            frames.append(unpatchify(out, vae.config.patch_size) if vae.config.patch_size is not None else out)
        return torch.cat(frames, 2)
    result = {}
    for index, camera in enumerate(contract_cameras):
        cache = [None] * vae._cached_conv_counts["decoder"]
        run(features(initial, index), cache, first=True)
        chunks = []
        for k, sample in enumerate(predicted):
            future = sample[:, :, 1:] if k == 0 else sample
            frames = run(features(future, index), copy.deepcopy(cache)).clamp(-1, 1)[0].permute(1, 2, 3, 0).float()
            chunks.append(((frames + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu().numpy())
            if k < len(observed):
                run(features(observed[k], index), cache)
        result[camera] = np.concatenate(chunks)
    return result


def compose(predicted, real, observer):
    """Left: predicted ego|task above simulator ego|task; right: simulator observer."""
    from PIL import Image, ImageDraw
    tile = 128
    left = np.concatenate([np.concatenate([predicted[EGO], predicted[TASK]], 1),
                           np.concatenate([real[EGO], real[TASK]], 1)], 0)
    right = np.asarray(Image.fromarray(observer).resize((2 * tile, 2 * tile), Image.BILINEAR))
    frame = Image.fromarray(np.concatenate([left, right], 1))
    draw = ImageDraw.Draw(frame)
    for (x, y), text in (((2, 1), "PRED ego"), ((tile + 2, 1), "PRED task"), ((2, tile + 1), "SIM ego"),
                         ((tile + 2, tile + 1), "SIM task"), ((2 * tile + 2, 1), "SIM observer")):
        draw.rectangle([x - 1, y, x + 6 * len(text) + 2, y + 11], fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 255))
    return np.asarray(frame)


def rollout(role, encoder, prompts, case, *, config, output, update, seed):
    """One closed-loop episode with the current student; returns the media record."""
    import imageio.v2 as imageio
    from arm_opsd.sim.libero.worker_client import WorkerClient
    from arm_opsd.evaluation.episode import evaluate_case
    from arm_opsd.model.codec import ObservationEncoder
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"update-{update:06d}"
    captured = {"initial": None, "predicted": [], "observed": [], "real": [], "observer": []}
    def on_chunk(result, receipt, latents):
        if result is None:  # Reset: the conditioning frame's latent warms the decoder only.
            captured["initial"] = latents.detach().cpu()
            return
        captured["predicted"].append(result.state.detach().cpu())
        captured["real"].extend({k: o[k] for k in (EGO, TASK)} for o in receipt["observations"])
        captured["observer"].extend(o[OBSERVER] for o in receipt["observations"])
        if latents is not None:
            captured["observed"].append(latents.detach().cpu())
    started = time.perf_counter()
    env = json.loads(Path(config.machine.worker_config).read_text())
    with torch.random.fork_rng(devices=[role.device] if role.device.type == "cuda" else []), \
            stem.with_suffix(".worker.log").open("wb") as stderr, \
            WorkerClient(python=config.machine.cpu_python, env=env, timeout=config.machine.worker_timeout_s,
                         stderr=stderr) as worker:
        # Own streaming encoder state on the same frozen VAE: the training stream is untouched.
        result = evaluate_case(role, worker, ObservationEncoder(encoder.vae), prompts, case,
                               resource_hash="visualization", generation_seed=seed, on_chunk=on_chunk)
    rollout_s = time.perf_counter() - started
    decoded = decode_predictions(encoder.vae, captured["initial"], captured["observed"], captured["predicted"],
                                 config.native.cameras)
    count = min(len(captured["real"]), len(decoded[EGO]))  # A terminal chunk executes only a prefix.
    with imageio.get_writer(str(stem.with_suffix(".mp4")), fps=FPS, codec="libx264", quality=7,
                            macro_block_size=8) as writer:
        for i in range(count):
            writer.append_data(compose({k: decoded[k][i] for k in (EGO, TASK)}, captured["real"][i],
                                       captured["observer"][i]))
    record = {"event": "prediction_video", "optimizer_step": update, "path": str(stem.with_suffix(".mp4")),
              "fps": FPS, "frames": count, "task_id": case["task_id"], "case_id": case["case_id"],
              "instruction": case["instruction"], "generation_seed": seed, "role": "student",
              "actor_version": role.version, "success": result["success"], "chunks": result["chunks"],
              "rollout_s": rollout_s, "decode_and_write_s": time.perf_counter() - started - rollout_s,
              "layout": "top-left predicted ego/task (decoded sampled latents); bottom-left simulator ego/task; "
                        "right simulator observer (frontview, not a model input)"}
    stem.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    return record
