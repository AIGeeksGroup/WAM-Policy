"""Optional prediction videos during SFT: causal full-episode rollouts of fixed validation episodes.

Enabled with ``visualize_every`` > 0 (the paper used every 500 steps on three validation
episodes). Each video shows the VAE reconstruction of the ground truth (left) next to the
model's causal prediction from the initial observation only (right); ego on top, task camera
below. Rendering runs on rank 0 with isolated RNG and never affects training; failures are
logged and ignored.
"""
from contextlib import contextmanager, nullcontext
from pathlib import Path
import gc
import subprocess
import time

import numpy as np
import torch
from wan_va.modules.model import custom_sdpa
from wan_va.utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id
from wam_policy.common.distributed import isolated_rng
from wam_policy.common.video import DualViewVAE
from wam_policy.transfer.config import write_json

SEED = 42
FPS = 12.5
CACHE = "sft-periodic-predicted-video"


class _SDPA(torch.nn.Module):
    def forward(self, query, key, value):
        return custom_sdpa(query, key, value)


@contextmanager
def sampling_context(model, seed):
    """Borrow the current parameters, restoring all training-local state afterwards."""
    modules = list(model.modules())
    modes = [(m, m.training) for m in modules]
    attention = [(m, m.attn_op) for m in modules if hasattr(m, "attn_op")]
    old_config = model.config.attn_mode
    device = next(model.parameters()).device
    for block in model.blocks:
        if CACHE in block.attn1.attn_caches:
            raise ValueError("Visualization cache name already in use")
    with isolated_rng(device, seed):
        try:
            model.eval()
            model.register_to_config(attn_mode="torch")
            for module, _ in attention:
                delattr(module, "attn_op")
                module.attn_op = _SDPA()
            yield
        finally:
            for block in model.blocks:
                block.attn1.attn_caches.pop(CACHE, None)
            for module, original in attention:
                delattr(module, "attn_op")
                module.attn_op = original
            model.register_to_config(attn_mode=old_config)
            for module, training in modes:
                module.training = training


def _query(model, logical):
    for block in model.blocks:
        block.attn1.attn_caches[CACHE]["query_logical_id"] = logical


def _forward(model, value, text, group, timestep, *, action=False, update=0):
    _query(model, 2 * group + int(action))
    grid = get_mesh_id(1, 16 if action else 12, 1 if action else 10,
                       int(action), f_shift=group, action=action).to(value.device)[None]
    payload = dict(noisy_latents=value, text_emb=text,
                   timesteps=torch.full((1,1), float(timestep), device=value.device), grid_id=grid)
    return model(payload, action_mode=action, update_cache=update, cache_name=CACHE)


def _sample(model, text, group, generator, *, action, dtype, device):
    value = torch.randn((1,9,1,16,1) if action else (1,48,1,24,20),
                        generator=generator, dtype=dtype, device=device)
    scheduler = FlowMatchScheduler(shift=1. if action else 5., sigma_min=0., extra_one_step=True)
    scheduler.set_timesteps(50 if action else 25)
    for t in scheduler.timesteps:
        field = _forward(model, value, text, group, t, action=action)
        field = (field.transpose(1,2).reshape_as(value) if action else
                 data_seq_to_patch(model.patch_size, field, 1, 24, 20, batch_size=1))
        value = scheduler.step(field, t, value).to(dtype)
    _forward(model, value, text, group, 0, action=action, update=1)
    return value


@torch.no_grad()
def predict_video(model, batch, *, seed):
    """Causal full-episode rollout; only the initial measured state is conditioning."""
    device = next(model.parameters()).device
    dtype = torch.bfloat16 if device.type == "cuda" else next(model.parameters()).dtype
    with sampling_context(model, seed), (torch.autocast("cuda", dtype=torch.bfloat16)
                                        if device.type == "cuda" else nullcontext()):
        model.create_empty_cache(CACHE, 12, 120, 16, device, dtype, 1)
        for block in model.blocks:
            cache = block.attn1.attn_caches[CACHE]
            cache["logical_id"] = torch.full_like(cache["id"], -1)
            cache["window_units"] = 8
            cache["query_logical_id"] = 0
        text = batch["text_emb"].to(device=device, dtype=dtype)
        video = batch["latents"][:, :, :1].to(device=device, dtype=dtype)
        generator = torch.Generator(device=device).manual_seed(seed)
        _forward(model, video, text, 0, 0, update=2)
        _forward(model, torch.zeros(1,9,1,16,1, device=device, dtype=dtype), text, 0, 0, action=True, update=2)
        frames = [video.cpu()]
        for group in range(1, batch["latents"].shape[2]):
            video = _sample(model, text, group, generator, action=False, dtype=dtype, device=device)
            _sample(model, text, group, generator, action=True, dtype=dtype, device=device)
            frames.append(video.detach().cpu())
        result = torch.cat(frames, dim=2)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite predicted video")
        return result


def _load_decoder(config, device, rgb_profile):
    from diffusers import AutoencoderKLWan
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKLWan.from_pretrained(str(Path(config.base_model) / "vae"), torch_dtype=dtype,
                                           local_files_only=True).to(device)
    return DualViewVAE(vae, rgb_profile=rgb_profile)


def decoded_frames(decoder, latent):
    views = decoder.decode(latent, remove_letterbox=True)
    # Each view is [1,3,N,180,320]; the review layout preserves ego above task.
    image = torch.cat([views["go2_ego"], views["d435i_rgb_task"]], dim=-2)
    if not torch.isfinite(image).all():
        raise ValueError("Nonfinite decoded video")
    return ((image[0].permute(1, 2, 3, 0).float().cpu() + 1) * 127.5).round().clamp(0,255).byte().numpy()


def write_video(frames, path, ffmpeg):
    frames = np.ascontiguousarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8 or not len(frames):
        raise ValueError("Expected nonempty uint8 RGB video")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    command = [str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(FPS),
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    subprocess.run(command, input=frames.tobytes(), stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, check=True, timeout=180)
    return path


def render_predictions(model, data, config, directory, step, tracker):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    missing = set(config.visualize_episodes) - set(data.split["validation"])
    if missing:
        raise ValueError(f"Visualization episodes must be validation episodes: {sorted(missing)}")
    device = next(model.parameters()).device
    decoder = _load_decoder(config, device, data.rgb_profile)
    videos, records = {}, []
    try:
        for index, uid in enumerate(config.visualize_episodes):
            batch = data.batch(uid, split="validation")
            prediction = predict_video(model, batch, seed=SEED + index)
            truth = decoded_frames(decoder, batch["latents"])
            predicted = decoded_frames(decoder, prediction)
            path = write_video(np.concatenate([truth, predicted], axis=2), directory / (uid + ".mp4"), config.ffmpeg)
            videos["visualization/predicted_video/" + uid] = path
            records.append(dict(episode_uid=uid, frames=len(predicted), fps=FPS, seed=SEED + index, path=str(path)))
    finally:
        del decoder
        gc.collect()
    caption = "Left: ground-truth VAE reconstruction; right: causal prediction. Top: ego; bottom: task."
    tracker.log_videos(videos, step, fps=FPS, caption=caption)
    report = dict(passed=True, step=step, episodes=records, video_steps=25, action_steps=50,
                  window_units=8, guidance=1.0, caption=caption)
    write_json(directory / "metadata.json", report)
    return report


def maybe_visualize(model, data, config, world, tracker, output, *, step):
    def run():
        started = time.monotonic()
        directory = Path(output) / "visualization" / f"step_{step:08d}"
        with isolated_rng(world.device, SEED):
            try:
                result = render_predictions(model, data, config, directory, step, tracker)
            except Exception as error:
                result = dict(passed=False, step=step, error=f"{type(error).__name__}: {error}")
                write_json(directory / "metadata.json", result)
        tracker.log({"visualization/failed": int(not result["passed"]),
                     "visualization/seconds": time.monotonic() - started}, step)
        return result
    return world.leader(run)
