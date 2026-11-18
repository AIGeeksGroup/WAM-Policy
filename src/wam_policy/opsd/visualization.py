"""Optional periodic closed-loop prediction video on one fixed episode (rank 0, config.visualization_every).

The current student rolls out the whole episode through a separate RPC endpoint, with its own
actor cache and streaming codec state; the QLM CPU process uses the same native physics, RAMBO
controller and policy cameras as training plus the evaluation observer camera. The video latent
the student actually sampled for every chunk (ActorSampler.last_plan) is decoded with the
observed causal history: top-left predicted ego/task, bottom-left simulator ego/task at the same
ticks, right the simulator observer (never a policy input). Every failure is reported and never
interrupts training.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import numpy as np
import torch

CAMERAS = ('go2_ego', 'd435i_rgb_task')
FPS = 12.5       # Four policy observation ticks per 0.32 s chunk.
TILE = (180, 320)
# Ranks 1-7 wait in a collective, and the training CPU workers on RPC, while rank 0 rolls out;
# keep room for the round update, checkpoint save, decode and encode.
TIMEOUT_MARGIN_S = 600


def schedule(config):
    """Student updates to visualize: the initial policy and every config.visualization_every updates."""
    every = config.visualization_every
    return set() if every <= 0 else {0, *range(every, config.student_updates + 1, every)}


def encoder_available():
    """The MP4 writer needs imageio-ffmpeg; check once instead of after a full rollout."""
    try:
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
        return True
    except Exception as exc:
        print(f'WARNING: closed-loop video disabled, MP4 encoder unavailable: {type(exc).__name__}: {exc}',
              file=sys.stderr, flush=True)
        return False


def rollout_timeout(config):
    """A longer rollout would trip the collective/RPC timeouts and abort training."""
    return min(config.collective_timeout_s, config.rpc_timeout_s) - TIMEOUT_MARGIN_S


def _tile(image):
    from PIL import Image
    return np.asarray(Image.fromarray(np.asarray(image)).resize(TILE[::-1], Image.BILINEAR))


@torch.no_grad()
def decode_predictions(vae, initial, observed, predicted):
    """uint8 [chunks,4,views,180,320,3]: each sampled latent decoded from a copy of the decoder
    state warmed on the actually observed history up to that chunk."""
    from wam_policy.common.video import split_latents, normalize_latents
    from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
    weight = next(vae.parameters())
    def features(latent, view):
        raw = normalize_latents(split_latents(latent)[view].to(weight.device, weight.dtype), vae, inverse=True)
        return vae.post_quant_conv(raw)
    views = []
    for view in CAMERAS:
        cache = [None] * vae._cached_conv_counts['decoder']; chunks = []
        vae.decoder(features(initial, view), feat_cache=cache, feat_idx=[0], first_chunk=True)
        for k, z in enumerate(predicted):
            frames = vae.decoder(features(z, view), feat_cache=copy.deepcopy(cache), feat_idx=[0])
            if vae.config.patch_size is not None:
                frames = unpatchify(frames, vae.config.patch_size)
            frames = frames.clamp(-1, 1)[0, :, :, 6:-6, :].permute(1, 2, 3, 0).float()
            chunks.append(((frames + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu().numpy())
            if k < len(observed):
                vae.decoder(features(observed[k], view), feat_cache=cache, feat_idx=[0])
        views.append(np.stack(chunks))
    return np.stack(views, axis=2)


def compose(predicted, real, observer):
    """Left: predicted ego|task above simulator ego|task; right: simulator observer."""
    from PIL import Image, ImageDraw
    h, w = TILE
    left = np.concatenate([np.concatenate(list(predicted), 1), np.concatenate(list(real), 1)], 0)
    right = np.asarray(Image.fromarray(np.asarray(observer)).resize((2 * w, 2 * h), Image.BILINEAR))
    frame = Image.fromarray(np.concatenate([left, right], 1))
    draw = ImageDraw.Draw(frame)
    for (x, y), text in (((4, 2), 'PREDICTED ego'), ((w + 4, 2), 'PREDICTED task'), ((4, h + 2), 'SIM ego'),
                         ((w + 4, h + 2), 'SIM task'), ((2 * w + 4, 2), 'SIM observer')):
        draw.rectangle([x - 2, y, x + 8 * len(text), y + 12], fill=(0, 0, 0)); draw.text((x, y), text, fill=(255, 255, 255))
    return np.asarray(frame)


def rollout(learner, codec, prompt, normalizer, config, output, step):
    """One complete closed-loop episode with the current student; returns the media record."""
    import imageio.v2 as imageio
    from PIL import Image
    from qlm_bench.live import RPCServer
    from wam_policy.inference.session import OnlineSession
    from wam_policy.common.video import DualViewVAE
    from wam_policy.opsd.launch import child_environment
    from wam_policy.opsd.runtime import actor_contract
    from wam_policy.opsd.streaming import ActorSampler
    out = Path(output) / f'step-{step:06d}'
    out.mkdir(parents=True, exist_ok=False)
    actor = ActorSampler(learner.student, normalizer, contract=actor_contract(config))
    session = OnlineSession(actor, DualViewVAE(codec.vae, rgb_profile='dual-rgb-360p-v1'), prompt,
                            info={'role': 'visualization'})
    real, predicted = {}, []
    def handle(request):
        method = request.get('method')
        if method == 'reset':
            actor.begin_episode(request['payload']['reset'])
        if method in ('reset', 'feedback'):
            for sample in ((request.get('payload') or {}).get('observation') or {}).get('samples', []):
                real[int(sample['capture_tick'])] = np.stack([_tile(sample['rgb'][k]) for k in CAMERAS])
        response = session.handle(request)
        if method == 'predict' and actor.last_plan is not None and (not predicted or predicted[-1] is not actor.last_plan):
            predicted.append(actor.last_plan)
        return response
    started = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[learner.student.device]), RPCServer(handle, port=0, timeout_s=config.rpc_timeout_s) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            try:
                command = [config.resources['qlm_python'], '-m', 'qlm_bench.training_env.visualization',
                           '--pool', config.resources['train_pool'], '--port', str(server.address[1]), '--output', str(out / 'sim'),
                           '--rpc-timeout-s', str(config.rpc_timeout_s)]
                with (out / 'sim.log').open('w') as log:
                    subprocess.run(command, env=child_environment(config, cpu=True), stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=rollout_timeout(config))
            finally:
                server.shutdown(); thread.join(timeout=10)
        initial = actor.journal._initial
        observed = [entry[0] for entry in actor.journal._entries]
    finally:
        actor.clear()  # Release the cache lease before the next learner update.
    rollout_s = time.perf_counter() - started
    sim = json.loads((out / 'sim/result.json').read_text())
    decoded = decode_predictions(codec.vae, initial, observed, predicted)
    frames = 0
    with imageio.get_writer(str(out / 'video.mp4'), fps=FPS, codec='libx264', quality=7, macro_block_size=8) as writer:
        for k, chunk in enumerate(decoded):
            ticks = [k * 160 + 40 * (j + 1) for j in range(4)]
            if not all(t in real for t in ticks):
                continue  # Terminal chunk without complete feedback.
            for j, t in enumerate(ticks):
                path = out / 'sim/observer' / f'{t:08d}.jpg'
                observer = np.asarray(Image.open(path).convert('RGB')) if path.is_file() else np.zeros((360, 640, 3), np.uint8)
                writer.append_data(compose(chunk[j], real[t], observer)); frames += 1
    record = {'step': step, 'path': str(out / 'video.mp4'), 'fps': FPS, 'frames': frames, 'task': sim['task'],
              'episode_uid': sim['episode_uid'], 'split': sim['split'], 'policy_seed': sim['policy_seed'], 'role': 'student',
              'actor_version': learner.student.version, 'success': bool(sim['outcome'].get('success')),
              'fallen': bool(sim['outcome'].get('fallen')), 'chunks': len(predicted), 'rollout_s': rollout_s,
              'decode_and_write_s': time.perf_counter() - started - rollout_s,
              'layout': 'top-left predicted ego/task (decoded sampled latents); bottom-left simulator ego/task; right simulator observer'}
    (out / 'video.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def caption(record):
    return (f"{record['task']} {record['split']} {record['episode_uid']} step {record['step']} seed {record['policy_seed']} "
            f"{record['role']} success={record['success']}; {record['layout']}")
