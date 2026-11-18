"""One complete closed-loop episode of a frozen policy (both benchmarks and the training-time video)."""
import numpy as np
import torch
from arm_opsd.model.history import EpisodeJournal, HistoryStore
from arm_opsd.model.codec import ActionCodec
from arm_opsd.model.sampler import NativeSampler


@torch.no_grad()
def evaluate_case(role, worker, encoder, prompt_lookup, case, *, resource_hash, generation_seed,
                  terminal_writer=None, video_sink=None, on_chunk=None, sampler=None):
    """Reset `worker` to `case`, then alternate policy chunks and simulator feedback until the episode ends.
    `sampler` defaults to the 1/2-NFE student sampler; `generation_seed` fixes the policy noise."""
    from time import perf_counter
    from uuid import uuid4
    episode = uuid4().hex
    reset = worker.reset(case, episode)
    encoder.reset()
    initial = encoder.encode(reset["observation"])
    if on_chunk is not None:  # Training-time visualization only; evaluation results are unchanged.
        on_chunk(None, reset, initial)
    video_seconds = 0.
    if video_sink is not None:
        video_start = perf_counter()
        video_sink(reset["observation"])
        video_seconds += perf_counter() - video_start
    journal = EpisodeJournal(stream_id="evaluation", session_id=worker.session_id, episode_id=episode,
        task_id=case["task_id"], instruction=case["instruction"], initial_latent=initial,
        resource_hash=resource_hash, reset_receipt=reset.get("reset_info"))
    history = HistoryStore()
    history.add(journal)
    sampler = sampler or NativeSampler(state_steps=1, action_steps=2)
    rng = torch.Generator(device=role.device).manual_seed(generation_seed)
    chunks, actions, generation_ms, feedback_ms, replay_ms, vae_ms = 0, 0, [], [], [], []
    call_counts = {"denoising": 0, "clean": 0, "observed_commits": 0}
    cpu_physics_ms, cpu_render_ms = [], []
    while True:
        context = journal.snapshot(actor_version=role.version, round_id=0, slot_id=chunks)
        positive, negative = prompt_lookup(context.task_id, context.instruction)
        start = perf_counter()
        view = history.view(role, context, text_emb=positive, negative_text_emb=negative)
        if context.prefix:
            call_counts["observed_commits"] += 2
        if role.device.type == "cuda":
            torch.cuda.synchronize(role.device)
        replay_ms.append((perf_counter() - start) * 1000)
        start = perf_counter()
        result = sampler.joint(role, view, first=context.first,
            initial_latent=journal.initial_latent.to(role.device, role.dtype), generator=rng)
        if role.device.type == "cuda":
            torch.cuda.synchronize(role.device)
        generation_ms.append((perf_counter() - start) * 1000)
        commands = ActionCodec().decode(result.action, first=context.first)
        feedback_start = perf_counter()
        receipt = worker.step_chunk(commands, episode_id=episode, chunk_id=chunks, actor_version=role.version)
        applied = np.asarray(receipt["applied_actions"])
        if not np.array_equal(applied, commands[:len(applied)]):
            raise ValueError("The simulator applied other actions than requested")
        chunks += 1
        actions += len(applied)
        call_counts["denoising"] += result.denoising_calls
        call_counts["clean"] += result.clean_calls
        feedback_ms.append((perf_counter() - feedback_start) * 1000)
        cpu_physics_ms.append(receipt.get("cpu_physics_seconds", 0.) * 1000)
        cpu_render_ms.append(receipt.get("cpu_render_seconds", 0.) * 1000)
        if video_sink is not None:
            video_start = perf_counter()
            for observation in receipt["observations"]:
                video_sink(observation)
            video_seconds += perf_counter() - video_start
        if receipt["terminal"]:
            if on_chunk is not None:
                on_chunk(result, receipt, None)
            if terminal_writer is not None:
                terminal_writer({"receipt": {k: v for k, v in receipt.items() if k != "observations"},
                                 "terminal_observation": receipt["observations"][-1]})
            break
        vae_start = perf_counter()
        latents = encoder.encode(receipt["observations"])
        if role.device.type == "cuda":
            torch.cuda.synchronize(role.device)
        vae_ms.append((perf_counter() - vae_start) * 1000)
        if on_chunk is not None:
            on_chunk(result, receipt, latents)
        journal.append(receipt_id=receipt.get("receipt_id") or receipt["request_id"], latents=latents,
            normalized_actions=ActionCodec().encode_executed(applied, first=context.first),
            applied_actions=applied, actor_version=role.version)
        # The current prefix suffices as a replay ancestor; old prefixes cannot
        # accumulate a full GPU KV copy for every step of a long episode.
        history.prune_views({(journal.journal_id, context.prefix)})
    return {"case_id": case["case_id"], "task_id": case["task_id"], "generation_seed": generation_seed,
            "policy_version": role.version, "success": bool(receipt["success"]),
            "timeout": bool(receipt["timeout"]), "policy_actions": actions, "chunks": chunks,
            "model_generation_ms": generation_ms, "feedback_ipc_ms": feedback_ms,
            "history_replay_ms": replay_ms, "vae_encode_ms": vae_ms, "backbone_calls": call_counts,
            "cpu_physics_ms": cpu_physics_ms, "cpu_render_ms": cpu_render_ms,
            "diagnostic_video_encoding_ms": video_seconds * 1000,
            "reset_info": {k: v for k, v in reset.get("reset_info", {}).items() if k in ("reset_seed", "settle_steps", "reset_success")}}
