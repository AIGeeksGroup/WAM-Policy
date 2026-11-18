"""Deployment sampler of a Stage 1.2 OPSD / Stage 2.2 M-OPD student.

``student_actor(model, normalizer, student_dir)`` injects the student LoRA (unmerged) into an
already loaded Stage 1.1 teacher model and returns ``(actor, metadata)``. ``actor`` is the
sampler the student was distilled for (1 video / 2 action Euler steps, CFG 1, window 8) and is
used as the ``sampler`` of ``wam_policy.inference.session.OnlineSession``.

Contract for the caller:
  * before every ``session.handle(request)`` whose ``request["method"] == "reset"``, call
    ``actor.begin_episode(request["payload"]["reset"])`` (the reset body names the episode and the
    task; ``actor.reset`` refuses to start without it);
  * after each predict, ``actor.last_plan`` holds the sampled video latent plan of that chunk
    (CPU tensor [1, 48, 1, 24, 20]) and ``actor.last_model_action`` the normalized action sample
    ([1, 9, 1, 16, 1]);
  * ``actor.spawn()`` returns another independent sampler on the same resident model (own KV cache
    leases, journal, RNG and episode state; same ``runtime``, ``normalizer``, ``contract``, ``steps``)
    for concurrent episodes. Calls into one model must not overlap (serialize GPU work).
"""
from wam_policy.opsd.cache import RoleRuntime
from wam_policy.opsd.config import Config
from wam_policy.opsd.export import load_student, read_student
from wam_policy.opsd.models import inject_lora
from wam_policy.opsd.runtime import actor_contract
from wam_policy.opsd.streaming import ActorSampler


def student_actor(model, normalizer, student_dir):
    meta, _ = read_student(student_dir)
    lora, sampling = meta["lora"], meta["sampling"]
    config = Config(world_size=1, lora_rank=lora["rank"], lora_alpha=lora["alpha"],
                    video_steps=sampling["video_steps"], action_steps=sampling["action_steps"],
                    window_units=sampling.get("window_units", 8)).validate()
    inject_lora(model, config.lora_rank, config.lora_alpha)
    load_student(model, student_dir, config=config)
    model.eval().requires_grad_(False)
    actor = ActorSampler(RoleRuntime(model, "deployment", config), normalizer, contract=actor_contract(config))
    return actor, meta

