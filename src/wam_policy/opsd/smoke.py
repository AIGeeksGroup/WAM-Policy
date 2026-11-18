"""Tiny CPU smoke of the distillation chain: a two-layer synthetic transformer, synthetic journals,
teacher steps 4/8, no weights and no simulator. It runs the real code path of every stage:

  teacher target cache (teacher actor -> cache entries) -> streaming updates with cached contexts and
  held-out evaluation -> checkpoint and resume -> export in the release student format -> deployment
  sampler (``deploy.student_actor``) on the exported student.

``python -m wam_policy.opsd smoke --output DIR`` writes DIR/report.json.
"""
from dataclasses import replace
from pathlib import Path
import copy
import time
import torch
from wam_policy.opsd.config import Config, write_json
from wam_policy.opsd.history import EpisodeJournal, ContextRecord
from wam_policy.opsd.models import build_roles, adapter_state

TINY = dict(world_size=1, lora_rank=4, lora_alpha=4, teacher_video_steps=4, teacher_action_steps=8,
            student_updates=4, checkpoint_every=2, lr_warmup_steps=1, ema_start_step=1, heldout_contexts=2,
            heldout_every=2, offline_contexts_per_rank=2, cache_contexts=6, tiny=True)


def tiny_config(**fields):
    return Config(**{**TINY, **fields}).validate()


def tiny_model(device='cpu'):
    from wan_va.modules.model import WanTransformer3DModel
    return WanTransformer3DModel(num_layers=2, num_attention_heads=2, attention_head_dim=16, ffn_dim=64, freq_dim=16,
                                 text_dim=16, action_dim=9, attn_mode='torch').to(device=device, dtype=torch.float32)


def tiny_roles(config, seed=123):
    torch.manual_seed(seed)
    base = tiny_model()
    return build_roles(lambda: copy.deepcopy(base), config)


def synthetic_records(version, seed=312, task='push_box'):
    """Three consecutive contexts (prefixes 0, 1, 2) of one synthetic episode."""
    gen = torch.Generator().manual_seed(seed)
    journal = EpisodeJournal(('synthetic', f'episode-{seed}', 1), torch.randn(1, 48, 1, 24, 20, generator=gen),
                             torch.randn(1, 512, 16, generator=gen), contract={'task': task})
    records = []
    for i in range(3):
        records.append(ContextRecord(journal.snapshot(actor_version=version, round_id=version,
                                                      observation_id=f'synthetic:{seed}:{i}', tick=i * 160), journal))
        a = torch.randn(1, 9, 1, 16, 1, generator=gen)
        a[:, 6:] = 0
        journal.append(torch.randn(1, 48, 1, 24, 20, generator=gen), a, tick=(i + 1) * 160, receipt=f'synthetic:{i}')
    return records


def identity_normalizer():
    from wam_policy.common.action_space import ActionNormalizer
    return ActionNormalizer((0.,) * 9, (1.,) * 9)


def actor_cache(config, roles, root, episodes, *, seed=777, source='teacher_rollout'):
    """The teacher acting at its own steps (the cache builder's ActorSampler path) on synthetic
    episodes; its own noise/plan/action records become the cache. episodes: [(task, chunks)]; ids use
    the builder's r{rank}-{round}-{j} form."""
    from wam_policy.opsd.cache import RoleRuntime
    from wam_policy.opsd.runtime import actor_contract
    from wam_policy.opsd.streaming import ActorSampler
    from wam_policy.opsd.targets import TargetGenerator, write_cache
    runtime = RoleRuntime(roles.teacher, 'teacher', config, history=True, experts=roles.experts)
    actor = ActorSampler(runtime, identity_normalizer(), contract=actor_contract(config),
                         steps=(config.teacher_video_steps, config.teacher_action_steps))
    generator, samples, gen = TargetGenerator(runtime, config), [], torch.Generator().manual_seed(seed)
    for e, (task, chunks) in enumerate(episodes):
        actor.begin_episode({'session_id': 'smoke', 'episode_id': f'{task}-{e}', 'reset_epoch': 1,
                             'body': {'task': {'id': task, 'instruction': task}}})
        actor.reset(torch.randn(1, 48, 1, 24, 20, generator=gen), torch.randn(1, 512, 16, generator=gen), seed=42 + e)
        for i in range(chunks):
            commands = actor.predict()
            samples.append(generator.from_actor_record(actor.last_record, source='offline', id=f'r{e}-{i:05d}-0'))
            actor.commit(torch.randn(1, 48, 1, 24, 20, generator=gen), commands)
        actor.clear()
    return write_cache(root, config, samples, source=source)


def train_rounds(learner, rounds, seed=500, task='push_box'):
    losses = []
    for k in range(rounds):
        records = synthetic_records(learner.student_steps, seed=seed + k, task=task) if learner.expects_records() else ()
        m = learner.train_round(records)
        losses.append(sum(u['loss/total'] for u in m['units']) / len(m['units']))
    return losses


def self_check(config, roles, world):
    """The untrained student (LoRA B = 0) at the teacher's steps reproduces the teacher target (loss 0)."""
    from wam_policy.opsd.learner import Learner
    from wam_policy.opsd.losses import regression_loss
    from wam_policy.opsd.sampling import generate
    learner = Learner(roles, config, world)
    record = synthetic_records(0)[1]
    sample = learner._target(record, (0, record.snapshot.key, 0), source='onpolicy')
    with torch.no_grad(), learner.student.prepare(record) as c:
        z = generate(c, None, initial_noise=sample.eps_video, steps=config.teacher_video_steps)
        loss, _ = regression_loss(z, sample.target_video, sample.mask_video, c=config.huber_c)
    return float(loss)


def tiny_smoke(output):
    from wam_policy.opsd import checkpoint
    from wam_policy.opsd.deploy import student_actor
    from wam_policy.opsd.distributed import World
    from wam_policy.opsd.export import export_student, read_student
    from wam_policy.opsd.learner import Learner
    from wam_policy.opsd.targets import TargetCache
    torch.set_num_threads(2)
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    world = World('cpu')
    config = tiny_config()
    report = {'self_check_loss': self_check(config, tiny_roles(config), world)}

    # 1. Teacher target cache through the actor path, then the reuse-key check.
    built = actor_cache(config, tiny_roles(config), output / 'teacher_cache',
                        [('push_box', 4), ('open_room_door', 4), ('turn_valve', 4)])
    cfg = config.with_resources(offline_cache=built['path'])
    try:
        TargetCache(built['path'], replace(cfg, teacher_action_steps=16), source='offline')
        report['cache_key_rejects_other_sampler'] = False
    except ValueError:
        report['cache_key_rejects_other_sampler'] = True
    report['teacher_cache_contexts'] = built['count']

    # 2. Streaming updates (on-policy + cached contexts) with full prefix replay and with history reuse.
    results = {}
    for reuse in (False, True):
        c = replace(cfg, history_reuse=reuse)
        learner = Learner(tiny_roles(c), c, world, offline=TargetCache(built['path'], c, source='offline'))
        heldout0 = learner.evaluate_heldout()
        losses = train_rounds(learner, 2)
        results[reuse] = (learner, losses, heldout0)
    (plain, losses, heldout0), (reused, _, _) = results[False], results[True]
    a, b = adapter_state(plain.roles.student), adapter_state(reused.roles.student)
    report.update(losses=losses, heldout_s0=heldout0, heldout_after=plain.evaluate_heldout(),
                  history_reuse_equal=all(torch.equal(a[k], b[k]) for k in a),
                  history_reuse_stats=dict(reused.student.history.stats))

    # 3. Checkpoint, resume into fresh roles, one more identical round on both.
    checkpoint.save(plain, output / 'checkpoint')
    resumed = Learner(tiny_roles(cfg), cfg, world, offline=TargetCache(built['path'], cfg, source='offline'))
    checkpoint.restore(resumed, output / 'checkpoint')
    try:
        checkpoint.restore(Learner(tiny_roles(cfg), replace(cfg, learning_rate=1e-4), world), output / 'checkpoint')
        report['resume_rejects_other_config'] = False
    except ValueError:
        report['resume_rejects_other_config'] = True
    train_rounds(plain, 1, seed=900)
    train_rounds(resumed, 1, seed=900)
    a, b = adapter_state(plain.roles.student), adapter_state(resumed.roles.student)
    report['resume_equal'] = all(torch.equal(a[k], b[k]) for k in a)

    # 4. Export (release student format) and the deployment sampler on the exported student.
    exported = export_student(plain.ema_state(), output / 'student', config=cfg, variant='ema',
                              updates=plain.student_steps)
    meta, state = read_student(exported['path'])
    model = tiny_model()
    actor, _ = student_actor(model, identity_normalizer(), exported['path'])
    actor.config = replace(actor.config, tiny=True)  # the tiny model's 16-wide prompt
    actor.begin_episode({'session_id': 'deploy', 'episode_id': 'e', 'reset_epoch': 1,
                         'body': {'task': {'id': 'push_box', 'instruction': 'push'}}})
    actor.reset(torch.randn(1, 48, 1, 24, 20), torch.randn(1, 512, 16), seed=1)
    commands = actor.predict()
    actor.commit(torch.randn(1, 48, 1, 24, 20), commands)
    actor.clear()
    report.update(export={'path': exported['path'], 'method': meta['method'], 'variant': meta['variant'],
                          'tensors': len(state)},
                  deploy={'commands_shape': list(commands.shape), 'last_plan_shape': list(actor.last_plan.shape)})
    report['passed'] = bool(report['self_check_loss'] == 0 and report['cache_key_rejects_other_sampler'] and
                            report['history_reuse_equal'] and report['resume_rejects_other_config'] and
                            report['resume_equal'] and losses[-1] == losses[-1])
    report['elapsed_s'] = time.monotonic() - started
    write_json(output / 'report.json', report)
    return report
