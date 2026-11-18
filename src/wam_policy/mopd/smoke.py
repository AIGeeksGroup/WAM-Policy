"""Tiny CPU smoke of M-OPD: synthetic specialists in the release format on the tiny teacher, task
routing, the specialist teacher cache (collection by the routed teacher actor + merge into an OPSD
cache), and a few updates of a student initialized from an OPSD-style export.

``python -m wam_policy.mopd smoke --output DIR`` writes DIR/report.json.
"""
from dataclasses import replace
from pathlib import Path
import time
import torch
from wam_policy.models import metadata
from wam_policy.opsd.config import write_json
from wam_policy.opsd.smoke import tiny_config, tiny_roles, actor_cache, train_rounds, synthetic_records

EXPERT_TASKS = ("push_box", "open_room_door")


def synthetic_specialist(root, task, layers, seed):
    """A release-format specialist directory with a random rank-2 adapter for `layers` Linear layers."""
    gen = torch.Generator().manual_seed(seed)
    lora = {}
    for i, layer in enumerate(layers):
        lora[f'{i}.old_A'] = torch.randn(2, layer.base.in_features, generator=gen) * .1
        lora[f'{i}.old_B'] = torch.randn(layer.base.out_features, 2, generator=gen) * .1
    root.mkdir(parents=True)
    torch.save({'lora': lora}, root / 'adapter.pt')
    metadata.write(root, {'kind': 'specialist', 'task': task, 'adapter_keys': 'old', 'round': 'r0',
                          'lora': {'rank': 2, 'alpha': 2, 'targets': list(metadata.SPECIALIST_TARGETS)},
                          'sampling': {'video_steps': 25, 'action_steps': 50}})
    return str(root)


def teacher_endpoint(learner, record):
    from wam_policy.opsd.sampling import draw_noise
    eps_v = draw_noise(learner.rng.generator('route-v'), learner.student)
    eps_a = draw_noise(learner.rng.generator('route-a'), learner.student, action=True)
    return learner.targets.generate(record, eps_v, eps_a, source='onpolicy', id='route').target_action


def tiny_smoke(output):
    from wam_policy.mopd.experts import ExpertBank
    from wam_policy.mopd.expert_cache import merge_report
    from wam_policy.opsd.distributed import World
    from wam_policy.opsd.export import export_student, read_student
    from wam_policy.opsd.learner import Learner
    from wam_policy.opsd.models import adapter_state
    from wam_policy.opsd.runtime import attach_experts_and_student
    from wam_policy.opsd.targets import TargetCache
    torch.set_num_threads(2)
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    world = World('cpu')
    plain = tiny_config()
    report = {}

    # Synthetic specialists (layer shapes from a scratch bank on the tiny teacher).
    layers = ExpertBank(tiny_roles(plain).teacher).layers
    experts = {task: synthetic_specialist(output / 'specialists' / task, task, layers, seed)
               for seed, task in enumerate(EXPERT_TASKS)}
    routed = plain.with_resources(experts=experts)

    # 1. Routing: an expert task gets its specialist, any other task the plain teacher.
    roles_plain, roles_routed = tiny_roles(plain), attach_experts_and_student(tiny_roles(routed), routed)
    learner_plain, learner_routed = Learner(roles_plain, plain, world), Learner(roles_routed, routed, world)
    route = {}
    for task in ('push_box', 'turn_valve'):
        record = synthetic_records(0, seed=41, task=task)[1]
        route[task] = not torch.equal(teacher_endpoint(learner_plain, record), teacher_endpoint(learner_routed, record))
    report['routing'] = {'expert_task_changes_target': route['push_box'], 'other_task_unchanged': not route['turn_valve']}

    # 2. OPSD cache, specialist collection (routed teacher actor, expert tasks only), merge.
    previous = actor_cache(plain, tiny_roles(plain), output / 'opsd_cache',
                           [('push_box', 3), ('open_room_door', 3), ('turn_valve', 3)])
    collected = actor_cache(routed, attach_experts_and_student(tiny_roles(routed), routed), output / 'collected',
                            [('push_box', 4), ('open_room_door', 4)], seed=778)
    merged = merge_report(previous['path'], collected['path'], output / 'merged', routed)
    try:
        TargetCache(merged['path'], plain, source='offline')
        rejects = False
    except ValueError:
        rejects = True
    report['merge'] = {'count': merged['count'], 'tasks': merged['tasks'], 'plain_run_rejects_merged_cache': rejects}

    # 3. OPSD-style student export -> initialized M-OPD student -> updates on the merged cache.
    opsd = Learner(tiny_roles(plain), plain, world)
    train_rounds(opsd, 1)
    start = export_student(adapter_state(opsd.roles.student), output / 'opsd_student', config=plain,
                           variant='raw', updates=1)
    config = replace(routed.with_resources(offline_cache=merged['path'], initialized_student=start['path']))
    roles = attach_experts_and_student(tiny_roles(config), config)
    initialized = all(torch.equal(adapter_state(roles.student)[k], v)
                      for k, v in read_student(start['path'])[1].items())
    learner = Learner(roles, config, world, offline=TargetCache(merged['path'], config, source='offline'))
    losses = train_rounds(learner, 2, task='open_room_door')
    exported = export_student(learner.ema_state(), output / 'mopd_student', config=config, variant='ema',
                              updates=learner.student_steps)
    report.update(initialized_from_export=initialized, losses=losses,
                  export={'path': exported['path'], 'method': read_student(exported['path'])[0]['method']})
    report['passed'] = bool(all(report['routing'].values()) and rejects and initialized and
                            report['export']['method'] == 'mopd' and
                            all(t['merged'] == t['previous'] for t in merged['tasks'].values()))
    report['elapsed_s'] = time.monotonic() - started
    write_json(output / 'report.json', report)
    return report
