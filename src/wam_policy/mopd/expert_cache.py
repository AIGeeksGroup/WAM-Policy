"""Specialist teacher cache (Stage 2.2 M-OPD): each specialist teacher fills the teacher target cache
only with contexts of its own task, and those contexts replace the entries of their tasks one for one;
every other task keeps its entries of the OPSD teacher cache (build that cache first with
``python -m wam_policy.opsd cache``).

Collection runs the streams on a QLM task-subset profile (only the expert tasks, at
world_size / len(tasks) active episodes per task), so every rank always holds an episode of one expert
task and each round's contexts split evenly over the tasks. It collects MULTIPLE times what the merge
keeps, so each stream runs several episodes. `merge` then drops the previous cache's entries of the
expert tasks, takes the same number of new entries per task at evenly spaced positions over the rounds
(fewer when the collection fell short, reported, never an error), prefixes their ids (the builder's
r{rank}-{round}-{j} would collide with the previous ids) and writes a new manifest whose header carries
the specialist teachers (so only a run with the same specialists reuses it)."""
import json
import os
from pathlib import Path
import re
import shutil
import sys
from wam_policy.common.tasks import TASKS

# The collection holds MULTIPLE times the entries the merge keeps, so the merge can take them evenly over the
# rounds (episode positions as in a long collection) instead of the earliest, mostly early-episode, entries.
MULTIPLE = 2
# Bound on the resets a task consumes: its streams (max_active_per_task) run chunks_per_clip chunks a round, an
# episode lasts at least MIN_EPISODE_CHUNKS (push_box's shortest are about 30), and MAX_TASK_EPISODES stays well
# below the 80-reset deck (QLM reserve can find no reset near 78). It caps the rounds when MULTIPLE would exceed it.
MIN_EPISODE_CHUNKS = 30
MAX_TASK_EPISODES = 60
PREFIX = 'specialist-'
BUILDER_ID = re.compile(r'r(\d+)-(\d+)-(\d+)')


def expert_tasks(config):
    """The configured expert tasks in TASKS order (QLM keeps the same order for a profile subset)."""
    return tuple(t for t in TASKS if t in config.resources.get('experts', {}))


def collection_profile(config):
    tasks = expert_tasks(config)
    if not tasks or config.world_size % len(tasks) or config.world_size // len(tasks) < 2:
        raise ValueError('Expert cache collection needs world_size / len(expert tasks) >= 2 active episodes per task')
    return {'tasks': list(tasks), 'max_active_per_task': config.world_size // len(tasks)}


def task_counts(manifest, tasks):
    rows = manifest['entries']
    return {task: sum(row['task_id'] == task for row in rows) for task in tasks}


def collection_contexts(config, manifest):
    """Contexts to collect: the expert tasks split every round evenly, so the largest previous count of an
    expert task sets what the merge needs; MULTIPLE times that, unless a task could then exceed
    MAX_TASK_EPISODES episodes (never below what the merge needs). Every rank derives the same fixed round
    count from it (engine.cache_rounds)."""
    counts = task_counts(manifest, expert_tasks(config))
    if not all(counts.values()):
        raise ValueError(f'Previous cache holds no entries of an expert task: {counts}')
    need = len(counts) * max(counts.values())
    streams = collection_profile(config)['max_active_per_task']
    rounds = MAX_TASK_EPISODES * MIN_EPISODE_CHUNKS // (streams * config.chunks_per_clip)
    return max(need, min(need * MULTIPLE, rounds * config.chunks_per_clip * config.world_size))


def _link(source, target):
    try:
        os.link(source, target)
    except OSError:  # Another filesystem: copy.
        shutil.copyfile(source, target)


def _builder_order(row):
    match = BUILDER_ID.fullmatch(row['id'])
    if match is None:
        raise ValueError('Collected entry id is not a cache builder id: ' + row['id'])
    rank, round, j = map(int, match.groups())
    return round, rank, j


def merge(previous, collected, output, config):
    """Merged cache at `output` (a new directory); returns write_manifest's result plus per-task counts."""
    from wam_policy.opsd.targets import TargetCache, write_manifest
    tasks = expert_tasks(config)
    old = TargetCache(previous, plain(config), source='offline')  # the OPSD cache: same teacher/samplers/pool
    new = TargetCache(collected, config, source='offline')        # labelled by this run's specialists
    output = Path(output)
    (output / 'entries').mkdir(parents=True, exist_ok=False)
    rows, counts = [], {}
    for row in old.rows.values():
        if row['task_id'] not in tasks:
            _link(old.root / 'entries' / (row['id'] + '.pt'), output / 'entries' / (row['id'] + '.pt'))
            rows.append(dict(row))
    for task in tasks:
        wanted = sum(row['task_id'] == task for row in old.rows.values())
        found = sorted((row for row in new.rows.values() if row['task_id'] == task), key=_builder_order)
        # Evenly spaced over the rounds (deterministic): the episode positions of the whole collection.
        chosen = found if len(found) <= wanted else [found[int((i + .5) * len(found) / wanted)] for i in range(wanted)]
        for row in chosen:
            name = PREFIX + row['id']
            _link(new.root / 'entries' / (row['id'] + '.pt'), output / 'entries' / (name + '.pt'))
            rows.append({**row, 'id': name})
        counts[task] = {'previous': wanted, 'collected': len(found), 'merged': min(wanted, len(found)),
                        'shortfall': max(0, wanted - len(found))}
    header = {k: v for k, v in new.manifest.items() if k not in ('count', 'entries')}
    header['merged_from'] = {'previous': str(old.root.resolve()), 'collected': str(new.root.resolve())}
    result = write_manifest(output, header, rows)
    return {**result, 'tasks': counts, 'shortfall': any(c['shortfall'] for c in counts.values())}


def plain(config):
    """The same config without specialist teachers (the key of the OPSD cache it extends)."""
    return config.with_resources(experts={})


def build(config, output, previous, *, execute=True):
    """Collect the expert tasks with their specialist teachers, then merge with `previous` (the OPSD
    teacher cache) into <output>/cache. Returns the merge report (or the process plan when not executing)."""
    from wam_policy.opsd.launch import launch
    from wam_policy.opsd.targets import TargetCache
    TargetCache(previous, plain(config), source='offline')  # Fail before collecting if it cannot be extended.
    contexts = collection_contexts(config, json.loads((Path(previous) / 'manifest.json').read_text()))
    plan = collection_profile(config)
    output = Path(output).absolute()
    result = launch(config, output / 'collect', execute=execute, cache_contexts=contexts,
                    profile_tasks=(plan['tasks'], plan['max_active_per_task']))
    if not execute:
        return {**result, 'contexts': contexts, 'profile': plan}
    if result.get('passed') is not True:
        raise RuntimeError('Specialist cache collection stopped early; finalize it with '
                           '`python -m wam_policy.opsd finalize-cache` and run `python -m wam_policy.mopd merge`')
    return merge_report(previous, output / 'collect/learner/cache', output / 'cache', config)


def merge_report(previous, collected, output, config):
    merged = merge(previous, collected, output, config)
    if merged['shortfall']:
        print(f"WARNING: specialist cache collection fell short; merged counts {merged['tasks']}",
              file=sys.stderr, flush=True)
    return merged
