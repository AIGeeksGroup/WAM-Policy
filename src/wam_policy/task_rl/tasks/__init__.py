"""Task profiles of Stage 2.1 Task-Specific RL, selected by task name."""
from importlib import import_module

PROFILES = {'push_box': 'wam_policy.task_rl.tasks.push_box',
            'open_room_door': 'wam_policy.task_rl.tasks.open_room_door'}


def get(name):
    """The profile class of a task."""
    if name not in PROFILES:
        raise ValueError(f'No Task RL profile for {name!r}; available: {sorted(PROFILES)}')
    return import_module(PROFILES[name]).PROFILE
