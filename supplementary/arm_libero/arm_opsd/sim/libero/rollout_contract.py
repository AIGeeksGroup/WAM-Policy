"""Shared lightweight rollout contract; safe to import without torch or GL."""
# Native evaluation (native_chunk800_v1): the episode ends once the environment timestep reaches 800.
# A reset settles for 5 zero-action steps; the first joint chunk executes 12 actions, every later one 16.
EVAL_TIMESTEPS = 800
SETTLE_STEPS = 5
FIRST_CHUNK_ACTIONS, CHUNK_ACTIONS = 12, 16
# Chunks an evaluation episode can execute: the last chunk is the first to reach timestep 800 (50).
EVAL_CHUNKS = 1 + -(-(EVAL_TIMESTEPS - SETTLE_STEPS - FIRST_CHUNK_ACTIONS) // CHUNK_ACTIONS)
# Training episodes run to the evaluation horizon on every task: late-episode contexts,
# where most failures happen, are collected and cached like early ones.
TRAIN_CHUNKS = (EVAL_CHUNKS,) * 10
TRAIN_PROFILE = "libero_long_eval_aligned_chunks_v2"
EVAL_PROFILE = "native_chunk800_v1"


def training_profile(task):
    if type(task) is not int or not 0 <= task < 10:
        raise ValueError("Unknown LIBERO task")
    return {"name": TRAIN_PROFILE, "task_id": task, "max_joint_chunks": TRAIN_CHUNKS[task]}


def validate_profile(profile, task):
    if profile is None:
        return {"name": EVAL_PROFILE}
    if profile != training_profile(task):
        raise ValueError("Training horizon differs from the fixed task contract")
    return dict(profile)
