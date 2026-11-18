"""Stage 2.1 Task-Specific RL: window-level DiffusionNFT specialist LoRA adapters per task on the Stage 1.1 teacher.

``core`` holds the task-independent machinery (samplers with a switchable LoRA, start pool, candidate windows, scoring,
the DiffusionNFT objective and learner, scene-worker client, round driver); ``tasks`` the task profiles (push_box,
open_room_door). Command line: ``python -m wam_policy.task_rl --help``. Inference with a released specialist:
``wam_policy.task_rl.adapters.attach_specialist(model, specialist_dir)``.
"""
