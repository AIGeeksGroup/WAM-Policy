# Interfaces for Policy and Training Code

[QLM-Bench README](../README.md)

**Standard100** (`python -m qlm_bench.benchmark`):

```
fetch   --output <resource_root> [--case-id ID ...] [--repo-id R] [--revision REV]
verify  --resource-root <resource_root> --output <report.json> [--case-id ID ...]
run     --resource-root <resource_root> --policy-port P --output <new dir> [--policy-seed {42,43,44}]
        [--cases id1,id2] [--policy-name NAME] [--record-video --ffmpeg F --ffprobe P]
        [--previous-result <result.json>] [--rpc-timeout-s 600] [--suite <dir>]
prepare --snapshot <dataset snapshot> --controller-root <dir> --output <dir> [--resource-map <json>]   # maintainers
```
Transport session of an episode: `evaluation-<task>/<case_id>/<trial_id>/attempt-NNN`. Exit code 0 iff every
expected trial has a task outcome (success / failure / timeout). Python: `qlm_bench.benchmark.run(resource_root,
output, *, policy_port, policy_seed=42, cases=None, ...)` (sets single-threaded torch and `MUJOCO_GL=egl` if
unset); `qlm_bench.benchmark.suite.{load_suite, reconstruct, runtime_for}` (`runtime_for` is Standard100-only and
requires hardware EGL; the Task RL scene worker uses only `reconstruct`).

**Streaming training environment** (`python -m qlm_bench.training_env <command>`):

```
worker --pool <manifest.json> --pool-sha256 <sha> --rank R --world-size W --port P --output <dir>
       --rpc-timeout-s T --wall-timeout-s T --segment S --training-profile <profile.json> --schedule-db <sqlite>
       [--max-episodes N] [--diagnostic] [--render-mode {full,sampled}]
make-profile --pool <manifest.json> --profile <out.json> [--tasks a,b] [--max-active-per-task N]
inspect-profile | schedule-init | schedule-snapshot | schedule-restore | schedule-stop | inspect-pool
prepare-pool --snapshot <dir> --output <dir> --controller-root <dir> [--resource-map <json>]   # maintainers
python -m qlm_bench.training_env.visualization --pool <manifest.json> --port P --output <dir>
```
The worker connects to the learner's `qlm-live-v1` server. Its `reset` request carries
`extensions['training'] = {train_pool_sha256, training_profile_sha256, train_episode_uid, segment,
schedule_ticket, render_mode, runtime: {physics, controller, rendering, rgb_profile}}` (`render_mode` and
`runtime.rendering` are informational). The learner stops the worker with `stop_after_feedback` in a feedback
response. Training profile (`qlm_bench.training_env.schedule.profile(pool_sha256, tasks=None,
max_active_per_task=2)`): `{schema: qlm-training-profile-v1, train_pool_sha256, training_horizons_s,
training_tasks, seed: 42, sampling, max_active_per_task, runtime}`; default = the nine training tasks
(all except `sweep_block_to_zone`) with two active episodes each. Python: `training_env.worker.select_cpu_environment`,
`training_env.schedule.EpisodeSchedule`.

**Task RL scene worker** (`python -m qlm_bench.task_rl.scene_worker --controller <model_2000.pt> --state-root <dir>
[--render-mode {full,sampled}]`, or in process `qlm_bench.task_rl.scene_worker.SceneWorker`): length-prefixed
JSON frames on stdin/stdout, envelope `{schema: qlm-task-rl-scene-ipc-v1, id, operation, session, payload}`.
Operations: `reset {case, model}`, `restore_state {case, model, path}`, `execute {chunk, commands (16×9), ring}`,
`capture_state {path}`, `persist_ring {chunk, path, origin}`, `inspect`. Execution results use schema
`qlm-task-rl-execution-v1` (RGB samples every 40 ticks, proprioception, privileged rows
`qlm-task-rl-privileged-v1`, terminal, progress); rollout states are `<name>.pt` + `<name>.json` +
`<name>.own.json` (`qlm-task-rl-rollout-state-v1`) under `--state-root`.

**Task RL inputs** (`inputs/task_rl/<task>/`): recorded absolute paths start with `@WORKSPACE@`, to be replaced by
the local data root; `scene-files.json` lists every scene file (and the controller under
`assets/rambo-go2-controller/iteration-2000/runtime/`) by sha256 so missing files can be fetched from the dataset.
