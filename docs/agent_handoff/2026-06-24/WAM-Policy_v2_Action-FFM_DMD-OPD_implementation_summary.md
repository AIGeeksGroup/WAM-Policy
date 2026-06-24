# WAM-Policy v2 Action-FFM + DMD-OPD Implementation Summary

> Session date: 2026-06-24  
> Scope: WAM-Policy v2 local/static implementation only. No Mac-side checkpoint
> loading, RoboTwin simulation, GPU work, or real LingBot training was run.

## 1. What Was Completed

This session implemented the handoff plan in eight local commits:

```text
0eb2b2a Add v2 stage static dispatch configs
980c6cb Add action flowmap data and loss
cbb1a87 Add action student and LingBot action hooks
287e8b0 Add action FFM trainer path
3bff56d Add rollout schema and DMD surrogate loss
5e307d3 Add video DMD OPD trainer and probes
79995ff Add action DMD OPD trainer path
363929e Document v2 local and server smoke checks
```

A later repository commit, `39f5145 Add: 06/24 implementation plan`, added the
original plan document under this same handoff directory.

A later single commit, `9adfb6e Prepare v2 full pipeline admin debug`, folded in
the remaining server-debug readiness work from
`WAM-Policy_v2_remaining_stages_server_debug_handoff.md`.

Implemented stage coverage now includes:

- `video_ffm`: existing video Forward Flow-map Training path remains intact.
- `action_ffm`: config, static dispatch, action flow-map pair, action dataset,
  action loss, action student wrapper, LingBot action adapter skeleton,
  ActionFFMTrainer, probe script, Slurm diagnostic, and fake CPU tests.
- `video_dmd`: config, static dispatch, rollout manifest dataset, DMD surrogate
  loss, VideoDMDOPDTrainer, DMD probe script, Slurm diagnostic, and fake CPU
  tests. The follow-up commit added explicit DMD source configuration,
  cached-label teacher fallback, small-interval student score probing, and
  clearer non-dry-run failure semantics.
- `action_dmd`: config, static dispatch, ActionDMDOPDTrainer, action DMD probe
  support, and fake CPU tests. The follow-up commit added the same DMD source
  configuration and kept the student-rollout `z_condition` invariant explicit.

## 2. Key Implementation Details

The v2 pipeline now dispatches all four stage names:

```text
video_ffm
action_ffm
video_dmd
action_dmd
```

The code-facing time convention remains unchanged:

```text
s = 0: noise
s = 1: clean data
0 <= s_src < s_tgt <= 1
target_u = (x_tgt - x_src) / (s_tgt - s_src)
```

Action FFM uses the same average-velocity target as video FFM:

```text
A_s = (1 - s) * eps + s * A_clean
target_u = (A_tgt - A_src) / (s_tgt - s_src)
```

LingBot two-time conditioning was changed from a single global state to
mode-specific state:

```text
model._wam_current_flowmap_mode = "video" | "action" | None
model.wam_video_time_pair_conditioner
model.wam_action_time_pair_conditioner
model.wam_time_pair_conditioner  # retained as video alias
```

This keeps the existing video FFM path compatible while allowing action FFM to
attach a separate conditioner.

Action DMD intentionally requires rollout `z_condition` from the student video
rollout path. It rejects missing `z_condition` instead of silently falling back
to teacher video condition.

The follow-up single commit made DMD configuration explicit:

```text
opd.dmd.surrogate_state: pred_u | clean_sample
opd.dmd.teacher_score_source: not_implemented | lingbot_velocity_forward | cached_label
opd.dmd.student_score_source: flowmap_small_interval | current_interval_average_velocity
opd.dmd.small_delta_s: 0.02
```

Important semantics:

- `teacher_score_source=not_implemented` is the default and intentionally fails
  DMD non-dry-run paths immediately with a clear remediation message.
- `teacher_score_source=cached_label` reads `teacher_video_label` or
  `teacher_action_label` from rollout manifest samples, then checks shape,
  device, and finiteness.
- `teacher_score_source=lingbot_velocity_forward` still depends on the real
  server-side LingBot teacher velocity binding.
- `student_score_source=flowmap_small_interval` now actually runs an additional
  bounded student flow-map call using `opd.dmd.small_delta_s`.
- `surrogate_state=pred_u` is a debug surrogate for server interface bring-up,
  not the final clean-sample DMD objective. `clean_sample` is implemented as a
  selectable branch for fake/static validation.

## 3. New Runtime Entrypoints

Local/static probes:

```bash
python scripts/tools/probe_action_ffm_runtime.py probe-static-source \
  --config configs/experiments/v2/anyflow_action_ffm.yaml

python scripts/tools/probe_dmd_opd_runtime.py probe-static-source \
  --config configs/experiments/v2/anyflow_video_dmd_opd.yaml

python scripts/tools/probe_dmd_opd_runtime.py probe-static-source \
  --config configs/experiments/v2/anyflow_action_dmd_opd.yaml
```

Server Slurm diagnostics:

```text
scripts/slurm/v2_full_pipeline_admin_diagnostic.sbatch
scripts/slurm/action_ffm_admin_diagnostic.sbatch
scripts/slurm/dmd_opd_admin_diagnostic.sbatch
```

The new admin-facing primary workflow is:

```bash
sbatch --export=ALL scripts/slurm/v2_full_pipeline_admin_diagnostic.sbatch
```

It covers the full intended v2 staged long task:

```text
00_env
01_static_all
02_import
03_load_model
04_video_manifest_debug
05_video_forward_debug
06_action_manifest_debug
07_action_forward_debug
08_rollout_manifest_debug
09_video_dmd_forward_debug
10_action_dmd_forward_debug
11_video_ffm_train_smoke
12_action_ffm_train_smoke
13_video_dmd_train_smoke
14_action_dmd_train_smoke
15_video_ffm_train_long_full_manifest
16_action_ffm_train_long_full_manifest
17_video_dmd_train_long_full_manifest
18_action_dmd_train_long_full_manifest
19_collect_artifacts
```

Required environment variables for the unified script:

```text
REPO_ROOT
TEACHER_CKPT
VIDEO_DEBUG_MANIFEST
VIDEO_FULL_MANIFEST
ACTION_DEBUG_MANIFEST
ACTION_FULL_MANIFEST
ROLLOUT_DEBUG_MANIFEST
ROLLOUT_FULL_MANIFEST
```

Legacy video aliases `DEBUG_MANIFEST` and `FULL_MANIFEST` are still accepted for
the video manifest paths.

The stage-specific Slurm scripts remain as lower-level debug fallbacks.

The DMD diagnostic supports:

```text
STAGE=video_dmd
STAGE=action_dmd
```

The runtime notes are documented in:

```text
docs/experiments/v2_runtime.md
```

## 4. Verification Performed

Static dry-run checks passed for all four stage configs:

```bash
python scripts/train.py --config configs/experiments/v2/anyflow_video_ffm.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_action_ffm.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_video_dmd_opd.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_action_dmd_opd.yaml --dry-run static
```

Static probe checks passed:

```bash
python scripts/tools/probe_action_ffm_runtime.py probe-static-source --config configs/experiments/v2/anyflow_action_ffm.yaml
python scripts/tools/probe_dmd_opd_runtime.py probe-static-source --config configs/experiments/v2/anyflow_video_dmd_opd.yaml
python scripts/tools/probe_dmd_opd_runtime.py probe-static-source --config configs/experiments/v2/anyflow_action_dmd_opd.yaml
```

Py compile passed for:

```text
scripts/train.py
scripts/tools/probe_action_ffm_runtime.py
scripts/tools/probe_dmd_opd_runtime.py
src/wam_policy/adapters/anyflow.py
src/wam_policy/adapters/lingbot_va.py
src/wam_policy/adapters/robotwin.py
src/wam_policy/data/action_dataset.py
src/wam_policy/data/rollout_manifest_dataset.py
src/wam_policy/opd/losses.py
src/wam_policy/opd/stages.py
src/wam_policy/opd/trainer.py
```

Targeted pytest passed:

```text
69 passed
```

The targeted tests covered pipeline static dispatch, action flow-map pair,
action dataset/loss/trainer, LingBot student/adapter fake paths, rollout
manifest dataset, DMD surrogate loss, video/action DMD fake trainers, and probe
script static checks.

After the follow-up single commit, local lightweight verification was rerun for
the changed surface:

```bash
python scripts/train.py --config configs/experiments/v2/anyflow_video_ffm.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_action_ffm.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_video_dmd_opd.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_action_dmd_opd.yaml --dry-run static

python scripts/tools/probe_action_ffm_runtime.py probe-static-source --config configs/experiments/v2/anyflow_action_ffm.yaml
python scripts/tools/probe_dmd_opd_runtime.py probe-static-source --config configs/experiments/v2/anyflow_video_dmd_opd.yaml
python scripts/tools/probe_dmd_opd_runtime.py probe-static-source --config configs/experiments/v2/anyflow_action_dmd_opd.yaml

python -m py_compile scripts/train.py scripts/tools/probe_action_ffm_runtime.py scripts/tools/probe_dmd_opd_runtime.py scripts/tools/upload_admin_logs_to_wandb.py src/wam_policy/adapters/anyflow.py src/wam_policy/adapters/lingbot_va.py src/wam_policy/adapters/robotwin.py src/wam_policy/data/action_dataset.py src/wam_policy/data/rollout_manifest_dataset.py src/wam_policy/opd/losses.py src/wam_policy/opd/stages.py src/wam_policy/opd/trainer.py

bash -n scripts/slurm/v2_full_pipeline_admin_diagnostic.sbatch

python -m pytest tests/test_pipeline_static_dispatch.py tests/test_action_flowmap_pair.py tests/test_action_ffm_trainer_static.py tests/test_dmd_losses.py tests/test_video_dmd_trainer_static.py tests/test_action_dmd_trainer_static.py tests/test_probe_scripts_static.py
```

The latest targeted pytest result was:

```text
34 passed
```

Additional test coverage was added for:

- `surrogate_state=pred_u` and `surrogate_state=clean_sample`.
- `student_score_source=flowmap_small_interval` using bounded `small_delta_s`.
- `teacher_score_source=not_implemented` failing clearly in non-dry-run DMD.
- `teacher_score_source=cached_label` using rollout teacher labels.
- action forward probe diagnostic formatting and fix-location guidance.

## 5. Remaining Server-Side Work

The following are intentionally not solved by Mac/local implementation:

- Real LingBot action forward contract still needs compute-node validation.
- Real LingBot teacher video/action score or velocity binding still needs
  server-side probe work.
- Real action manifest shape and rollout manifest shape still need validation
  against produced server data.
- RoboTwin online rollout generation is still a server/runtime integration task.
- Distributed training, FSDP, DeepSpeed/ZeRO, and long training are not
  implemented or validated.

Expected next server workflow:

1. Prepare real `TEACHER_CKPT`, video latent debug/full manifests, action chunk
   debug/full manifests, and student rollout debug/full manifests.
2. Submit `scripts/slurm/v2_full_pipeline_admin_diagnostic.sbatch` as the
   primary staged admin workflow.
3. Set `DMD_TEACHER_SCORE_SOURCE=cached_label` if rollout manifests contain
   teacher labels, or set `DMD_TEACHER_SCORE_SOURCE=lingbot_velocity_forward`
   only after the real teacher velocity binding probe succeeds.
4. If action binding fails, fix only `LingBotVAAdapter` action input/output
   binding.
5. If DMD teacher score/velocity binding fails, fix only the relevant
   `LingBotVAAdapter.teacher_*_score_or_velocity` method.
6. Use `action_ffm_admin_diagnostic.sbatch` or `dmd_opd_admin_diagnostic.sbatch`
   only as narrower fallback tools for isolated stage debug.

DMD long training cannot run without rollout manifests because this repo path
does not yet implement RoboTwin online rollout collection.

## 6. Worktree Notes

At the end of the implementation session, the code changes were committed in
the eight listed commits and local lightweight acceptance passed. No external
repos, checkpoints, datasets, rollouts, logs, caches, or wandb files were added.

At the end of the follow-up server-debug readiness pass, the new local commit
was:

```text
9adfb6e Prepare v2 full pipeline admin debug
```

That commit also updated `README.md` and `AGENTS.md` so future users see the
unified full v2 admin pipeline as the primary server workflow.
