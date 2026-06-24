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

Implemented stage coverage now includes:

- `video_ffm`: existing video Forward Flow-map Training path remains intact.
- `action_ffm`: config, static dispatch, action flow-map pair, action dataset,
  action loss, action student wrapper, LingBot action adapter skeleton,
  ActionFFMTrainer, probe script, Slurm diagnostic, and fake CPU tests.
- `video_dmd`: config, static dispatch, rollout manifest dataset, DMD surrogate
  loss, VideoDMDOPDTrainer, DMD probe script, Slurm diagnostic, and fake CPU
  tests.
- `action_dmd`: config, static dispatch, ActionDMDOPDTrainer, action DMD probe
  support, and fake CPU tests.

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
scripts/slurm/action_ffm_admin_diagnostic.sbatch
scripts/slurm/dmd_opd_admin_diagnostic.sbatch
```

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

1. Run `action_ffm_admin_diagnostic.sbatch` through `05_action_forward`.
2. If action binding fails, fix only `LingBotVAAdapter` action input/output
   binding.
3. Run `dmd_opd_admin_diagnostic.sbatch` for `STAGE=video_dmd` and
   `STAGE=action_dmd`.
4. Expect teacher score/velocity binding to fail clearly at `05_dmd_forward`
   until the real LingBot teacher score contract is confirmed.
5. After forward probes pass, run `TRAIN_MAX_STEPS=2` smoke jobs before longer
   training.

## 6. Worktree Notes

At the end of the implementation session, the code changes were committed in
the eight listed commits and local lightweight acceptance passed. No external
repos, checkpoints, datasets, rollouts, logs, caches, or wandb files were added.
