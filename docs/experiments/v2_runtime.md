# WAM-Policy v2 Runtime Notes

This note covers the current v2 local smoke checks and server Slurm entrypoints.
The Mac/local path is limited to config dispatch, static probes, fake tensor tests,
and py_compile. Real LingBot checkpoints, RoboTwin rollout, ROCm, and teacher
score bindings are validated only on a compute node.

## Implemented v2 Stages

- `video_ffm`: video-stream Forward Flow-map Training.
- `action_ffm`: action-stream Forward Flow-map Training interface, dataset,
  fake-testable trainer, and server probe entrypoint.
- `video_dmd`: video DMD-style OPD trainer skeleton with fake CPU tests.
- `action_dmd`: action DMD-style OPD trainer skeleton with fake CPU tests.

All trainer-facing code uses `s_src`, `s_tgt`, and `delta_s` semantics where
`s=0` is noise and `s=1` is clean data.

## Local Acceptance

Run only lightweight checks locally:

```bash
python scripts/train.py --config configs/experiments/v2/anyflow_video_ffm.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_action_ffm.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_video_dmd_opd.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_action_dmd_opd.yaml --dry-run static

python scripts/tools/probe_action_ffm_runtime.py probe-static-source \
  --config configs/experiments/v2/anyflow_action_ffm.yaml
python scripts/tools/probe_dmd_opd_runtime.py probe-static-source \
  --config configs/experiments/v2/anyflow_video_dmd_opd.yaml
python scripts/tools/probe_dmd_opd_runtime.py probe-static-source \
  --config configs/experiments/v2/anyflow_action_dmd_opd.yaml
```

Targeted tests:

```bash
pytest \
  tests/test_pipeline_static_dispatch.py \
  tests/test_action_flowmap_pair.py \
  tests/test_action_dataset.py \
  tests/test_action_ffm_loss.py \
  tests/test_action_ffm_trainer_static.py \
  tests/test_dmd_losses.py \
  tests/test_rollout_manifest_dataset.py \
  tests/test_video_dmd_trainer_static.py \
  tests/test_action_dmd_trainer_static.py \
  tests/test_probe_scripts_static.py
```

Do not run full training, RoboTwin simulation, GPU work, or full pytest in a
lightweight local session.

## Server Smoke Jobs

Action FFM:

```bash
sbatch --export=ALL,TEACHER_CKPT="$TEACHER_CKPT",ACTION_DEBUG_MANIFEST="$ACTION_DEBUG_MANIFEST",ACTION_FULL_MANIFEST="$ACTION_FULL_MANIFEST",TRAIN_MAX_STEPS=2 \
  scripts/slurm/action_ffm_admin_diagnostic.sbatch
```

DMD OPD:

```bash
sbatch --export=ALL,STAGE=video_dmd,TEACHER_CKPT="$TEACHER_CKPT",ROLLOUT_DEBUG_MANIFEST="$ROLLOUT_DEBUG_MANIFEST",ROLLOUT_FULL_MANIFEST="$ROLLOUT_FULL_MANIFEST",TRAIN_MAX_STEPS=2 \
  scripts/slurm/dmd_opd_admin_diagnostic.sbatch

sbatch --export=ALL,STAGE=action_dmd,TEACHER_CKPT="$TEACHER_CKPT",ROLLOUT_DEBUG_MANIFEST="$ROLLOUT_DEBUG_MANIFEST",ROLLOUT_FULL_MANIFEST="$ROLLOUT_FULL_MANIFEST",TRAIN_MAX_STEPS=2 \
  scripts/slurm/dmd_opd_admin_diagnostic.sbatch
```

Expected DMD behavior: config, import, model load, and rollout manifest probes
should pass first. Real teacher score or velocity binding may still fail at
`05_dmd_forward` with a clear adapter error until the server-side LingBot
contract is confirmed.

For `action_dmd`, rollout records must provide `z_condition_path` from student
video rollout. The action DMD trainer intentionally rejects missing
`z_condition` instead of falling back to teacher video condition.
