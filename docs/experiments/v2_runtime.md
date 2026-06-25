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
- `offline_video_stream` visualization: LingBot-VA teacher 25-step video
  sampling vs WAM-Policy_v2 student 4-step flow-map sampling, stitched when
  continuous chunk metadata exists and rendered as a grid otherwise.
- `robotwin_rollout` visualization: W&B video logging and admin workflow are
  wired through `RoboTwinAdapter.rollout_student_episode`; server deployments
  must bind that adapter to the local RoboTwin environment.

All trainer-facing code uses `s_src`, `s_tgt`, and `delta_s` semantics where
`s=0` is noise and `s=1` is clean data.

## Local Acceptance

Run only lightweight checks locally:

```bash
python scripts/train.py --config configs/experiments/v2/anyflow_video_ffm.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_action_ffm.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_video_dmd_opd.yaml --dry-run static
python scripts/train.py --config configs/experiments/v2/anyflow_action_dmd_opd.yaml --dry-run static
python scripts/visualize_v2.py offline-video-stream --config configs/experiments/v2/anyflow_visualization.yaml --dry-run static
python scripts/visualize_v2.py robotwin-rollout --config configs/experiments/v2/anyflow_visualization.yaml --dry-run static

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
  tests/test_probe_scripts_static.py \
  tests/test_visualization_static.py
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

## Server Visualization Jobs

The unified admin script runs visualization after long training by default:

```text
19_offline_video_stream_visualization
20_robotwin_rollout_visualization
21_collect_artifacts
```

The output root is `results/v2/visualization_admin/<JOB_ID>/`. W&B logs videos
under:

```text
offline_video_stream/teacher_25step
offline_video_stream/student_4step
offline_video_stream/teacher_vs_student
robotwin_rollout/actual_observation
robotwin_rollout/student_imagined_video
robotwin_rollout/actual_vs_imagined
```

Use `RUN_OFFLINE_VISUALIZATION=false` or `RUN_ROBOTWIN_VISUALIZATION=false` only
when isolating earlier training/runtime failures. Offline visualization uses the
manifest latent frame shape directly; WAM-Policy does not force chunks to two
frames.

For visualization-only diagnostics, use:

```bash
sbatch --export=ALL scripts/slurm/v2_visualization_admin_diagnostic.sbatch
```

The visualization admin job runs:

```text
00_env
01_import
02_video_encoder_probe
03_vae_decode_probe
04_teacher_student_sampling_probe
05_decode_and_encode_probe
06_visualization_full
07_wandb_upload
08_collect_artifacts
```

`01_import` imports WAM-Policy, LingBot-VA/Wan VAE, and W&B when available, and
scans LingBot/Wan VAE-adjacent source files for CUDA-only patterns such as
`.cuda(`, `torch.device("cuda")`, `flash_attn`, and CUDA extensions.
`03_vae_decode_probe` loads the VAE through `LingBotVAAdapter` and decodes a
manifest latent on the configured ROCm PyTorch device. Failures write a summary
with `failed_stage=vae_decode`, the concrete import/op/device error, traceback,
and adapter location.

Video encoding probes prefer GPU ffmpeg backends first (`h264_vaapi` or
`h264_amf`), then CPU ffmpeg/libx264 or imageio, then GIF, PNG frame sequence,
and finally a saved RGB tensor. Summary JSON records backend, codec, fps, output
path, and fallback level.

RoboTwin online rollout uses `visualization.robotwin_rollout.runner`. The
server-ready mode is `external_command`: set `ROBOTWIN_ROLLOUT_COMMAND` to a
site-specific closed-loop rollout command. The command receives
`WAM_ROBOTWIN_REQUEST_JSON`, `WAM_ROBOTWIN_RESULT_JSON`, and
`WAM_ROBOTWIN_OUTPUT_DIR`; it must write a result JSON pointing to
actual-observation, student-imagined, and side-by-side videos.
