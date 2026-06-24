# WAM-Policy: World Action Models with On-Policy Distillation

WAM-Policy is a research repo for distilling LingBot-VA style video-action world-action models with on-policy distillation.

The tracked repo contains source code, configs, scripts, docs, tests, and lightweight adapters. External source repos, pretrained models, datasets, checkpoints, run outputs, logs, caches, and W&B files live under ignored runtime directories and must not be committed. See `WAM-Policy_repo_structure.md` and `AGENTS.md` for repo boundaries and agent rules.

## Run WAM-Policy v2 Full Pipeline

The admin-facing v2 workflow is a staged Slurm job:

```text
video_ffm -> action_ffm -> video_dmd -> action_dmd
```

It runs diagnostics first, then smoke training, then long training:

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

Each stage stops on first failure and appends the first failing stage to:

```text
logs/v2_full_pipeline_admin/<JOB_ID>/status.txt
```

Use this workflow on a compute node through Slurm. Do not run real LingBot loading, checkpoint loading, GPU work, RoboTwin simulation, or long training from a lightweight login/agent session.

### 1. Prepare Runtime Directories

From the repo root on the training server:

```bash
mkdir -p \
  external \
  experiments/pretrained_models/lingbot-va \
  experiments/checkpoints/v2 \
  experiments/runs/v2 \
  datasets/raw \
  datasets/processed \
  datasets/latents \
  datasets/rollouts \
  datasets/manifests \
  results \
  logs \
  cache/huggingface \
  wandb
```

Do not add `.gitkeep` files or placeholders to ignored runtime directories.

### 2. Install External Repos

WAM-Policy expects LingBot-VA at `external/lingbot-va` and AnyFlow at `external/AnyFlow`:

```bash
git clone https://github.com/Robbyant/lingbot-va.git external/lingbot-va
test -d external/lingbot-va/wan_va

git clone <AnyFlow repo URL> external/AnyFlow
test -d external/AnyFlow
```

The current local/fake v2 path does not import AnyFlow runtime internals directly, but the config and bootstrap path keep it available for server work.

### 3. Prepare Teacher Checkpoint

Download the LingBot-VA RoboTwin post-trained checkpoint into the ignored pretrained-model directory on the server:

```bash
python3 -m pip install -U huggingface_hub
export HF_HOME="$PWD/cache/huggingface"

python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="robbyant/lingbot-va-posttrain-robotwin",
    repo_type="model",
    local_dir="experiments/pretrained_models/lingbot-va/lingbot-va-posttrain-robotwin",
)
PY
```

The v2 scripts usually point at the `transformer/` subdirectory:

```bash
export TEACHER_CKPT="$PWD/experiments/pretrained_models/lingbot-va/lingbot-va-posttrain-robotwin/transformer"
test -d "$TEACHER_CKPT"
```

### 4. Prepare Manifests

The full pipeline requires six manifests.

Video FFM manifests contain precomputed video latents and text context:

```bash
export VIDEO_DEBUG_MANIFEST="$PWD/datasets/manifests/video_latents_debug.jsonl"
export VIDEO_FULL_MANIFEST="$PWD/datasets/manifests/video_latents_full.jsonl"
```

The legacy video aliases are still accepted by the unified Slurm script:

```bash
export DEBUG_MANIFEST="$VIDEO_DEBUG_MANIFEST"
export FULL_MANIFEST="$VIDEO_FULL_MANIFEST"
```

Action FFM manifests contain clean action chunks, context, and optional `z_condition`:

```bash
export ACTION_DEBUG_MANIFEST="$PWD/datasets/manifests/action_chunks_debug.jsonl"
export ACTION_FULL_MANIFEST="$PWD/datasets/manifests/action_chunks_full.jsonl"
```

DMD OPD manifests contain student rollouts:

```bash
export ROLLOUT_DEBUG_MANIFEST="$PWD/datasets/manifests/student_rollouts_debug.jsonl"
export ROLLOUT_FULL_MANIFEST="$PWD/datasets/manifests/student_rollouts_full.jsonl"
```

The DMD long-training stages cannot run without rollout manifests. This repo path does not yet implement RoboTwin online rollout collection, so those manifests must be prepared ahead of time. For `teacher_score_source=cached_label`, rollout records must also point to cached `teacher_video_label` and `teacher_action_label` tensors.

### 5. Choose DMD Teacher Source

DMD config defaults to:

```text
opd.dmd.teacher_score_source=not_implemented
```

That default intentionally fails at DMD forward/training time until the teacher score path is selected.

Use cached labels when rollout manifests already contain teacher labels:

```bash
export DMD_TEACHER_SCORE_SOURCE=cached_label
```

Use real LingBot teacher velocity only after the server probe confirms the binding:

```bash
export DMD_TEACHER_SCORE_SOURCE=lingbot_velocity_forward
```

Current DMD surrogate defaults are debug-oriented:

```bash
export DMD_SURROGATE_STATE=pred_u
export DMD_STUDENT_SCORE_SOURCE=flowmap_small_interval
export DMD_SMALL_DELTA_S=0.02
```

`DMD_SURROGATE_STATE=pred_u` is a server interface debug surrogate, not the final clean-sample DMD objective. Switch to `clean_sample` only when the differentiable rollout path is validated.

### 6. Submit the Full Pipeline

Export the required environment variables, then submit:

```bash
export REPO_ROOT="$PWD"

sbatch --export=ALL scripts/slurm/v2_full_pipeline_admin_diagnostic.sbatch
```

For a short end-to-end smoke before long training, reduce both smoke and long steps:

```bash
export TRAIN_SMOKE_MAX_STEPS=1
export TRAIN_MAX_STEPS=2

sbatch --export=ALL scripts/slurm/v2_full_pipeline_admin_diagnostic.sbatch
```

Default long-training controls:

```text
TRAIN_MAX_STEPS=5000
TRAIN_SAVE_EVERY=500
TRAIN_LOG_EVERY=10
TRAIN_BATCH_SIZE=1
```

Training stages use isolated directories:

```text
experiments/runs/v2/video_ffm_admin/<JOB_ID>/
experiments/runs/v2/action_ffm_admin/<JOB_ID>/
experiments/runs/v2/video_dmd_admin/<JOB_ID>/
experiments/runs/v2/action_dmd_admin/<JOB_ID>/

experiments/checkpoints/v2/video_ffm_admin/<JOB_ID>/
experiments/checkpoints/v2/action_ffm_admin/<JOB_ID>/
experiments/checkpoints/v2/video_dmd_admin/<JOB_ID>/
experiments/checkpoints/v2/action_dmd_admin/<JOB_ID>/
```

### 7. W&B Safety

W&B is optional. With `WANDB_ENABLED=auto`, it activates only when `WANDB_API_KEY` is present:

```bash
export WANDB_API_KEY="<private-api-key>"
export WANDB_PROJECT="WAM-Policy"
export WANDB_ENTITY="<team-or-user>"
```

Do not commit secrets. The admin script may upload a log artifact containing `status.txt`, per-stage logs, Slurm output/error files, and a checkpoint file listing. It does not upload checkpoint tensors, manifests, datasets, or secrets. Disable log artifact upload while keeping metrics enabled with:

```bash
export WANDB_UPLOAD_LOGS=false
```

Disable W&B entirely with:

```bash
export WANDB_ENABLED=false
```

### 8. Debug Fallback Scripts

Use the unified full-pipeline script first. Stage-specific scripts remain available for narrower debugging:

```text
scripts/slurm/video_ffm_admin_diagnostic.sbatch
scripts/slurm/action_ffm_admin_diagnostic.sbatch
scripts/slurm/dmd_opd_admin_diagnostic.sbatch
```

They are useful when a single stage fails and you want to iterate only on that binding or manifest path.

## Local Lightweight Checks

Local checks should stay static/fake/lightweight:

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

Do not run full pytest, real checkpoint loading, LingBot/RoboTwin runtime, GPU, ROCm, torchrun, or RoboTwin simulation in lightweight sessions.

References:

- LingBot-VA source repo: https://github.com/Robbyant/lingbot-va
- LingBot-VA RoboTwin checkpoint: https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin
- LingBot-VA RoboTwin dataset: https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot
