# WAM-Policy: World Action Models with On-Policy Distillation for Quadruped Loco-Manipulation

WAM-Policy is a research repo for distilling LingBot-VA style video-action world-action models with on-policy distillation.

This repo uses a single-root layout. Source code, configs, scripts, docs, and tests are tracked by Git. External source repos, pretrained models, datasets, checkpoints, runs, logs, and caches live under fixed local directories but are ignored by Git.

See `WAM-Policy_repo_structure.md` for the current repo structure and maintenance rules.

## Setup WAM-Policy Video-stream Forward Flow-map Training

### 1. Clone WAM-Policy on the training server

Clone the WAM-Policy repository on the server where Slurm jobs will be submitted, then enter the repo root:

```bash
git clone git@github.com:AIGeeksGroup/WAM-Policy.git
cd WAM-Policy
```

### 2. Create ignored local/runtime directories

WAM-Policy keeps external repos, pretrained models, datasets, run outputs, logs, caches, and Weights & Biases files under fixed paths inside the repo root. These paths are intentionally ignored by Git and must not be committed.

Create the local directory layout described in `WAM-Policy_repo_structure.md`:

```bash
mkdir -p \
  external \
  experiments/pretrained_models/lingbot-va \
  experiments/checkpoints/v2/video_ffm \
  experiments/runs/v2/video_ffm \
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

Do not add `.gitkeep` files or placeholder files to these directories.

### 3. Install the LingBot-VA source repo

WAM-Policy expects the LingBot-VA source repo at `external/lingbot-va`. The runtime config uses this path as `paths.lingbot_repo`.

```bash
git clone https://github.com/Robbyant/lingbot-va.git external/lingbot-va
test -d external/lingbot-va/wan_va
```

### 4. Download the LingBot-VA RoboTwin teacher checkpoint

Download the public LingBot-VA RoboTwin post-trained checkpoint from Hugging Face into the ignored pretrained-model directory. This downloads the full model snapshot, including the `transformer/`, `text_encoder/`, `tokenizer/`, `vae/`, and asset files.

Run large downloads only on a server or node where administrator policy allows them.

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

For WAM-Policy video Forward Flow-map Training, the teacher checkpoint path should point to the downloaded `transformer/` subdirectory:

```bash
export TEACHER_CKPT="$PWD/experiments/pretrained_models/lingbot-va/lingbot-va-posttrain-robotwin/transformer"

test -d "$TEACHER_CKPT"
ls -lh "$TEACHER_CKPT"
```

Keep `TEACHER_CKPT` exported in the shell used to submit the Slurm job. The admin Slurm script passes it into WAM-Policy as `teacher.checkpoint_path`.

### 5. Prepare the video latent manifest

WAM-Policy video Forward Flow-map Training uses precomputed LingBot-VA/WAN 2.2 video latents. You do not need to run LingBot-VA on a Mac to create these latents. Administrators should prepare them on the server from the official RoboTwin dataset.

Download the public RoboTwin dataset with precomputed WAN 2.2 latents:

```bash
huggingface-cli download \
  --repo-type dataset \
  robbyant/robotwin-clean-and-aug-lerobot \
  --local-dir datasets/raw/robotwin-clean-and-aug-lerobot
```

Convert a small debug subset into the WAM-Policy manifest format:

```bash
python3 scripts/tools/convert_lingbot_robotwin_latents.py \
  --repo-root "$PWD" \
  --source-root datasets/raw/robotwin-clean-and-aug-lerobot \
  --output-root datasets/latents/robotwin_video_ffm_debug \
  --manifest datasets/manifests/video_latents_debug.jsonl \
  --max-samples 16 \
  --overwrite
```

Use the debug manifest for the diagnostic stages:

```bash
export DEBUG_MANIFEST="$PWD/datasets/manifests/video_latents_debug.jsonl"
```

Convert the full manifest for the training stage:

```bash
python3 scripts/tools/convert_lingbot_robotwin_latents.py \
  --repo-root "$PWD" \
  --source-root datasets/raw/robotwin-clean-and-aug-lerobot \
  --output-root datasets/latents/robotwin_video_ffm_full \
  --manifest datasets/manifests/video_latents_full.jsonl \
  --max-samples 0 \
  --overwrite

export FULL_MANIFEST="$PWD/datasets/manifests/video_latents_full.jsonl"
```

The converter reads LingBot `.pth` files under `latents/**/*.pth`, writes split `z_clean.pt` and `context.pt` files under `datasets/latents/`, and writes JSONL manifests that the current WAM-Policy dataset loader can read. Keep both `DEBUG_MANIFEST` and `FULL_MANIFEST` exported in the shell used to submit the Slurm job.

### 6. Enable Weights & Biases and submit the Slurm job

WAM-Policy keeps Weights & Biases optional. The default config has `logging.wandb.enabled=auto`: wandb turns on when `WANDB_API_KEY` is present in the environment and stays off when no key is present.

Install the optional dependency only when remote experiment tracking is needed:

```bash
python3 -m pip install -e '.[wandb]'
```

Set the API key as an environment variable. Do not commit it to Git or write it into config files:

```bash
export WANDB_API_KEY="<private-api-key>"
```

Optionally set the entity or project if the run should go to a specific W&B team/workspace:

```bash
export WANDB_ENTITY="<team-or-user>"
export WANDB_PROJECT="WAM-Policy"
```

Submit the admin Slurm script once. It runs diagnostics first and automatically enters long training only if all diagnostic stages pass:

```bash
sbatch --export=ALL,TEACHER_CKPT="$TEACHER_CKPT",DEBUG_MANIFEST="$DEBUG_MANIFEST",FULL_MANIFEST="$FULL_MANIFEST" \
  scripts/slurm/video_ffm_admin_diagnostic.sbatch
```

The script runs:

```text
00_env -> 01_static -> 02_import -> 03_load_model -> 04_manifest -> 05_video_forward -> 06_train_long
```

`00_env` through `05_video_forward` use `DEBUG_MANIFEST`. If any diagnostic stage fails, the job stops at the first failing stage. If all diagnostics pass, `06_train_long` starts automatically with `FULL_MANIFEST`. By default the training stage uses:

```text
TRAIN_MAX_STEPS=5000
TRAIN_SAVE_EVERY=500
TRAIN_LOG_EVERY=10
TRAIN_BATCH_SIZE=1
```

For a shorter first smoke run, submit with:

```bash
sbatch --export=ALL,TEACHER_CKPT="$TEACHER_CKPT",DEBUG_MANIFEST="$DEBUG_MANIFEST",FULL_MANIFEST="$FULL_MANIFEST",TRAIN_MAX_STEPS=2 \
  scripts/slurm/video_ffm_admin_diagnostic.sbatch
```

Logs and checkpoints are written under ignored runtime directories:

```text
logs/admin_diagnostic/<JOB_ID>/
experiments/runs/v2/video_ffm_admin_debug/<JOB_ID>/
experiments/checkpoints/v2/video_ffm_admin_debug/<JOB_ID>/
```

When wandb is enabled, the script also uploads admin logs to the same W&B run as an `admin-logs` artifact:

```text
logs/admin_diagnostic/<JOB_ID>/status.txt
logs/admin_diagnostic/<JOB_ID>/*.log
slurm-wam-video-ffm-admin-<JOB_ID>.out
slurm-wam-video-ffm-admin-<JOB_ID>.err
logs/admin_diagnostic/<JOB_ID>/checkpoint_files_<JOB_ID>.txt
```

The upload includes a checkpoint file listing, not the checkpoint tensors themselves. Set `WANDB_UPLOAD_LOGS=false` to keep wandb metrics enabled but skip log artifact upload.

Check the first status file after submission:

```bash
cat logs/admin_diagnostic/<JOB_ID>/status.txt
```

Set `WANDB_ENABLED=false` explicitly if a run should not report to wandb even when `WANDB_API_KEY` is present.

References:

- LingBot-VA source repo: https://github.com/Robbyant/lingbot-va
- LingBot-VA RoboTwin checkpoint: https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin
- LingBot-VA RoboTwin dataset: https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot
- Hugging Face Hub download guide: https://huggingface.co/docs/huggingface_hub/en/guides/download
