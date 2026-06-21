#!/usr/bin/env bash
#SBATCH --job-name=wam-video-ffm-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Usage examples:
#   sbatch --export=ALL,MODE=probe_static_source scripts/slurm/video_ffm_debug.sbatch
#   sbatch --export=ALL,MODE=probe_import scripts/slurm/video_ffm_debug.sbatch
#   sbatch --export=ALL,MODE=probe_load_model scripts/slurm/video_ffm_debug.sbatch \
#     --set teacher.checkpoint_path=experiments/pretrained_models/lingbot-va/<ckpt>
#   sbatch --export=ALL,MODE=probe_manifest_one_sample scripts/slurm/video_ffm_debug.sbatch \
#     --set data.manifest=datasets/manifests/video_latents_debug.jsonl
#   sbatch --export=ALL,MODE=probe_video_forward scripts/slurm/video_ffm_debug.sbatch \
#     --set teacher.checkpoint_path=experiments/pretrained_models/lingbot-va/<ckpt> \
#     --set data.manifest=datasets/manifests/video_latents_debug.jsonl
#   sbatch --export=ALL,MODE=train_tiny scripts/slurm/video_ffm_debug.sbatch \
#     --set teacher.checkpoint_path=experiments/pretrained_models/lingbot-va/<ckpt> \
#     --set data.manifest=datasets/manifests/video_latents_debug.jsonl
#
# Optional environment variables:
#   REPO_ROOT=/path/to/WAM-Policy
#   CONFIG=configs/experiments/v2/anyflow_video_ffm.yaml
#   PYTHON_BIN=/path/to/python
#   ENV_SETUP=/path/to/env_setup.sh

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONFIG="${CONFIG:-configs/experiments/v2/anyflow_video_ffm.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${MODE:-probe_import}"

cd "$REPO_ROOT"

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_SETUP"
fi

echo "repo_root: $REPO_ROOT"
echo "config: $CONFIG"
echo "mode: $MODE"
echo "python: $PYTHON_BIN"
echo "hostname: $(hostname)"
echo "slurm_job_id: ${SLURM_JOB_ID:-local}"

case "$MODE" in
  probe_import)
    "$PYTHON_BIN" scripts/tools/probe_video_ffm_runtime.py \
      probe-import \
      --config "$CONFIG" \
      "$@"
    ;;
  probe_static_source)
    "$PYTHON_BIN" scripts/tools/probe_video_ffm_runtime.py \
      probe-static-source \
      --config "$CONFIG" \
      "$@"
    ;;
  probe_load_model)
    "$PYTHON_BIN" scripts/tools/probe_video_ffm_runtime.py \
      probe-load-model \
      --config "$CONFIG" \
      "$@"
    ;;
  probe_manifest_one_sample)
    "$PYTHON_BIN" scripts/tools/probe_video_ffm_runtime.py \
      probe-manifest-one-sample \
      --config "$CONFIG" \
      "$@"
    ;;
  probe_video_forward)
    "$PYTHON_BIN" scripts/tools/probe_video_ffm_runtime.py \
      probe-video-forward \
      --config "$CONFIG" \
      "$@"
    ;;
  train_tiny)
    "$PYTHON_BIN" scripts/train.py \
      --config "$CONFIG" \
      "$@" \
      --set train.max_steps=2 \
      --set data.batch_size=1 \
      --set data.num_workers=0 \
      --set train.save_every=1 \
      --set train.log_every=1
    ;;
  *)
    echo "Unknown MODE: $MODE" >&2
    exit 2
    ;;
esac
