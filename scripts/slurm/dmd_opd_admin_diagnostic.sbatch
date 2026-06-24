#!/bin/bash
#SBATCH --job-name=wam-dmd-opd
#SBATCH --output=logs/slurm/dmd_opd_%j.out
#SBATCH --error=logs/slurm/dmd_opd_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
STAGE="${STAGE:-video_dmd}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${MODE:-01_static}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-2}"
TRAIN_SAVE_EVERY="${TRAIN_SAVE_EVERY:-1}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"

if [[ "$STAGE" == "video_dmd" ]]; then
  CONFIG="${CONFIG:-configs/experiments/v2/anyflow_video_dmd_opd.yaml}"
elif [[ "$STAGE" == "action_dmd" ]]; then
  CONFIG="${CONFIG:-configs/experiments/v2/anyflow_action_dmd_opd.yaml}"
else
  echo "STAGE must be video_dmd or action_dmd, got: $STAGE" >&2
  exit 2
fi

cd "$REPO_ROOT"

print_header() {
  printf '\n== %s ==\n' "$1"
}

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is missing: $name" >&2
    exit 2
  fi
}

common_overrides=(
  --set runtime.device=rocm
  --set runtime.lingbot_attn_mode=torch
)

if [[ -n "${TEACHER_CKPT:-}" ]]; then
  common_overrides+=(--set "teacher.checkpoint_path=$TEACHER_CKPT")
fi

rollout_manifest_for_mode() {
  if [[ "$MODE" == "06_train_short_or_long" && -n "${ROLLOUT_FULL_MANIFEST:-}" ]]; then
    printf '%s\n' "$ROLLOUT_FULL_MANIFEST"
  else
    printf '%s\n' "${ROLLOUT_DEBUG_MANIFEST:-}"
  fi
}

case "$MODE" in
  00_env)
    print_header "00_env"
    "$PYTHON_BIN" - <<'PY'
import torch
print(f"torch.__version__: {torch.__version__}")
print(f"torch.version.hip: {torch.version.hip}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
PY
    ;;
  01_static)
    print_header "01_static"
    "$PYTHON_BIN" scripts/train.py --config "$CONFIG" --dry-run static "${common_overrides[@]}"
    "$PYTHON_BIN" scripts/tools/probe_dmd_opd_runtime.py probe-static-source \
      --config "$CONFIG" "${common_overrides[@]}"
    ;;
  02_import)
    print_header "02_import"
    "$PYTHON_BIN" scripts/tools/probe_dmd_opd_runtime.py probe-import \
      --config "$CONFIG" --require-external-exists "${common_overrides[@]}"
    ;;
  03_load_model)
    require_var TEACHER_CKPT
    print_header "03_load_model"
    "$PYTHON_BIN" scripts/tools/probe_dmd_opd_runtime.py probe-load-model \
      --config "$CONFIG" --require-external-exists "${common_overrides[@]}"
    ;;
  04_rollout_manifest)
    require_var ROLLOUT_DEBUG_MANIFEST
    manifest="$(rollout_manifest_for_mode)"
    print_header "04_rollout_manifest"
    "$PYTHON_BIN" scripts/tools/probe_dmd_opd_runtime.py probe-rollout-manifest \
      --config "$CONFIG" --require-external-exists \
      "${common_overrides[@]}" --set "opd.rollout_manifest=$manifest"
    ;;
  05_dmd_forward)
    require_var TEACHER_CKPT
    require_var ROLLOUT_DEBUG_MANIFEST
    manifest="$(rollout_manifest_for_mode)"
    print_header "05_dmd_forward"
    "$PYTHON_BIN" scripts/tools/probe_dmd_opd_runtime.py probe-dmd-forward \
      --config "$CONFIG" --require-external-exists \
      "${common_overrides[@]}" --set "opd.rollout_manifest=$manifest"
    ;;
  06_train_short_or_long)
    require_var TEACHER_CKPT
    require_var ROLLOUT_DEBUG_MANIFEST
    manifest="$(rollout_manifest_for_mode)"
    print_header "06_train_short_or_long"
    "$PYTHON_BIN" scripts/train.py --config "$CONFIG" \
      "${common_overrides[@]}" \
      --set "opd.rollout_manifest=$manifest" \
      --set "data.batch_size=$TRAIN_BATCH_SIZE" \
      --set "train.max_steps=$TRAIN_MAX_STEPS" \
      --set "train.save_every=$TRAIN_SAVE_EVERY" \
      --set "train.log_every=$TRAIN_LOG_EVERY"
    ;;
  *)
    echo "Unknown MODE: $MODE" >&2
    exit 2
    ;;
esac
