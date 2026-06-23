#!/usr/bin/env bash
#SBATCH --job-name=wam-video-ffm-admin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Admin submission example:
#   sbatch --export=ALL,TEACHER_CKPT=/abs/path/to/transformer,DEBUG_MANIFEST=/abs/path/to/video_latents_debug.jsonl,FULL_MANIFEST=/abs/path/to/video_latents_full.jsonl \
#     /shared/amdgpu/home/zeyu_zhang_ck1/WAM-Policy/scripts/slurm/video_ffm_admin_diagnostic.sbatch
#
# Required:
#   TEACHER_CKPT=/abs/path/to/lingbot/transformer_or_checkpoint
#   DEBUG_MANIFEST=/abs/path/to/video_latents_debug.jsonl
#   FULL_MANIFEST=/abs/path/to/video_latents_full.jsonl
#
# Optional:
#   REPO_ROOT=/shared/amdgpu/home/zeyu_zhang_ck1/WAM-Policy
#   CONFIG=configs/experiments/v2/anyflow_video_ffm.yaml
#   PYTHON_BIN=python3
#   ENV_SETUP=/path/to/conda_or_module_setup.sh
#   TRAIN_MAX_STEPS=5000
#   TRAIN_SAVE_EVERY=500
#   TRAIN_LOG_EVERY=10
#   TRAIN_BATCH_SIZE=1
#   WANDB_ENABLED=auto
#   WANDB_PROJECT=WAM-Policy
#   WANDB_ENTITY=
#   WANDB_RUN_NAME=
#   WANDB_RUN_ID=
#   WANDB_MODE=online
#   WANDB_DIR=wandb
#   WANDB_UPLOAD_LOGS=true

DEFAULT_REPO_ROOT="/shared/amdgpu/home/zeyu_zhang_ck1/WAM-Policy"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
CONFIG="${CONFIG:-configs/experiments/v2/anyflow_video_ffm.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-5000}"
TRAIN_SAVE_EVERY="${TRAIN_SAVE_EVERY:-500}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
JOB_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
WANDB_ENABLED="${WANDB_ENABLED:-auto}"
WANDB_PROJECT="${WANDB_PROJECT:-WAM-Policy}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-video-ffm-${JOB_ID}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-wam-video-ffm-${JOB_ID}}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DIR="${WANDB_DIR:-wandb}"
WANDB_UPLOAD_LOGS="${WANDB_UPLOAD_LOGS:-true}"

cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs/admin_diagnostic/$JOB_ID"
RUN_DIR="$REPO_ROOT/experiments/runs/v2/video_ffm_admin_debug/$JOB_ID"
CKPT_DIR="$REPO_ROOT/experiments/checkpoints/v2/video_ffm_admin_debug/$JOB_ID"
STATUS_FILE="$LOG_DIR/status.txt"
SLURM_OUTPUT_DIR="${SLURM_SUBMIT_DIR:-$REPO_ROOT}"
SLURM_OUT_FILE="$SLURM_OUTPUT_DIR/slurm-wam-video-ffm-admin-${JOB_ID}.out"
SLURM_ERR_FILE="$SLURM_OUTPUT_DIR/slurm-wam-video-ffm-admin-${JOB_ID}.err"

mkdir -p "$LOG_DIR" "$RUN_DIR" "$CKPT_DIR"

mark_failed() {
  local stage="$1"
  local exit_code="$2"
  echo "FAILED:${stage}:${exit_code}" | tee -a "$STATUS_FILE"
  echo "First failing stage: ${stage}" | tee -a "$STATUS_FILE"
  echo "See log: ${LOG_DIR}/${stage}.log" | tee -a "$STATUS_FILE"
  exit "$exit_code"
}

run_stage() {
  local stage="$1"
  shift
  local log_file="$LOG_DIR/${stage}.log"

  echo "START:${stage}:$(date -Is)" | tee -a "$STATUS_FILE"
  {
    echo "stage: ${stage}"
    echo "started_at: $(date -Is)"
    echo "workdir: $(pwd)"
    echo "command: $*"
    echo
    set +e
    "$@"
    local command_status=$?
    set -e
    echo
    echo "finished_at: $(date -Is)"
    echo "exit_code: ${command_status}"
    exit "$command_status"
  } >"$log_file" 2>&1 || mark_failed "$stage" "$?"

  echo "PASSED:${stage}:$(date -Is)" | tee -a "$STATUS_FILE"
}

run_shell_stage() {
  local stage="$1"
  local script="$2"
  run_stage "$stage" bash -lc "$script"
}

echo "STARTED:${JOB_ID}:$(date -Is)" | tee "$STATUS_FILE"
echo "repo_root: $REPO_ROOT" | tee -a "$STATUS_FILE"
echo "config: $CONFIG" | tee -a "$STATUS_FILE"
echo "log_dir: $LOG_DIR" | tee -a "$STATUS_FILE"
echo "run_dir: $RUN_DIR" | tee -a "$STATUS_FILE"
echo "checkpoint_dir: $CKPT_DIR" | tee -a "$STATUS_FILE"

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_SETUP"
fi

DEBUG_MANIFEST="${DEBUG_MANIFEST:-${DIAG_MANIFEST:-${MANIFEST:-}}}"
FULL_MANIFEST="${FULL_MANIFEST:-${TRAIN_MANIFEST:-${MANIFEST:-}}}"

if [[ -z "${TEACHER_CKPT:-}" ]]; then
  echo "TEACHER_CKPT is required." >"$LOG_DIR/00_env.log"
  mark_failed "00_env" 2
fi
if [[ -z "${DEBUG_MANIFEST:-}" ]]; then
  echo "DEBUG_MANIFEST is required." >"$LOG_DIR/00_env.log"
  mark_failed "00_env" 2
fi
if [[ -z "${FULL_MANIFEST:-}" ]]; then
  echo "FULL_MANIFEST is required." >"$LOG_DIR/00_env.log"
  mark_failed "00_env" 2
fi

DIAG_OVERRIDES=(
  --set runtime.device=rocm
  --set runtime.lingbot_attn_mode=torch
  --set "teacher.checkpoint_path=${TEACHER_CKPT}"
  --set "data.manifest=${DEBUG_MANIFEST}"
  --set "paths.output_dir=${RUN_DIR}"
  --set "paths.checkpoint_dir=${CKPT_DIR}"
)

TRAIN_BASE_OVERRIDES=(
  --set runtime.device=rocm
  --set runtime.lingbot_attn_mode=torch
  --set "teacher.checkpoint_path=${TEACHER_CKPT}"
  --set "data.manifest=${FULL_MANIFEST}"
  --set "paths.output_dir=${RUN_DIR}"
  --set "paths.checkpoint_dir=${CKPT_DIR}"
)

WANDB_OVERRIDES=()
WANDB_EFFECTIVE_ENABLED=false
case "$WANDB_ENABLED" in
  1|true|True|TRUE|yes|Yes|YES|on|On|ON)
    WANDB_EFFECTIVE_ENABLED=true
    WANDB_OVERRIDES=(
      --set logging.wandb.enabled=true
      --set "logging.wandb.project=${WANDB_PROJECT}"
      --set "logging.wandb.name=${WANDB_RUN_NAME}"
      --set "logging.wandb.id=${WANDB_RUN_ID}"
      --set "logging.wandb.resume=${WANDB_RESUME}"
      --set "logging.wandb.mode=${WANDB_MODE}"
      --set "logging.wandb.dir=${WANDB_DIR}"
    )
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
      WANDB_OVERRIDES+=(--set "logging.wandb.entity=${WANDB_ENTITY}")
    fi
    ;;
  auto|Auto|AUTO)
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
      WANDB_EFFECTIVE_ENABLED=true
      WANDB_OVERRIDES=(
        --set logging.wandb.enabled=true
        --set "logging.wandb.project=${WANDB_PROJECT}"
        --set "logging.wandb.name=${WANDB_RUN_NAME}"
        --set "logging.wandb.id=${WANDB_RUN_ID}"
        --set "logging.wandb.resume=${WANDB_RESUME}"
        --set "logging.wandb.mode=${WANDB_MODE}"
        --set "logging.wandb.dir=${WANDB_DIR}"
      )
      if [[ -n "${WANDB_ENTITY:-}" ]]; then
        WANDB_OVERRIDES+=(--set "logging.wandb.entity=${WANDB_ENTITY}")
      fi
    fi
    ;;
  0|false|False|FALSE|no|No|NO|off|Off|OFF)
    WANDB_OVERRIDES=(--set logging.wandb.enabled=false)
    ;;
esac

TRAIN_OVERRIDES=(
  "${TRAIN_BASE_OVERRIDES[@]}"
  "${WANDB_OVERRIDES[@]}"
  --set "data.batch_size=${TRAIN_BATCH_SIZE}"
  --set data.num_workers=0
  --set "train.max_steps=${TRAIN_MAX_STEPS}"
  --set "train.save_every=${TRAIN_SAVE_EVERY}"
  --set "train.log_every=${TRAIN_LOG_EVERY}"
)

upload_wandb_logs() {
  local exit_code="$1"
  if [[ "$WANDB_EFFECTIVE_ENABLED" != "true" ]]; then
    return 0
  fi
  case "$WANDB_UPLOAD_LOGS" in
    0|false|False|FALSE|no|No|NO|off|Off|OFF)
      return 0
      ;;
  esac

  set +e
  local upload_args=(
    scripts/tools/upload_admin_logs_to_wandb.py
    --job-id "$JOB_ID" \
    --log-dir "$LOG_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    --project "$WANDB_PROJECT" \
    --name "$WANDB_RUN_NAME" \
    --run-id "$WANDB_RUN_ID" \
    --resume "$WANDB_RESUME" \
    --mode "$WANDB_MODE" \
    --dir "$WANDB_DIR" \
    --exit-code "$exit_code" \
    --slurm-out "$SLURM_OUT_FILE" \
    --slurm-err "$SLURM_ERR_FILE"
  )
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    upload_args+=(--entity "$WANDB_ENTITY")
  fi
  "$PYTHON_BIN" "${upload_args[@]}"
  local upload_status=$?
  set -e
  if [[ "$upload_status" -ne 0 ]]; then
    echo "WARN: failed to upload admin logs to wandb; upload_exit_code=${upload_status}" | tee -a "$STATUS_FILE"
  fi
  return 0
}

on_exit() {
  local exit_code="$1"
  trap - EXIT
  upload_wandb_logs "$exit_code" || true
  exit "$exit_code"
}

trap 'on_exit $?' EXIT

run_shell_stage "00_env" "
set -euo pipefail
echo hostname: \$(hostname)
echo date: \$(date -Is)
echo slurm_job_id: ${SLURM_JOB_ID:-}
echo slurm_nodelist: ${SLURM_NODELIST:-}
echo repo_root: ${REPO_ROOT}
echo config: ${CONFIG}
echo teacher_ckpt: ${TEACHER_CKPT}
echo debug_manifest: ${DEBUG_MANIFEST}
echo full_manifest: ${FULL_MANIFEST}
echo train_max_steps: ${TRAIN_MAX_STEPS}
echo train_save_every: ${TRAIN_SAVE_EVERY}
echo train_log_every: ${TRAIN_LOG_EVERY}
echo train_batch_size: ${TRAIN_BATCH_SIZE}
echo wandb_enabled: ${WANDB_ENABLED}
echo wandb_effective_enabled: ${WANDB_EFFECTIVE_ENABLED}
echo wandb_project: ${WANDB_PROJECT}
echo wandb_entity: ${WANDB_ENTITY:-}
echo wandb_run_name: ${WANDB_RUN_NAME}
echo wandb_run_id: ${WANDB_RUN_ID}
echo wandb_resume: ${WANDB_RESUME}
echo wandb_mode: ${WANDB_MODE}
echo wandb_dir: ${WANDB_DIR}
echo wandb_upload_logs: ${WANDB_UPLOAD_LOGS}
echo slurm_out_file: ${SLURM_OUT_FILE}
echo slurm_err_file: ${SLURM_ERR_FILE}
echo python_bin: ${PYTHON_BIN}
echo
test -d ${REPO_ROOT@Q}
test -f ${CONFIG@Q}
test -e ${TEACHER_CKPT@Q}
test -f ${DEBUG_MANIFEST@Q}
test -f ${FULL_MANIFEST@Q}
command -v ${PYTHON_BIN@Q}
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi
else
  echo 'rocm-smi not found'
fi
echo
env | sort | sed -E 's/^(.*(TOKEN|PASSWORD|SECRET|API_KEY|ACCESS_KEY).*=).*/\1<redacted>/'
"

run_shell_stage "01_static" "
set -euo pipefail
${PYTHON_BIN@Q} scripts/train.py --config ${CONFIG@Q} --dry-run static
${PYTHON_BIN@Q} scripts/tools/probe_video_ffm_runtime.py probe-static-source --config ${CONFIG@Q}
${PYTHON_BIN@Q} -m py_compile \
  scripts/train.py \
  scripts/tools/convert_lingbot_robotwin_latents.py \
  scripts/tools/probe_video_ffm_runtime.py \
  scripts/tools/upload_admin_logs_to_wandb.py \
  src/wam_policy/adapters/lingbot_va.py \
  src/wam_policy/utils/logging.py \
  src/wam_policy/utils/runtime.py
"

run_shell_stage "02_import" "
set -euo pipefail
${PYTHON_BIN@Q} - <<'PY'
import torch
print('torch.__version__:', torch.__version__)
print('torch.version.hip:', getattr(torch.version, 'hip', None))
print('torch.version.cuda:', getattr(torch.version, 'cuda', None))
print('torch.cuda.is_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('torch.cuda.device_count:', torch.cuda.device_count())
    print('torch.cuda.current_device:', torch.cuda.current_device())
    print('torch.cuda.get_device_name:', torch.cuda.get_device_name(0))
PY
${PYTHON_BIN@Q} scripts/tools/probe_video_ffm_runtime.py probe-import --config ${CONFIG@Q} \
  --set runtime.device=rocm \
  --set runtime.lingbot_attn_mode=torch \
  --set teacher.checkpoint_path=${TEACHER_CKPT@Q} \
  --set data.manifest=${DEBUG_MANIFEST@Q} \
  --set paths.output_dir=${RUN_DIR@Q} \
  --set paths.checkpoint_dir=${CKPT_DIR@Q}
"

run_stage "03_load_model" \
  "$PYTHON_BIN" scripts/tools/probe_video_ffm_runtime.py \
  probe-load-model \
  --config "$CONFIG" \
  "${DIAG_OVERRIDES[@]}"

run_stage "04_manifest" \
  "$PYTHON_BIN" scripts/tools/probe_video_ffm_runtime.py \
  probe-manifest-one-sample \
  --config "$CONFIG" \
  "${DIAG_OVERRIDES[@]}"

run_stage "05_video_forward" \
  "$PYTHON_BIN" scripts/tools/probe_video_ffm_runtime.py \
  probe-video-forward \
  --config "$CONFIG" \
  "${DIAG_OVERRIDES[@]}"

run_stage "06_train_long" \
  "$PYTHON_BIN" scripts/train.py \
  --config "$CONFIG" \
  "${TRAIN_OVERRIDES[@]}"

echo "SUCCESS:${JOB_ID}:$(date -Is)" | tee -a "$STATUS_FILE"
echo "Admin diagnostic completed. Logs: $LOG_DIR" | tee -a "$STATUS_FILE"
echo "Checkpoints: $CKPT_DIR" | tee -a "$STATUS_FILE"
