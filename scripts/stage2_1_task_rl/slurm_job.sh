#!/usr/bin/env bash
# Thin Slurm wrapper around run_task_rl.sh (one node, all of its GPUs). Site specifics (partition, account, modules,
# environment activation) go into your own sbatch options / a site file; this script only exports the run settings.
#
#   sbatch --nodes 1 --gpus-per-node 8 --cpus-per-task 64 --time 12:00:00 \
#     --export ALL,TASK=push_box,RUN=<run dir>,QLM_BENCH=...,QLM_PYTHON=...,TEACHER=...,BASE_MODEL=...,PROMPT_ROOT=... \
#     scripts/stage2_1_task_rl/slurm_job.sh
#
# The round driver stops cleanly at a round boundary before the time limit (DEADLINE = job end - 10 min); submit the
# same job again (or with --dependency=afterany:<id>) to continue.
#SBATCH --job-name=task-rl
#SBATCH --output=%x-%j.log
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -e "$SLURM_SUBMIT_DIR/scripts/stage2_1_task_rl/run_task_rl.sh" ] && HERE="$SLURM_SUBMIT_DIR/scripts/stage2_1_task_rl"
export GPUS="${GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${DEADLINE:-}" ]; then
  end=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null || true)
  if [ -n "$end" ] && [ "$end" != "N/A" ]; then export DEADLINE=$(( $(date -d "$end" +%s) - 600 )); fi
fi
exec bash "$HERE/run_task_rl.sh"
