#!/usr/bin/env bash
# Optional Slurm wrapper of scripts/train/student.sh (or teacher_cache.sh) on one 8-GPU node.
# Submit from supplementary/arm_libero (Slurm runs a copy of this file, so the project root is the submit
# directory), or pass ARM_LIBERO_ROOT=<path to supplementary/arm_libero> in --export.
#   cd supplementary/arm_libero
#   sbatch --partition=<p> --account=<a> --export=ALL,SITE=site.json,PROMPTS=prompts.pt,OUTPUT=runs/opsd \
#     scripts/slurm/student.sbatch configs/students/opsd.json
#   sbatch ... --export=ALL,SITE=site.json,PROMPTS=prompts.pt,STEP=teacher_cache scripts/slurm/student.sbatch
# ENV_SETUP may name a file to source first (module loads, the Python environment).
#SBATCH --job-name=arm-opsd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
[[ -n "${ENV_SETUP:-}" ]] && source "$ENV_SETUP"
root=${ARM_LIBERO_ROOT:-$SLURM_SUBMIT_DIR}
[[ -f "$root/scripts/train/student.sh" ]] || { echo "No scripts/train/student.sh under $root: submit from supplementary/arm_libero or set ARM_LIBERO_ROOT" >&2; exit 2; }
if [[ "${STEP:-student}" == teacher_cache ]]; then
    bash "$root/scripts/train/teacher_cache.sh"
else
    bash "$root/scripts/train/student.sh" "$@"
fi
