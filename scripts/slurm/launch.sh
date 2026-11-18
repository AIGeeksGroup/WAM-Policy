#!/usr/bin/env bash
# Optional generic Slurm launcher: run one single-node command (e.g. a stage script) as a batch job.
#
#   scripts/slurm/launch.sh --name teacher-sft --gpus 8 --cpus 64 --mem 512G --time 36:00:00 \
#       [--partition P] [--account A] [--dry-run] -- bash scripts/stage1_1_transfer/train.sh configs/...
#
# The job sources configs/site.env (if present) and runs the command from the repository root.
# Nothing here is required: every stage also runs directly on a single node.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
name=wam-policy; gpus=8; cpus=64; mem=512G; time=24:00:00; partition=; account=; dry=0
while [ $# -gt 0 ]; do
  case $1 in
    --name) name=$2; shift 2;; --gpus) gpus=$2; shift 2;; --cpus) cpus=$2; shift 2;;
    --mem) mem=$2; shift 2;; --time) time=$2; shift 2;; --partition) partition=$2; shift 2;;
    --account) account=$2; shift 2;; --dry-run) dry=1; shift;; --) shift; break;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[ $# -gt 0 ] || { echo "usage: $0 [options] -- <command ...>" >&2; exit 2; }
mkdir -p "$ROOT/outputs/slurm"
script=$(mktemp "$ROOT/outputs/slurm/${name}.XXXX.sbatch")
{
  echo "#!/usr/bin/env bash"
  echo "#SBATCH --job-name=$name"
  echo "#SBATCH --nodes=1"
  echo "#SBATCH --ntasks=1"
  echo "#SBATCH --gpus=$gpus"
  echo "#SBATCH --cpus-per-task=$cpus"
  echo "#SBATCH --mem=$mem"
  echo "#SBATCH --time=$time"
  echo "#SBATCH --output=$ROOT/outputs/slurm/%x-%j.log"
  [ -n "$partition" ] && echo "#SBATCH --partition=$partition"
  [ -n "$account" ] && echo "#SBATCH --account=$account"
  echo "set -euo pipefail"
  echo "cd $(printf %q "$ROOT")"
  echo "[ -f configs/site.env ] && source configs/site.env"
  printf '%q ' "$@"; echo
} > "$script"
if [ $dry = 1 ]; then cat "$script"; else sbatch "$script"; fi
