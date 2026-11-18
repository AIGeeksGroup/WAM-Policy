#!/usr/bin/env bash
# Stage 1.1 ablation (RGB + proprioception): whole-episode SFT, 5000 steps, single node.
# The paper run used 8 GPUs (one episode per GPU = global batch 8); CUDA and ROCm both work.
#
# One-time inputs (see configs/ablations/proprioception/sft5000.json):
#   export QLM_BENCH_SNAPSHOT=/path/to/qlm-bench SFT_CACHE=/path/to/frozen-cache
#   export WAM_PROPRIO_RESOURCES=/path/to/resources WAM_PROPRIO_OUTPUT=/path/to/runs
#   python -m wam_policy.data download --snapshot "$QLM_BENCH_SNAPSHOT"      # Stage 1.1 data (shared)
#   python -m wam_policy.data fit   --snapshot "$QLM_BENCH_SNAPSHOT" --cache "$SFT_CACHE"
#   python -m wam_policy.data cache --snapshot "$QLM_BENCH_SNAPSHOT" --cache "$SFT_CACHE" --base-model <lingbot-va-base>
#   python -m wam_policy.models.native9 --source <lingbot-va-base>/transformer --output <native9> --seed 0
#   python -m wam_policy.ablations.proprioception convert --native9-transformer <native9> \
#       --output "$WAM_PROPRIO_RESOURCES/transformer-proprioception" [--symlink]
#   python -m wam_policy.ablations.proprioception data --config configs/ablations/proprioception/sft5000.json
#
# Usage: scripts/ablations/proprioception/train.sh [extra train args, e.g. --resume <checkpoint> --run-name NAME]
# Env: NPROC (default 8), CONFIG (default configs/ablations/proprioception/sft5000.json), PYTHON (default python).
# W&B is off unless WANDB_MODE=online|offline (entity/project from WANDB_ENTITY / WANDB_PROJECT).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
NPROC=${NPROC:-8}
CONFIG=${CONFIG:-$REPO/configs/ablations/proprioception/sft5000.json}
PYTHON=${PYTHON:-python}
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" \
    -m wam_policy.ablations.proprioception train --config "$CONFIG" "$@"
