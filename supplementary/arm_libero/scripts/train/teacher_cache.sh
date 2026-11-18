#!/usr/bin/env bash
# The teacher cache shared by every student (about 5,000 teacher contexts; it also supplies every
# student's held-out set). One node, method.world_size (8) GPUs. Writes to machine.offline_cache.
#   SITE=site.json PROMPTS=/data/arm/prompts.pt bash scripts/train/teacher_cache.sh
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${SITE:?SITE required}" "${PROMPTS:?PROMPTS required}"
PYTHON=${PYTHON:-python}
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
config=$root/configs/students/opsd.json
cache=$("$PYTHON" -c "import sys; from arm_opsd.config import ArmConfig; print(ArmConfig.load(sys.argv[1], site=sys.argv[2]).machine.offline_cache)" "$config" "$SITE")
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NPROC:-8}" --max_restarts=0 \
    -m arm_opsd.cli build-teacher-cache --config "$config" --site "$SITE" --prompts "$PROMPTS" \
    --output "$cache" --contexts "${CONTEXTS:-5000}"
