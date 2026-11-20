#!/usr/bin/env bash
# Download the Unitree Go2 model of MuJoCo Menagerie (BSD-3-Clause) at a fixed commit into
# third_party/mujoco_menagerie/ for the MuJoCo closed loop (sim/):
#
#   scripts/fetch_mujoco_menagerie.sh
set -euo pipefail
DEPLOY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
URL="https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT="8161bba264d7fa7c99ca301e91e7fb44737676ad"
OUT="$DEPLOY/third_party/mujoco_menagerie"

if [[ -f "$OUT/unitree_go2/scene.xml" && "$(git -C "$OUT" rev-parse HEAD 2>/dev/null)" == "$COMMIT" ]]; then
  echo "fetch_mujoco_menagerie: $OUT is already at $COMMIT"
  exit 0
fi
[[ ! -e "$OUT" ]] || { echo "fetch_mujoco_menagerie: $OUT exists but is not at $COMMIT; remove it first" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
git clone -q --filter=blob:none --no-checkout "$URL" "$OUT"
git -C "$OUT" sparse-checkout set unitree_go2
git -C "$OUT" checkout -q "$COMMIT"
echo "fetch_mujoco_menagerie: $OUT/unitree_go2 at $COMMIT"
