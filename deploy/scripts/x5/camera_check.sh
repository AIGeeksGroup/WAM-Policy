#!/usr/bin/env bash
# Laptop-side acceptance check for the X5 camera host:
#   1. camera topics visible over the wired link, publisher QoS, colour/depth rates
#   2. /lowstate receive timing WHILE the camera streams (lowstate_probe.py)
# Read-only: no publisher, no controller. Results -> data/logs/camera_checks_<date>/<label>_*.txt
#   scripts/x5/camera_check.sh camera_and_lowstate [probe_seconds=120]
set -o pipefail   # no -u: ROS setup.bash reads unset variables
LABEL="${1:-camera_and_lowstate}"; DUR="${2:-120}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT="$REPO/data/logs/camera_checks_$(date +%F)"; mkdir -p "$OUT"
# shellcheck disable=SC1091
source "$HERE/laptop_camera_env.bash"
NS=/camera/d435i
{
  echo "# $(date '+%F %T') label=$LABEL"
  echo "== camera topics =="; timeout 10 ros2 topic list 2>/dev/null | grep '^/camera' || echo "NO camera topics visible"
  echo "== publisher QoS: color =="; timeout 10 ros2 topic info -v $NS/color/image_raw 2>/dev/null | grep -E "Node name|Reliability|Durability|Publisher count" | head -5
  echo "== camera_info once =="; timeout 10 ros2 topic echo --once --qos-reliability best_effort $NS/color/camera_info 2>/dev/null | grep -E "width|height|frame_id" | head -3
} | tee "$OUT/${LABEL}_camera.txt"
echo "== CONCURRENT ${DUR}s: colour+depth receive probe  ||  /lowstate timing probe =="
python3 "$HERE/camera_sub_probe.py" --duration "$DUR" --label "$LABEL" --out "$OUT/${LABEL}_camera_rx.txt" &
CAM_PID=$!
python3 "$HERE/lowstate_probe.py" --duration "$DUR" --label "$LABEL" --out "$OUT/${LABEL}_lowstate.txt"
wait "$CAM_PID"
echo "saved: $OUT/${LABEL}_{camera,camera_rx,lowstate}.txt"
