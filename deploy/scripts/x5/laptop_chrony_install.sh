#!/usr/bin/env bash
# ONE-TIME laptop setup (needs sudo):
#   scripts/x5/laptop_chrony_install.sh
# Installs chrony (replaces systemd-timesyncd as the NTP client; the laptop keeps
# syncing to the Ubuntu pools when online), adds the Go2-rig drop-in
# (chrony_laptop_go2.conf -> /etc/chrony/conf.d/go2_rig.conf: serve the switch
# subnet, keep serving offline), opens UDP 123 for the subnet if ufw is active,
# restarts chronyd and shows both hosts' view.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../load_config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../load_config.sh"     # X5_IP, DEMO_LAPTOP_IP (config/deploy.env)
X5="${X5:-${X5_IP:+root@$X5_IP}}"
[[ -n "$X5" ]] || { echo "set X5_IP (config/deploy.env) or X5=root@<address of the RDK X5>" >&2; exit 2; }
[[ -n "$DEMO_LAPTOP_IP" ]] || { echo "set DEMO_LAPTOP_IP (config/deploy.env)" >&2; exit 2; }
SUBNET="${DEMO_LAPTOP_IP%.*}.0/24"
if ! command -v chronyd >/dev/null; then
  sudo apt-get update -qq || true
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y chrony
fi
sed "s|@ROBOT_SUBNET@|$SUBNET|g" "$HERE/chrony_laptop_go2.conf" | sudo tee /etc/chrony/conf.d/go2_rig.conf >/dev/null
if systemctl is-active --quiet ufw; then
  sudo ufw allow from "$SUBNET" to any port 123 proto udp comment 'Go2 rig NTP (chrony)' || true
fi
sudo systemctl enable chrony >/dev/null 2>&1 || true
sudo systemctl restart chrony
sleep 3
echo "== laptop: chronyc tracking =="; chronyc tracking
echo "== laptop: chronyc sources =="; chronyc sources
echo "== X5 view (may need ~10 s to lock) =="
sleep 6
# force the X5 to step onto the laptop clock now (its camera script restarts the camera node after
# any step over CLOCK_STEP_THRESHOLD_S = 0.05 s; a smaller one passes librealsense's 100 ms gate and
# fades out of the stamps within ~10 s)
ssh -o BatchMode=yes -o ConnectTimeout=5 "$X5" 'chronyc makestep >/dev/null 2>&1; sleep 2' || true
ssh -o BatchMode=yes -o ConnectTimeout=5 "$X5" 'chronyc sources; chronyc tracking | grep -E "Reference ID|Stratum|System time|Last offset|RMS offset|Leap"' || echo "X5 not reachable over ssh"
