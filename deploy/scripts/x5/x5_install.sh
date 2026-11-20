#!/usr/bin/env bash
# Deploy / update the X5 camera host from the laptop (key login as root@X5):
# copies go2_camera.sh + the systemd unit, installs camera.conf.example as the X5's
# camera.conf when it has none (an existing one is kept), enables the service at boot,
# starts it, pushes the laptop's time (only while chrony is not running there), prints
# status. Re-run after editing either file. chrony itself: x5_chrony_setup.sh.
#   scripts/x5/x5_install.sh            # X5=root@$X5_IP (config/deploy.env)
set -euo pipefail
# shellcheck source=../load_config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../load_config.sh"     # X5_IP, DEMO_LAPTOP_IP (config/deploy.env)
X5="${X5:-${X5_IP:+root@$X5_IP}}"
[[ -n "$X5" ]] || { echo "set X5_IP (config/deploy.env) or X5=root@<address of the RDK X5>" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ssh -o BatchMode=yes "$X5" 'mkdir -p /root/go2_camera_deploy/{bin,config,logs}'
scp -q -o BatchMode=yes "$HERE/go2_camera.sh" "$X5:/root/go2_camera_deploy/bin/go2_camera.sh"
scp -q -o BatchMode=yes "$HERE/go2-camera.service" "$X5:/etc/systemd/system/go2-camera.service"
ssh -o BatchMode=yes "$X5" 'test -f /root/go2_camera_deploy/config/camera.conf' \
  || scp -q -o BatchMode=yes "$HERE/camera.conf.example" "$X5:/root/go2_camera_deploy/config/camera.conf"
# time: only push the laptop clock when chrony is NOT running on the X5 (x5_chrony_setup.sh);
# a manual 1-s-granularity step under a synced chrony would be slewed away for minutes
ssh -o BatchMode=yes "$X5" "chmod +x /root/go2_camera_deploy/bin/go2_camera.sh && { systemctl is-active --quiet chrony || date -s @$(date +%s) >/dev/null; } && systemctl daemon-reload && systemctl enable go2-camera.service >/dev/null 2>&1 && systemctl restart go2-camera.service && sleep 3 && /root/go2_camera_deploy/bin/go2_camera.sh status"
