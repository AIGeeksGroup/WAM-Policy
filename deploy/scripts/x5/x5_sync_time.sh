#!/usr/bin/env bash
# Push the laptop's wall clock to the X5 (RTC stuck in 2000, no NTP upstream on
# the robot subnet). Normally the X5 runs chrony as a client of the laptop
# (scripts/x5/x5_chrony_setup.sh + laptop_chrony_install.sh) - when that is
# active this script only SHOWS the sync state and does nothing: a manual
# `date -s` has 1 s granularity and chrony would slew the error away at
# 1000 ppm, leaving camera stamps wrong for minutes. Use --force to push anyway
# (e.g. chrony present but the laptop is not serving yet).
set -euo pipefail
# shellcheck source=../load_config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../load_config.sh"     # X5_IP, DEMO_LAPTOP_IP (config/deploy.env)
X5="${X5:-${X5_IP:+root@$X5_IP}}"
[[ -n "$X5" ]] || { echo "set X5_IP (config/deploy.env) or X5=root@<address of the RDK X5>" >&2; exit 2; }
if [ "${1:-}" != "--force" ] && ssh -o BatchMode=yes -o ConnectTimeout=5 "$X5" 'systemctl is-active --quiet chrony'; then
  echo "chrony is active on $X5 - not pushing time. State:"
  ssh -o BatchMode=yes "$X5" 'chronyc tracking | grep -E "Reference ID|Stratum|System time|Last offset|Leap"; date'
  echo "(use --force to push the laptop clock anyway)"
  exit 0
fi
ssh -o BatchMode=yes "$X5" "date -s @$(date +%s) && hwclock -w 2>/dev/null; date"
