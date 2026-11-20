#!/usr/bin/env bash
# Provision chrony on the X5 as a client of the laptop (the rig's time source).
# Idempotent. Run from the laptop; key login as root@X5 (set up by x5_install.sh).
#  - installs chrony when missing, through an HTTP proxy on the laptop if the X5 has no
#    internet (PROXY=http://<laptop address>:3128 after  python3 -m proxy --hostname 0.0.0.0 --port 3128);
#    apt refuses Release files "from the future" while the X5 clock is in 2000,
#    so the laptop's time is pushed first
#  - installs scripts/x5/chrony_x5.conf (the laptop's address and the subnet filled in) as
#    /etc/chrony/chrony.conf (original -> chrony.conf.orig), restarts chronyd, prints sources + tracking
# The laptop side is scripts/x5/laptop_chrony_install.sh (needs sudo).
#   scripts/x5/x5_chrony_setup.sh        # X5=root@$X5_IP; the laptop is DEMO_LAPTOP_IP (config/deploy.env)
set -euo pipefail
# shellcheck source=../load_config.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../load_config.sh"     # X5_IP, DEMO_LAPTOP_IP (config/deploy.env)
X5="${X5:-${X5_IP:+root@$X5_IP}}"
[[ -n "$X5" ]] || { echo "set X5_IP (config/deploy.env) or X5=root@<address of the RDK X5>" >&2; exit 2; }
[[ -n "$DEMO_LAPTOP_IP" ]] || { echo "set DEMO_LAPTOP_IP (config/deploy.env)" >&2; exit 2; }
PROXY="${PROXY:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8 "$X5")
if ! "${SSH[@]}" 'command -v chronyd >/dev/null'; then
  echo "installing chrony on $X5${PROXY:+ via $PROXY} (replaces the image's ntpd)"
  proxy_opt="${PROXY:+-o Acquire::http::Proxy=$PROXY}"
  "${SSH[@]}" "date -s @$(date +%s) >/dev/null; apt-get $proxy_opt update 2>&1 | tail -2; \
    DEBIAN_FRONTEND=noninteractive apt-get -y $proxy_opt install chrony 2>&1 | grep -E 'Setting up|Removing|^E:' || true"
fi
"${SSH[@]}" '[ -f /etc/chrony/chrony.conf.orig ] || cp /etc/chrony/chrony.conf /etc/chrony/chrony.conf.orig'
conf="$(mktemp)"; trap 'rm -f "$conf"' EXIT
sed -e "s|@LAPTOP_IP@|$DEMO_LAPTOP_IP|g" -e "s|@ROBOT_SUBNET@|${DEMO_LAPTOP_IP%.*}.0/24|g" "$HERE/chrony_x5.conf" > "$conf"
scp -q -o BatchMode=yes "$conf" "$X5:/etc/chrony/chrony.conf"
"${SSH[@]}" 'systemctl enable chrony >/dev/null 2>&1 || true; systemctl restart chrony; sleep 5; \
  echo "== chronyc sources (X5) =="; chronyc sources; echo "== chronyc tracking (X5) =="; chronyc tracking'
echo "If sources shows the laptop with '?' the laptop is not serving yet: run scripts/x5/laptop_chrony_install.sh there (sudo)."
