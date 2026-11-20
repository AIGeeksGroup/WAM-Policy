#!/usr/bin/env bash
# shellcheck shell=bash
# scripts/demo/common.sh - shared robot bring-up for scripts/demo/wam_task.sh (sourced, never run).
#
# What the functions do, in the order wam_task.sh calls them:
#   demo_source_env          ROS 2 Jazzy + the unitree message and deploy workspaces + the deploy venv;
#                            DDS bound to the wired interface (FastDDS whitelist of DEMO_LAPTOP_IP, or
#                            CycloneDDS on DEMO_NIC). If the robot also learns a WiFi address it may
#                            stream /lowstate over WiFi, with 20-250 ms gaps.
#   demo_perf_env            torch thread cap (DEMO_TORCH_THREADS, default 4) and optional CPU pinning
#                            (DEMO_CPUSET): a 100 Hz controller next to many OpenMP threads can stall.
#   demo_preflight_network   DEMO_NIC carries DEMO_LAPTOP_IP/24 (set with sudo if not), the robot
#                            answers ping, and the route to it uses DEMO_NIC.
#   demo_preflight_processes no stray controller / launch / viewer from an earlier run.
#   demo_release_sport_mode  hand the motors over from Unitree's sport mode (must exit 0).
#   demo_launch_controller   ros2 launch wam_go2_deploy wam_go2.launch.py in its own process group.
#   demo_wait_mode           wait until the controller publishes rambo/mode.
#   demo_preflight_datapath  the 500 Hz /lowstate stream arrives on DEMO_NIC and nowhere else.
#   demo_check_controller    the running controller is bound to DEMO_NIC.
#   demo_view_start          the operator view (scripts/x5/ego_d435i_view.py): the Go2 front camera and
#                            the D435i in two windows, each recorded to its own video. It never fails
#                            the run: every problem is one warning.
#   demo_confirm             Enter on the terminal before the robot moves.
#   demo_cleanup             DAMP first, then stop the controller (it sends damping frames on exit),
#                            then the view.
#
# Output: the operator's terminal gets only the stage headers (between dashed lines) and a few short
# lines, on file descriptor 3 when the caller opened it (run_wam.sh points it at the terminal and
# everything else at the run's log); demo_say lines go to stdout (the log).
#
# Env: DEMO_NIC (required), DEMO_LAPTOP_IP, DEMO_ROBOT_IP, DEPLOY_WS, UNITREE_MSGS_WS, DEPLOY_VENV
# (deploy/config/deploy.env), RAMBO_CONTROLLER (controller checkpoint; empty = the bundled path),
# DEMO_LAUNCH_EXTRA (extra controller launch arguments), DEMO_LOG_DIR (default data/logs),
# DEMO_CONFIRM=0 (no Enter pause: only with someone watching the robot), RMW_IMPLEMENTATION /
# CYCLONEDDS_URI, FASTRTPS_DEFAULT_PROFILES_FILE (own FastDDS profile instead of the generated one),
# DEMO_DATAPATH_MIN_MBPS (0.4), DEMO_TORCH_THREADS (4; 0 = torch default), DEMO_CPUSET, and for the
# view: DEMO_VIEW (1; 0 = off), DEMO_VIEW_DIR (output directory), DEMO_VIEW_RES (360|720),
# DEMO_VIEW_CALIB (time-sync calibration JSON), DEMO_VIEW_CPUSET, DEMO_VIEW_EXTRA,
# DEMO_VIEW_EGO_SOURCE (frontvideo | topic), DEMO_VIEW_BIN / DEMO_SYSFS_NET (tests).

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_REPO="$(cd "$DEMO_DIR/../.." && pwd)"
# shellcheck source=../load_config.sh
source "$DEMO_REPO/scripts/load_config.sh"
DEMO_TORCH_THREADS="${DEMO_TORCH_THREADS:-4}"
DEMO_CPUSET="${DEMO_CPUSET-}"
DEMO_TASKSET=()
DEMO_LOG_DIR="${DEMO_LOG_DIR:-$DEMO_REPO/data/logs}"
DEMO_LAUNCH_PID=""
DEMO_LAUNCH_PGID=""
DEMO_LOG=""
DEMO_VIEW="${DEMO_VIEW:-1}"
DEMO_VIEW_RES="${DEMO_VIEW_RES:-360}"
DEMO_VIEW_DIR="${DEMO_VIEW_DIR:-}"
DEMO_VIEW_CALIB="${DEMO_VIEW_CALIB:-}"
DEMO_VIEW_CPUSET="${DEMO_VIEW_CPUSET-}"
DEMO_VIEW_EXTRA="${DEMO_VIEW_EXTRA:-}"
DEMO_VIEW_EGO_SOURCE="${DEMO_VIEW_EGO_SOURCE:-frontvideo}"
DEMO_VIEW_BIN="${DEMO_VIEW_BIN:-$DEMO_REPO/scripts/x5/ego_d435i_view.py}"
DEMO_VIEW_PID=""
DEMO_VIEW_LOG=""
DEMO_VIEW_VIDEO=""
DEMO_VIEW_VIDEO_EGO=""
DEMO_VIEW_VIDEO_D435I=""
DEMO_VIEW_PAIRS=""

# ---------------------------------------------------------------------------------------------- output
# A closed terminal (EIO) must never abort the cleanup path: every print ignores its errors.
demo_say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" 2>/dev/null || true; }
DEMO_RULE="------------------------------------------------------------"
demo_has_term() { { true >&3; } 2>/dev/null; }
demo_term() { if demo_has_term; then { printf '%s\n' "$*" >&3; } 2>/dev/null || true; fi; }
demo_stage() { demo_say "$*"; demo_term "$DEMO_RULE"; demo_term "$*"; }   # a stage header for the operator
demo_warn() { demo_say "WARNING: $*"; demo_term "      Warning: $*"; }
demo_die() {
  demo_say "ERROR: $*" >&2
  demo_term "      Error: $*"
  exit 1
}

# torch thread cap + cpu pinning for everything the run starts (launch tree and demo_sequence.py)
demo_perf_env() {
  if (( DEMO_TORCH_THREADS > 0 )); then
    export OMP_NUM_THREADS="$DEMO_TORCH_THREADS" MKL_NUM_THREADS="$DEMO_TORCH_THREADS"
  fi
  local cpus="$DEMO_CPUSET"
  DEMO_TASKSET=()
  if [[ -n "$cpus" ]] && command -v taskset >/dev/null 2>&1; then DEMO_TASKSET=(taskset -c "$cpus"); fi
  demo_say "perf env: torch threads=${OMP_NUM_THREADS:-torch default}, cpuset=${cpus:-none}"
}

# ---------------------------------------------------------------------------------------------- environment
demo_source_env() {
  set +u   # the ROS setup scripts reference unset variables
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash || demo_die "source /opt/ros/jazzy/setup.bash failed"
  # shellcheck disable=SC1091
  source "$UNITREE_MSGS_WS/install/setup.bash" \
    || demo_die "$UNITREE_MSGS_WS/install/setup.bash missing (scripts/install_unitree_msgs.sh)"
  # shellcheck disable=SC1091
  source "$DEPLOY_WS/install/setup.bash" \
    || demo_die "$DEPLOY_WS/install/setup.bash missing (scripts/build.sh)"
  # the deploy venv last: its python3 (numpy/scipy/osqp/torch/av, and ROS's rclpy and cv2 through its
  # system site packages) comes first for the sequencer, the view and the camera nodes
  # shellcheck disable=SC1091
  source "$DEPLOY_VENV/bin/activate" || demo_die "$DEPLOY_VENV/bin/activate missing (README: Installation)"
  set -u
  export PYTHONUNBUFFERED=1 RCUTILS_LOGGING_BUFFERED_STREAM=0
  [[ -n "$DEMO_NIC" ]] || demo_die "DEMO_NIC is not set: put the wired interface to the robot in config/deploy.env (ip -br link)"
  [[ -n "$DEMO_ROBOT_IP" && -n "$DEMO_LAPTOP_IP" ]] \
    || demo_die "DEMO_ROBOT_IP and DEMO_LAPTOP_IP must be set in config/deploy.env (docs/network.md)"
  if [[ "${RMW_IMPLEMENTATION:-}" == "rmw_cyclonedds_cpp" ]]; then
    ros2 pkg prefix rmw_cyclonedds_cpp >/dev/null 2>&1 \
      || demo_die "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp but it is not installed (sudo apt install ros-jazzy-rmw-cyclonedds-cpp)"
    : "${CYCLONEDDS_URI:=<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"$DEMO_NIC\" priority=\"default\" multicast=\"default\"/></Interfaces></General></Domain></CycloneDDS>}"
    export CYCLONEDDS_URI
    demo_say "CycloneDDS bound to $DEMO_NIC"
  else
    demo_bind_fastdds_wired
  fi
  ros2 pkg prefix wam_go2_deploy >/dev/null 2>&1 || demo_die "package wam_go2_deploy not found in $DEPLOY_WS (scripts/build.sh)"
}

demo_bind_fastdds_wired() {
  # The default FastDDS participant announces every interface; whitelisting the wired address makes
  # the robot only ever learn that one.
  if [[ -n "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" ]]; then
    [[ -r "$FASTRTPS_DEFAULT_PROFILES_FILE" ]] \
      || demo_die "FASTRTPS_DEFAULT_PROFILES_FILE=$FASTRTPS_DEFAULT_PROFILES_FILE is not readable"
    demo_say "FastDDS profile (yours): $FASTRTPS_DEFAULT_PROFILES_FILE"
    return 0
  fi
  local dir="${XDG_RUNTIME_DIR:-/tmp}"
  DEMO_FASTDDS_PROFILE="$dir/wam_go2_fastdds_wired_${DEMO_LAPTOP_IP}.xml"
  cat > "$DEMO_FASTDDS_PROFILE" <<EOF_XML || demo_die "cannot write the FastDDS profile $DEMO_FASTDDS_PROFILE"
<?xml version="1.0" encoding="UTF-8" ?>
<!-- generated by scripts/demo/common.sh from DEMO_LAPTOP_IP; template: config/fastdds_wired.xml -->
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors>
    <transport_descriptor>
      <transport_id>udp_wired</transport_id>
      <type>UDPv4</type>
      <interfaceWhiteList>
        <address>$DEMO_LAPTOP_IP</address>
      </interfaceWhiteList>
    </transport_descriptor>
  </transport_descriptors>
  <participant profile_name="wired_participant" is_default_profile="true">
    <rtps>
      <userTransports>
        <transport_id>udp_wired</transport_id>
      </userTransports>
      <useBuiltinTransports>false</useBuiltinTransports>
    </rtps>
  </participant>
  <!-- ros_discovery_info in its own partition: a ROS 2 Humble node (the X5 camera host) cannot
       parse Jazzy's node-name messages (config/fastdds_wired.xml) -->
  <data_writer profile_name="ros_discovery_info">
    <qos><partition><names><name>wam_jazzy_graph</name></names></partition></qos>
  </data_writer>
  <data_reader profile_name="ros_discovery_info">
    <qos><partition><names><name>wam_jazzy_graph</name></names></partition></qos>
  </data_reader>
</profiles>
EOF_XML
  export FASTRTPS_DEFAULT_PROFILES_FILE="$DEMO_FASTDDS_PROFILE"
  demo_say "FastDDS bound to $DEMO_LAPTOP_IP ($DEMO_NIC): $DEMO_FASTDDS_PROFILE"
  demo_restart_ros2_daemon
}

demo_restart_ros2_daemon() {
  # A ros2 CLI daemon started earlier without the wired binding cannot see the wired-only nodes
  # ('ros2 param get' -> 'Node not found'). Restart it with this environment, early, so it has the
  # whole bring-up to discover the controller.
  ros2 daemon stop >/dev/null 2>&1 || true
  ros2 daemon start >/dev/null 2>&1 || true
}

# ---------------------------------------------------------------------------------------------- pre-flight
demo_rx_bytes() {
  cat "${DEMO_SYSFS_NET:-/sys/class/net}/$1/statistics/rx_bytes" 2>/dev/null || echo 0
}

demo_preflight_datapath() {
  # The controller is up and subscribed: the 500 Hz LowState stream (~0.65 MB/s) must now arrive
  # on DEMO_NIC and on nothing else (a robot that still reaches the laptop over WiFi).
  local min_mbps="${DEMO_DATAPATH_MIN_MBPS:-0.4}" seconds="${DEMO_DATAPATH_SECONDS:-2}"
  local sys="${DEMO_SYSFS_NET:-/sys/class/net}" nic others=() b0=() b1=() i
  [[ -r "$sys/$DEMO_NIC/statistics/rx_bytes" ]] || demo_die "cannot read $sys/$DEMO_NIC/statistics/rx_bytes"
  for nic in "$sys"/*/; do
    nic="$(basename "$nic")"
    [[ "$nic" == "$DEMO_NIC" || "$nic" == "lo" ]] && continue
    [[ -r "$sys/$nic/statistics/rx_bytes" ]] || continue
    [[ "$(cat "$sys/$nic/operstate" 2>/dev/null || echo up)" == "down" ]] && continue
    others+=("$nic")
  done
  local nics=("$DEMO_NIC" "${others[@]}")
  for i in "${!nics[@]}"; do b0[i]="$(demo_rx_bytes "${nics[i]}")"; done
  sleep "$seconds"
  for i in "${!nics[@]}"; do b1[i]="$(demo_rx_bytes "${nics[i]}")"; done
  local report="" rate_nic="" rate
  for i in "${!nics[@]}"; do
    rate="$(awk -v a="${b0[i]}" -v b="${b1[i]}" -v s="$seconds" 'BEGIN{printf "%.2f", (b-a)/1e6/s}')"
    report+="${nics[i]} ${rate} MB/s  "
    (( i == 0 )) && rate_nic="$rate"
  done
  demo_say "LowState data path (${seconds} s): $report"
  if awk -v r="$rate_nic" -v m="$min_mbps" 'BEGIN{exit !(r < m)}'; then
    demo_die "/lowstate is not arriving on $DEMO_NIC ($rate_nic MB/s < $min_mbps): the robot streams it over another interface. Turn the laptop WiFi off (nmcli radio wifi off) and run again"
  fi
  for i in "${!nics[@]}"; do
    (( i == 0 )) && continue
    rate="$(awk -v a="${b0[i]}" -v b="${b1[i]}" -v s="$seconds" 'BEGIN{printf "%.2f", (b-a)/1e6/s}')"
    if awk -v r="$rate" -v m="$min_mbps" 'BEGIN{exit !(r >= m)}'; then
      demo_warn "${nics[i]} also receives $rate MB/s (make sure it is not a second /lowstate path)"
    fi
  done
}

demo_preflight_network() {
  demo_say "network pre-flight ($DEMO_NIC, $DEMO_LAPTOP_IP -> $DEMO_ROBOT_IP)"
  ip link show dev "$DEMO_NIC" >/dev/null 2>&1 || demo_die "network interface $DEMO_NIC not found: plug in the robot cable"
  if ! ip -4 -br addr show dev "$DEMO_NIC" | grep -qF "$DEMO_LAPTOP_IP/24"; then
    demo_term "      Configuring $DEMO_NIC as $DEMO_LAPTOP_IP/24 (needs sudo)"
    sudo ip addr flush dev "$DEMO_NIC"
    sudo ip addr add "$DEMO_LAPTOP_IP/24" dev "$DEMO_NIC"
    sudo ip link set "$DEMO_NIC" up
    ip -4 -br addr show dev "$DEMO_NIC" | grep -qF "$DEMO_LAPTOP_IP/24" \
      || demo_die "could not set $DEMO_LAPTOP_IP/24 on $DEMO_NIC"
  fi
  ping -c 3 -W 2 "$DEMO_ROBOT_IP" >/dev/null || demo_die "the robot ($DEMO_ROBOT_IP) does not answer: is it on and the cable plugged in?"
  local route
  route="$(ip route get "$DEMO_ROBOT_IP")"
  demo_say "route: $route"
  grep -qF "dev $DEMO_NIC" <<<"$route" || demo_die "the route to $DEMO_ROBOT_IP does not use $DEMO_NIC"
  grep -qF "src $DEMO_LAPTOP_IP" <<<"$route" || demo_die "the route to $DEMO_ROBOT_IP does not use source $DEMO_LAPTOP_IP"
}

demo_view_stop_stray() {
  # A viewer left over from an earlier run keeps camera bytes flowing into DEMO_NIC and would satisfy
  # the data-path byte-rate check on its own. Stop it (INT, TERM, KILL); never die. Match only an
  # interpreter running the viewer, never an editor or a pytest whose command line names the file.
  local pat='^([^ ]*/)?python[0-9.]*( -[A-Za-z]+)* ([^ ]*/)?ego_d435i_view\.py( |$)' pids="" pid i b alive left
  b="$(basename -- "$DEMO_VIEW_BIN")"
  if [[ "$b" != ego_d435i_view.py ]]; then
    b="$(printf '%s' "$b" | sed 's/[][\.*^$|()+?{}]/\\&/g')"
    pat="$pat|^([^ ]*/)?python[0-9.]*( -[A-Za-z]+)* ([^ ]*/)?${b}( |$)"
  fi
  pids="$(pgrep -f -- "$pat" 2>/dev/null | grep -vx -- "${DEMO_VIEW_PID:-x}" | grep -vx -- "$$" || true)"
  [[ -n "$pids" ]] || return 0
  demo_say "stopping a camera viewer left over from an earlier run: $(echo "$pids" | paste -sd,)"
  for pid in $pids; do kill -INT "$pid" 2>/dev/null || true; done
  for i in $(seq 20); do alive=0; for pid in $pids; do kill -0 "$pid" 2>/dev/null && alive=1; done; (( alive )) || break; sleep 0.1; done
  for pid in $pids; do kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true; done
  for i in $(seq 10); do alive=0; for pid in $pids; do kill -0 "$pid" 2>/dev/null && alive=1; done; (( alive )) || break; sleep 0.1; done
  for pid in $pids; do kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true; done
  sleep 0.1
  left="$(pgrep -f -- "$pat" 2>/dev/null | grep -vx -- "${DEMO_VIEW_PID:-x}" | grep -vx -- "$$" || true)"
  [[ -z "$left" ]] || demo_warn "could not stop an old camera viewer (pid $(echo "$left" | paste -sd,))"
  return 0
}

demo_preflight_processes() {
  # bracket patterns never match this shell's own command line
  local stray
  if stray="$(pgrep -fa '[g]o2_low_level')"; then
    demo_die "a go2_low_level process is running (two LowCmd publishers would fight): $stray"
  fi
  if stray="$(pgrep -fa '[l]ib/wam_go2_deploy/controller')"; then
    demo_die "a controller is already running (two LowCmd publishers would fight); stop it first: $stray"
  fi
  # this package's launch files; only a running `ros2 launch` counts (not an editor with the file open)
  if stray="$(pgrep -fa '[r]os2 launch .*wam_(go2|bridge)\.launch\.py')"; then
    demo_die "a launch from an earlier run is still running; stop it first: $stray"
  fi
  demo_view_stop_stray
  demo_say "no stray LowCmd publishers"
}

# ---------------------------------------------------------------------------------------------- controller
demo_release_sport_mode() {
  demo_say "release sport mode (must exit 0)"
  timeout --foreground 60 ros2 run wam_go2_deploy release_sport_mode \
    || demo_die "release_sport_mode failed: the robot's sport mode still owns the motors"
}

demo_launch_controller() {
  local name="$1" ckpt=()
  mkdir -p "$DEMO_LOG_DIR"
  DEMO_LOG="$DEMO_LOG_DIR/controller_${name}_$(date +%Y%m%d_%H%M%S).log"
  [[ -z "$RAMBO_CONTROLLER" ]] || ckpt=("policy_checkpoint:=$RAMBO_CONTROLLER")
  demo_say "launching the controller (log: $DEMO_LOG)"
  # setsid: own process group, so the whole launch tree can be signalled. env --default-signal: a
  # background child of a non-interactive bash starts with SIGINT/SIGQUIT ignored, which survives
  # exec, so ros2 launch would never see the cleanup SIGINT.
  # shellcheck disable=SC2086
  setsid env --default-signal=INT,QUIT "${DEMO_TASKSET[@]}" ros2 launch wam_go2_deploy wam_go2.launch.py \
    task:=quadruped "${ckpt[@]}" ${DEMO_LAUNCH_EXTRA:-} "wired_nic:=$DEMO_NIC" > "$DEMO_LOG" 2>&1 3>&- &
  DEMO_LAUNCH_PID=$!
  sleep 1
  kill -0 "$DEMO_LAUNCH_PID" 2>/dev/null || demo_die "the controller exited at once (log: $DEMO_LOG)"
  DEMO_LAUNCH_PGID="$(ps -o pgid= -p "$DEMO_LAUNCH_PID" 2>/dev/null | tr -d ' ' || true)"
  local sigign
  sigign="$(awk '/^SigIgn:/{print $2}' "/proc/$DEMO_LAUNCH_PID/status" 2>/dev/null || true)"
  if [[ -n "$sigign" ]] && (( 16#$sigign & 2 )); then
    demo_die "the controller launch ignores SIGINT (SigIgn=$sigign): the cleanup could not stop it"
  fi
  demo_say "controller launch pid=$DEMO_LAUNCH_PID pgid=${DEMO_LAUNCH_PGID:-?}"
}

demo_wait_mode() {
  demo_say "waiting for rambo/mode (needs /lowstate)"
  local deadline=$((SECONDS + 25))
  while (( SECONDS < deadline )); do
    if [[ -n "$DEMO_LAUNCH_PID" ]] && ! kill -0 "$DEMO_LAUNCH_PID" 2>/dev/null; then
      demo_die "the controller exited during start-up (log: $DEMO_LOG)"
    fi
    if timeout --foreground 5 ros2 topic echo rambo/mode --once >/dev/null 2>&1; then
      demo_say "controller publishes rambo/mode"
      return 0
    fi
  done
  demo_warn "no controller status within 25 s (no /lowstate?); still waiting"
}

demo_confirm() {
  [[ "${DEMO_CONFIRM:-1}" == "1" ]] || { demo_say "DEMO_CONFIRM=0: no confirmation pause"; return 0; }
  local tty_in=""
  if [[ -t 0 ]]; then
    tty_in=/dev/stdin
  elif [[ -r /dev/tty && -w /dev/tty ]] && { : </dev/tty; } 2>/dev/null; then
    tty_in=/dev/tty
  else
    demo_die "no terminal for the safety confirmation (stdin is not a tty); DEMO_CONFIRM=0 only with someone watching the robot"
  fi
  # only an Enter pressed at THIS prompt starts the robot: discard anything typed ahead
  local _ahead
  while read -r -t 0.05 -n 4096 _ahead <"$tty_in"; do :; done
  local prompt="      The robot will stand up. Hold the leash; Ctrl+C stops the robot at any time.
      Press Enter to start (Ctrl+C to abort): "
  if demo_has_term; then { printf '%s' "$prompt" >&3; } 2>/dev/null || true
  else printf '%s' "$prompt"; fi
  read -r _ <"$tty_in" || return $?
  demo_say "operator confirmed"
}

demo_param_get() {
  # ros2 param get on the running controller. The CLI's discovery is the weak spot: the restarted
  # daemon first, then --no-daemon with a long spin time, up to four rounds (~30 s worst case).
  # Prints the answer, or "" if none.
  local param="$1" out="" i
  for i in 1 2 3 4; do
    out="$(timeout --foreground 12 ros2 param get /wam_go2_controller "$param" 2>/dev/null || true)"
    [[ "$out" == *value* ]] && { echo "$out"; return 0; }
    out="$(timeout --foreground 15 ros2 param get --no-daemon --spin-time 4 /wam_go2_controller "$param" 2>/dev/null || true)"
    [[ "$out" == *value* ]] && { echo "$out"; return 0; }
    sleep 1
  done
  echo ""
}

demo_check_controller() {
  # the running controller must check the same wired interface this script bound DDS to
  local actual_nic
  actual_nic="$(demo_param_get wired_nic)"
  [[ "$actual_nic" == "String value is: $DEMO_NIC" ]] \
    || demo_die "the running controller's wired_nic is not DEMO_NIC=$DEMO_NIC (got: ${actual_nic:-no answer})"
  demo_say "controller parameters ok (wired_nic=$DEMO_NIC)"
}

# Bring-up commands run with `timeout --foreground` so Ctrl+C reaches them at once; the two stop-path
# commands below keep plain `timeout`, whose own process group shields them from a second Ctrl+C.
demo_mode() {
  # current rambo/mode string; "" = unknown (nothing arrived within 5 s), never "DAMP"
  timeout 5 ros2 topic echo rambo/mode --once 2>/dev/null | sed -n 's/^data: //p' | tr -d "'\"" || true
}

demo_publish_damp() {
  timeout 5 ros2 topic pub --once -w 1 rambo/fsm_cmd std_msgs/msg/String "{data: DAMP}" >/dev/null 2>&1 || true
}

demo_request_damp() {
  # Publish DAMP first (the operator's stop must not wait for a mode query), then confirm it.
  demo_say "requesting DAMP"
  demo_publish_damp
  local i m=""
  for i in 1 2 3 4; do
    m="$(demo_mode)"
    [[ "$m" == DAMP* ]] && { demo_say "rambo/mode: $m"; return 0; }
    demo_publish_damp
  done
  demo_warn "DAMP not confirmed; stopping the controller anyway (it damps on exit)"
}

demo_signal_launch() {
  local sig="$1"
  if [[ -n "$DEMO_LAUNCH_PGID" ]]; then
    kill "-$sig" -- "-$DEMO_LAUNCH_PGID" 2>/dev/null || kill "-$sig" "$DEMO_LAUNCH_PID" 2>/dev/null || true
  else
    kill "-$sig" "$DEMO_LAUNCH_PID" 2>/dev/null || true
  fi
}

demo_wait_launch_gone() {
  local ticks="$1" i
  for i in $(seq "$ticks"); do
    kill -0 "$DEMO_LAUNCH_PID" 2>/dev/null || return 0
    sleep 0.1
  done
  return 1
}

# ---------------------------------------------------------------------------------------------- camera view
demo_view_start() {
  # The two camera windows (Ego, D435i) for the operator, one video each. Never fails the run.
  local name="$1"
  [[ "$DEMO_VIEW" == "1" ]] || { demo_say "DEMO_VIEW=$DEMO_VIEW: camera view off"; return 0; }
  if [[ ! -r "$DEMO_VIEW_BIN" ]]; then
    demo_warn "camera viewer not found; continuing without camera windows"; return 0
  fi
  if [[ "$DEMO_VIEW_BIN" == *ego_d435i_view.py ]] && ! python3 -c "import cv2, av" >/dev/null 2>&1; then
    demo_warn "python3 lacks cv2/av; continuing without camera windows"; return 0
  fi
  demo_view_stop_stray
  local cpus="$DEMO_VIEW_CPUSET" pin=()
  if [[ -n "$cpus" ]] && command -v taskset >/dev/null 2>&1; then pin=(taskset -c "$cpus"); fi
  local stamp view_dir log_dir stem
  stamp="$(date +%Y%m%d_%H%M%S)"
  if [[ -n "$DEMO_VIEW_DIR" ]]; then
    view_dir="$DEMO_VIEW_DIR"; log_dir="$DEMO_VIEW_DIR"; stem="camera"
  else
    view_dir="$DEMO_REPO/data/videos/demo_view"; log_dir="$DEMO_LOG_DIR/demo_view"; stem="demo_${name}_${stamp}_view"
  fi
  mkdir -p "$view_dir" "$log_dir" 2>/dev/null || { demo_warn "cannot create $view_dir; continuing without camera windows"; return 0; }
  DEMO_VIEW_VIDEO="$view_dir/$stem.mp4"                 # the viewer's recording name; it writes the two below
  DEMO_VIEW_VIDEO_EGO="$view_dir/${stem}_ego.mp4"
  DEMO_VIEW_VIDEO_D435I="$view_dir/${stem}_d435i.mp4"
  DEMO_VIEW_LOG="$log_dir/$stem.log"
  DEMO_VIEW_PAIRS="$log_dir/${stem}_pairs.csv"
  local args=(--res "$DEMO_VIEW_RES" --record "$DEMO_VIEW_VIDEO" --out-dir "$log_dir" --ego-source "$DEMO_VIEW_EGO_SOURCE")
  local calib_desc="uncalibrated"
  if [[ -n "$DEMO_VIEW_CALIB" ]]; then
    if [[ -r "$DEMO_VIEW_CALIB" ]]; then args+=(--calib "$DEMO_VIEW_CALIB"); calib_desc="calib $DEMO_VIEW_CALIB"
    else demo_warn "DEMO_VIEW_CALIB is not readable; uncalibrated view"; fi
  fi
  local extra=()
  if [[ -n "$DEMO_VIEW_EXTRA" ]]; then set -f; read -r -a extra <<<"$DEMO_VIEW_EXTRA"; set +f; fi
  if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    demo_say "no display: the view records without windows"
    args+=(--no-window)
  fi
  # shellcheck disable=SC2086
  setsid env --default-signal=INT,QUIT nice -n 10 "${pin[@]}" python3 "$DEMO_VIEW_BIN" "${args[@]}" "${extra[@]}" \
    > "$DEMO_VIEW_LOG" 2>&1 3>&- &
  DEMO_VIEW_PID=$!
  sleep 1
  if ! kill -0 "$DEMO_VIEW_PID" 2>/dev/null; then
    local rc=0; wait "$DEMO_VIEW_PID" 2>/dev/null || rc=$?
    demo_warn "camera viewer exited (rc=$rc); continuing without camera windows"
    DEMO_VIEW_PID=""; return 0
  fi
  demo_say "camera view: windows Ego and D435i (ego ${DEMO_VIEW_EGO_SOURCE}, res ${DEMO_VIEW_RES}, ${calib_desc}, cpuset ${cpus:-none}) pid=$DEMO_VIEW_PID"
  demo_say "  videos: $DEMO_VIEW_VIDEO_EGO, $DEMO_VIEW_VIDEO_D435I; log: $DEMO_VIEW_LOG; pair log: $DEMO_VIEW_PAIRS"
  demo_term "      Camera windows open; click this terminal before pressing Enter."
  return 0
}

demo_view_pairing_line() {
  # The viewer's one-line time-sync reports ("grid ...", "pairing: ..."), if it wrote them.
  local line=""
  [[ -n "$DEMO_VIEW_LOG" && -r "$DEMO_VIEW_LOG" ]] || return 0
  line="$(grep -a -m1 '^grid [0-9.]* Hz: ' "$DEMO_VIEW_LOG" 2>/dev/null || true)"
  [[ -z "$line" ]] || demo_say "  $line"
  line="$(grep -a -m1 '^pairing: ' "$DEMO_VIEW_LOG" 2>/dev/null || true)"
  [[ -z "$line" ]] || demo_say "  $line"
  return 0
}

demo_view_stop() {
  # Stop the view last; its exit status never changes the run's.
  [[ -n "$DEMO_VIEW_PID" ]] || return 0
  if ! kill -0 "$DEMO_VIEW_PID" 2>/dev/null; then
    local rc=0; wait "$DEMO_VIEW_PID" 2>/dev/null || rc=$?
    demo_warn "camera viewer stopped during the run (rc=$rc)"
    demo_view_pairing_line
    DEMO_VIEW_PID=""; return 0
  fi
  kill -INT "$DEMO_VIEW_PID" 2>/dev/null || true
  local i
  for i in $(seq 30); do kill -0 "$DEMO_VIEW_PID" 2>/dev/null || break; sleep 0.1; done
  if kill -0 "$DEMO_VIEW_PID" 2>/dev/null; then
    kill -TERM "$DEMO_VIEW_PID" 2>/dev/null || true
    for i in $(seq 20); do kill -0 "$DEMO_VIEW_PID" 2>/dev/null || break; sleep 0.1; done
    kill -0 "$DEMO_VIEW_PID" 2>/dev/null && kill -KILL "$DEMO_VIEW_PID" 2>/dev/null || true
  fi
  wait "$DEMO_VIEW_PID" 2>/dev/null || true
  local v
  for v in "$DEMO_VIEW_VIDEO_EGO" "$DEMO_VIEW_VIDEO_D435I"; do
    if [[ -s "$v" ]]; then demo_say "camera view stopped, video: $v"
    else demo_say "camera view stopped, no video $v (log $DEMO_VIEW_LOG)"; fi
  done
  demo_view_pairing_line
  DEMO_VIEW_PID=""
  return 0
}

# ---------------------------------------------------------------------------------------------- cleanup
demo_cleanup() {
  local rc=$?
  trap '' INT TERM HUP PIPE   # first: a second Ctrl+C (or a closed terminal's second SIGHUP) must not abandon the shutdown
  trap - EXIT
  if [[ -n "$DEMO_LAUNCH_PID" ]] && kill -0 "$DEMO_LAUNCH_PID" 2>/dev/null; then
    demo_request_damp
    demo_say "stopping the controller (SIGINT; it sends damping frames before it exits)"
    demo_signal_launch INT
    if ! demo_wait_launch_gone 50; then
      demo_say "controller still running after 5 s: SIGTERM"
      demo_signal_launch TERM
      if ! demo_wait_launch_gone 30; then
        demo_say "controller still running: SIGKILL"
        demo_signal_launch KILL
      fi
    fi
    wait "$DEMO_LAUNCH_PID" 2>/dev/null || true
  fi
  [[ -n "$DEMO_LOG" ]] && demo_say "controller log: $DEMO_LOG"
  demo_view_stop
  exit "$rc"
}
