#!/usr/bin/env bash
# The robot side of one WAM-Policy task run (normally started by run_wam.sh / tasks/<task>.sh,
# which also start the policy server and create the run directory):
#
#   1. bring-up (scripts/demo/common.sh): ROS env, wired-only DDS, network and process pre-flight,
#      sport-mode release, the controller (heading hold off while walking, on while stepping in
#      place between chunks), data-path and controller checks, the two camera windows;
#   2. the real-to-sim D435i stream (scripts/x5/d435i_sim_camera.py --scale 0.5, 640x360);
#   3. the policy server must answer on 127.0.0.1:WAM_PORT (WAM_MOCK=1 starts the scripted mock);
#   4. Enter, then `ros2 launch wam_go2_deploy wam_bridge.launch.py` (ego_stream + wam_bridge) in the
#      background: the bridge waits for the camera streams and for POLICY. It is started before the
#      robot stands, and the robot only stands once the bridge is up;
#   5. scripts/demo_sequence.py --demo wam_entry: STAND -> front-left foot to the task's entry
#      target -> POLICY -> settle; then the bridge runs the episode (reset, predict, execute,
#      feedback, ...) until its budget, a stop condition, or Ctrl+C;
#   6. cleanup: stop the bridge, request DAMP, stop the sidecars and the controller.
#
#   WAM_TASK_ID=push_box scripts/demo/wam_task.sh            # the server must already run
#   WAM_TASK_ID=push_box WAM_MOCK=1 scripts/demo/wam_task.sh # rehearsal with the mock (steps in place)
#
# Env: WAM_TASK_ID (push_box; instruction, entry target and task profile come from
# config/wam_tasks.json unless WAM_INSTRUCTION / WAM_EE / WAM_TASK_VERSION are set), WAM_PORT (8862),
# WAM_SESSION (go2-real-<task>), WAM_MAX_CHUNKS (1000), WAM_MAX_WALL_S (0 = none), WAM_SAVE_IMAGES (0),
# WAM_GO2_LATENCY_MS (0), WAM_CALIB (time-sync calibration JSON), WAM_OUTPUT_DIR (bridge records,
# ~/.ros/wam_bridge), WAM_HEADING_HOLD (false), WAM_IN_PLACE_HOLD (true), WAM_COMMAND_HANDLING
# (sim: commands executed exactly as in the QLM-Bench training simulation, unclipped | rambo: clipped
# to RAMBO's training ranges), WAM_MOCK (0), WAM_MOCK_ARGS (e.g. "--vx 0.1"), WAM_SIM_CAMERA (1),
# WAM_START_TIMEOUT_S (90), WAM_SETTLE_S (2.0), WAM_BRIDGE_READY_S (30), QLM_BENCH_SRC (optional, configs/site.env),
# plus the DEMO_* settings of common.sh. Extra arguments go to demo_sequence.py (--settle, --gap, ...).
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

WAM_TASK_ID="${WAM_TASK_ID:-push_box}"
wam_task_field() { PYTHONPATH="$DEMO_REPO" python3 -m wam_go2.wam.tasks get "$WAM_TASK_ID" "$1"; }
WAM_INSTRUCTION="${WAM_INSTRUCTION:-$(wam_task_field instruction)}"
WAM_EE="${WAM_EE:-$(wam_task_field entry_ee)}"
WAM_TASK_VERSION="${WAM_TASK_VERSION:-$(wam_task_field task_profile_version)}"
WAM_HOST=127.0.0.1
WAM_SESSION="${WAM_SESSION:-go2-real-$WAM_TASK_ID}"
WAM_MAX_CHUNKS="${WAM_MAX_CHUNKS:-1000}"
WAM_MAX_WALL_S="${WAM_MAX_WALL_S:-0}"
WAM_SAVE_IMAGES="${WAM_SAVE_IMAGES:-0}"
WAM_GO2_LATENCY_MS="${WAM_GO2_LATENCY_MS:-0}"
WAM_CALIB="${WAM_CALIB:-}"
WAM_HEADING_HOLD="${WAM_HEADING_HOLD:-false}"     # training had no yaw hold while walking
WAM_IN_PLACE_HOLD="${WAM_IN_PLACE_HOLD:-true}"     # hold the yaw while stepping in place between chunks
WAM_COMMAND_HANDLING="${WAM_COMMAND_HANDLING:-sim}"
WAM_MOCK="${WAM_MOCK:-0}"
WAM_MOCK_ARGS="${WAM_MOCK_ARGS:-}"
WAM_SIM_CAMERA="${WAM_SIM_CAMERA:-1}"
WAM_OUTPUT_DIR="${WAM_OUTPUT_DIR:-$HOME/.ros/wam_bridge}"
WAM_START_TIMEOUT_S="${WAM_START_TIMEOUT_S:-90}"
WAM_SETTLE_S="${WAM_SETTLE_S:-2.0}"
WAM_BRIDGE_READY_S="${WAM_BRIDGE_READY_S:-30}"
WAM_BRIDGE_LOG=""
WAM_BRIDGE_T0=0
WAM_SIDECAR_PIDS=()
WAM_BRIDGE_PID=""

wam_stop() {  # wam_stop <pid> <ticks of 0.1 s after SIGINT>: INT, then TERM, then KILL (its process group)
  local pid="$1" ticks="$2" _
  kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null; return 0; }
  kill -INT "$pid" 2>/dev/null || true
  for _ in $(seq "$ticks"); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
  fi
  if kill -0 "$pid" 2>/dev/null; then
    demo_say "process $pid still running: SIGKILL"
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

wam_sidecar_stop() {
  local pid
  for pid in "${WAM_SIDECAR_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && wam_stop "$pid" 20
  done
  WAM_SIDECAR_PIDS=()
}

wam_cleanup() {
  local rc=$?
  # first: no further signal may cut the shutdown short (a closed terminal sends SIGHUP twice;
  # a terminal or log pipe that went away must not kill a write)
  trap '' INT TERM HUP PIPE
  trap - EXIT
  set +e                        # the shutdown runs to its end whatever fails on the way
  demo_stage "[5/5] Shutdown: robot damped, controller and cameras stopped"
  local bridge_up=0
  if [[ -n "$WAM_BRIDGE_PID" ]] && kill -0 "$WAM_BRIDGE_PID" 2>/dev/null; then
    bridge_up=1
    kill -INT "$WAM_BRIDGE_PID" 2>/dev/null || true      # the bridge stops commanding at once
  fi
  # damp right away: an operator's Ctrl+C must not wait for the bridge or the sidecars
  if [[ -n "$DEMO_LAUNCH_PID" ]] && kill -0 "$DEMO_LAUNCH_PID" 2>/dev/null; then
    demo_request_damp
  fi
  if (( bridge_up )); then
    wam_stop "$WAM_BRIDGE_PID" 100
  fi
  wam_sidecar_stop
  (exit "$rc")                  # demo_cleanup reports $? as this script's exit code
  demo_cleanup                  # (DAMP confirmed), stop the controller and the view; exits with rc
}

wam_check_qlm() {
  # the QLM-Bench live package: installed in the deploy venv (pip install -e "<QLM-Bench>[live]"), or the
  # src/ of a checkout in QLM_BENCH_SRC (configs/site.env)
  local src="${QLM_BENCH_SRC:-}"
  [[ -z "$src" || ! -d "$src/qlm_bench" ]] || export PYTHONPATH="$src${PYTHONPATH:+:$PYTHONPATH}"
  python3 -c 'import qlm_bench.live' >/dev/null 2>&1 \
    || demo_die "the QLM-Bench live package is not importable: pip install -e \"<QLM-Bench>[live]\" in the deploy venv (README: Installation)"
}

wam_shm_profile() {
  # the camera nodes and the bridge exchange frames locally: wired whitelist + shared memory
  local base="$DEMO_REPO/config/fastdds_wired_shm.xml"
  [[ -r "$base" ]] || demo_die "$base missing"
  WAM_SHM_PROFILE="${XDG_RUNTIME_DIR:-/tmp}/wam_go2_fastdds_wired_shm_${DEMO_LAPTOP_IP}.xml"
  sed "s/@LAPTOP_IP@/${DEMO_LAPTOP_IP}/g" "$base" > "$WAM_SHM_PROFILE"
}

wam_service_check() {
  if [[ "$WAM_MOCK" == "1" ]]; then
    local log="$DEMO_LOG_DIR/mock_policy_$(date +%Y%m%d_%H%M%S).log"
    demo_say "starting the mock policy server (scripted command, no model; log: $log)"
    # shellcheck disable=SC2086
    setsid env --default-signal=INT,QUIT python3 -m wam_go2.wam.mock_service --host "$WAM_HOST" --port "$WAM_PORT" \
      --instruction "$WAM_INSTRUCTION" --ee $WAM_EE $WAM_MOCK_ARGS > "$log" 2>&1 3>&- &
    WAM_SIDECAR_PIDS+=("$!")
    sleep 1
  fi
  python3 - "$WAM_HOST" "$WAM_PORT" <<'EOF' || demo_die "no policy server answers on $WAM_HOST:$WAM_PORT (run_wam.sh starts it; WAM_MOCK=1 for a rehearsal)"
import socket, sys
s = socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=3)
s.close()
EOF
  demo_say "policy server reachable: $WAM_HOST:$WAM_PORT"
}

wam_sim_camera_start() {
  [[ "$WAM_SIM_CAMERA" == "1" ]] || { demo_say "WAM_SIM_CAMERA=0: real-to-sim D435i stream not started"; return 0; }
  local cpus="$DEMO_VIEW_CPUSET" pin=()
  [[ -n "$cpus" ]] && command -v taskset >/dev/null 2>&1 && pin=(taskset -c "$cpus")
  local log="$DEMO_LOG_DIR/d435i_sim_camera_$(date +%Y%m%d_%H%M%S).log"
  demo_say "starting the real-to-sim D435i stream (640x360, --scale 0.5; log: $log)"
  FASTRTPS_DEFAULT_PROFILES_FILE="$WAM_SHM_PROFILE" setsid env --default-signal=INT,QUIT nice -n 5 "${pin[@]}" \
    python3 "$DEMO_REPO/scripts/x5/d435i_sim_camera.py" --scale 0.5 > "$log" 2>&1 3>&- &
  WAM_SIDECAR_PIDS+=("$!")
}

wam_bridge_start() {
  # Started BEFORE the entry sequence: the bridge waits for the streams, POLICY + settle and the
  # sequence's exit, so its Python start-up never runs next to a controller in POLICY.
  local cpus="$DEMO_VIEW_CPUSET" pin=()
  [[ -n "$cpus" ]] && command -v taskset >/dev/null 2>&1 && pin=(taskset -c "$cpus")
  local save=false; [[ "$WAM_SAVE_IMAGES" == "1" ]] && save=true
  WAM_BRIDGE_LOG="$DEMO_LOG_DIR/wam_bridge_$(date +%Y%m%d_%H%M%S).log"
  # every argument must be name:=value with a value (ros2 launch refuses 'calib:=' and starts nothing)
  local largs=("host:=$WAM_HOST" "port:=$WAM_PORT" "instruction:=$WAM_INSTRUCTION"
               "task_id:=$WAM_TASK_ID" "task_version:=$WAM_TASK_VERSION" "session_id:=$WAM_SESSION"
               "max_chunks:=$WAM_MAX_CHUNKS" "max_wall_s:=$WAM_MAX_WALL_S" "save_images:=$save"
               "go2_latency_ms:=$WAM_GO2_LATENCY_MS" "output_dir:=$WAM_OUTPUT_DIR"
               "start_timeout_s:=$WAM_START_TIMEOUT_S" "settle_s:=$WAM_SETTLE_S")
  [[ -n "$WAM_CALIB" ]] && largs+=("calib:=$WAM_CALIB")
  demo_say "starting ego_stream + wam_bridge (records: $WAM_OUTPUT_DIR, log: $WAM_BRIDGE_LOG)"
  WAM_BRIDGE_T0="$(date +%s)"
  FASTRTPS_DEFAULT_PROFILES_FILE="$WAM_SHM_PROFILE" setsid env --default-signal=INT,QUIT "${pin[@]}" \
    ros2 launch wam_go2_deploy wam_bridge.launch.py "${largs[@]}" > "$WAM_BRIDGE_LOG" 2>&1 3>&- &
  WAM_BRIDGE_PID=$!
}

wam_bridge_ready() {
  # the robot stands up only with a running bridge: wait for its start line; the launch dying first,
  # or WAM_BRIDGE_READY_S without the line, is a failure
  local i rc=0
  for i in $(seq $(( WAM_BRIDGE_READY_S * 10 ))); do
    if ! kill -0 "$WAM_BRIDGE_PID" 2>/dev/null; then    # first: a dead launch closes the gate whatever it logged
      wait "$WAM_BRIDGE_PID" 2>/dev/null || rc=$?
      WAM_BRIDGE_PID=""
      demo_say "the bridge launch exited before the robot stood up (exit $rc; log $WAM_BRIDGE_LOG)"
      tail -n 5 "$WAM_BRIDGE_LOG" 2>/dev/null | sed 's/^/    /' || true
      return 1
    fi
    if grep -q "waiting for rambo/state" "$WAM_BRIDGE_LOG" 2>/dev/null; then
      sleep 0.5                                    # ... and still running a moment later
      if kill -0 "$WAM_BRIDGE_PID" 2>/dev/null; then
        demo_say "bridge up, waiting for POLICY"
        return 0
      fi
      continue                                     # gone: the check above reaps and reports it
    fi
    sleep 0.1
  done
  demo_say "the bridge did not come up within ${WAM_BRIDGE_READY_S} s (log $WAM_BRIDGE_LOG)"
  return 1
}

wam_bridge_summary() {  # "<exit_code> <stop_reason>" of the newest bridge summary.json written since $1 (epoch s)
  python3 - "$WAM_OUTPUT_DIR" "$1" <<'EOF'
import json, sys
from pathlib import Path
root, t0 = Path(sys.argv[1]).expanduser(), float(sys.argv[2])
found = sorted((p.stat().st_mtime, str(p)) for p in root.glob("*/summary.json") if p.stat().st_mtime >= t0 - 1)
if found:
    data = json.loads(Path(found[-1][1]).read_text())
    if isinstance(data.get("exit_code"), int):
        print(data["exit_code"], data.get("stop_reason") or "")
EOF
}

wam_bridge_wait() {
  # ros2 launch returns 0 whatever the bridge returned: the bridge's own exit code is in its summary.json
  local rc=0 got
  wait "$WAM_BRIDGE_PID" || rc=$?
  WAM_BRIDGE_PID=""
  got="$(wam_bridge_summary "$WAM_BRIDGE_T0" 2>/dev/null || true)"
  if [[ -n "$got" ]]; then
    demo_say "bridge summary: exit_code ${got%% *}, stop_reason ${got#* } (launch exit $rc)"
    rc="${got%% *}"
  else                                             # killed before it could write its summary: not a clean end
    demo_warn "bridge summary missing (bridge killed?)"
    demo_say "no bridge summary.json for this run in $WAM_OUTPUT_DIR (launch exit $rc)"
    (( rc != 0 )) || rc=1
  fi
  return "$rc"
}

main() {
  trap '' PIPE                  # a write to a terminal or log pipe that went away fails instead of killing
  trap wam_cleanup EXIT
  trap 'trap "" INT TERM HUP PIPE; demo_say "interrupted"; exit 130' INT TERM HUP
  case "$WAM_COMMAND_HANDLING" in
    sim|rambo) ;;
    *) demo_die "WAM_COMMAND_HANDLING=$WAM_COMMAND_HANDLING: must be sim or rambo" ;;
  esac
  demo_stage "[3/5] Robot bring-up: network, controller, cameras"
  demo_say "task $WAM_TASK_ID: \"$WAM_INSTRUCTION\" (entry target $WAM_EE, command handling $WAM_COMMAND_HANDLING)"
  demo_source_env
  wam_check_qlm
  wam_shm_profile
  demo_perf_env
  mkdir -p "$DEMO_LOG_DIR"
  demo_preflight_network
  demo_preflight_processes
  DEMO_LAUNCH_EXTRA="${DEMO_LAUNCH_EXTRA:-} heading_hold:=$WAM_HEADING_HOLD heading_hold_in_place:=$WAM_IN_PLACE_HOLD"
  DEMO_LAUNCH_EXTRA="$DEMO_LAUNCH_EXTRA command_handling:=$WAM_COMMAND_HANDLING"
  demo_release_sport_mode
  demo_launch_controller "$WAM_TASK_ID"
  demo_wait_mode
  demo_preflight_datapath
  demo_check_controller
  wam_service_check
  demo_view_start "$WAM_TASK_ID"
  wam_sim_camera_start
  demo_confirm
  wam_bridge_start
  wam_bridge_ready || demo_die "the bridge is not running, so the robot is not stood up (log $WAM_BRIDGE_LOG)"
  demo_stage "[4/5] Policy running (Ctrl+C to stop the robot)"
  local result_file="$DEMO_LOG_DIR/wam_entry_$(date +%Y%m%d_%H%M%S)_result.json"
  # shellcheck disable=SC2086
  "${DEMO_TASKSET[@]}" python3 "$DEMO_REPO/scripts/demo_sequence.py" --demo wam_entry --ee $WAM_EE \
    --result-file "$result_file" "$@" || demo_die "entering POLICY failed (exit $?; $result_file)"
  demo_say "controller in POLICY: the bridge drives the robot"
  local rc=0
  wam_bridge_wait || rc=$?
  demo_say "bridge exit code $rc (0 budget reached, 2 controller left POLICY, 4 could not start, 6 observation missing, 7 policy server failure, 8 prediction refused, 130 interrupted)"
  exit "$rc"
}

main "$@"
