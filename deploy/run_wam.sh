#!/usr/bin/env bash
# run_wam.sh - run one WAM-Policy task on the real Unitree Go2 (tasks/<task>.sh call this).
#
#   ./run_wam.sh <task> [demo_sequence.py arguments]    # e.g. ./run_wam.sh push_box
#   ./run_wam.sh --list                                 # the tasks
#
# Stages (a header each on the terminal, between dashed lines; everything else goes to run.log):
#   [1/5] the policy server (scripts/wam/policy_server.sh: wam_policy.inference.serve of this
#         repository, 127.0.0.1:WAM_PORT) starts loading the model in the background
#   [2/5] this package is built into DEPLOY_WS and verified (scripts/build.sh); then the run waits
#         until the policy server answers
#   [3/5] robot bring-up (scripts/demo/wam_task.sh); press Enter to let the robot stand up
#   [4/5] the policy drives the robot until its chunk budget, a stop condition or Ctrl+C
#   [5/5] shutdown: DAMP, bridge, cameras and controller stopped, then the policy server
#
# Ctrl+C at any time stops the run; once the robot is up it is put into damping first.
# Exit code: 0 chunk budget reached, 1 execution error, 2 the controller left POLICY (DAMP or a safety
# trip), 4 could not start, 6 observation gap, 7 policy server failure or timeout, 8 prediction refused,
# 130 interrupted (Ctrl+C); 64 bad settings, 3 the policy server could not be used.
#
# Every run gets result/<task>/<YYYYmmdd_HHMMSS>/ with run.log, run.json (task, policy, times,
# exit code), logs/ (controller, bridge, policy server, camera processes), server/ (the policy
# server's request and event logs), bridge/ (episode records) and view/ (camera_ego.mp4,
# camera_d435i.mp4 and their time-sync records).
#
# Env (also read from deploy/config/deploy.env): WAM_POLICY (the server's --policy: opsd_student
# default | mopd_student | teacher | specialist), WAM_MOCK=1 (rehearsal with the scripted mock
# server instead of a model: the robot steps in place), WAM_PORT (8862),
# WAM_COMMAND_HANDLING (sim default | rambo), WAM_SERVER_WAIT_S (900), WAM_VERBOSE=1 (also show
# run.log on the terminal), WAM_RUN_ROOT (default deploy/result), plus the settings of
# scripts/wam/policy_server.sh and scripts/demo/wam_task.sh.
set -euo pipefail
trap '' PIPE      # a write to a terminal or log pipe that went away fails instead of killing the run

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/load_config.sh
source "$HERE/scripts/load_config.sh"
BUILD="${WAM_RUN_BUILD:-$HERE/scripts/build.sh}"                       # tests substitute fakes
POLICY_SERVER="${WAM_RUN_POLICY_SERVER:-$HERE/scripts/wam/policy_server.sh}"
WAM_TASK="${WAM_RUN_TASK:-$HERE/scripts/demo/wam_task.sh}"
SS_BIN="${WAM_RUN_SS:-ss}"
PROC_NET_TCP="${WAM_RUN_PROC_NET:-/proc/net/tcp /proc/net/tcp6}"
WAM_SERVER_WAIT_S="${WAM_SERVER_WAIT_S:-900}"
WAM_RUN_ROOT="${WAM_RUN_ROOT:-$HERE/result}"
WAM_VERBOSE="${WAM_VERBOSE:-0}"
WAM_COMMAND_HANDLING="${WAM_COMMAND_HANDLING:-sim}"                       # sim | rambo (scripts/demo/wam_task.sh)
export WAM_COMMAND_HANDLING
WAM_MOCK="${WAM_MOCK:-0}"
WAM_HOST=127.0.0.1
TASK=""
RUN_DIR=""
SERVER_PID=""
SERVER_LOG=""
TERM_OPEN=0
RULE="------------------------------------------------------------"

# ---------------------------------------------------------------------------------------------- output
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" 2>/dev/null || true; }      # run.log
term() {                                                                          # the terminal
  if (( TERM_OPEN )); then { printf '%s\n' "$*" >&3; } 2>/dev/null || true; else printf '%s\n' "$*" 2>/dev/null || true; fi
}
stage() { (( TERM_OPEN )) && say "$*"; term "$RULE"; term "$*"; }
shown_path() { case "$1" in "$PWD"/*) printf '%s' "${1#"$PWD"/}" ;; *) printf '%s' "$1" ;; esac; }
die() {
  local rc="${2:-1}"
  if (( TERM_OPEN )); then
    say "ERROR: $1"
    term "      Error: $1"
  else
    term "error: $1"
  fi
  exit "$rc"
}
usage() { awk 'NR > 1 { if (!/^#/) exit; print substr($0, 3) }' "${BASH_SOURCE[0]}"; }

tasks_py() { PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" python3 -m wam_go2.wam.tasks "$@"; }

run_json() {  # run_json key=value ... : merged into $RUN_DIR/run.json (numbers / JSON values kept typed)
  python3 - "$RUN_DIR/run.json" "$@" <<'EOF'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text()) if path.exists() else {}
for item in sys.argv[2:]:
    key, value = item.split("=", 1)
    try:
        data[key] = json.loads(value) if value[:1] in '[{"-0123456789' or value in ("true", "false", "null") else value
    except ValueError:
        data[key] = value
path.write_text(json.dumps(data, indent=2) + "\n")
EOF
}

# ---------------------------------------------------------------------------------------------- policy server
server_reachable() { timeout 3 bash -c "exec 3<>/dev/tcp/$WAM_HOST/$WAM_PORT" 2>/dev/null; }

proc_listener_pids() {  # without ss: LISTEN sockets on the port from /proc/net/tcp*, their inodes -> our pids
  local hexport inodes args=() ino
  hexport="$(printf '%04X' "$WAM_PORT")"
  # shellcheck disable=SC2086
  inodes="$(awk -v suffix=":$hexport" 'FNR > 1 && $4 == "0A" && substr($2, length($2) - length(suffix) + 1) == suffix { print $10 }' \
            $PROC_NET_TCP 2>/dev/null)"
  [[ -n "$inodes" ]] || return 0
  for ino in $inodes; do args+=(-o -lname "socket:\\[$ino\\]"); done
  find /proc/[0-9]*/fd -maxdepth 1 \( "${args[@]:1}" \) -printf '%h\n' 2>/dev/null | sed 's|^/proc/||; s|/fd$||' | sort -u
}

listeners() {  # "pid cmd" per process of ours listening on the port: ss, else /proc; "" when none can be seen
  local pids="" pid
  if command -v "$SS_BIN" >/dev/null 2>&1; then
    pids="$("$SS_BIN" -H -ltnp "sport = :$WAM_PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)"
  elif [[ -r "${PROC_NET_TCP%% *}" ]]; then
    pids="$(proc_listener_pids)"
  fi
  for pid in $pids; do printf '%s %s\n' "$pid" "$(tr '\0\n' '  ' < "/proc/$pid/cmdline" 2>/dev/null)"; done
}

refuse_busy_port() {  # something already answers on the port: never used, whatever it is
  local who; who="$(listeners)"
  say "listening on $WAM_HOST:$WAM_PORT: ${who:-(not visible)}"
  die "$WAM_HOST:$WAM_PORT is already in use (${who:+pid ${who%% *}; }a server from an earlier run?): stop it or set WAM_PORT" 3
}

server_is_ours() {  # every visible listener on the port belongs to the session of the server started here
  local who sid pid
  who="$(listeners)"
  [[ -n "$who" ]] || return 1
  sid="$(ps -o sid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')"
  [[ -n "$sid" ]] || return 1
  for pid in $(echo "$who" | cut -d' ' -f1); do
    [[ "$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')" == "$sid" ]] || return 1
  done
  return 0
}

start_server() {
  server_reachable && refuse_busy_port
  [[ -x "$POLICY_SERVER" ]] || die "missing $POLICY_SERVER" 3
  SERVER_LOG="$RUN_DIR/logs/policy_server.log"
  # its own session: Ctrl+C in this terminal ends the run, and this script stops the server
  WAM_SERVER_LOG_DIR="$RUN_DIR/server" setsid env --default-signal=INT,QUIT "$POLICY_SERVER" "$TASK" \
    > "$SERVER_LOG" 2>&1 3>&- &
  SERVER_PID=$!
  say "policy server pid $SERVER_PID (log $SERVER_LOG)"
}

wait_server() {
  local waited=0 t0=$SECONDS told=0
  while ! server_reachable; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      wait "$SERVER_PID" 2>/dev/null || true
      SERVER_PID=""
      tail -n 20 "$SERVER_LOG" 2>/dev/null | sed 's/^/    /' || true
      term "$(tail -n 3 "$SERVER_LOG" 2>/dev/null | cut -c1-150 | sed 's/^/      | /')"
      die "the policy server exited before it was ready (log: $SERVER_LOG)" 3
    fi
    if (( ! told )); then term "      Waiting for the model to load ..."; told=1; fi
    waited=$((SECONDS - t0))
    (( waited < WAM_SERVER_WAIT_S )) || { stop_server; die "the policy server was not ready within ${WAM_SERVER_WAIT_S} s (log: $SERVER_LOG)" 3; }
    sleep 1
  done
  server_is_ours || { stop_server; die "cannot confirm that the server started here answers on $WAM_HOST:$WAM_PORT (another process on the port?)" 3; }
  term "      Policy server ready ($((SECONDS - SERVER_T0)) s)"
}

stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    say "stopping the policy server (pid $SERVER_PID)"
    kill -INT -- "-$SERVER_PID" 2>/dev/null || kill -INT "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 200); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.1; done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
      for _ in $(seq 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.1; done
    fi
    kill -0 "$SERVER_PID" 2>/dev/null && { kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true; }
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

# ---------------------------------------------------------------------------------------------- run
check_settings() {  # everything that would otherwise fail only after the model has loaded
  [[ -n "$DEMO_NIC" ]] || die "DEMO_NIC is not set: the wired interface to the robot, in config/deploy.env (ip -br link)" 64
  local ipv4='^([0-9]{1,3}\.){3}[0-9]{1,3}$'
  [[ "$DEMO_ROBOT_IP" =~ $ipv4 ]] || die "DEMO_ROBOT_IP is not set: the robot's Ethernet address, in config/deploy.env (docs/network.md)" 64
  [[ "$DEMO_LAPTOP_IP" =~ $ipv4 ]] || die "DEMO_LAPTOP_IP is not set: the laptop's address on the robot cable, in config/deploy.env" 64
  ip link show dev "$DEMO_NIC" >/dev/null 2>&1 || die "network interface DEMO_NIC=$DEMO_NIC not found (ip -br link)" 64
  [[ -x "$DEPLOY_VENV/bin/python" ]] || die "deploy venv not found: DEPLOY_VENV=$DEPLOY_VENV (README: Installation)" 64
  local name value
  for name in WAM_HEADING_HOLD WAM_IN_PLACE_HOLD; do
    value="${!name:-}"
    [[ -z "$value" || "$value" == true || "$value" == false ]] || die "$name=$value: must be true or false" 64
  done
  [[ -z "${WAM_MAX_CHUNKS:-}" || "$WAM_MAX_CHUNKS" =~ ^[1-9][0-9]*$ ]] \
    || die "WAM_MAX_CHUNKS=$WAM_MAX_CHUNKS: must be a positive integer" 64
  [[ -z "${WAM_SAVE_IMAGES:-}" || "$WAM_SAVE_IMAGES" == 0 || "$WAM_SAVE_IMAGES" == 1 ]] \
    || die "WAM_SAVE_IMAGES=$WAM_SAVE_IMAGES: must be 0 or 1" 64
  for name in WAM_MAX_WALL_S WAM_SETTLE_S WAM_START_TIMEOUT_S WAM_BRIDGE_READY_S WAM_GO2_LATENCY_MS; do
    value="${!name:-}"
    [[ -z "$value" || "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "$name=$value: must be a non-negative number" 64
  done
  [[ -z "${WAM_CALIB:-}" || -r "$WAM_CALIB" ]] || die "WAM_CALIB=$WAM_CALIB is not readable" 64
  return 0
}

outcome() {  # the meaning of a run's exit code
  case "$1" in
    0) echo "chunk budget reached" ;;
    2) echo "the controller left POLICY (stop request or safety trip)" ;;
    130|143|129) echo "interrupted" ;;
    *) echo "failed" ;;
  esac
}

finish() {
  local rc="$1"
  trap "" INT TERM HUP PIPE
  set +e
  stop_server
  [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] || return 0
  run_json "ended_at=$(date -Iseconds)" "exit_code=$rc"
  say "finished: exit code $rc"
  term "$RULE"
  term "Done: $(outcome "$rc") (exit $rc)"
  term "Results: $(shown_path "$RUN_DIR")"
  case "$rc" in 0|2|129|130|143) ;; *) term "Log: $(shown_path "$RUN_DIR/run.log")" ;; esac
  term "$RULE"
}

setup_run() {
  local stamp n=1
  stamp="$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$WAM_RUN_ROOT/$TASK"
  RUN_DIR="$WAM_RUN_ROOT/$TASK/$stamp"
  while ! mkdir "$RUN_DIR" 2>/dev/null; do          # two runs within one second: -2, -3, ...
    n=$((n + 1))
    (( n <= 99 )) || die "cannot create a run directory under $WAM_RUN_ROOT/$TASK"
    RUN_DIR="$WAM_RUN_ROOT/$TASK/$stamp-$n"
  done
  RUN_DIR="$(cd "$RUN_DIR" && pwd)"
  mkdir -p "$RUN_DIR/logs"
  # the terminal stays on fd 3 for the stage lines; everything else goes to run.log
  exec 3>&1
  TERM_OPEN=1
  # (the tee of WAM_VERBOSE=1 ignores Ctrl+C: it must outlive the shutdown it logs)
  if [[ "$WAM_VERBOSE" == 1 ]]; then exec > >(trap '' INT TERM HUP; exec tee -i -a "$RUN_DIR/run.log") 2>&1
  else exec >>"$RUN_DIR/run.log" 2>&1; fi
}

main() {
  case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    --list) tasks_py list | awk -F'\t' '{ printf "  %-20s %s\n", $1, $2 }'; exit 0 ;;
    ""|-*) usage >&2; echo; echo "tasks:"; tasks_py list | awk -F'\t' '{ printf "  %-20s %s\n", $1, $2 }'; exit 64 ;;
  esac
  TASK="$1"; shift
  local instruction
  instruction="$(tasks_py get "$TASK" instruction 2>/dev/null)" \
    || die "unknown task '$TASK' (./run_wam.sh --list)" 64
  case "$WAM_POLICY" in
    opsd_student|mopd_student|teacher|specialist) ;;
    *) die "WAM_POLICY=$WAM_POLICY: must be opsd_student, mopd_student, teacher or specialist" 64 ;;
  esac
  [[ "$WAM_MOCK" == 0 || "$WAM_MOCK" == 1 ]] || die "WAM_MOCK=$WAM_MOCK: must be 0 or 1" 64
  [[ "$WAM_PORT" =~ ^[0-9]+$ ]] || die "WAM_PORT=$WAM_PORT is not a port number" 64
  case "$WAM_COMMAND_HANDLING" in
    sim|rambo) ;;
    *) die "WAM_COMMAND_HANDLING=$WAM_COMMAND_HANDLING: must be sim (as the training simulation) or rambo" 64 ;;
  esac
  [[ -x "$BUILD" && -x "$WAM_TASK" ]] || die "missing $BUILD or $WAM_TASK" 1
  check_settings

  setup_run
  local rc=0 interrupted=""
  # the first signal ignores the next ones, so nothing can skip finish (which stops the policy server)
  trap 'rc=$?; trap "" INT TERM HUP PIPE; finish "$rc"; exit "$rc"' EXIT
  trap 'trap "" INT TERM HUP; exit 130' INT
  trap 'trap "" INT TERM HUP; exit 143' TERM
  trap 'trap "" INT TERM HUP; exit 129' HUP
  local shown="$WAM_POLICY"; [[ "$WAM_MOCK" == 0 ]] || shown="mock (rehearsal, no model)"
  term "$RULE"
  term "WAM-Policy | task $TASK | policy $shown"
  term "\"$instruction\""
  term "Run: $(shown_path "$RUN_DIR")"
  run_json "task=$TASK" "instruction=\"$instruction\"" "policy=$WAM_POLICY" "mock=$([[ "$WAM_MOCK" == 1 ]] && echo true || echo false)" \
    "command_handling=$WAM_COMMAND_HANDLING" \
    "started_at=$(date -Iseconds)"
  say "task $TASK, policy $WAM_POLICY, command handling $WAM_COMMAND_HANDLING, port $WAM_PORT, run $RUN_DIR"

  SERVER_T0=$SECONDS
  if [[ "$WAM_MOCK" == 1 ]]; then
    stage "[1/5] Policy server: scripted mock (rehearsal, no model)"
    server_reachable && refuse_busy_port
  else
    stage "[1/5] Policy server: loading $WAM_POLICY in the background"
    start_server
  fi
  stage "[2/5] Build: ROS 2 package"
  "$BUILD" || die "build failed"
  [[ "$WAM_MOCK" == 1 ]] || wait_server

  # the robot side handles Ctrl+C / TERM / HUP itself (bridge stop, DAMP, controller stop): its exit code is the run's
  trap 'interrupted=1' INT TERM HUP
  WAM_TASK_ID="$TASK" WAM_INSTRUCTION="$instruction" WAM_OUTPUT_DIR="$RUN_DIR/bridge" DEMO_VIEW_DIR="$RUN_DIR/view" \
    DEMO_LOG_DIR="$RUN_DIR/logs" WAM_MOCK="$WAM_MOCK" \
    "$WAM_TASK" "$@" || rc=$?
  trap 'trap "" INT TERM HUP; exit 130' INT; trap 'trap "" INT TERM HUP; exit 143' TERM
  trap 'trap "" INT TERM HUP; exit 129' HUP
  [[ -z "$interrupted" ]] || run_json "interrupted=true"
  exit "$rc"
}

main "$@"
