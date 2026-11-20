#!/usr/bin/env bash
# Task open_room_door: "Press down the lever handle, then push the door open."
# One real-robot run of WAM-Policy on the Unitree Go2 (see run_wam.sh for the stages and outputs).
#
#   tasks/open_room_door.sh                        # the released OPSD student (default)
#   WAM_POLICY=teacher tasks/open_room_door.sh     # another released policy: opsd_student | mopd_student | teacher | specialist
#   WAM_MOCK=1 tasks/open_room_door.sh             # rehearsal without a model (the robot steps in place)
#
# Before: the robot lying down, the task's objects in front of it, the laptop on the robot's
# Ethernet; press Ctrl+C at any time to stop the robot.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run_wam.sh" open_room_door "$@"
