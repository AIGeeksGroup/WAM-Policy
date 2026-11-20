#!/usr/bin/env bash
# Task pull_open_drawer: "Slide your foot into the drawer handle opening and pull the drawer outward."
# One real-robot run of WAM-Policy on the Unitree Go2 (see run_wam.sh for the stages and outputs).
#
#   tasks/pull_open_drawer.sh                        # the released OPSD student (default)
#   WAM_POLICY=teacher tasks/pull_open_drawer.sh     # another released policy: opsd_student | mopd_student | teacher
#   WAM_MOCK=1 tasks/pull_open_drawer.sh             # rehearsal without a model (the robot steps in place)
#
# Before: the robot lying down, the task's objects in front of it, the laptop on the robot's
# Ethernet; press Ctrl+C at any time to stop the robot.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run_wam.sh" pull_open_drawer "$@"
