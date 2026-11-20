#!/usr/bin/env bash
# Task kick_ball_to_goal: "Kick the ball through the goal."
# One real-robot run of WAM-Policy on the Unitree Go2 (see run_wam.sh for the stages and outputs).
#
#   tasks/kick_ball_to_goal.sh                        # the released OPSD student (default)
#   WAM_POLICY=teacher tasks/kick_ball_to_goal.sh     # another released policy: opsd_student | mopd_student | teacher
#   WAM_MOCK=1 tasks/kick_ball_to_goal.sh             # rehearsal without a model (the robot steps in place)
#
# Before: the robot lying down, the task's objects in front of it, the laptop on the robot's
# Ethernet; press Ctrl+C at any time to stop the robot.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run_wam.sh" kick_ball_to_goal "$@"
