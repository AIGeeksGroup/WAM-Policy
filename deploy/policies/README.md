# RAMBO controller checkpoint

The controller (`wam_go2/control/`) runs a RAMBO quadruped policy, task `quadruped`
(`Isaac-RAMBO-Quadruped-Go2-v0`, observation 405, action 18) with the front-left leg as the
manipulation leg; the deployment constants in `wam_go2/constants.py` and `wam_go2/tasks.py` match
that task's training environment.

No checkpoint is included. `scripts/fetch_controller.sh` downloads `model_2000.pt` from the
QLM-Bench dataset (`assets/rambo-go2-controller/iteration-2000/runtime/`, CC BY-NC 4.0) to
`policies/quadruped/model_2000.pt` (git-ignored) and checks its SHA-256. `RAMBO_CONTROLLER` in
`configs/site.env` points the scripts at another file instead.
