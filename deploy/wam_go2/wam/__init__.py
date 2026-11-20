"""WAM-Policy online bridge: the real-robot side of the qlm-live-v1 protocol.

The policy server (``python -m wam_policy.inference.serve`` of this repository)
is a qlm-live-v1 RPC server: reset, predict, feedback, end. In simulation QLM's
``SynchronousDriver`` is its client and freezes physical time while the model
computes. On the real Go2 this package plays the client: it keeps a logical
2 ms-tick clock that advances only while a command is held, executes every
16-command chunk on a 20 ms wall-clock schedule through the RAMBO controller,
resolves the four 80 ms feedback samples from buffered camera frames and
controller states, and holds zero velocity while the next prediction is made.

ROS-free modules (unit-tested without ROS):
  timeline      logical clock, chunk/step/sample schedule, nearest-sample lookup
  rootlink      controller state -> QLM root-link fields (they place the camera frames)
  observation   frame/state buffers -> protocol snapshots at the grid instants
  limits        prediction bounds (stop, never clamp)
  episode       the client driver (reset -> predict -> execute -> feedback -> end)
  node_support  message decoding, parameter mapping and the start gate of the ROS node
  mock_service  protocol-faithful stand-in for the policy server (tests, MuJoCo)
  settings      the default instruction, the 640x360 RGB profile, the policy names
  tasks         the ten QLM-Bench tasks (config/wam_tasks.json)
  model_check   checks of the released model directories before the server starts
The ROS node is wam_go2/nodes/wam_bridge_node.py.
"""
