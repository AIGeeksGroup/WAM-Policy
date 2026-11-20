"""Shared constants of the controller on the Unitree Go2.

Every value here matches its source (line references below):
- RAMBO env constants: rambo/source/isaaclab_tasks/isaaclab_tasks/direct/rambo_quadruped/qp_env.py
- Go2 hardware constants: unitree_sdk2 / unitree_ros2 examples and go2_description.urdf
Do not tune these blindly; the RAMBO values must match the training-time
configuration of the checkpoint being deployed.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Joint ordering
#
# RAMBO / IsaacLab order (BFS, qp_env.py:1422-1425):
#   [FL_hip, FR_hip, RL_hip, RR_hip,
#    FL_thigh, FR_thigh, RL_thigh, RR_thigh,
#    FL_calf, FR_calf, RL_calf, RR_calf]
# Go2 hardware order (unitree_sdk2 go2/defines.h JointIndex, Unitree Go2 low-level control docs):
#   [FR_hip(0), FR_thigh(1), FR_calf(2),
#    FL_hip(3), FL_thigh(4), FL_calf(5),
#    RR_hip(6), RR_thigh(7), RR_calf(8),
#    RL_hip(9), RL_thigh(10), RL_calf(11)]
# ---------------------------------------------------------------------------
RAMBO_JOINT_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
GO2_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
# RAMBO_TO_GO2[i] = index in Go2 motor array for RAMBO joint i
RAMBO_TO_GO2 = np.array([GO2_JOINT_NAMES.index(n) for n in RAMBO_JOINT_NAMES])
# GO2_TO_RAMBO[i] = index in RAMBO array for Go2 motor i
GO2_TO_RAMBO = np.array([RAMBO_JOINT_NAMES.index(n) for n in GO2_JOINT_NAMES])

# ---------------------------------------------------------------------------
# RAMBO controller constants (qp_env.py module-level constants)
# ---------------------------------------------------------------------------
NOMINAL_BASE_HEIGHT = 0.3          # qp_env.py:29
FOOT_HEIGHT = 0.15                 # qp_env.py:30
FOOT_CLEARANCE = 0.0               # qp_env.py:31

BASE_ACTION_SCALE = 5.0            # qp_env.py:39
JOINT_ACTION_SCALE = 0.15          # qp_env.py:42
NUM_ACTIONS = 18                   # 6 base acc + 12 joint offsets
ACTION_SCALE = np.array([BASE_ACTION_SCALE] * 6 + [JOINT_ACTION_SCALE] * 12)

POS_LIMIT_MARGIN = 0.1             # qp_env.py:48 [rad]
TORQUE_LIMIT_SCALE = 0.9           # qp_env.py:49
# qp_env.py:623-625 (hip x4, thigh x4, calf x4 in RAMBO grouping)
JOINT_TORQUE_LIMIT_RAW = np.array([23.7] * 8 + [45.43] * 4)
JOINT_TORQUE_LIMIT = JOINT_TORQUE_LIMIT_RAW * TORQUE_LIMIT_SCALE

POS_ALPHA = 0.5                    # qp_env.py:52-55 (EMA on desired pos)
TOR_ALPHA = 0.5                    # EMA on feedforward torque

CONTROL_DT = 0.01                  # policy step: sim dt 0.002 * decimation 5 (100 Hz)
HISTORY_LENGTH = 5                 # qp_env.py:463
NUM_OBS_PER_STEP = 81              # qp_env.py:464
NUM_OBSERVATIONS = NUM_OBS_PER_STEP * HISTORY_LENGTH  # 405

# PD gains sent to the Go2 low-level controller during policy control
# (DelayedDCMotorCfg stiffness/damping, qp_env.py:395-421)
POLICY_KP = 40.0
POLICY_KD = 1.0

# Default joint angles, RAMBO order (qp_env.py:425-438)
DEFAULT_JOINT_POS = np.array([0.0] * 4 + [0.8] * 4 + [-1.5] * 4)

# Joint position limits in RAMBO order (go2_description.urdf; front/rear thigh differ)
JOINT_POS_LOWER = np.array(
    [-1.0472] * 4 + [-1.5708, -1.5708, -0.5236, -0.5236] + [-2.7227] * 4)
JOINT_POS_UPPER = np.array(
    [1.0472] * 4 + [3.4907, 3.4907, 4.5379, 4.5379] + [-0.83776] * 4)

# Command ranges (qp_env.py:63-78); used for clipping user commands
VEL_RANGE_X = (-0.5, 0.5)
VEL_RANGE_Y = (-0.5, 0.5)
AVEL_RANGE_Z = (-0.5, 0.5)
FL_POS_X = (0.1934, 0.5)
FL_POS_Y = (0.0, 0.2)
FL_POS_Z = (0.0, 0.4)
FL_FORCE_LIM = 30.0
DEFAULT_EE_POS_CMD = np.array([0.1934, 0.142, 0.05])  # qp_env.py:660-662

# ---------------------------------------------------------------------------
# Kinematic constants (qp_env.py:480-486, "values taken from urdf files";
# verified against go2_description.urdf joint origins)
# ---------------------------------------------------------------------------
HIP_OFFSET = np.array([
    [0.1934, 0.0465, 0.0],    # FL
    [0.1934, -0.0465, 0.0],   # FR
    [-0.1934, 0.0465, 0.0],   # RL
    [-0.1934, -0.0465, 0.0],  # RR
])
LINK_LENGTHS = np.array([0.0955, 0.213, 0.213])  # l_hip(abduction offset), l_thigh, l_calf
L_HIP_SIGN = np.array([1.0, -1.0, 1.0, -1.0])    # left legs +, right legs -

# Raibert heuristic hip positions (qp_env.py:571-574; FR row carries the
# upstream "TODO" value [0.1934, 0.0, 0.0] on purpose - it is what the policy
# was trained with; do NOT "fix" it without retraining)
HIP_POSITIONS_IN_BODY_FRAME = np.array([
    [0.1934, 0.142, 0.0],    # FL
    [0.1934, 0.0, 0.0],      # FR (upstream TODO, kept as trained)
    [-0.1934, 0.142, 0.0],   # RL
    [-0.1934, -0.142, 0.0],  # RR
])

DESIRED_JOINT_POS_STANCE = np.array([0.0] * 4 + [0.8] * 4 + [-1.5] * 4)  # qp_env.py:559-563
DESIRED_JOINT_POS_SWING = np.array([0.0] * 4 + [1.1] * 4 + [-2.2] * 4)   # qp_env.py:564-568

# ---------------------------------------------------------------------------
# QP constants (qp_env.py:577-587, qp_torque_optimizer.py:53-57)
# ---------------------------------------------------------------------------
QP_BASE_POSITION_KP = np.array([50.0, 50.0, 50.0])
QP_BASE_POSITION_KD = np.array([10.0, 10.0, 10.0])
QP_BASE_ORIENTATION_KP = np.array([50.0, 50.0, 50.0])
QP_BASE_ORIENTATION_KD = np.array([10.0, 10.0, 10.0])
QP_WEIGHT_DDQ = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
QP_WEIGHT_GRF = 1e-4
QP_WEIGHT_EE_FORCE = 1.0
QP_FRICTION_COEF = 0.6
QP_BODY_MASS = 16.0870                                   # matches go2 urdf total
QP_BODY_INERTIA = np.diag(np.array([0.14, 0.35, 0.35]) * 1.5)
QP_ACC_CLIP_LIN = 30.0
QP_ACC_CLIP_ANG = 20.0
QP_FZ_MAX = 130.0
QP_FZ_MIN = 10.0
QP_EPS = 1e-3
QP_GRAVITY = 9.8   # solve_grf_qpth uses 9.8 (gravity comp uses 9.81)

# ---------------------------------------------------------------------------
# Gait: "walking" contact sequence (qp_env.py:530-548)
# Each entry: [mode, duration_s, init_phase, gait_duration_s, stance_ratio]
# ---------------------------------------------------------------------------
WALKING_CONTACT_SEQUENCE = {
    "FL": [["stance", 1.0, 0.0, 0.0, 0.0], ["swing", 9.0, 0.0, 0.0, 0.0]],
    "FR": [["stance", 1.0, 0.0, 0.0, 0.0], ["phase", 9.0, 0.0, 0.7, 0.8]],
    "RL": [["stance", 1.0, 0.0, 0.0, 0.0], ["phase", 9.0, 0.33, 0.7, 0.8]],
    "RR": [["stance", 1.0, 0.0, 0.0, 0.0], ["phase", 9.0, 0.67, 0.7, 0.8]],
}

# ---------------------------------------------------------------------------
# Gravity compensation point-mass model (go2_description.urdf inertials).
# Chain per leg, expressed relative to the hip joint frame; sigma = +1 for
# left legs (FL, RL), -1 for right legs (FR, RR); front = +1 (FL, FR), -1 rear.
# Fixed rotor links are merged into their URDF parent as in the simulator:
#   thigh_rotor -> hip body, calf_rotor -> thigh body.
# ---------------------------------------------------------------------------
GRAV_COMP = {
    "hip_mass": 0.678,
    "hip_com": np.array([-0.0054, 0.00194, -0.000105]),   # x sign flips for rear, y*sigma
    "thigh_rotor_mass": 0.089,
    "thigh_rotor_pos": np.array([0.0, -0.00015, 0.0]),    # in hip frame, y*sigma
    "thigh_mass": 1.152,
    "thigh_com": np.array([-0.00374, -0.0223, -0.0327]),  # in thigh frame, y*sigma
    "calf_rotor_mass": 0.089,
    "calf_rotor_pos": np.array([0.0, -0.03235, 0.0]),     # in thigh frame, y*sigma
    "calf_mass": 0.154,
    "calf_com": np.array([0.00548, -0.000975, -0.115]),   # in calf frame, y*sigma
    "foot_mass": 0.04,
    "foot_pos": np.array([0.0, 0.0, -0.213]),             # in calf frame
    "gravity": 9.81,                                       # qp_env.py:1317
}

# ---------------------------------------------------------------------------
# Go2 hardware constants (unitree_sdk2 examples, Unitree Go2 low-level control docs)
# ---------------------------------------------------------------------------
POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0
LOWLEVEL_FLAG = 0xFF
GO2_NUM_MOTORS = 20      # slots in LowCmd/LowState, first 12 used
FOOT_RADIUS = 0.022      # go2 foot collision sphere radius [m], for height estimate

# Soft emergency stop (unitree_sdk2 example/state_machine: kp=0, kd=2)
DAMPING_KD = 2.0

# Stand-up interpolation gains (unitree_sdk2 go2_stand_example: kp=60, kd=5)
STAND_KP = 60.0
STAND_KD = 5.0

# Stand-up phase 1 target: legs gathered under the body while still prone,
# hips at 0 so the feet sit at the default stance width before any load goes
# onto them. Thigh/calf are the official fold pose _targetPos_1 from
# unitree_sdk2 example/go2/go2_stand_example.cpp:61 (official stand is the
# same fold-then-rise). DELIBERATE DEVIATION: official _targetPos_1 keeps the
# rear hips at -0.2/+0.2; we fold them to 0 so the loaded rise needs zero hip
# sweep - on our high-friction floor a loaded hip sweep stalls instead of
# sliding (measured: rear hips pinned at +/-16 Nm for 30 s until the
# firmware overcurrent protection cut all motors). The belly carries the weight during
# the fold (measured tau ~0.2 Nm prone), so zeroing the hips there is nearly
# free. The rise target stays RAMBO's DEFAULT_JOINT_POS (policy training
# pose), not the official stand pose.
GATHER_POSE = np.concatenate([np.zeros(4), np.full(4, 1.36), np.full(4, -2.65)])
GATHER_DURATION = 1.5  # [s]
# The fold is staged so the leg is SHORT before it sweeps: calves fold first
# (leg cannot reach the ground), then thighs/hips. A single linear blend from
# a stretched-leg pose (post-collapse "sploot") would ram the foot 13-16 cm
# into the ground and rake it sideways at kp=60 (verified by FK path scan).
GATHER_CALF_FRACTION = 0.4  # calves reach the fold in the first 40 %
