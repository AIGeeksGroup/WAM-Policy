"""Main deployment node: Go2 <-> RAMBO closed loop.

FSM (requests on rambo/fsm_cmd, std_msgs/String):
    DAMP   -- STAND -->  STAND  -- POLICY -->  POLICY
    any state -- DAMP --> DAMP (soft e-stop: kp=0, kd=2)
    safety trip -> DAMP (latched; a REARM request clears it)
    stopping the node (SIGINT / Ctrl+C) sends damping frames before it exits

Loops:
- /lowstate subscription (~500 Hz): parse, run the state-estimator KF
- control timer (100 Hz): FSM step, publish /lowcmd

Commands while in POLICY:
- /cmd_vel (geometry_msgs/Twist), plus /rambo/ee_pos_cmd, /rambo/ee_force_cmd
  (geometry_msgs/Point / Vector3) for the FL end-effector task.
"""

import math
import os
import signal
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Twist, Vector3
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, Float64MultiArray, String

from ..adapter.command_adapter import CommandError, adapt_command, policy_gains
from ..constants import (
    DEFAULT_JOINT_POS,
    GATHER_CALF_FRACTION,
    GATHER_DURATION,
    GATHER_POSE,
    STAND_KD,
    STAND_KP,
)
from ..control import kinematics
from ..control.math_utils import quat_rotate_inverse
from ..control.policy import RamboPolicy
from ..control.rambo_controller import RamboController, command_handling_options
from ..control.robot_state import RobotState
from ..realtime import freeze_startup_heap
from ..estimator.state_estimator import StateEstimator
from ..go2.interface import Go2Interface
from ..go2.motion_switcher import MotionSwitcherClient
from ..safety.hardware_health import hardware_fault_reason
from ..safety.safety_manager import SafetyManager
from ..control.heading_hold import HeadingHold, yaw_from_quat
from ..control.force_command import ForceCommandWatchdog
from .. import state_stream
from ..safety.state_freshness import StateFreshnessMonitor
from ..safety.wired_link import WiredLinkMonitor
from ..tasks import get_task
from ..run_metadata import run_metadata

DAMP = "DAMP"
STAND = "STAND"
POLICY = "POLICY"

# The stand-up gather pose / staging constants live in constants.py; the
# STAND branch below runs the verified gather-then-rise.


class RamboGo2Controller(Node):
    def __init__(self):
        super().__init__("wam_go2_controller")

        # Parameters are cached at startup. Reject live edits rather than
        # reporting values that differ from the actual control objects.
        declared_names = []

        def declare(name, value):
            parameter = self.declare_parameter(
                name, value, descriptor=ParameterDescriptor(read_only=True))
            declared_names.append(name)
            return parameter
        # ---- parameters
        declare("task", "quadruped")  # RAMBO task (tasks.py)
        declare("policy_checkpoint", "")
        declare("control_rate", 100.0)
        declare("stand_duration", 2.0)
        declare("auto_release_sport_mode", True)
        declare("lowstate_timeout", 0.1)
        declare("source_stale_timeout", 0.05)
        declare("control_gap_timeout", 0.05)
        declare("estimator_timeout", 0.05)
        declare("max_tilt_gravity_z", -0.3)
        declare("stand_kp", STAND_KP)
        declare("stand_kd", STAND_KD)
        declare("gravity_compensation", True)
        declare("joint_limit_soft_factor", 1.0)
        # rambo (default: clipped to the RAMBO training ranges) | sim (WAM runs: unclipped, exactly as the
        # QLM-Bench MuJoCo simulation of the training data)
        declare("command_handling", "rambo")
        # directory for a 100 Hz trace of every POLICY run (raw policy
        # action, JPC target, commanded q / tau_ff, GRF, estimator state),
        # saved as policy_trace_<unix>.npz when POLICY ends; "" = off
        declare("policy_trace_dir", "")
        # Handshake compliance (RAMBO paper Sec. V-D): while rambo/ee_compliance
        # is true in POLICY the manipulation-leg joints (FL) run
        # these PD gains instead of the training 40 / 1. The QP feedforward
        # (gravity compensation + stance torque) still holds the leg up.
        # Values chosen in MuJoCo (sim/mj_sim.py quad_demo).
        declare("ee_compliant_kp", 15.0)
        declare("ee_compliant_kd", 0.8)
        # soft: while compliant the manipulation leg also loses the policy
        # joint residual and the QP stance torque (gravity compensation
        # stays), so the hand only works against the lowered PD
        declare("ee_compliant_soft", True)
        # /cmd_vel older than this is treated as zero
        # (a dead publisher must not leave the robot walking; the demo
        # sequencer republishes at 20 Hz)
        declare("cmd_vel_timeout", 0.5)
        # /rambo/ee_force_cmd staleness: without it a publisher that dies
        # mid-run leaves the last force applied for the rest of the POLICY run
        # (30 N on 15.2 kg is 2 m/s^2 of unbalanced thrust). 0 = off.
        declare("ee_force_cmd_timeout", 0.5)
        # quadruped heading hold: the three-leg gait yaws right ~1-2 deg/s on
        # its own; while walking straight (operator yaw ~0) a PI loop on the
        # IMU yaw commands the yaw rate instead (wam_go2/control/heading_hold.py)
        # rambo/state (std_msgs/Float64MultiArray, wam_go2/state_stream.py layout): the
        # estimator/IMU state once per control tick for consumers such as the WAM bridge
        # (additive; the control path never depends on it)
        declare("publish_state", True)
        declare("heading_hold", True)
        # ... and with the whole command ~zero (stepping in place: the turn happens there too).
        # Off by default; WAM runs switch it on
        declare("heading_hold_in_place", False)
        declare("heading_hold_kp", 1.0)
        declare("heading_hold_ki", 0.3)
        declare("heading_hold_max_wz", 0.3)
        declare("heading_hold_deadband_deg", 1.0)
        # Host <-> robot over the Ethernet cable ONLY: the LowState bytes must
        # arrive on this NIC (set it to your wired interface, see `ip -br link`)
        # or the node refuses to move. "none" disables (mock robot on loopback).
        declare("wired_nic", "eth0")
        declare("wired_min_mbps", 0.3)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.task = get_task(str(p("task")))
        self._stand_duration = float(p("stand_duration"))
        self._stand_kp = float(p("stand_kp"))
        self._stand_kd = float(p("stand_kd"))
        control_rate = float(p("control_rate"))
        self._dt = 1.0 / control_rate
        self._trace_dir = os.path.expanduser(str(p("policy_trace_dir")))
        self._trace = []
        # ticks that ran on an unchanged LowState frame
        self._freshness = StateFreshnessMonitor(self._dt)
        self._prev_mode = None
        self._compliant_kp = float(p("ee_compliant_kp"))
        self._compliant_kd = float(p("ee_compliant_kd"))
        self._compliant_soft = bool(p("ee_compliant_soft"))
        self._ee_compliant = False     # POLICY only; forced off on entry/exit
        self._cmd_vel_timeout = float(p("cmd_vel_timeout"))
        self._ee_force_timeout = float(p("ee_force_cmd_timeout"))
        self._force_req = np.zeros(3)  # the FL force last requested on rambo/ee_force_cmd [N]
        self._ee_force_time = None     # monotonic time of the last force message
        self._heading_hold = HeadingHold(
            enabled=bool(p("heading_hold")),
            kp=float(p("heading_hold_kp")), ki=float(p("heading_hold_ki")),
            max_wz=float(p("heading_hold_max_wz")),
            deadband_deg=float(p("heading_hold_deadband_deg")), dt=self._dt,
            in_place=bool(p("heading_hold_in_place")))
        self.get_logger().info(
            f"heading hold {'ON' if self._heading_hold.enabled else 'OFF'}: while walking straight the "
            f"IMU yaw at walk start is held (kp {self._heading_hold.kp:g}, ki {self._heading_hold.ki:g}, "
            f"|wz| <= {self._heading_hold.max_wz:g} rad/s, deadband "
            f"{math.degrees(self._heading_hold.deadband):.1f} deg); operator yaw input releases it; "
            f"in place (zero command) {'ON' if self._heading_hold.in_place else 'OFF'}")
        self._wired = WiredLinkMonitor(str(p("wired_nic")), float(p("wired_min_mbps")))
        self._wired_logged = -1e9
        self._vel_stale_logged = False

        # ---- components
        checkpoint = str(p("policy_checkpoint")) or self._default_checkpoint()
        self.get_logger().info(
            f"task={self.task.name} ({self.task.gym_id}, obs={self.task.num_obs}); "
            f"loading policy from {checkpoint}")
        policy = RamboPolicy(checkpoint, num_obs=self.task.num_obs)
        self._trace_metadata = run_metadata(checkpoint, {
            name: self.get_parameter(name).value
            for name in declared_names})
        handling = command_handling_options(str(p("command_handling")), self.task)
        self.controller = RamboController(
            policy,
            task=self.task,
            gravity_compensation=bool(p("gravity_compensation")),
            joint_limit_soft_factor=float(p("joint_limit_soft_factor")),
            **handling,
        )
        if handling["clip_commands"]:
            self.get_logger().info(
                f"command_handling={p('command_handling')}: commands clipped to the RAMBO training ranges "
                "(vx, vy, wz +-0.5; FL x 0.1934..0.5, y 0..0.2, z 0..0.4 m), zeroed while all feet are in stance")
        else:
            self.get_logger().info(
                f"command_handling={p('command_handling')}: as the QLM-Bench MuJoCo simulation - velocity and FL "
                "target unclipped, not zeroed in stance (the WAM bridge refuses implausible predictions)")
        self.estimator = StateEstimator(nominal_height=self.task.nominal_base_height)
        # Staleness watchdog of the FL force command (native9 fx/fy/fz).
        self._force_watchdog = ForceCommandWatchdog(command_timeout=self._ee_force_timeout)
        self.safety = SafetyManager(
            lowstate_timeout=float(p("lowstate_timeout")),
            source_stale_timeout=float(p("source_stale_timeout")),
            control_gap_timeout=float(p("control_gap_timeout")),
            estimator_timeout=float(p("estimator_timeout")),
            max_tilt_gravity_z=float(p("max_tilt_gravity_z")))
        self.go2 = Go2Interface(self, state_callback=self._on_state)
        self.motion_switcher = MotionSwitcherClient(self)

        # ---- command inputs
        self._topic_vel_cmd = np.zeros(3)
        self._topic_vel_time = None    # monotonic time of the last /cmd_vel
        self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(Point, "rambo/ee_pos_cmd", self._on_ee_pos, 10)
        self.create_subscription(Vector3, "rambo/ee_force_cmd", self._on_ee_force, 10)
        self.create_subscription(String, "rambo/fsm_cmd", self._on_fsm_cmd, 10)
        self.create_subscription(Bool, "rambo/ee_compliance",
                                 self._on_ee_compliance, 10)
        self._mode_pub = self.create_publisher(String, "rambo/mode", 10)
        self._state_pub = (self.create_publisher(Float64MultiArray, state_stream.TOPIC, 10)
                           if bool(p("publish_state")) else None)
        self._state_pub_failed = False

        # ---- FSM state
        self.mode = DAMP
        self._policy_time = 0.0
        self._stand_progress = 0.0
        self._stand_start_q = DEFAULT_JOINT_POS.copy()
        self._stand_hold_q = None
        self._stand_track_bad = 0
        self._gather_progress = 1.0
        self._last_est_tick = None
        self._last_contact = None  # resolved contact, shared estimator/controller
        self._released_sport_mode = False
        self._first_state_seen = False
        self.motion_switcher.start_release(allow_release=bool(p("auto_release_sport_mode")))
        self._fsm_request = None
        self._damp_requested = False
        self._last_control_time = None
        self._exit_reason = "startup"
        self._policy_fault = None

        # Physical control and its stop/watchdog checks must keep running if
        # use_sim_time changes or /clock pauses. Match the monotonic ages used
        # throughout the safety path instead of the node's ROS clock.
        self._control_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self._timer = self.create_timer(self._dt, self._control_tick,
                                        clock=self._control_clock)
        self.get_logger().info("wam_go2_controller up; state=DAMP; rambo/fsm_cmd: STAND, POLICY, DAMP, REARM")

    # ------------------------------------------------------------------
    @staticmethod
    def _bundled(rel, required=True):
        """Path of a package data file: installed share dir first, then the
        source tree (running from the repo without colcon install)."""
        candidates = []
        try:
            from ament_index_python.packages import get_package_share_directory
            candidates.append(os.path.join(
                get_package_share_directory("wam_go2_deploy"), rel))
        except Exception:
            pass
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.normpath(os.path.join(here, "..", "..", rel)))
        for c in candidates:
            if os.path.isfile(c):
                return c
        if required:
            raise RuntimeError(f"bundled file {rel} not found (looked in {candidates})")
        return ""

    def _default_checkpoint(self):
        """The task's checkpoint under the package's policies/ directory (downloaded separately)."""
        rel = os.path.join("policies", self.task.checkpoint_relpath)
        path = self._bundled(rel, required=False)
        if not path:
            raise RuntimeError(
                f"no policy_checkpoint given and {rel} was not found; run "
                "scripts/fetch_controller.sh and scripts/build.sh, or pass policy_checkpoint:=/path/model.pt")
        return path

    # ------------------------------------------------------------------
    # inputs
    def _on_cmd_vel(self, msg):
        self._topic_vel_cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self._topic_vel_time = time.monotonic()

    def _topic_velocity(self):
        """/cmd_vel for the controller: zero once the publisher has gone quiet."""
        vel = self._topic_vel_cmd
        stale = (self._topic_vel_time is None
                 or time.monotonic() - self._topic_vel_time > self._cmd_vel_timeout)
        if not stale:
            self._vel_stale_logged = False
            return vel
        if np.any(vel != 0.0) and not self._vel_stale_logged:
            self._vel_stale_logged = True
            self.get_logger().warn(
                f"/cmd_vel silent for > {self._cmd_vel_timeout:.1f} s while commanding "
                f"{np.round(vel, 2).tolist()}: velocity zeroed (publisher died?)")
        return np.zeros(3)

    def _on_ee_pos(self, msg):
        self.controller.set_commands(ee_pos_cmd=[msg.x, msg.y, msg.z], leg_slot=0)

    def _on_ee_force(self, msg):
        self._force_req = np.array([msg.x, msg.y, msg.z], dtype=float)
        self._ee_force_time = time.monotonic()
        self.controller.set_commands(ee_force_cmd=[msg.x, msg.y, msg.z], leg_slot=0)

    def _on_fsm_cmd(self, msg):
        request = msg.data.strip().upper()
        if request == "DAMP":
            self._damp_requested = True
            self._fsm_request = None
        elif not self._damp_requested:
            self._fsm_request = request

    def _on_ee_compliance(self, msg):
        want = bool(msg.data)
        if self.mode != POLICY:
            if want:
                self.get_logger().warn("ee compliance request ignored: not in POLICY")
            return
        self._set_ee_compliance(want)

    def _set_ee_compliance(self, on):
        """Lower / restore the manipulation-leg PD gains (POLICY only)."""
        if on == self._ee_compliant:
            return
        self._ee_compliant = on
        self.controller.ee_soft = bool(on and self._compliant_soft)
        if on:
            soft = ("; soft: no policy residual / QP torque on the leg, gravity comp only"
                    if self._compliant_soft else "")
            self.get_logger().info(
                f"ee compliance ON (FL kp {self._compliant_kp:.1f} / "
                f"kd {self._compliant_kd:.2f}; other joints 40 / 1{soft})")
        else:
            self.get_logger().info("ee compliance OFF")

    def _on_state(self, state):
        """LowState callback (~500 Hz): run the estimator."""
        # Go2Interface only calls us for an accepted forward source tick.
        # The modulo delta also handles uint32 wrap; reboot requires restart.
        dt = self.go2.accepted_dt
        self._last_est_tick = state.tick_ms

        # Contacts come from the gait schedule in POLICY; in every other mode the
        # robot rests on (or rises from) all four feet. No foot contact force is
        # read: the policy has no foot-force input.
        if self.mode == POLICY:
            contact = np.asarray(self.controller.contact_generator.desired_contact_state, dtype=bool)
        else:
            contact = np.ones(4, dtype=bool)
        self._last_contact = contact
        self.estimator.update(dt, state.quat, state.gyro, state.accel,
                              state.q, state.dq, contact)

    # ------------------------------------------------------------------
    def _force_step(self, vel_cmd):
        """FL force command after the staleness watchdog (same driver as simulation)."""
        moving = bool(np.linalg.norm(np.asarray(vel_cmd)[:2]) > 1e-6)
        force = self._force_watchdog.step(
            self._force_req, moving=moving,
            now=time.monotonic(), command_time=self._ee_force_time)
        self.controller.set_commands(ee_force_cmd=force, leg_slot=0)

    def _make_robot_state(self, s):
        # Use the same resolved contact as the estimator (_on_state): the gait
        # scheduler's plan in POLICY, all four feet otherwise.
        if self._last_contact is not None:
            contacts = self._last_contact
        else:
            contacts = np.ones(4, dtype=bool)
        rs = RobotState(
            joint_pos=s.q.copy(),
            joint_vel=s.dq.copy(),
            base_quat=s.quat.copy(),
            base_ang_vel_b=s.gyro.copy(),
            base_lin_vel_b=self.estimator.base_lin_vel_b(s.quat),
            base_height=self.estimator.base_height,
            foot_contacts=contacts,
        )
        return rs.finalize()

    @staticmethod
    def _gravity_z(s):
        return quat_rotate_inverse(s.quat, np.array([0.0, 0.0, -1.0]))[2]

    def _standing(self, s):
        """Genuinely standing on four legs: EVERY leg extended (min, not
        mean - a collapsed "sitting" pose with front legs propped reads high
        on average) AND the base near level."""
        leg_heights = -kinematics.foot_positions_b(s.q)[:, 2]
        return leg_heights.min() > 0.2 and self._gravity_z(s) < -0.85

    def _apply_fsm_request(self):
        if self._damp_requested:
            self._fsm_request = None
            self._damp_requested = False
            self._to_damp("topic request")
            return True  # caller sends damping and skips ALL other actions
        req, self._fsm_request = self._fsm_request, None
        if req is None:
            return
        if req == "DAMP":
            self._to_damp("topic request")
        elif req == "STAND" and self.mode == DAMP:
            self._to_stand()
        elif req == "POLICY" and self.mode == STAND:
            self._to_policy()
        elif req == "REARM":
            self.safety.rearm()
            self.get_logger().info("safety re-armed")

    # ------------------------------------------------------------------
    # transitions
    def _to_damp(self, reason):
        if self.mode != DAMP:
            self.get_logger().warn(f"-> DAMP ({reason})")
        if reason != "topic request" and not self.safety.tripped:
            self.safety._trip(reason)
        self.mode = DAMP
        self._exit_reason = reason
        self._set_ee_compliance(False)   # every exit path, not only the tick's

    def _wired_ok_or_refuse(self, what):
        """Require confirmed ownership, no fault latch, and wired traffic.

        NIC counters supplement the required DDS binding; they cannot prove
        the source of an individual topic.
        """
        if self.safety.tripped:
            self.get_logger().error(f"{what} refused: safety latched ({self.safety.reason})")
            return False
        if not self._released_sport_mode:
            self.get_logger().error(f"{what} refused: motion release not confirmed "
                                    f"({self.motion_switcher.release_state})")
            return False
        if self._wired.ok is True:
            return True
        self.get_logger().error(
            f"{what} refused: {self._wired.reason} (parameter wired_nic; "
            "mock robot: wired_nic:=none)")
        return False

    def _to_stand(self):
        s = self.go2.last_state
        if s is None:
            return
        if not self._wired_ok_or_refuse("STAND"):
            return
        self._stand_start_q = s.q.copy()
        self._stand_progress = 0.0
        self._stand_hold_q = None
        self._stand_track_bad = 0
        # Skip the prone gather phase only when genuinely standing (see
        # _standing). Any other pose folds first - folding is the safe
        # recovery from arbitrary collapsed postures.
        self._gather_progress = 1.0 if self._standing(s) else 0.0
        self.mode = STAND
        self.get_logger().info(
            "-> STAND (%s)" % ("rise only" if self._gather_progress >= 1.0
                               else "gather legs, then rise"))

    def _to_policy(self):
        s = self.go2.last_state
        if s is None:
            return
        if not self._wired_ok_or_refuse("POLICY"):
            return
        if self.mode != STAND:
            return
        if self._gather_progress < 1.0 or self._stand_progress < 1.0:
            self.get_logger().warn(
                "POLICY refused: the stand-up is not complete yet")
            return
        rs = self._make_robot_state(s)
        self._policy_time = 0.0
        self._ee_compliant = False
        self._freshness.reset()        # per-run LowState freshness summary
        self._heading_hold.reset()
        # drop any stale operator commands from a previous session
        self._topic_vel_cmd = np.zeros(3)
        self._topic_vel_time = None
        self.controller.set_commands(vel_cmd=[0.0, 0.0, 0.0])
        for slot in range(self.task.num_ee):
            self.controller.set_commands(ee_pos_cmd=self.task.ee_pos_default[slot],
                                         ee_force_cmd=[0.0, 0.0, 0.0], leg_slot=slot)
        self._force_req = np.zeros(3)
        self._ee_force_time = None
        self._force_watchdog.reset()
        self._policy_fault = None
        self._exit_reason = ""
        self.controller.reset(rs)
        self.mode = POLICY
        self.get_logger().info(
            f"-> POLICY (RAMBO {self.task.name} controller active)")

    def _send_pose(self, q_des, dq_des=None, tau_ff=None):
        try:
            cmd = adapt_command(q_des,
                                np.zeros(12) if dq_des is None else dq_des,
                                np.zeros(12) if tau_ff is None else tau_ff,
                                kp=self._stand_kp, kd=self._stand_kd)
            self.go2.send(*cmd)
        except CommandError as e:
            self._to_damp(str(e))
            self.go2.send_damping()

    def _control_tick(self):
        now = time.monotonic()
        control_gap = 0.0 if self._last_control_time is None else now - self._last_control_time
        self._last_control_time = now
        was_active = self.mode != DAMP
        # Ownership checks progress even before the first LowState arrives.
        waiting_for_release = not self._released_sport_mode
        if waiting_for_release:
            self._released_sport_mode = self.motion_switcher.poll_release(now)
        s = self.go2.last_state
        if s is None:
            self._fsm_request = None
            self._damp_requested = False
            return  # nothing received yet; do not publish

        first_state = not self._first_state_seen
        self._first_state_seen = True
        hardware_fault = hardware_fault_reason(s.bit_flag, s.motor_errors)
        if waiting_for_release or first_state or hardware_fault:
            # Requests made before ownership/first state or during a fault
            # must never become delayed motion/rearm when the condition clears.
            self._fsm_request = None
            self._damp_requested = False
        if hardware_fault:
            # Fault bits outrank REARM requests as well as motion. Do not
            # run a posture recovery on faulted motors.
            self._to_damp(f"hardware fault: {hardware_fault} — latched; "
                          "re-arm with a REARM request after the fault clears")
            if self._released_sport_mode:
                self.go2.send_damping()
            self._track_mode_exit()
            self._publish_mode()
            return
        if not self._released_sport_mode:
            # Damping is still a LowCmd: it may conflict with the official
            # controller until CheckMode confirms that service is released.
            self._publish_mode()
            return
        stale_warning = self._freshness.update(self.go2.state_count)
        if stale_warning:
            self.get_logger().warn(stale_warning)
        self._wired.update(time.monotonic(), self.go2.state_count)
        if self._wired.ok is False and time.monotonic() - self._wired_logged > 5.0:
            self._wired_logged = time.monotonic()
            self.get_logger().error(f"WIRED LINK CHECK FAILED: {self._wired.reason}"
                                    + ("" if self.mode == DAMP else
                                       " - active motion will enter latched damping"))

        if self._apply_fsm_request():
            self.go2.send_damping()
            self._track_mode_exit()
            self._publish_mode()
            return

        rs = self._make_robot_state(s)
        if getattr(self, "_state_pub", None) is not None:      # test doubles borrow this tick without it
            self._publish_state(s, rs, control_gap)
        # The estimated-state divergence bounds (speed/height) only make sense
        # in POLICY, where the estimator output drives control. In DAMP/STAND
        # the commands are fixed poses and the estimator legitimately reports a
        # low FK height when the robot is prone with folded legs (feet are not
        # on the ground as FK assumes) - checking it there false-trips.
        in_policy = self.mode == POLICY
        ok = self.safety.check(self.go2.seconds_since_last_state(),
                               rs.projected_gravity_b, rs.joint_pos, rs.joint_vel,
                               base_lin_vel_b=rs.base_lin_vel_b if in_policy else None,
                               base_height=rs.base_height if in_policy else None,
                               base_ang_vel_b=rs.base_ang_vel_b,
                               active=was_active or self.mode != DAMP,
                               control_gap=control_gap, wired_ok=self._wired.ok,
                               source_fault=self.go2.source_fault,
                               estimator_age=self.estimator.age(now) if in_policy else None)
        if not ok:
            if self.mode != DAMP:
                self._to_damp(f"safety: {self.safety.reason} — latched; "
                              "re-arm with a REARM request")
            self.go2.send_damping()
            self._track_mode_exit()   # save the trace of the run that tripped
            self._publish_mode()
            return

        if self.mode == DAMP:
            self.go2.send_damping()
        elif self.mode == STAND:
            if self._gather_progress < 1.0:
                # phase 1: fold legs under the body while prone (hips -> 0),
                # calves first so the leg is short before thigh/hip sweep
                self._gather_progress = min(
                    1.0, self._gather_progress + self._dt / GATHER_DURATION)
                p = self._gather_progress
                pc = min(1.0, p / GATHER_CALF_FRACTION)
                ph = max(0.0, (p - GATHER_CALF_FRACTION)
                         / (1.0 - GATHER_CALF_FRACTION))
                q_des = self._stand_start_q.copy()
                q_des[8:12] = ((1.0 - pc) * self._stand_start_q[8:12]
                               + pc * GATHER_POSE[8:12])
                q_des[0:8] = ((1.0 - ph) * self._stand_start_q[0:8]
                              + ph * GATHER_POSE[0:8])
                if self._gather_progress >= 1.0:
                    # re-anchor the rise on the measured gathered pose
                    self._stand_start_q = s.q.copy()
                    self.get_logger().info("legs gathered (hips %s); rising"
                                           % np.round(s.q[0:4], 3).tolist())
                self._send_pose(q_des)
                self._publish_mode()
                return
            self._stand_progress = min(
                1.0, self._stand_progress + self._dt / self._stand_duration)
            if self._stand_progress >= 1.0:
                if self._stand_hold_q is None:
                    # Feet can end up friction-pinned wider than the default
                    # stance; holding kp*err there stalls the hip motors
                    # (measured on the real robot: rear hips at -14.6/+16.4 Nm
                    # for 30 s until firmware overcurrent protection cut all
                    # motors and the robot collapsed). Latch hip targets to
                    # the achieved posture - zero hip stall torque; thigh/calf
                    # keep default height under normal gravity load. POLICY's
                    # stepping re-places the feet toward nominal stance.
                    hold = DEFAULT_JOINT_POS.copy()
                    hold[0:4] = np.clip(s.q[0:4], -0.45, 0.45)
                    self._stand_hold_q = hold
                    self.get_logger().info(
                        "stand complete; hip targets latched at "
                        f"{np.round(hold[0:4], 3).tolist()}")
                q_des = self._stand_hold_q
                # motors-not-tracking watchdog: firmware protection can cut
                # motor power while lowstate keeps streaming; stop blindly
                # commanding kp=60 into a limp robot (snap hazard on resume).
                if np.abs(s.q - q_des).max() > 0.5:
                    self._stand_track_bad += 1
                else:
                    self._stand_track_bad = 0
                if self._stand_track_bad > int(1.0 / self._dt):
                    self._to_damp("motors not tracking stand target "
                                  "(firmware protection?) — check the robot; this "
                                  "latches: re-arm (REARM), then STAND")
                    self.go2.send_damping()
                    self._publish_mode()
                    return
            else:
                q_des = ((1.0 - self._stand_progress) * self._stand_start_q
                         + self._stand_progress * DEFAULT_JOINT_POS)
            self._send_pose(q_des)
        elif self.mode == POLICY:
            vel_cmd = list(self._topic_velocity())
            vel_cmd = self._heading_hold.update(vel_cmd, yaw_from_quat(s.quat))
            if self._heading_hold.event == "latched":
                what = "stepping in place" if self._heading_hold.mode == "in_place" else "walking straight"
                self.get_logger().info(
                    f"heading hold: {what}, holding yaw {math.degrees(self._heading_hold.yaw_ref):+.1f} deg")
            elif self._heading_hold.event == "released":
                self.get_logger().info("heading hold: released (operator / policy command changed)")
            try:
                self.controller.set_commands(vel_cmd=vel_cmd)
                self._force_step(vel_cmd)
                q_des, dq_des, tau_ff = self.controller.step(self._policy_time, rs)
            except Exception as exc:
                # Keep the raw failed output separate from normal trace rows:
                # it may have a wrong length, or fail on the very first step.
                self._policy_fault = dict(
                    wall=time.time(), t=self._policy_time,
                    reason=f"{type(exc).__name__}: {exc}",
                    raw_action=np.asarray(self.controller.last_raw_action).copy())
                self.get_logger().error(f"POLICY fault: {self._policy_fault['reason']}; "
                                        f"raw_action={self._policy_fault['raw_action']!r}")
                self._to_damp(f"POLICY fault: {self._policy_fault['reason']}")
                self.go2.send_damping()
                self._save_policy_trace()
                self._track_mode_exit()
                self._publish_mode()
                return
            self._policy_time += self._dt
            if self._trace_dir:
                c = self.controller
                self._trace.append(dict(
                    wall=time.time(), t=self._policy_time - self._dt,
                    s=c.entry_fraction, raw_action=c.last_raw_action.copy(),
                    action=c._last_action.copy(), jpc_target=c._desired_joint_pos.copy(),
                    q_des=q_des.copy(), tau_ff=tau_ff.copy(), grf=c.last_grf.copy(),
                    qp_cost=c.last_qp_cost, h_des=c.qp.desired_base_position[2],
                    contact=c.contact_generator.desired_contact_state.astype(np.int8),
                    q=s.q.copy(), dq=s.dq.copy(), quat=s.quat.copy(), gyro=s.gyro.copy(),
                    base_height=rs.base_height, base_lin_vel_b=rs.base_lin_vel_b.copy(),
                    fresh=np.int8(not self._freshness.last_stale), tick_ms=np.int64(s.tick_ms),
                    source_age=self.go2.seconds_since_last_state(),
                    control_gap=control_gap,
                    receive_count=self.go2.receive_count,
                    estimator_valid=np.int8(self.estimator.valid),
                    estimator_age=self.estimator.age(),
                    estimator_failures=self.estimator.consecutive_failures,
                    qp_status=c.qp.status, qp_solve_ms=1000.0 * c.qp.solve_duration,
                    qp_failures=c.qp.consecutive_failures,
                    vel_cmd=np.asarray(vel_cmd, dtype=float), yaw=yaw_from_quat(s.quat),
                    heading_hold=np.int8(self._heading_hold.active),
                    heading_hold_in_place=np.int8(self._heading_hold.mode == "in_place"),
                    # the FL commands actually applied (grf is the QP's solution,
                    # not what was asked for)
                    ee_force_cmd=np.asarray(c.ee_force_commands[0], dtype=np.float32),
                    ee_pos_cmd=np.asarray(c.ee_pos_commands[0], dtype=np.float32),
                    tau_est=s.tau_est.astype(np.float32)))
                if len(self._trace) > 60000:          # keep the last 10 min
                    del self._trace[:len(self._trace) - 60000]
            try:
                kp, kd = policy_gains(self.task.ee_joint_indices, self._ee_compliant,
                                      self._compliant_kp, self._compliant_kd)
                cmd = adapt_command(q_des, dq_des, tau_ff, kp=kp, kd=kd)
                self.go2.send(*cmd)
            except CommandError as e:
                self._to_damp(str(e))
                self.go2.send_damping()

        self._track_mode_exit()
        self._publish_mode()

    def _track_mode_exit(self):
        """POLICY -> anything: save the 100 Hz trace, drop compliance."""
        if self._prev_mode == POLICY and self.mode != POLICY:
            self.get_logger().info(
                f"POLICY run LowState freshness: {self._freshness.summary()}")
            self._save_policy_trace()
            self._set_ee_compliance(False)
        self._prev_mode = self.mode

    def _save_policy_trace(self):
        rows, self._trace = self._trace, []
        fault, self._policy_fault = self._policy_fault, None
        if not self._trace_dir or (not rows and fault is None):
            return
        try:
            os.makedirs(self._trace_dir, exist_ok=True)
            wall = rows[0]["wall"] if rows else fault["wall"]
            path = os.path.join(self._trace_dir, f"policy_trace_{int(wall * 1e6)}.npz")
            payload = {k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else {}
            payload.update(metadata_json=np.array(self._trace_metadata),
                           exit_reason=np.array(self._exit_reason),
                           safety_reason=np.array(self.safety.reason))
            if fault is not None:
                payload.update({f"fault_{key}": np.asarray(value) for key, value in fault.items()})
            np.savez_compressed(path, **payload)
            self.get_logger().info(f"POLICY trace saved: {path} ({len(rows)} ticks)")
        except Exception as e:  # never let diagnostics take the controller down
            self.get_logger().warn(f"POLICY trace not saved: {e}")

    def _publish_state(self, s, rs, control_gap):
        """rambo/state for consumers (the WAM bridge); it can never disturb control."""
        if self._state_pub is None:
            return
        try:
            action = getattr(self.controller, "_last_action", None) if self.mode == POLICY else None
            msg = Float64MultiArray()
            msg.data = state_stream.encode(
                wall=time.time(), tick_ms=int(s.tick_ms), mode=self.mode, tripped=bool(self.safety.tripped),
                fresh=not self._freshness.last_stale, estimator_valid=bool(self.estimator.valid),
                base_height=float(rs.base_height), quat_wxyz=rs.base_quat, lin_vel_b=rs.base_lin_vel_b,
                ang_vel_b=rs.base_ang_vel_b, source_age=float(self.go2.seconds_since_last_state()),
                control_gap=float(control_gap),
                action18=None if action is None else np.asarray(action, dtype=float).reshape(-1).tolist())
            self._state_pub.publish(msg)
        except Exception as exc:  # diagnostics only
            if not self._state_pub_failed:
                self._state_pub_failed = True
                self.get_logger().warn(f"rambo/state not published: {exc}")

    def _publish_mode(self):
        msg = String()
        msg.data = self.mode if not self.safety.tripped else \
            f"{self.mode} (safety: {self.safety.reason})"
        self._mode_pub.publish(msg)


def main(args=None):
    # Own SIGINT/SIGTERM: rclpy's default handler shuts the context down
    # before `finally` runs, which silently drops the damping frames on exit.
    # With the flag below the spin loop ends cleanly and the frames go out
    # first.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    stop = []

    def _on_signal(signum, _frame):
        stop.append(signal.Signals(signum).name)

    # installed before the (seconds-long) policy load so an early Ctrl+C is a
    # clean exit too; SIGHUP = terminal / SSH gone while the node runs
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _on_signal)
    node = RamboGo2Controller()
    # the start-up heap out of the cyclic GC before any motion (wam_go2/realtime.py: its first full collection
    # would otherwise be a ~40 ms control gap early in the first POLICY run)
    frozen, collect_ms = freeze_startup_heap()
    node.get_logger().info(f"GC: start-up heap frozen ({frozen} objects, collect {collect_ms:.1f} ms)")
    try:
        while rclpy.ok() and not stop:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        can_damp = node._released_sport_mode and node.go2.last_state is not None
        if stop:
            action = ("sending damping frames (kp 0 / kd 2), exiting" if can_damp
                      else "exiting without LowCmd (handoff or first state not confirmed)")
            node.get_logger().info(f"{stop[0]}: {action}")
        try:
            if can_damp:
                for _ in range(3):
                    node.go2.send_damping()
                    time.sleep(0.01)
        except Exception:
            pass
        try:
            if node.mode == POLICY:      # a run cut short by the signal
                node.get_logger().info(
                    f"POLICY run LowState freshness: {node._freshness.summary()}")
            node._exit_reason = "shutdown: " + (stop[0] if stop else "executor stopped")
            node._save_policy_trace()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
