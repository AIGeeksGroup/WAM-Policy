#!/usr/bin/env python3
"""Bounded fault injection through the real ROS node and an isolated mock Go2.

After sourcing ROS2 and unitree message environments:
    python3 test/integration/mock_fault_test.py

Uses the quadruped checkpoint (policies/quadruped/model_2000.pt, downloaded
separately) and the production controller, estimator,
interface, FSM, safety manager and publisher. Only the robot and the injected
failed policy/solver call are fake. All cases run in one executor, without
controller subprocesses. Nonzero ROS domain and localhost-only DDS are enforced
before ROS initialization; no real-robot interface or service is used.
"""

import os
from pathlib import Path
import sys
import time

# Set these before importing/initializing rclpy. Refuse an explicitly selected
# robot/default domain. Changes affect this process, not the calling shell.
_DOMAIN = int(os.environ.get("ROS_DOMAIN_ID", "42"))
if not 1 <= _DOMAIN <= 232:
    raise SystemExit("mock_fault_test requires ROS_DOMAIN_ID in 1..232 (default 42)")
os.environ["ROS_DOMAIN_ID"] = str(_DOMAIN)
os.environ["ROS_LOCALHOST_ONLY"] = "1"
os.environ.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)
os.environ.pop("FASTDDS_DEFAULT_PROFILES_FILE", None)
os.environ.pop("CYCLONEDDS_URI", None)

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from unitree_api.msg import Response  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from mock_robot_test import DEFAULT_Q_GO2, MockRobot  # noqa: E402
from wam_go2.constants import DAMPING_KD  # noqa: E402
from wam_go2.go2.motion_switcher import API_CHECK_MODE  # noqa: E402
from wam_go2.nodes.controller_node import (  # noqa: E402
    DAMP, POLICY, STAND, RamboGo2Controller,
)


class FaultPublisher:
    """Mutate synthetic LowState immediately before its actual ROS publish."""
    def __init__(self, publisher, robot):
        self._publisher = publisher
        self._robot = robot

    def publish(self, msg):
        robot = self._robot
        if robot.repeat_tick is not None:
            msg.tick = robot.repeat_tick
        if robot.bad_accel:
            msg.imu_state.accelerometer[0] = float("nan")
        if robot.bad_gyro:
            msg.imu_state.gyroscope[0] = float("nan")
        msg.bit_flag = robot.status_flags
        msg.motor_state[0].reserve[0] = robot.motor_error
        self._publisher.publish(msg)


class FaultRobot(MockRobot):
    def __init__(self):
        super().__init__()
        self.bad_accel = False
        self.bad_gyro = False
        self.repeat_tick = None
        self.status_flags = 0x04  # documented normal, not charging
        self.motor_error = 0
        self.motion_name = "mcf"  # exercise ownership gating before becoming clear
        self.pub = FaultPublisher(self.pub, self)

    def _on_motion_request(self, request):
        response = Response()
        response.header.identity.id = request.header.identity.id
        response.header.identity.api_id = request.header.identity.api_id
        response.header.status.code = 0 if request.header.identity.api_id == API_CHECK_MODE else 1
        response.data = '{"name": "' + self.motion_name + '"}'
        self.motion_pub.publish(response)


class FaultChecks:
    def __init__(self, executor, robot, node):
        self.executor, self.robot, self.node = executor, robot, node
        self.deadline = time.monotonic() + 65.0
        self.passed = []

    def spin(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            assert time.monotonic() < self.deadline, "fault suite exceeded 65 s bound"
            self.executor.spin_once(timeout_sec=0.001)

    def until(self, condition, seconds, what):
        end = time.monotonic() + seconds
        while not condition() and time.monotonic() < end:
            self.spin(0.002)
        assert condition(), (f"{what}: mode={self.node.mode}, "
                             f"safety={self.node.safety.reason!r}, "
                             f"handoff={self.node.motion_switcher.release_state}")

    def damping_only(self, what):
        cmds = self.robot.cmds
        assert len(cmds) >= 3, f"{what}: expected at least three actual LowCmd messages"
        assert all(np.all(c["kp"] == 0.0) and np.allclose(c["kd"], DAMPING_KD)
                   and np.all(c["tau"] == 0.0) and c["crc_ok"] for c in cmds), (
                       f"{what}: active/invalid command escaped damping")

    def passed_check(self, name):
        self.passed.append(name)
        print(f"PASS {name}", flush=True)

    def stand(self):
        robot, node = self.robot, self.node
        assert not robot.bad_accel and not robot.bad_gyro and robot.repeat_tick is None
        robot.q = DEFAULT_Q_GO2.copy()
        robot.request("DAMP")
        self.spin(0.06)
        robot.request("REARM")
        self.until(lambda: not node.safety.tripped, 0.5, "explicit rearm")
        robot.request("STAND")
        self.until(lambda: node.mode == STAND and node._stand_progress >= 1.0,
                   2.5, "standing with healthy sensors")
        assert not node.safety.tripped, node.safety.reason

    def policy(self):
        self.stand()
        self.robot.cmds.clear()
        self.robot.request("POLICY")
        self.until(lambda: self.node.mode == POLICY or self.node.safety.tripped,
                   0.5, "policy entry")
        assert self.node.mode == POLICY, f"healthy policy entry failed: {self.node.safety.reason}"
        self.spin(0.06)
        assert self.node.mode == POLICY, f"healthy policy could not run: {self.node.safety.reason}"
        assert any(np.allclose(c["kp"], 40.0) for c in self.robot.cmds), (
            "healthy policy did not publish active commands")

    def stops_have_priority(self):
        self.stand()
        # Deliver both requests before the next timer through ROS subscriptions.
        self.node._timer.cancel()
        self.robot.request("DAMP")
        self.robot.request("STAND")
        self.spin(0.012)
        self.node._timer.reset()
        self.until(lambda: self.node.mode == DAMP, 0.2, "DAMP then STAND")
        self.spin(0.03)
        self.robot.cmds.clear()
        self.spin(0.06)
        self.damping_only("DAMP then STAND in one control period")
        self.passed_check("topic DAMP cannot be overwritten by STAND")

    def fault(self, name, reason, inject, restore, reject_rearm=False):
        self.policy()
        started = time.monotonic()
        inject()
        try:
            self.until(lambda: self.node.safety.tripped and self.node.mode == DAMP,
                       0.4, name + " failed to latch DAMP")
            delay = time.monotonic() - started
            assert reason in self.node.safety.reason, (
                f"{name} exited for wrong reason: {self.node.safety.reason}")
            self.spin(0.03)
            self.robot.cmds.clear()
            if reject_rearm:
                self.robot.request("REARM")
                self.robot.request("STAND")
                self.spin(0.07)
                assert self.node.safety.tripped and self.node.mode == DAMP, (
                    name + " accepted REARM/STAND while the fault was still present")
                self.damping_only(name + " during attempted REARM")
        finally:
            restore()
        self.spin(0.07)
        assert self.node.safety.tripped and self.node.mode == DAMP, (
            name + " automatically rearmed when valid input returned")
        self.robot.request("STAND")
        self.robot.request("POLICY")
        self.spin(0.06)
        assert self.node.safety.tripped and self.node.mode == DAMP, (
            name + " accepted motion without explicit REARM")
        self.damping_only(name + " after valid input returned")
        self.passed_check(f"{name}: latched in {delay * 1000:.0f} ms, recovery stays DAMP")

    def paused_ros_clock(self):
        """The physical-control watchdog must not depend on /clock updates."""
        node, robot = self.node, self.robot
        for name, inject, restore, reason in (
            ("MCU timeout", lambda: setattr(robot, "status_flags", 0x44),
             lambda: setattr(robot, "status_flags", 0x04), "MCU timeout"),
            ("motor communication interrupted", lambda: setattr(robot, "motor_error", 0x100),
             lambda: setattr(robot, "motor_error", 0), "communication interrupted"),
        ):
            self.policy()
            result = node.set_parameters([Parameter("use_sim_time", value=True)])[0]
            assert result.successful, f"could not enable ROS simulated time: {result.reason}"
            try:
                # No /clock publisher exists in this isolated graph. The mock
                # retains its wall-clock timer while only the controller's ROS
                # clock is paused. LowState reception must remain a positive
                # control, not merely an absence of LowCmd.
                assert node.get_clock().now().nanoseconds == 0
                self.spin(0.03)
                robot.cmds.clear()
                before_state, before_policy = node.go2.state_count, node._policy_time
                self.spin(0.08)
                assert node.go2.state_count >= before_state + 5, "mock LowState stopped"
                assert node.mode == POLICY and not node.safety.tripped, node.safety.reason
                assert node._policy_time > before_policy and len(robot.cmds) >= 3, (
                    "paused ROS clock stopped the physical control timer while LowState remained fresh")
                assert any(np.allclose(c["kp"], 40.0) and c["crc_ok"] for c in robot.cmds)
                started = time.monotonic()
                inject()
                self.until(lambda: node.safety.tripped and node.mode == DAMP,
                           0.3, name + " with paused ROS clock")
                delay = time.monotonic() - started
                assert reason in node.safety.reason, node.safety.reason
                self.spin(0.03)
                robot.cmds.clear()
                self.spin(0.06)
                self.damping_only(name + " with paused ROS clock")
                restore()
                self.spin(0.06)
                assert node.safety.tripped and node.mode == DAMP, "clock/fault recovery auto-rearmed"
            finally:
                restore()
                result = node.set_parameters([Parameter("use_sim_time", value=False)])[0]
                assert result.successful, "failed to restore wall-clock ROS time"
            self.spin(0.03)
            assert node.safety.tripped and node.mode == DAMP, "restoring ROS time cleared the latch"
            self.stand()  # real topic REARM followed by STAND; no implicit recovery
            self.passed_check(
                f"paused ROS clock: POLICY keeps ticking, {name} latched in {delay * 1000:.0f} ms; "
                "explicit REARM restores STAND")

    def run(self):
        self.until(lambda: self.node.go2.state_count >= 10, 3.0, "mock LowState discovery")
        self.robot.request("REARM")
        self.robot.request("STAND")
        self.spin(0.20)
        assert not self.node._released_sport_mode, "mock still has its high-level controller"
        assert not self.robot.cmds, "LowCmd escaped before ownership was confirmed"
        self.robot.motion_name = ""
        self.node.motion_switcher.start_release(allow_release=False)
        self.until(lambda: self.node._released_sport_mode and self.node.go2.state_count >= 10,
                   3.0, "confirmed isolated mock ownership")
        self.spin(0.05)
        assert self.node.mode == DAMP, "motion queued before ownership executed after handoff"
        self.passed_check("no LowCmd before ownership; waiting requests do not auto-start")
        self.paused_ros_clock()
        self.stops_have_priority()
        for name, flag, reason in (("MCU timeout", 0x40, "MCU"),
                                   ("motor status timeout", 0x80, "motor")):
            self.fault(name, reason,
                       lambda flag=flag: setattr(self.robot, "status_flags", 0x04 | flag),
                       lambda: setattr(self.robot, "status_flags", 0x04), reject_rearm=True)
        for name, flag in (("motor driver overheat", 0x04),
                           ("motor communication interrupted", 0x100)):
            self.fault(name, "motor",
                       lambda flag=flag: setattr(self.robot, "motor_error", flag),
                       lambda: setattr(self.robot, "motor_error", 0), reject_rearm=True)
        self.fault("invalid acceleration", "estimator",
                   lambda: setattr(self.robot, "bad_accel", True),
                   lambda: setattr(self.robot, "bad_accel", False))
        self.fault("invalid gyroscope", "angular velocity",
                   lambda: setattr(self.robot, "bad_gyro", True),
                   lambda: setattr(self.robot, "bad_gyro", False))

        def repeated_source():
            self.robot.repeat_tick = self.node.go2.last_state.tick_ms
        self.fault("repeated source tick", "source stale", repeated_source,
                   lambda: setattr(self.robot, "repeat_tick", None))

        def pause_control():
            self.node._timer.cancel()
            self.spin(0.085)  # LowState and estimator remain fresh throughout.
            self.node._timer.reset()
        self.fault("control scheduling gap", "control loop stalled", pause_control,
                   lambda: None)

        original_policy = self.node.controller.policy
        self.fault("non-finite policy output", "PolicyOutputError",
                   lambda: setattr(self.node.controller, "policy", lambda obs: np.full(18, np.nan)),
                   lambda: setattr(self.node.controller, "policy", original_policy))
        qp = self.node.controller.qp
        original_solve = qp._solve_once
        self.fault("QP solver failure", "QPSolveError",
                   lambda: setattr(qp, "_solve_once", lambda *args: None),
                   lambda: setattr(qp, "_solve_once", original_solve))
        self.stand()  # explicit REARM really restores healthy admission
        self.passed_check("explicit REARM restores healthy STAND admission")
        self.robot.request("DAMP")
        self.spin(0.05)


def main():
    params = ["--ros-args", "--log-level", "error"]
    for key, value in {
        "task": "quadruped", "auto_release_sport_mode": "false", "wired_nic": "none",
        "stand_duration": "0.1",
    }.items():
        params.extend(["-p", f"{key}:={value}"])
    rclpy.init(args=params)
    executor = SingleThreadedExecutor()
    robot = node = None
    try:
        robot = FaultRobot()
        node = RamboGo2Controller()
        executor.add_node(robot)
        executor.add_node(node)
        checks = FaultChecks(executor, robot, node)
        checks.run()
        print(f"ALL {len(checks.passed)} FAULT CHECKS PASSED "
              f"(ROS domain {_DOMAIN}, localhost only)", flush=True)
        return 0
    except Exception as exc:
        print(f"FAULT CHECK FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if node is not None:
            node._timer.cancel()
            if node._released_sport_mode and node.go2.last_state is not None:
                node.go2.send_damping()
            executor.remove_node(node)
            node.destroy_node()
        if robot is not None:
            executor.remove_node(robot)
            robot.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
