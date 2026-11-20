"""Demo, launch and simulator failure handling around the FL force command.

This release reads no foot contact force: the force part of the native9 command
passes only the staleness watchdog (wam_go2/control/force_command.py), which
the node and the MuJoCo harness share.
"""
import ast
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from wam_go2.control.force_command import ForceCommandWatchdog

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import demo_sequence as ds


def test_actual_harness_push_step_matches_shared_driver_replay():
    """Execute the real Harness method without importing MuJoCo or ROS."""
    source = Path(__file__).resolve().parents[1] / "sim/mj_sim.py"
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "Harness")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "push_step")
    ns = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), ns)
    controller = SimpleNamespace(set_commands=lambda **kw: setattr(controller, "command", kw))
    sim_driver, node_driver = ForceCommandWatchdog(), ForceCommandWatchdog()
    h = SimpleNamespace(force_watchdog=sim_driver, controller=controller,
                        sim=SimpleNamespace(), push_target=0., push_moving=False)
    seen_force = False
    for i in range(500):
        t = i * .01
        h.sim.t = t
        h.push_target = 14. if 50 <= i < 400 else 0.
        h.push_moving = 50 <= i < 400
        expected = node_driver.step([h.push_target, 0., 0.], h.push_moving, now=t, command_time=t)
        ns["push_step"](h)
        np.testing.assert_array_equal(h.controller.command["ee_force_cmd"], expected)
        assert h.push_cmd == expected[0]
        seen_force |= h.push_cmd > 0.
    assert seen_force
    assert h.push_cmd == 0.


def test_harness_without_force_watchdog_sends_nothing():
    source = Path(__file__).resolve().parents[1] / "sim/mj_sim.py"
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "Harness")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "push_step")
    ns = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), ns)
    controller = SimpleNamespace(set_commands=lambda **kw: pytest.fail("no command expected"))
    ns["push_step"](SimpleNamespace(force_watchdog=None, controller=controller))


def test_launch_forwards_typed_settings_and_wired_nic(monkeypatch):
    """Exercise generate_launch_description with ROS-free launch stand-ins."""
    settings = {"ee_force_cmd_timeout": "0.25", "wired_nic": "test_ethernet"}
    class Config:
        def __init__(self, name):
            self.name = name
    class Value:
        def __init__(self, value, value_type):
            self.value, self.kind = value, value_type
        def resolve(self):
            raw = settings[self.value.name]
            return raw == "true" if self.kind is bool else self.kind(raw)
    def declare(name, default_value, **kw):
        settings.setdefault(name, default_value)
        return SimpleNamespace(argument=name)
    modules = {
        "ament_index_python.packages": dict(get_package_share_directory=lambda _: "/package"),
        "launch": dict(LaunchDescription=lambda actions: actions),
        "launch.actions": dict(DeclareLaunchArgument=declare, LogInfo=lambda **kw: kw,
                               SetEnvironmentVariable=lambda *a: a),
        "launch.substitutions": dict(LaunchConfiguration=Config),
        "launch_ros.actions": dict(Node=lambda **kw: SimpleNamespace(**kw)),
        "launch_ros.parameter_descriptions": dict(ParameterValue=Value),
    }
    for name, contents in modules.items():
        mod = ModuleType(name)
        mod.__dict__.update(contents)
        monkeypatch.setitem(sys.modules, name, mod)
    source = Path(__file__).resolve().parents[1] / "launch/wam_go2.launch.py"
    description = runpy.run_path(str(source))["generate_launch_description"]()
    node = next(a for a in description if getattr(a, "executable", None) == "controller")
    params = {}
    for value in node.parameters:
        if isinstance(value, dict):
            params.update(value)
    assert params["wired_nic"].resolve() == "test_ethernet"
    assert params["ee_force_cmd_timeout"].resolve() == .25
    assert not any(name.startswith(("push_force", "foot_force")) for name in params)


def test_box_push_plan_moves_without_any_force_command():
    plan = ds.build_plan(ds.parse_args(["--demo", "box_push", "--walk-time", "2"]))
    moves = [s for s in plan if s.kind == "vel"]
    assert moves[0].value[0] > 0. and moves[0].force == (0., 0., 0.)
    assert moves[1].value == (0., 0., 0.) and moves[1].force == (0., 0., 0.)
    assert not any(s.kind == "ee_force" for s in plan)


def test_push_force_option_is_gone():
    with pytest.raises(SystemExit):
        ds.parse_args(["--demo", "box_push", "--push-force", "14"])


@pytest.mark.parametrize("mode", ["DAMP (safety: LowState timeout (0.251s))",
    "DAMP (safety: tilt)", "DAMP (safety: estimator invalid)"])
def test_infinite_demo_reports_actual_safety_trip_as_failure(mode):
    node = object.__new__(ds.DemoSequencer)
    node.result = None
    node._mode = mode
    node._mode_time = __import__("time").monotonic()
    node._interrupted = None
    node._reached = ds.POLICY
    node._leaving = False
    node._plan = [ds.Step("sleep", "wait", until_leave=True)]
    node._idx = 0
    recorded = []
    node.abort = lambda code, reason, **kw: recorded.append((code, reason))
    node.finish_by_operator = lambda: recorded.append((0, "operator"))
    node._tick()
    assert recorded[0][0] != 0
    assert mode in recorded[0][1]


@pytest.mark.parametrize("mode,expected", [("DAMP", 0),
    ("DAMP (safety: estimator update expired)", 2)])
def test_demo_termination_retains_operator_or_fault_reason(monkeypatch, mode, expected):
    for name in ("Twist", "Vector3", "Bool"):
        monkeypatch.setattr(ds, name, SimpleNamespace, raising=False)
    node = object.__new__(ds.DemoSequencer)
    node.result = None
    node._mode = mode
    messages = []
    for name in ("_vel_pub", "_comp_pub", "_force_pub"):
        setattr(node, name, SimpleNamespace(publish=messages.append))
    node.finish_by_operator()
    assert node.result == expected
    assert mode in node.result_reason
    assert len(messages) == 13  # zero velocity/force and compliance-off on either path


@pytest.mark.parametrize("answer,expected", [("String value is: usb_wired", 0),
    ("String value is: other_nic", 1), ("", 1)])
def test_the_running_controller_must_check_the_wired_nic(answer, expected):
    source = Path(__file__).resolve().parents[1] / "scripts/demo/common.sh"
    # Only these fake query results are used; no ROS processes or network calls.
    script = '''source "$1"
DEMO_NIC=usb_wired
demo_param_get() {
  if [[ "$1" == wired_nic ]]; then
    printf '%s\\n' "$TEST_NIC_ANSWER"
  else
    printf '%s\\n' 'String value is: topic'
  fi
}
demo_check_controller
'''
    import os
    result = subprocess.run(["bash", "-c", script, "check", str(source)],
        env={**os.environ, "TEST_NIC_ANSWER": answer, "WAM_NO_CONFIG": "1"}, capture_output=True, text=True)
    assert result.returncode == expected
