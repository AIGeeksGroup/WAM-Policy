"""Launch and bring-up settings around the FL force command and the wired link.

No foot contact force is read: the force part of the native9 command passes only
the staleness watchdog (wam_go2/control/force_command.py).
"""
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


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
