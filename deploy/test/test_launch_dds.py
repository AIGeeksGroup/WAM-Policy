"""The launch files bind DDS to the wired interface: unless the operator set a profile, they write the
config/ template with DEMO_LAPTOP_IP filled in and point FastDDS at it (ROS launch modules stubbed)."""
import importlib.util
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
NS = {"f": "http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles"}
LAPTOP_IP = "192.0.2.10"                                   # a reserved documentation address


class _Action:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


def _load(name, monkeypatch):
    stubs = {
        "ament_index_python": {}, "ament_index_python.packages": {"get_package_share_directory": lambda _: str(PKG)},
        "launch": {"LaunchDescription": _Action}, "launch.substitutions": {"LaunchConfiguration": _Action},
        "launch.actions": {k: type(k, (_Action,), {}) for k in ("DeclareLaunchArgument", "LogInfo",
                                                                 "SetEnvironmentVariable", "Shutdown")},
        "launch_ros": {}, "launch_ros.actions": {"Node": _Action},
        "launch_ros.parameter_descriptions": {"ParameterValue": _Action},
    }
    for mod, attrs in stubs.items():
        monkeypatch.setitem(sys.modules, mod, types.SimpleNamespace(**attrs))
    spec = importlib.util.spec_from_file_location(name, PKG / "launch" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CASES = [("wam_go2.launch", "wired_dds_actions", "fastdds_wired.xml"),
         ("wam_bridge.launch", "dds_actions", "fastdds_wired_shm.xml")]


def _whitelists(path):
    root = ET.parse(path).getroot()
    return [a.text for a in root.iter(f"{{{NS['f']}}}address")]


@pytest.fixture
def env(monkeypatch, tmp_path):
    for key in ("FASTRTPS_DEFAULT_PROFILES_FILE", "CYCLONEDDS_URI", "RMW_IMPLEMENTATION", "DEMO_LAPTOP_IP"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    return monkeypatch


@pytest.mark.parametrize("name,func,template", CASES)
def test_the_template_is_filled_with_the_laptop_address(env, tmp_path, name, func, template):
    env.setenv("DEMO_LAPTOP_IP", LAPTOP_IP)
    actions = getattr(_load(name, env), func)(str(PKG))
    profile = [a.args[1] for a in actions if type(a).__name__ == "SetEnvironmentVariable"]
    assert len(profile) == 1 and Path(profile[0]).parent == tmp_path
    assert _whitelists(PKG / "config" / template) and set(_whitelists(PKG / "config" / template)) == {"@LAPTOP_IP@"}
    assert set(_whitelists(profile[0])) == {LAPTOP_IP}


@pytest.mark.parametrize("name,func,template", CASES)
def test_no_laptop_address_binds_nothing_and_warns(env, name, func, template):
    actions = getattr(_load(name, env), func)(str(PKG))
    assert not [a for a in actions if type(a).__name__ == "SetEnvironmentVariable"]
    assert any("DEMO_LAPTOP_IP" in a.kwargs.get("msg", "") for a in actions)


@pytest.mark.parametrize("name,func,template", CASES)
def test_an_operator_profile_is_kept(env, tmp_path, name, func, template):
    env.setenv("DEMO_LAPTOP_IP", LAPTOP_IP)
    env.setenv("FASTRTPS_DEFAULT_PROFILES_FILE", str(tmp_path / "mine.xml"))
    actions = getattr(_load(name, env), func)(str(PKG))
    assert not [a for a in actions if type(a).__name__ == "SetEnvironmentVariable"]
    assert not list(tmp_path.glob("wam_go2_*.xml"))
