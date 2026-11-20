"""WAM runs handle the native9 command exactly like the QLM-Bench MuJoCo simulation that produced the training data.

QLM prepare_command (contracts/command/runtime.py) keeps float32(requested) with the force zeroed and
filter 'identity'; RAMBOCore.submit (controllers/rambo/core.py) stores the base velocity and the FL target UNCLIPPED
and never zeroes them while all four feet are in stance. RAMBO's own deployment clips to its Isaac sampling ranges
(vx, vy, wz +-0.5; FL x 0.1934..0.5, y 0..0.2, z 0..0.4) and zeroes velocity / force in all-stance, which would
alter a large share of the training commands on the robot.

  * command_handling "sim" (WAM runs: WAM_COMMAND_HANDLING of run_wam.sh / scripts/demo/wam_task.sh, default sim)
    = the simulation: no clip, no all-stance zeroing - stored command, observation and FL IK target are the QLM ones;
  * command_handling "rambo" (config/wam_go2.yaml default for the controller on its own) = RAMBO's behaviour.
The QLM-Bench source comes from QLM_BENCH_SRC (test/conftest.py); its tests skip without it.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from conftest import QLM_SRC  # noqa: E402
QLM_RAMBO = Path(QLM_SRC or "/nonexistent") / "qlm_bench" / "controllers" / "rambo"
needs_qlm = pytest.mark.skipif(not QLM_RAMBO.is_dir(), reason="QLM-Bench source not found (set QLM_BENCH_SRC)")

from wam_go2.constants import DEFAULT_JOINT_POS, NUM_OBS_PER_STEP  # noqa: E402
from wam_go2.control import kinematics  # noqa: E402
from wam_go2.control.rambo_controller import RamboController, command_handling_options  # noqa: E402
from wam_go2.control.robot_state import RobotState  # noqa: E402
from wam_go2.tasks import QUADRUPED  # noqa: E402

from test_run_wam import fakes  # noqa: E402,F401  (fixture)

# the RAMBO controller files of the qlm-bench branch (sha256 prefixes); their control code is the one the
# training data were generated with (they differ from that version only in imports and checkpoint checks)
QLM_RAMBO_SHA256 = {
    "core.py": "34996f9abd72",
    "control.py": "1c17a4586a86",
    "joint_position_controller.py": "92950351d6d3",
    "kinematics.py": "2d5fa5a02d64",
}

# the extremes of the training commands, a far out-of-range probe and an in-range one
CASES = [
    [0.213, -0.0805, -0.3087, 0.5845, 0.2506, -0.034, 0.0, 0.0, 0.0],
    [-0.0354, 0.0767, 0.208, 0.1439, -0.0085, 0.3977, 0.0, 0.0, 0.0],
    [0.6, -0.55, 0.7, 0.546, 0.251, -0.02, 0.0, 0.0, 0.0],
    [0.2, 0.0, 0.0, 0.3, 0.14, 0.2, 0.0, 0.0, 0.0],
]


class ZeroPolicy:
    last_obs = None

    def __call__(self, obs):
        self.last_obs = np.asarray(obs).copy()
        return np.zeros(18)


def _state():
    return RobotState(joint_pos=DEFAULT_JOINT_POS.copy(), base_height=0.30,
                      foot_contacts=np.ones(4, dtype=bool)).finalize()


def _qlm_prepare(requested):
    from qlm_bench.contracts.command.runtime import prepare_command
    return np.array(prepare_command(requested, "c:0", 0)["filtered"])


@needs_qlm
def test_the_qlm_rambo_files_are_the_pinned_ones():
    for name, prefix in QLM_RAMBO_SHA256.items():
        assert hashlib.sha256((QLM_RAMBO / name).read_bytes()).hexdigest().startswith(prefix), name


@needs_qlm
@pytest.mark.parametrize("requested", CASES)
def test_sim_handling_executes_the_qlm_filtered_command(requested):
    filtered = _qlm_prepare(requested)
    assert np.array_equal(filtered[:6], np.float32(requested[:6]).astype(np.float64))   # QLM: no clip at all
    policy = ZeroPolicy()
    ctrl = RamboController(policy, **command_handling_options("sim", QUADRUPED))
    rs = _state()
    ctrl.reset(rs)
    ctrl.set_commands(vel_cmd=filtered[:3], ee_pos_cmd=filtered[3:6], ee_force_cmd=filtered[6:])
    assert np.array_equal(ctrl.velocity_commands, filtered[:3])
    assert np.array_equal(ctrl.ee_pos_commands[0], filtered[3:6])
    for t in (0.5, 1.5):                       # all four feet in stance (FL planted), then FL in swing
        ctrl.step(t, rs)
        latest = policy.last_obs[4 * NUM_OBS_PER_STEP:]
        assert np.allclose(latest[54:63], filtered, atol=1e-7)   # the policy observes the submitted command
    # FL swing: the joint target is the IK of the UNCLIPPED target (QLM joint_position_controller)
    d = filtered[3:6].copy()
    d[2] -= rs.base_height
    q_fl = kinematics.ik_leg(0, rs.base_rot_mat_rp_t @ d)
    assert np.allclose(ctrl._desired_joint_pos[[0, 4, 8]], q_fl, atol=1e-12)


def test_rambo_handling_is_the_previous_behaviour_and_the_default():
    for ctrl in (RamboController(ZeroPolicy()), RamboController(ZeroPolicy(), **command_handling_options("rambo", QUADRUPED))):
        ctrl.set_commands(vel_cmd=[0.6, -0.55, 0.7], ee_pos_cmd=[0.5845, 0.2506, -0.034], ee_force_cmd=[100, 0, 0])
        assert np.allclose(ctrl.velocity_commands, [0.5, -0.5, 0.5])
        assert np.allclose(ctrl.ee_pos_commands[0], [0.5, 0.2, 0.0])
        assert ctrl.ee_force_commands[0, 0] == 30.0
        assert ctrl.clip_commands and ctrl.zero_commands_in_all_stance


def test_sim_handling_keeps_the_force_clip():
    # the sim zeroes the force; WAM's bridge refuses any non-zero force - the +-30 N clip stays as it was
    ctrl = RamboController(ZeroPolicy(), **command_handling_options("sim", QUADRUPED))
    ctrl.set_commands(ee_force_cmd=[100, 0, 0])
    assert ctrl.ee_force_commands[0, 0] == 30.0


def test_command_handling_names():
    assert command_handling_options("sim", QUADRUPED) == {"clip_commands": False, "zero_commands_in_all_stance": False}
    assert command_handling_options("rambo", QUADRUPED) == {"clip_commands": True, "zero_commands_in_all_stance": True}
    with pytest.raises(ValueError, match="rambo"):
        command_handling_options("none", QUADRUPED)


# ---- the wiring: WAM runs pass "sim"; the controller on its own defaults to "rambo" ----------------------------

def test_the_controller_parameter_defaults_to_rambo():
    yaml = (ROOT / "config/wam_go2.yaml").read_text(encoding="utf-8")
    assert "      command_handling: rambo\n" in yaml
    node = (ROOT / "wam_go2/nodes/controller_node.py").read_text(encoding="utf-8")
    assert 'declare("command_handling", "rambo")' in node
    assert 'command_handling_options(str(p("command_handling")), self.task)' in node
    launch = (ROOT / "launch/wam_go2.launch.py").read_text(encoding="utf-8")
    assert '"command_handling"' in launch and 'default_value="rambo"' in launch
    assert '"command_handling": LaunchConfiguration("command_handling")' in launch


def test_wam_runs_pass_sim():
    script = (ROOT / "scripts/demo/wam_task.sh").read_text(encoding="utf-8")
    assert 'WAM_COMMAND_HANDLING="${WAM_COMMAND_HANDLING:-sim}"' in script
    assert "command_handling:=$WAM_COMMAND_HANDLING" in script
    run = (ROOT / "run_wam.sh").read_text(encoding="utf-8")
    assert 'WAM_COMMAND_HANDLING="${WAM_COMMAND_HANDLING:-sim}"' in run


@pytest.mark.parametrize("value", ["", "sim", "rambo"])
def test_run_json_records_the_command_handling(fakes, value):
    env = {"WAM_MOCK": "1"}
    if value:
        env["WAM_COMMAND_HANDLING"] = value
    r = fakes("push_box", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    (run_dir,) = list((fakes.tmp / "result" / "push_box").iterdir())
    assert json.loads((run_dir / "run.json").read_text())["command_handling"] == (value or "sim")


def test_a_wrong_command_handling_stops_before_anything_starts(fakes):
    r = fakes("push_box", env={"WAM_MOCK": "1", "WAM_COMMAND_HANDLING": "none"})
    assert r.returncode == 64, r.stdout + r.stderr
    assert not (fakes.tmp / "result").exists() and not (fakes.tmp / "task.json").exists()
