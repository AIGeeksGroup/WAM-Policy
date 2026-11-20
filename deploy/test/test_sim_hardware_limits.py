"""The virtual motors must see the same position fence as the real adapter."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def test_simulated_motor_targets_obey_documented_hip_limits():
    pytest.importorskip("mujoco")
    path = Path(__file__).resolve().parents[1] / "sim/mj_sim.py"
    spec = importlib.util.spec_from_file_location("sim_hardware_limits", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not Path(module.SCENE).is_file():
        pytest.skip(f"MuJoCo Go2 model not installed: {module.SCENE} (scripts/fetch_mujoco_menagerie.sh)")
    sim = module.Sim(module.load_model())
    # RAMBO order: four hip joints first, followed by thighs and calves.
    target = np.array([1., -1., .25, -.25] + [.8] * 4 + [-1.5] * 4)
    sim.command(target, kp=40., kd=1.)
    assert np.max(np.abs(sim.q_des[:4])) <= np.radians(57.)          # hip output fence +-57 deg
    np.testing.assert_allclose(sim.q_des[:2], [np.radians(57.), -np.radians(57.)])   # 1 rad = 57.3 deg is cut
    np.testing.assert_array_equal(sim.q_des[2:], target[2:])
    # The motor boundary must not mutate the policy's own target buffer.
    np.testing.assert_array_equal(target[:2], [1., -1.])
