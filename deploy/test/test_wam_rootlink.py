"""wam_go2/wam/rootlink.py: the controller's state as QLM's five root-link fields.

QLM's ``make_observation`` needs the measured root state of every sample: it places each
camera frame (world position and orientation from the root link and the fixed camera mount).
WAM-Policy takes no body state as policy input, so the packet itself carries none; the fields
only serve the frame poses. Root position, xyzw orientation, WORLD linear and angular
velocity, body-frame projected gravity, as plain JSON floats.
"""
import math

import numpy as np
import pytest

from wam_go2.wam import rootlink as rl


def _quat_wxyz(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * axis])


def test_field_names_are_the_five_root_fields_in_protocol_order():
    assert rl.ROOT_FIELDS == (
        "observation.state.base.position", "observation.state.base.orientation",
        "observation.state.base.linear_velocity", "observation.state.base.angular_velocity",
        "observation.state.projected_gravity")
    assert rl.GROUND_Z == 0.0


def test_level_robot_fields_are_trivial():
    f = rl.root_state(height=0.30, quat_wxyz=[1, 0, 0, 0], lin_vel_b=[0.15, 0, 0], ang_vel_b=[0, 0, 0.1])
    assert f["observation.state.base.position"] == [0.0, 0.0, 0.30]
    assert f["observation.state.base.orientation"] == [0.0, 0.0, 0.0, 1.0]
    assert f["observation.state.base.linear_velocity"] == pytest.approx([0.15, 0, 0])
    assert f["observation.state.base.angular_velocity"] == pytest.approx([0, 0, 0.1])
    assert f["observation.state.projected_gravity"] == pytest.approx([0, 0, -1])
    for value in f.values():
        assert all(type(x) is float for x in value)        # plain JSON floats, no numpy


@pytest.mark.parametrize("axis,angle", [([0, 1, 0], 0.3), ([1, 0, 0], -0.2), ([0.3, -0.5, 0.8], 1.1),
                                        ([0, 0, 1], 2.9)])
def test_world_velocities_rotate_back_to_the_body_frame(axis, angle):
    q = _quat_wxyz(axis, angle)
    v_b, w_b = [0.12, -0.03, 0.02], [0.05, -0.2, 0.3]
    f = rl.root_state(height=0.27, quat_wxyz=q, lin_vel_b=v_b, ang_vel_b=w_b)
    q_xyzw = f["observation.state.base.orientation"]
    assert rl.rotate_inverse_xyzw(q_xyzw, f["observation.state.base.linear_velocity"]) == pytest.approx(v_b, abs=1e-9)
    assert rl.rotate_inverse_xyzw(q_xyzw, f["observation.state.base.angular_velocity"]) == pytest.approx(w_b, abs=1e-9)
    assert f["observation.state.projected_gravity"] == pytest.approx(rl.rotate_inverse_xyzw(q_xyzw, [0, 0, -1]))
    assert np.linalg.norm(q_xyzw) == pytest.approx(1.0, abs=1e-12)


def test_quaternion_is_normalised_and_converted_from_wxyz():
    assert rl.quat_xyzw_from_wxyz([2.0, 0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0, 1.0]
    assert rl.quat_xyzw_from_wxyz([0.5, 0.5, 0.5, 0.5]) == pytest.approx([0.5, 0.5, 0.5, 0.5])


@pytest.mark.parametrize("bad", [dict(height=float("nan")), dict(quat_wxyz=[0, 0, 0, 0]),
                                 dict(lin_vel_b=[0, float("inf"), 0]), dict(ang_vel_b=[0, 0]),
                                 dict(quat_wxyz=[1, 0, 0])])
def test_invalid_inputs_raise(bad):
    kw = dict(height=0.3, quat_wxyz=[1, 0, 0, 0], lin_vel_b=[0, 0, 0], ang_vel_b=[0, 0, 0])
    kw.update(bad)
    with pytest.raises(ValueError):
        rl.root_state(**kw)


def test_ground_z_shifts_position_only():
    f = rl.root_state(height=0.3, quat_wxyz=[1, 0, 0, 0], lin_vel_b=[0, 0, 0], ang_vel_b=[0, 0, 0], ground_z=0.05)
    assert f["observation.state.base.position"][2] == pytest.approx(0.35)
