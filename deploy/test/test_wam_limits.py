"""wam_go2/wam/limits.py: prediction bounds that STOP the episode, never clamp.

The acknowledgement tells the model what was executed; a command outside the
configured bounds is therefore refused (terminal), not silently altered.
"""
import numpy as np
import pytest

from wam_go2.wam import limits


def test_default_bounds_are_the_documented_envelope():
    b = limits.Bounds()
    assert b.vx == 0.75 and b.vy == 0.75 and b.wz == 0.75
    assert b.ee_x == (0.0, 0.65) and b.ee_y == (-0.15, 0.35) and b.ee_z == (-0.05, 0.50)


def test_in_bounds_command_has_no_violation():
    a = [0.15, 0.0, 0.05, 0.44, 0.142, 0.16, 0.0, 0.0, 0.0]
    assert limits.violations(a, limits.Bounds()) == []


@pytest.mark.parametrize("index,value,name", [
    (0, 0.76, "base_vx"), (0, -0.76, "base_vx"), (1, 0.76, "base_vy"), (2, -0.76, "base_yaw_rate"),
    (3, -0.01, "fl_ee_x"), (3, 0.66, "fl_ee_x"), (4, 0.36, "fl_ee_y"), (5, -0.06, "fl_ee_z"),
])
def test_each_dimension_is_checked_by_name(index, value, name):
    a = [0.0, 0.0, 0.0, 0.3, 0.14, 0.2, 0.0, 0.0, 0.0]
    a[index] = value
    v = limits.violations(a, limits.Bounds())
    assert len(v) == 1 and v[0].startswith(name)


def test_nonfinite_and_nonzero_force_are_violations():
    a = [float("nan"), 0.0, 0.0, 0.3, 0.14, 0.2, 0.0, 0.0, 0.0]
    assert any("finite" in v for v in limits.violations(a, limits.Bounds()))
    a = [0.0, 0.0, 0.0, 0.3, 0.14, 0.2, 0.0, 0.0, 1e-6]
    assert any("force" in v for v in limits.violations(a, limits.Bounds()))
    with pytest.raises(ValueError):
        limits.violations([0.0] * 8, limits.Bounds())


def test_chunk_check_reports_the_first_bad_command():
    good = [0.1, 0.0, 0.0, 0.3, 0.14, 0.2, 0.0, 0.0, 0.0]
    bad = list(good); bad[0] = 2.0
    chunk = [good] * 5 + [bad] + [good] * 10
    index, problems = limits.first_violation(chunk, limits.Bounds())
    assert index == 5 and problems and problems[0].startswith("base_vx")
    assert limits.first_violation([good] * 16, limits.Bounds()) == (None, [])


def test_bounds_from_mapping_and_validation():
    b = limits.Bounds.from_mapping({"vx": 0.2, "ee_z": (0.0, 0.3)})
    assert b.vx == 0.2 and b.ee_z == (0.0, 0.3) and b.vy == 0.75
    with pytest.raises(ValueError):
        limits.Bounds(vx=-0.1)
    with pytest.raises(ValueError):
        limits.Bounds(ee_x=(0.5, 0.1))
    with pytest.raises(ValueError):
        limits.Bounds.from_mapping({"nope": 1.0})
    assert np.isfinite(list(b.as_dict().values())[0])
