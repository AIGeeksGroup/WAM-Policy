"""wam_go2/state_stream.py: the controller's state as one flat Float64MultiArray.

The bridge needs, at 100 Hz, the quantities the controller already computes
(estimator height, body velocity, IMU quaternion and gyro) together with the
mode, freshness and timing it needs to judge them. One flat array with a
versioned layout keeps the controller free of new message packages; the
encoder and decoder live in one ROS-free module so both sides agree by test.
"""
import pytest

from wam_go2 import state_stream as ss


def _sample():
    return dict(wall=1.7e9 + 0.123456789, tick_ms=123456, mode="POLICY", tripped=False, fresh=True,
                estimator_valid=True, base_height=0.31, quat_wxyz=(0.99, 0.01, -0.02, 0.1),
                lin_vel_b=(0.15, -0.01, 0.0), ang_vel_b=(0.01, 0.0, -0.05), source_age=0.003,
                control_gap=0.0101, action18=[0.1 * i for i in range(18)])


def test_round_trip_preserves_every_field():
    values = ss.encode(**_sample())
    assert len(values) == ss.LENGTH and all(isinstance(v, float) for v in values)
    assert values[0] == ss.VERSION
    d = ss.decode(values)
    s = _sample()
    assert d.wall == pytest.approx(s["wall"], abs=1e-6) and d.tick_ms == 123456
    assert d.mode == "POLICY" and d.tripped is False and d.fresh is True and d.estimator_valid is True
    assert d.base_height == pytest.approx(0.31)
    assert d.quat_wxyz == pytest.approx(s["quat_wxyz"]) and d.lin_vel_b == pytest.approx(s["lin_vel_b"])
    assert d.ang_vel_b == pytest.approx(s["ang_vel_b"])
    assert d.source_age == pytest.approx(0.003) and d.control_gap == pytest.approx(0.0101)
    assert d.action18 == pytest.approx(s["action18"])
    assert d.state_time == pytest.approx(s["wall"] - 0.003, abs=1e-6)     # the LowState's arrival


@pytest.mark.parametrize("mode", ["DAMP", "STAND", "POLICY"])
def test_every_mode_has_a_code(mode):
    d = ss.decode(ss.encode(**dict(_sample(), mode=mode, tripped=True)))
    assert d.mode == mode and d.tripped is True


def test_decode_rejects_other_versions_and_lengths():
    values = ss.encode(**_sample())
    with pytest.raises(ValueError):
        ss.decode(values[:-1])
    bad = list(values); bad[0] = 99.0
    with pytest.raises(ValueError):
        ss.decode(bad)
    with pytest.raises(ValueError):
        ss.encode(**dict(_sample(), mode="NOPE"))
    with pytest.raises(ValueError):
        ss.encode(**dict(_sample(), action18=[0.0] * 17))


def test_no_action_outside_policy_is_zeros_and_nan_is_refused():
    d = ss.decode(ss.encode(**dict(_sample(), mode="STAND", action18=None)))
    assert d.action18 == [0.0] * 18
    with pytest.raises(ValueError):
        ss.encode(**dict(_sample(), base_height=float("nan")))


def test_to_state_record_maps_onto_the_observation_record():
    d = ss.decode(ss.encode(**_sample()))
    rec = d.state_record()
    assert rec.time == pytest.approx(d.state_time) and rec.mode == "POLICY" and rec.fresh is True
    assert rec.height == pytest.approx(0.31) and tuple(rec.quat_wxyz) == pytest.approx(_sample()["quat_wxyz"])
    d2 = ss.decode(ss.encode(**dict(_sample(), tripped=True)))
    assert d2.state_record().mode == "POLICY (safety)"
