"""wam_go2/wam/observation.py: buffered frames and states -> one protocol snapshot.

A feedback sample is "the two RGB views and the root state at tick T". The
cameras are free-running, so the bridge takes, per camera, the frame whose
capture time is nearest the sample instant within a tolerance, and the
controller state nearest the instant. Frame ids must keep increasing per
camera (the protocol rejects stale or reused frames), both views are 640x360
and pass unchanged (dual-rgb-360p-v1, the only profile: the policy's input),
and anything missing is an ``ObservationGap`` (never fill in, never reuse).
"""
import numpy as np
import pytest

from wam_go2.wam import observation as ob


def _img(h, w, value):
    return np.full((h, w, 3), value, np.uint8)


def _state(t, mode="POLICY", fresh=True, h=0.3, v=(0.1, 0.0, 0.0)):
    return ob.StateRecord(time=t, height=h, quat_wxyz=(1.0, 0.0, 0.0, 0.0), lin_vel_b=v,
                          ang_vel_b=(0.0, 0.0, 0.0), mode=mode, fresh=fresh, action18=[0.0] * 18)


def _filled(ego_period=0.07, d_period=1 / 30, state_period=0.01, t_end=1.0, ego_phase=0.0):
    b = ob.ObservationBuffers(keep_s=2.0)
    i = 0
    t = ego_phase
    while t < t_end:
        b.add_frame("ego", ob.FrameRecord(capture_time=t, frame_id=1000 + i, image=_img(360, 640, i),
                                          receive_time=t + 0.05))
        i += 1; t += ego_period
    j, t = 0, 0.0
    while t < t_end:
        b.add_frame("task_centric", ob.FrameRecord(capture_time=t, frame_id=j, image=_img(360, 640, j),
                                                   receive_time=t + 0.08))
        j += 1; t += d_period
    t = 0.0
    while t < t_end:
        b.add_state(_state(t))
        t += state_period
    return b


TOL = ob.Tolerances(ego_s=0.040, d435i_s=0.020, state_s=0.015)


def test_resolve_picks_nearest_frames_and_state():
    b = _filled()
    snap = b.resolve(tick=40, sim_time_ns=80_000_000, reset_epoch=3, grid_wall=0.50, tolerances=TOL)
    assert snap.physics_step == 40 and snap.simulation_time_ns == 80_000_000 and snap.reset_epoch == 3
    # ego frames at 0.00, 0.07, ..., 0.49, 0.56 -> 0.49 is nearest (|dt| 10 ms)
    assert snap.ego_frame_id == 1007 and snap.ego_offset_s == pytest.approx(-0.010)
    # d435i frames every 33.3 ms: 0.5 exactly is frame 15
    assert snap.d435i_frame_id == 15 and abs(snap.d435i_offset_s) < 1e-9
    assert snap.state_offset_s == pytest.approx(0.0, abs=1e-9)
    packet = snap.snapshot()
    assert set(packet) == {"physics_step", "simulation_time_ns", "reset_epoch", "state",
                           "sensor_frame_ids", "rgb", "rgb_spatial_profile"}
    assert packet["sensor_frame_ids"] == {"ego": 1007, "task_centric": 15}
    assert packet["rgb"]["ego"].shape == (360, 640, 3) and packet["rgb"]["ego"].dtype == np.uint8
    assert int(packet["rgb"]["ego"][0, 0, 0]) == 7
    assert packet["rgb"]["task_centric"].shape == (360, 640, 3)
    assert packet["state"]["observation.state.base.position"] == [0.0, 0.0, 0.3]
    assert packet["state"]["observation.state.base.linear_velocity"] == pytest.approx([0.1, 0, 0])


def test_resolve_refuses_a_frame_outside_tolerance_or_reused():
    b = _filled(ego_period=0.07, t_end=1.0)
    # a 140 ms hole in the ego stream around 0.5: remove the frames at 0.49 and 0.56
    b.frames["ego"] = [f for f in b.frames["ego"] if not (0.48 < f.capture_time < 0.57)]
    with pytest.raises(ob.ObservationGap) as e:
        b.resolve(tick=40, sim_time_ns=80_000_000, reset_epoch=1, grid_wall=0.50, tolerances=TOL)
    assert e.value.reason == "ego_frame_missing"
    b = _filled()
    first = b.resolve(tick=40, sim_time_ns=80_000_000, reset_epoch=1, grid_wall=0.50, tolerances=TOL)
    # a second sample 10 ms later would want the same ego frame: refused as reused
    with pytest.raises(ob.ObservationGap) as e:
        b.resolve(tick=45, sim_time_ns=90_000_000, reset_epoch=1, grid_wall=0.51, tolerances=TOL,
                  last_ids={"ego": first.ego_frame_id, "task_centric": first.d435i_frame_id})
    assert e.value.reason == "ego_frame_reused"


def test_resolve_requires_policy_mode_and_fresh_state():
    b = _filled()
    b.states = [s for s in b.states if not (0.485 < s.time < 0.525)]     # 50 ms state hole around 0.505
    with pytest.raises(ob.ObservationGap) as e:
        b.resolve(tick=40, sim_time_ns=80_000_000, reset_epoch=1, grid_wall=0.505, tolerances=TOL)
    assert e.value.reason == "state_missing"
    b = _filled()
    b.states = [_state(0.50, mode="DAMP")]
    with pytest.raises(ob.ObservationGap) as e:
        b.resolve(tick=40, sim_time_ns=80_000_000, reset_epoch=1, grid_wall=0.50, tolerances=TOL)
    assert e.value.reason == "controller_not_in_policy"
    b = _filled()
    b.states = [_state(0.50, fresh=False)]
    with pytest.raises(ob.ObservationGap) as e:
        b.resolve(tick=40, sim_time_ns=80_000_000, reset_epoch=1, grid_wall=0.50, tolerances=TOL)
    assert e.value.reason == "state_stale"


def test_frames_arriving_late_are_awaited_by_ready_check():
    b = _filled(t_end=0.52)      # the newest ego frame is at 0.49, newest d435i at ~0.50
    # a sample at 0.50 needs frames up to 0.50 + tolerance before it is decidable
    assert not b.ready(grid_wall=0.50, tolerances=TOL)
    b.add_frame("ego", ob.FrameRecord(capture_time=0.56, frame_id=9000, image=_img(360, 640, 1),
                                      receive_time=0.61))
    b.add_frame("task_centric", ob.FrameRecord(capture_time=0.533, frame_id=9001, image=_img(360, 640, 1),
                                               receive_time=0.61))
    b.add_state(_state(0.53))
    assert b.ready(grid_wall=0.50, tolerances=TOL)


def test_buffers_are_bounded_and_reject_bad_images():
    b = ob.ObservationBuffers(keep_s=0.5)
    for i in range(200):
        b.add_frame("task_centric", ob.FrameRecord(capture_time=i * 0.01, frame_id=i,
                                                   image=_img(360, 640, 0), receive_time=i * 0.01))
    assert b.frames["task_centric"][0].capture_time >= 1.99 - 0.5 - 1e-9
    with pytest.raises(ValueError):
        b.add_frame("ego", ob.FrameRecord(capture_time=0.0, frame_id=1, image=np.zeros((10, 10), np.uint8),
                                          receive_time=0.0))
    with pytest.raises(ValueError):
        b.add_frame("nope", ob.FrameRecord(capture_time=0.0, frame_id=1, image=_img(360, 640, 0),
                                           receive_time=0.0))
    with pytest.raises(ValueError):
        b.add_frame("task_centric", ob.FrameRecord(capture_time=0.0, frame_id=1, image=_img(100, 100, 0),
                                                   receive_time=0.0))


# --- the one profile: dual-rgb-360p-v1 (the policy's input) -----------------------------------------

def _filled_360(t_end=1.0):
    b = ob.ObservationBuffers(keep_s=2.0)
    i, t = 0, 0.0
    while t < t_end:
        b.add_frame("ego", ob.FrameRecord(capture_time=t, frame_id=1000 + i, image=_img(360, 640, i),
                                          receive_time=t + 0.03))
        i += 1; t += 1 / 60
    j, t = 0, 0.0
    while t < t_end:
        b.add_frame("task_centric", ob.FrameRecord(capture_time=t, frame_id=j, image=_img(360, 640, 100 + j % 100),
                                                   receive_time=t + 0.05))
        j += 1; t += 1 / 60
    t = 0.0
    while t < t_end:
        b.add_state(_state(t))
        t += 0.01
    return b


def test_profile_constants():
    assert ob.RGB_360 == "dual-rgb-360p-v1" and ob.PROFILE_HW == (360, 640)


def test_buffers_take_640x360_on_both_views_and_refuse_720p():
    b = ob.ObservationBuffers(keep_s=2.0)
    b.add_frame("ego", ob.FrameRecord(0.0, 1, _img(360, 640, 1), 0.0))
    b.add_frame("task_centric", ob.FrameRecord(0.0, 1, _img(360, 640, 1), 0.0))
    for camera in ("ego", "task_centric"):
        with pytest.raises(ValueError):
            b.add_frame(camera, ob.FrameRecord(0.1, 2, _img(720, 1280, 1), 0.1))


def test_snapshot_passes_images_unchanged_and_names_the_profile():
    b = _filled_360()
    snap = b.resolve(tick=40, sim_time_ns=80_000_000, reset_epoch=3, grid_wall=0.50, tolerances=TOL)
    packet = snap.snapshot()
    assert packet["rgb_spatial_profile"] == ob.RGB_360
    assert packet["rgb"]["ego"].shape == (360, 640, 3) and packet["rgb"]["task_centric"].shape == (360, 640, 3)
    assert packet["rgb"]["ego"] is snap.ego.image and packet["rgb"]["task_centric"] is snap.d435i.image


def test_buffers_have_no_profile_choice():
    with pytest.raises(TypeError):
        ob.ObservationBuffers(keep_s=2.0, rgb_profile="dual-rgb-720p-v1")
