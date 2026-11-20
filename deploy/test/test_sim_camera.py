"""Tests for wam_go2/sim_camera.py: real D435i colour image -> the simulation D435i camera.

The simulation export in config/d435i_rgb_simulation/ IS the D435i model; a real frame is
resampled so that every pixel holds the ray the simulated camera would see there (same 1280x720,
same K, no distortion, rotation identity: the bracket is taken as verified). Pixels the real
sensor did not see are black and reported in a validity mask, never filled in.
No ROS, no camera.
"""
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from wam_go2 import sim_camera as sc  # noqa: E402

PKG = Path(__file__).resolve().parents[1]
EXPORT = PKG / "config" / "d435i_rgb_simulation"

# a measured real colour camera_info (640x360x30 on the X5), and the same at 1280x720
REAL_360 = dict(width=640, height=360, k=[455.798, 0.0, 329.013, 0.0, 455.983, 193.844, 0.0, 0.0, 1.0],
                d=[0.0] * 5, distortion_model="plumb_bob")
REAL_720 = dict(width=1280, height=720, k=[911.596, 0.0, 658.026, 0.0, 911.966, 387.688, 0.0, 0.0, 1.0],
                d=[0.0] * 5, distortion_model="plumb_bob")


def _real(**kw):
    return sc.camera_from_info(**kw)


def _project(cam, ray):
    """Pixel of a ray (x, y, z) in the optical frame for an undistorted pinhole `cam`."""
    p = cam.K @ np.asarray(ray, float)
    return p[:2] / p[2]


def _spot(width, height, uv, channels=3, sigma=2.0):
    """A black image with a smooth Gaussian spot (peak 250) centred exactly on the real-valued pixel `uv`
    (a smooth spot keeps the resampled centroid unbiased; a hard-edged square aliases by ~0.1 px)."""
    ys, xs = np.mgrid[0:height, 0:width]
    g = 250.0 * np.exp(-((xs - uv[0]) ** 2 + (ys - uv[1]) ** 2) / (2 * sigma ** 2))
    g = np.round(g).astype(np.uint8)
    return np.repeat(g[..., None], channels, axis=2) if channels > 1 else g


def _dot_image(cam, ray, channels=3):
    return _spot(cam.width, cam.height, _project(cam, ray), channels)


def _centroid(img):
    g = (img if img.ndim == 2 else img[..., 0]).astype(float)
    ys, xs = np.nonzero(g > 2)
    w = g[ys, xs]
    return np.array([np.sum(xs * w) / w.sum(), np.sum(ys * w) / w.sum()])


# ------------------------------------------------------------------ the stored reference

def test_sim_camera_is_the_stored_export():
    sim = sc.load_sim_camera()
    assert (sim.width, sim.height) == (1280, 720)
    data = json.loads((EXPORT / "d435i_rgb_simulation.json").read_text())
    np.testing.assert_allclose(sim.K, np.array(data["intrinsics"]["K"]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(sim.K, np.loadtxt(EXPORT / "K.txt"), rtol=0, atol=1e-9)
    assert sim.K[0, 0] == pytest.approx(924.277379745929) and sim.K[1, 1] == pytest.approx(924.277379745929)
    assert (sim.K[0, 2], sim.K[1, 2]) == (640.0, 360.0)
    assert not np.any(sim.D)


def test_default_path_points_into_this_repo():
    assert sc.SIM_JSON == EXPORT / "d435i_rgb_simulation.json"


def test_load_rejects_a_distorted_or_malformed_sim_file(tmp_path):
    data = json.loads((EXPORT / "d435i_rgb_simulation.json").read_text())
    data["intrinsics"]["opencv_distortion_coefficients"] = [0.1, 0, 0, 0, 0]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="distortion"):
        sc.load_sim_camera(p)
    data = json.loads((EXPORT / "d435i_rgb_simulation.json").read_text())
    data["intrinsics"]["K"][0][0] = -1.0
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        sc.load_sim_camera(p)


def test_sim_camera_info_fields():
    sim = sc.load_sim_camera()
    f = sc.camera_info_fields(sim)
    assert (f["width"], f["height"], f["distortion_model"]) == (1280, 720, "plumb_bob")
    assert f["d"] == [0.0] * 5
    assert f["k"] == pytest.approx(sim.K.ravel().tolist())
    assert f["r"] == [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
    assert f["p"] == pytest.approx([sim.K[0, 0], 0, 640.0, 0, 0, sim.K[1, 1], 360.0, 0, 0, 0, 1.0, 0])


# ------------------------------------------------------------------ geometry

def test_identity_when_the_real_camera_equals_the_sim():
    sim = sc.load_sim_camera()
    r = sc.SimRemapper(sim)
    r.update(sc.camera_from_info(width=1280, height=720, k=sim.K.ravel().tolist(), d=[0.0] * 5,
                                 distortion_model="plumb_bob"))
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    out = r.apply(img)
    assert out.shape == (720, 1280, 3) and out.dtype == np.uint8
    assert r.valid.all()
    np.testing.assert_array_equal(out, img)


@pytest.mark.parametrize("ray", [(0.0, 0.0, 1.0), (0.3, -0.2, 1.0), (-0.5, 0.25, 1.0), (0.6, 0.3, 1.0)])
def test_a_ray_lands_where_the_sim_camera_sees_it(ray):
    sim = sc.load_sim_camera()
    real = _real(**REAL_720)
    r = sc.SimRemapper(sim)
    r.update(real)
    np.testing.assert_allclose(_centroid(r.apply(_dot_image(real, ray))), _project(sim, ray), atol=0.05)


def test_plumb_bob_distortion_is_removed():
    sim = sc.load_sim_camera()
    d = [-0.06, 0.08, 0.001, -0.0015, -0.02]
    real = sc.camera_from_info(width=1280, height=720, k=REAL_720["k"], d=d, distortion_model="plumb_bob",
                               allow_distortion=True)
    r = sc.SimRemapper(sim)
    r.update(real)
    for ray in [(0.4, 0.2, 1.0), (-0.55, -0.3, 1.0)]:
        uv, _ = cv2.projectPoints(np.array([ray], float), np.zeros(3), np.zeros(3), real.K, np.array(d))
        img = _spot(1280, 720, uv.ravel(), channels=1)
        # the lens distortion bends the spot a little, so a bit more slack than the pinhole case
        np.testing.assert_allclose(_centroid(r.apply(img)), _project(sim, ray), atol=0.15)


def test_unknown_distortion_model_is_refused_unless_all_zero():
    k = REAL_720["k"]
    with pytest.raises(ValueError, match="distortion"):
        sc.camera_from_info(width=1280, height=720, k=k, d=[0.1, 0, 0, 0, 0], distortion_model="inverse_brown_conrady")
    cam = sc.camera_from_info(width=1280, height=720, k=k, d=[0.0] * 5, distortion_model="inverse_brown_conrady")
    assert not np.any(cam.D)
    cam = sc.camera_from_info(width=1280, height=720, k=k, d=[], distortion_model="")
    assert not np.any(cam.D)


def test_invalid_intrinsics_are_refused():
    with pytest.raises(ValueError):
        sc.camera_from_info(width=1280, height=720, k=[0.0] * 9, d=[], distortion_model="plumb_bob")
    with pytest.raises(ValueError):
        sc.camera_from_info(width=0, height=720, k=REAL_720["k"], d=[], distortion_model="plumb_bob")


# ------------------------------------------------------------------ what the real sensor did not see

def test_unseen_border_of_the_measured_camera_is_black_and_masked():
    """The measured real principal point sits (+18, +28) px off centre at 1280x720, so the sim frame's
    right 9 columns and bottom 23 rows have no real pixel; they must be exactly 0 even for a white
    input, and the mask must say so."""
    sim = sc.load_sim_camera()
    r = sc.SimRemapper(sim)
    r.update(_real(**REAL_720))
    out = r.apply(np.full((720, 1280, 3), 255, np.uint8))
    bad_cols = np.nonzero((~r.valid).all(axis=0) | (~r.valid[:600]).any(axis=0))[0]
    bad_rows = np.nonzero((~r.valid[:, :1200]).any(axis=1))[0]
    assert bad_cols.tolist() == list(range(1271, 1280))
    assert bad_rows.tolist() == list(range(697, 720))
    assert np.all(out[~r.valid] == 0)
    assert np.all(out[r.valid] == 255)
    cov = r.coverage
    assert cov["valid_fraction"] == pytest.approx(1 - 0.0388, abs=0.001)
    assert (cov["right_cols"], cov["bottom_rows"], cov["left_cols"], cov["top_rows"]) == (9, 23, 0, 0)
    assert cov["upsampled"] is False
    mask = r.mask_u8()
    assert mask.dtype == np.uint8 and set(np.unique(mask)) == {0, 255}
    assert np.array_equal(mask == 255, r.valid)


def test_half_resolution_real_stream_is_flagged_upsampled():
    sim = sc.load_sim_camera()
    r = sc.SimRemapper(sim)
    r.update(_real(**REAL_360))
    out = r.apply(np.full((360, 640, 3), 200, np.uint8))
    assert out.shape == (720, 1280, 3)
    cov = r.coverage
    assert cov["upsampled"] is True
    assert cov["scale"] == pytest.approx(455.798 / (924.277379745929 / 2) / 2, rel=1e-6)
    # same field of view as the full-resolution profile, up to the half-pixel edge rule
    assert abs(cov["right_cols"] - 9) <= 2 and abs(cov["bottom_rows"] - 23) <= 2
    assert np.all(out[~r.valid] == 0) and np.all(out[r.valid] == 200)


# ------------------------------------------------------------------ remapper state

def test_update_rebuilds_only_on_a_new_calibration():
    sim = sc.load_sim_camera()
    r = sc.SimRemapper(sim)
    assert r.update(_real(**REAL_720)) is True
    assert r.update(_real(**REAL_720)) is False
    assert r.update(_real(**REAL_360)) is True
    assert r.version == 2


def test_apply_before_any_calibration_raises():
    r = sc.SimRemapper(sc.load_sim_camera())
    with pytest.raises(RuntimeError, match="camera_info"):
        r.apply(np.zeros((720, 1280, 3), np.uint8))


def test_apply_refuses_a_frame_of_another_size():
    r = sc.SimRemapper(sc.load_sim_camera())
    r.update(_real(**REAL_720))
    with pytest.raises(ValueError, match="640x360"):
        r.apply(np.zeros((360, 640, 3), np.uint8))


def test_mono_and_colour_frames():
    sim = sc.load_sim_camera()
    real = _real(**REAL_720)
    r = sc.SimRemapper(sim)
    r.update(real)
    ray = (0.2, 0.1, 1.0)
    for ch in (1, 3):
        out = r.apply(_dot_image(real, ray, channels=ch))
        assert out.shape == ((720, 1280) if ch == 1 else (720, 1280, 3))
        np.testing.assert_allclose(_centroid(out), _project(sim, ray), atol=0.05)


# ------------------------------------------------------------------ distortion models and identity

def test_nonzero_distortion_needs_an_explicit_opt_in():
    """realsense-ros labels every librealsense model except Kannala-Brandt 'plumb_bob', and
    the D4xx colour intrinsics are inverse Brown-Conrady: nonzero coefficients would be
    applied in the wrong direction (48 px at the corner for k1 = -0.05). Refuse them unless
    the caller states they are OpenCV forward coefficients."""
    k = REAL_720["k"]
    with pytest.raises(ValueError, match="distortion"):
        sc.camera_from_info(width=1280, height=720, k=k, d=[-0.05, 0, 0, 0, 0], distortion_model="plumb_bob")
    cam = sc.camera_from_info(width=1280, height=720, k=k, d=[-0.05, 0, 0, 0, 0], distortion_model="plumb_bob",
                              allow_distortion=True)
    assert cam.D[0] == -0.05


@pytest.mark.parametrize("bad", [[float("nan"), 0, 0, 0, 0], [0, float("inf"), 0, 0, 0]])
def test_non_finite_distortion_is_refused(bad):
    with pytest.raises(ValueError, match="finite"):
        sc.camera_from_info(width=1280, height=720, k=REAL_720["k"], d=bad, distortion_model="plumb_bob",
                            allow_distortion=True)


@pytest.mark.parametrize("model", ["equidistant", "fisheye", "kannala_brandt4"])
def test_fisheye_models_are_refused_even_with_zero_coefficients(model):
    """Kannala-Brandt with zero coefficients is r_d = atan(r), not a pinhole."""
    with pytest.raises(ValueError, match=model):
        sc.camera_from_info(width=1280, height=720, k=REAL_720["k"], d=[0.0] * 4, distortion_model=model)


def test_cameras_compare_by_identity():
    """Same size, different K: a field-wise == that skipped K would call them equal."""
    a = _real(**REAL_720)
    b = sc.camera_from_info(width=1280, height=720, k=[900.0, 0, 640, 0, 900.0, 360, 0, 0, 1],
                            d=[0.0] * 5, distortion_model="plumb_bob")
    assert a != b and a == a and a.key() != b.key()


# --- the 640x360 sim stream for WAM-Policy (dual-rgb-360p-v1) ---------------------------------------

def test_scaled_sim_camera_is_the_qlm_360p_d435i_profile():
    pytest.importorskip("qlm_bench.contracts.rgb_profiles", reason="QLM-Bench not on PYTHONPATH")
    from qlm_bench.contracts.command.validation import BUNDLE
    from qlm_bench.contracts.rgb_profiles import RGB_360, camera_profile
    half = sc.scaled(sc.load_sim_camera(), 0.5)
    qlm = camera_profile(BUNDLE["d435i_rgb_task"], RGB_360)
    assert (half.width, half.height) == (qlm["width"], qlm["height"]) == (640, 360)
    assert np.allclose(half.K, np.asarray(qlm["intrinsics"], float), atol=1e-3, rtol=0)
    assert np.all(half.D == 0) and half.distortion_model == "plumb_bob"


def test_scale_must_give_whole_pixels():
    with pytest.raises(ValueError):
        sc.scaled(sc.load_sim_camera(), 0.33)         # 422.4 x 237.6 pixels
    with pytest.raises(ValueError):
        sc.scaled(sc.load_sim_camera(), 0.0)
    assert sc.scaled(sc.load_sim_camera(), 1.0).key() == sc.load_sim_camera().key()
