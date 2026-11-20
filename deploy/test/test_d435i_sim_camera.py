"""ROS-free tests for scripts/x5/d435i_sim_camera.py: the laptop node that republishes the real
D435i colour stream as the simulation D435i camera (Real = Sim).
Fake CameraInfo / Image objects only; no ROS, no camera.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")

X5_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "x5"))
if X5_DIR not in sys.path:
    sys.path.insert(0, X5_DIR)
import d435i_sim_camera as node  # noqa: E402
from wam_go2 import sim_camera as sc  # noqa: E402

K720 = [911.596, 0.0, 658.026, 0.0, 911.966, 387.688, 0.0, 0.0, 1.0]
K360 = [455.798, 0.0, 329.013, 0.0, 455.983, 193.844, 0.0, 0.0, 1.0]


def _info(w=1280, h=720, k=K720, d=(0.0,) * 5, model="plumb_bob"):
    return SimpleNamespace(width=w, height=h, k=list(k), d=list(d), distortion_model=model)


def _image(arr, encoding="rgb8", pad=0):
    h, w = arr.shape[:2]
    ch = 1 if arr.ndim == 2 else arr.shape[2]
    step = w * ch + pad
    buf = np.zeros((h, step), np.uint8)
    buf[:, :w * ch] = arr.reshape(h, w * ch)
    return SimpleNamespace(width=w, height=h, encoding=encoding, step=step, data=buf.tobytes(),
                           header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=2), frame_id="d435i_color_optical_frame"))


def _rep():
    logs = []
    return node.Republisher(sc.load_sim_camera(), log=logs.append), logs


def test_output_topics():
    assert node.output_topics("/camera/d435i") == {
        "image": "/camera/d435i/sim/color/image_raw",
        "info": "/camera/d435i/sim/color/camera_info",
        "mask": "/camera/d435i/sim/color/valid_mask",
    }


def test_frame_before_camera_info_is_skipped():
    rep, _ = _rep()
    assert rep.on_image(_image(np.zeros((720, 1280, 3), np.uint8))) is None
    assert rep.stats["skipped_no_info"] == 1 and rep.stats["out"] == 0


def test_rgb8_frame_with_row_padding_becomes_a_sim_frame():
    rep, logs = _rep()
    assert rep.on_info(_info()) is True                     # first calibration -> remap built
    assert any("valid" in line for line in logs)            # coverage logged
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    out = rep.on_image(_image(arr, pad=64))
    assert out is not None
    img, encoding = out
    assert img.shape == (720, 1280, 3) and encoding == "rgb8"
    ref = sc.SimRemapper(sc.load_sim_camera())
    ref.update(sc.camera_from_info(width=1280, height=720, k=K720, d=[0.0] * 5, distortion_model="plumb_bob"))
    np.testing.assert_array_equal(img, ref.apply(arr))
    assert rep.stats["out"] == 1


@pytest.mark.parametrize("encoding,shape", [("bgr8", (720, 1280, 3)), ("mono8", (720, 1280))])
def test_other_8bit_encodings_pass_through(encoding, shape):
    rep, _ = _rep()
    rep.on_info(_info())
    img, enc = rep.on_image(_image(np.full(shape, 7, np.uint8), encoding=encoding))
    assert enc == encoding and img.shape == shape


def test_unsupported_encoding_is_skipped_and_warned_once():
    rep, logs = _rep()
    rep.on_info(_info())
    for _ in range(3):
        assert rep.on_image(_image(np.zeros((720, 1280, 2), np.uint8), encoding="yuv422")) is None
    assert rep.stats["skipped_encoding"] == 3
    assert sum("yuv422" in line for line in logs) == 1


def test_frame_of_another_size_than_camera_info_is_skipped():
    rep, logs = _rep()
    rep.on_info(_info())
    assert rep.on_image(_image(np.zeros((360, 640, 3), np.uint8))) is None
    assert rep.stats["skipped_size"] == 1
    assert sum("640x360" in line for line in logs) == 1


def test_calibration_change_rebuilds_and_flags_the_mask():
    rep, logs = _rep()
    assert rep.on_info(_info()) is True
    assert rep.take_mask() is not None                      # new mask to publish once
    assert rep.take_mask() is None
    assert rep.on_info(_info()) is False                    # same camera_info every frame: no rebuild
    assert rep.take_mask() is None
    assert rep.on_info(_info(640, 360, K360)) is True       # profile change on the X5
    mask = rep.take_mask()
    assert mask.shape == (720, 1280) and mask.dtype == np.uint8
    assert any("upsampled" in line for line in logs)       # a 640x360 stream cannot be the 1280x720 sim camera


def test_bad_camera_info_is_reported_not_raised():
    rep, logs = _rep()
    assert rep.on_info(_info(d=(0.1, 0, 0, 0, 0), model="inverse_brown_conrady")) is False
    assert rep.stats["bad_info"] == 1 and any("distortion" in line for line in logs)
    assert rep.on_image(_image(np.zeros((720, 1280, 3), np.uint8))) is None


def test_sim_camera_info_matches_the_export():
    f = node.sim_info_fields(sc.load_sim_camera())
    assert (f["width"], f["height"]) == (1280, 720)
    assert f["k"][0] == pytest.approx(924.277379745929) and f["k"][2] == 640.0 and f["k"][5] == 360.0


def test_summary_line():
    rep, _ = _rep()
    rep.on_info(_info())
    rep.on_image(_image(np.zeros((720, 1280, 3), np.uint8)))
    s = rep.summary()
    assert "in 1" in s and "out 1" in s and "valid 96.1 %" in s


def test_message_payload_is_an_array_not_bytes():
    """rosidl's uint8[] setter checks every element of a bytes value in Python: 107 ms for one
    1280x720 rgb8 frame (measured: the node then keeps only 9 of 15 fps); an
    array.array('B') is taken as is (1.1 ms)."""
    import array
    img = np.arange(720 * 1280 * 3, dtype=np.uint64).astype(np.uint8).reshape(720, 1280, 3)
    payload = node.msg_payload(img)
    assert isinstance(payload, array.array) and payload.typecode == "B"
    assert payload.tobytes() == img.tobytes()
    mask = np.zeros((720, 1280), np.uint8)
    assert len(node.msg_payload(mask)) == 720 * 1280


@pytest.mark.parametrize("change", [dict(step=1280), dict(truncate=100)])
def test_malformed_frame_is_skipped_not_raised(change):
    rep, logs = _rep()
    rep.on_info(_info())
    msg = _image(np.zeros((720, 1280, 3), np.uint8))
    if "step" in change:
        msg.step = change["step"]                       # shorter than width * 3
    else:
        msg.data = msg.data[:-change["truncate"]]       # fewer bytes than height * step
    assert rep.on_image(msg) is None
    assert rep.stats["skipped_malformed"] == 1
    assert any("malformed" in line for line in logs)


def test_main_limits_opencv_threads():
    import inspect
    assert "cv2.setNumThreads(1)" in inspect.getsource(node.main)


def test_the_default_output_is_the_640x360_sim_camera():
    # the policy takes dual-rgb-360p-v1: the default is its 640x360 camera
    sim = node.sim_camera(node.parse_args([]))
    assert node.parse_args([]).scale == 0.5 and (sim.width, sim.height) == (640, 360)
    assert sim.K[0, 0] == pytest.approx(924.277379745929 / 2) and (sim.K[0, 2], sim.K[1, 2]) == (320.0, 180.0)
    assert node.sim_camera(node.parse_args(["--scale", "1.0"])).width == 1280      # the full export, by hand only
