"""wam_go2/ego_stream.py: the Go2 front camera stream as timed RGB frames.

The firmware's /frontvideostream wire layout, the H.264 decode and the arrival
envelope time model are the field-verified ones of scripts/x5/ego_d435i_view.py;
this module makes them importable by the ego_stream node. A guard test keeps the copied wire parser identical to the
viewer's, and the time model is checked on synthetic arrivals.
"""
import os
import struct

import numpy as np
import pytest

from wam_go2 import ego_stream as es

TF_PER_FRAME = 1e5 / 15        # the counter's step per frame (1/15 s in 10 us units)


def _message(tf, res, payload):
    return b"\x00\x01\x00\x00" + struct.pack("<QII", tf, res, len(payload)) + payload


def test_parse_frontvideo_matches_the_firmware_layout():
    tf, res, data = es.parse_frontvideo(_message(123456789, 360, b"\x00\x00\x00\x01\x65abc"))
    assert (tf, res, data) == (123456789, 360, b"\x00\x00\x00\x01\x65abc")
    with pytest.raises(struct.error):
        es.parse_frontvideo(b"\x00\x01\x00")


def test_wire_parser_is_byte_identical_to_the_viewers():
    """The viewer is the field-verified reference; never let the two drift apart."""
    import inspect
    here = os.path.dirname(os.path.abspath(__file__))
    viewer = open(os.path.join(here, "..", "scripts", "x5", "ego_d435i_view.py"), encoding="utf-8").read()
    src = inspect.getsource(es.parse_frontvideo)
    assert src in viewer
    assert inspect.getsource(es.Go2Decoder) in viewer


def test_time_model_fits_the_arrival_envelope_and_applies_the_latency():
    rng = np.random.default_rng(0)
    model = es.EgoTimeModel(latency_s=0.044, min_samples=30)
    assert model.time_of(0) is None                      # no fit yet
    t0, tf0 = 1000.0, 5_000_000
    for i in range(120):
        tf = tf0 + int(round(i * TF_PER_FRAME))
        arrival = t0 + i * 0.0703 + rng.uniform(0.0, 0.004)      # 14.23 fps, +0..4 ms delivery jitter
        model.add(tf, arrival)
    t = model.time_of(tf0 + int(round(60 * TF_PER_FRAME)))
    assert t is not None
    # the envelope follows the EARLIEST arrivals (the true frame time) minus the latency
    assert abs((t + 0.044) - (t0 + 60 * 0.0703)) < 0.003
    assert model.fit is not None and model.fit.n_used >= 30
    assert model.calibrated is True
    assert es.EgoTimeModel(latency_s=0.0).calibrated is False


def test_time_model_refits_periodically_and_tolerates_bad_data():
    model = es.EgoTimeModel(latency_s=0.0, min_samples=10, refit_s=1.0)
    for i in range(20):
        model.add(100 + i, 50.0)            # constant arrivals: a degenerate fit must not raise
    assert model.time_of(100) is None or np.isfinite(model.time_of(100))


def test_decoder_decodes_a_synthetic_h264_stream():
    av = pytest.importorskip("av")
    try:
        enc = av.CodecContext.create("libx264", "w")
    except Exception:                      # no software encoder in this PyAV build
        pytest.skip("no H.264 encoder available")
    enc.width, enc.height, enc.pix_fmt = 64, 48, "yuv420p"
    from fractions import Fraction
    enc.time_base = Fraction(1, 15)
    enc.options = {"tune": "zerolatency", "preset": "ultrafast"}
    packets = []
    for i in range(6):
        frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 40 * i, np.uint8), format="rgb24")
        frame.pts = i
        for p in enc.encode(frame):
            packets.append(bytes(p))
    packets += [bytes(p) for p in enc.encode(None)]
    dec = es.Go2Decoder()
    frames = []
    for i, data in enumerate(packets):
        frames += dec.decode(data, 1000 + i)
    assert len(frames) >= 4
    tf, img = frames[0]
    assert img.shape == (48, 64, 3) and img.dtype == np.uint8
    assert dec.keyframe_seen is True
    rgb = es.to_rgb(img)
    assert rgb.shape == img.shape and rgb[..., 0].mean() == pytest.approx(img[..., 2].mean(), abs=1e-6)
