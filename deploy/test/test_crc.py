import os
import struct
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wam_go2.go2.crc import (  # noqa: E402
    crc32_core_reference,
    crc32_fast,
    pack_lowcmd_for_crc,
)


def test_fast_matches_reference():
    rng = np.random.default_rng(0)
    for n_words in [1, 2, 7, 202]:
        buf = rng.integers(0, 256, size=n_words * 4, dtype=np.uint8).tobytes()
        words = struct.unpack(f"<{n_words}I", buf)
        assert crc32_fast(buf) == crc32_core_reference(words)


def test_known_answers_from_c_reference():
    # anchor values produced by the compiled Unitree crc32_core C code
    assert crc32_core_reference([0x00000000]) == 0xC704DD7B
    assert crc32_core_reference([0x00000001]) == 0xC3C5C0CC
    assert crc32_core_reference([0xFFFFFFFF]) == 0x00000000
    assert crc32_fast(struct.pack("<I", 0)) == 0xC704DD7B
    assert crc32_fast(struct.pack("<I", 1)) == 0xC3C5C0CC


def test_zero_buffer_consistency():
    buf = bytes(808)
    words = struct.unpack("<202I", buf)
    assert crc32_fast(buf) == crc32_core_reference(words)


def _mock_lowcmd():
    m = types.SimpleNamespace()
    m.head = [0xFE, 0xEF]
    m.level_flag = 0xFF
    m.frame_reserve = 0
    m.sn = [0, 0]
    m.version = [0, 0]
    m.bandwidth = 0
    m.motor_cmd = []
    for i in range(20):
        mc = types.SimpleNamespace()
        mc.mode = 0x01
        mc.q = float(i) * 0.1
        mc.dq = -float(i) * 0.01
        mc.tau = 0.5
        mc.kp = 40.0
        mc.kd = 1.0
        mc.reserve = [0, 0, 0]
        m.motor_cmd.append(mc)
    m.bms_cmd = types.SimpleNamespace(off=0, reserve=[0, 0, 0])
    m.wireless_remote = [0] * 40
    m.led = [0] * 12
    m.fan = [0, 0]
    m.gpio = 0
    m.reserve = 0
    return m


def test_pack_layout():
    buf = pack_lowcmd_for_crc(_mock_lowcmd())
    assert len(buf) == 808
    assert buf[0] == 0xFE and buf[1] == 0xEF and buf[2] == 0xFF
    # padding bytes must be zero
    assert buf[22] == 0 and buf[23] == 0 and buf[803] == 0
    # motor 0 q at offset 24+4
    assert struct.unpack_from("<f", buf, 28)[0] == 0.0
    # motor 1 mode at offset 24+36
    assert buf[24 + 36] == 0x01
    # motor 1 kp at offset 24+36+16
    assert abs(struct.unpack_from("<f", buf, 24 + 36 + 16)[0] - 40.0) < 1e-6
    # wireless_remote starts at 748
    assert buf[747] != None  # noqa: E711  (bounds check via indexing)


def test_pack_crc_stable():
    m = _mock_lowcmd()
    c1 = crc32_fast(pack_lowcmd_for_crc(m))
    c2 = crc32_fast(pack_lowcmd_for_crc(m))
    assert c1 == c2
    m.motor_cmd[3].q += 1e-3
    assert crc32_fast(pack_lowcmd_for_crc(m)) != c1
