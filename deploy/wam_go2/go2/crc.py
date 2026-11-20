"""Unitree Go2 LowCmd CRC32.

The firmware validates LowCmd with a custom CRC computed over the raw C++
struct memory (unitree_sdk2 include/unitree/dds_wrapper/common/crc.h and
unitree_ros2 example/src/src/common/motor_crc.cpp):

    crc32_core(ptr, len): init 0xFFFFFFFF, poly 0x04C11DB7, processes each
    little-endian uint32 word MSB-first, no reflection, no final xor.
    len = sizeof(LowCmd)/4 - 1 = 202 words (808 bytes, crc field excluded).

That bit algorithm is exactly CRC-32/MPEG-2 applied to the byte stream with
each 4-byte group reversed (word-wise big-endian). `crc32_fast` uses a table
over that reordering; `crc32_core_reference` is the plain bit loop of that
algorithm, used to validate it in the tests.

The 812-byte struct layout (x86-64/aarch64 natural alignment, verified with
offsetof):
  head@0(2) level_flag@2 frame_reserve@3 sn@4(8) version@12(8) bandwidth@20(2)
  pad@22(2) motor_cmd@24 (20 x 36: mode@0 pad(3) q@4 dq@8 tau@12 kp@16 kd@20
  reserve@24(12)) bms@744 (off@744 reserve(3)) wireless_remote@748(40)
  led@788(12) fan@800(2) gpio@802(1) pad@803(1) reserve@804(4) crc@808(4)
"""

import struct

import numpy as np

_POLY = 0x04C11DB7


def crc32_core_reference(words):
    """The crc32_core bit loop, word by word (slow, for tests)."""
    crc = 0xFFFFFFFF
    for data in words:
        xbit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ _POLY
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if data & xbit:
                crc ^= _POLY
            xbit >>= 1
    return crc


def _make_table():
    table = []
    for byte in range(256):
        crc = byte << 24
        for _ in range(8):
            crc = ((crc << 1) & 0xFFFFFFFF) ^ _POLY if crc & 0x80000000 \
                else (crc << 1) & 0xFFFFFFFF
        table.append(crc)
    return table


_TABLE = _make_table()


def crc32_fast(buf):
    """CRC over a bytes-like of length multiple of 4 (little-endian words)."""
    # reverse each 4-byte group so the word's MSB comes first
    arr = np.frombuffer(bytes(buf), dtype="<u4")
    data = arr.astype(">u4").tobytes()
    crc = 0xFFFFFFFF
    table = _TABLE
    for b in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ table[((crc >> 24) ^ b) & 0xFF]
    return crc


def pack_lowcmd_for_crc(msg):
    """Serialize a unitree_go/LowCmd ROS message into the 808-byte C-struct
    image the firmware computes the CRC over (padding bytes are zero)."""
    buf = bytearray(808)
    struct.pack_into("<2B", buf, 0, msg.head[0], msg.head[1])
    struct.pack_into("<2B", buf, 2, msg.level_flag, msg.frame_reserve)
    struct.pack_into("<2I", buf, 4, msg.sn[0], msg.sn[1])
    struct.pack_into("<2I", buf, 12, msg.version[0], msg.version[1])
    struct.pack_into("<H", buf, 20, msg.bandwidth)
    # 2 bytes padding at 22
    off = 24
    for m in msg.motor_cmd:
        struct.pack_into("<B", buf, off, m.mode)
        struct.pack_into("<5f", buf, off + 4, m.q, m.dq, m.tau, m.kp, m.kd)
        struct.pack_into("<3I", buf, off + 24,
                         m.reserve[0], m.reserve[1], m.reserve[2])
        off += 36
    struct.pack_into("<4B", buf, 744, msg.bms_cmd.off, msg.bms_cmd.reserve[0],
                     msg.bms_cmd.reserve[1], msg.bms_cmd.reserve[2])
    struct.pack_into("<40B", buf, 748, *msg.wireless_remote)
    struct.pack_into("<12B", buf, 788, *msg.led)
    struct.pack_into("<2B", buf, 800, msg.fan[0], msg.fan[1])
    struct.pack_into("<B", buf, 802, msg.gpio)
    # 1 byte padding at 803
    struct.pack_into("<I", buf, 804, msg.reserve)
    return bytes(buf)


def compute_lowcmd_crc(msg):
    """CRC value to store into msg.crc before publishing."""
    return crc32_fast(pack_lowcmd_for_crc(msg))
