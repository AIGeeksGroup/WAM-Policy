"""Length-framed local IPC. No model, torch, MuJoCo or GL imports."""
from __future__ import annotations

import hashlib
import json
import struct
from typing import BinaryIO

import msgpack
import numpy as np

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 256 * 1024 * 1024
CAMERAS = ("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb")


class ProtocolError(RuntimeError):
    pass


def _pack_array(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ProtocolError("object arrays are forbidden")
        value = np.ascontiguousarray(value)
        return {"__ndarray__": True, "dtype": value.dtype.str,
                "shape": value.shape, "data": value.tobytes()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _unpack_array(value):
    if value.get("__ndarray__") is True:
        dtype = np.dtype(value["dtype"])
        if dtype.hasobject:
            raise ProtocolError("object arrays are forbidden")
        shape = tuple(int(n) for n in value["shape"])
        if any(n < 0 for n in shape):
            raise ProtocolError("negative array dimension")
        raw = value["data"]
        count = 1
        for n in shape:
            count *= n
        if count * dtype.itemsize != len(raw):
            raise ProtocolError("array shape/data mismatch")
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    return value


def pack(value) -> bytes:
    return msgpack.packb(value, default=_pack_array, use_bin_type=True)


def unpack(data: bytes):
    return msgpack.unpackb(data, object_hook=_unpack_array, raw=False,
                           strict_map_key=False)


def digest(value) -> str:
    """Order-independent mapping hash, including exact ndarray bytes/dtype/shape."""
    def canonical(v):
        if isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v)
            return {"array_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                    "dtype": a.dtype.str, "shape": list(a.shape)}
        if isinstance(v, dict):
            return {str(k): canonical(v[k]) for k in sorted(v)}
        if isinstance(v, (list, tuple)):
            return [canonical(x) for x in v]
        if isinstance(v, np.generic):
            return v.item()
        return v
    raw = json.dumps(canonical(value), sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def read_exact(stream: BinaryIO, count: int) -> bytes:
    chunks = []
    while count:
        part = stream.read(count)
        if not part:
            raise EOFError("worker IPC closed")
        chunks.append(part)
        count -= len(part)
    return b"".join(chunks)


def read_frame(stream: BinaryIO):
    length, = struct.unpack("!I", read_exact(stream, 4))
    if length > MAX_FRAME_BYTES:
        raise ProtocolError("IPC frame too large")
    return unpack(read_exact(stream, length))


def write_frame(stream: BinaryIO, value):
    raw = pack(value)
    if len(raw) > MAX_FRAME_BYTES:
        raise ProtocolError("IPC frame too large")
    stream.write(struct.pack("!I", len(raw)))
    stream.write(raw)
    stream.flush()
