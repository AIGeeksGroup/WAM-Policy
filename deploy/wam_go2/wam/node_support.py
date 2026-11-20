"""ROS-free helpers of the wam_bridge node (message decoding, parameter mapping)."""
from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np

from .limits import Bounds
from .observation import PROFILE_HW, RGB_360, Tolerances


def image_to_rgb(*, height: int, width: int, step: int, encoding: str, data) -> np.ndarray:
    """sensor_msgs/Image bytes -> contiguous uint8 HxWx3 RGB (rgb8 or bgr8, row padding honoured)."""
    if encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unsupported image encoding {encoding!r} (rgb8 or bgr8)")
    buf = np.frombuffer(bytes(data) if not isinstance(data, (bytes, bytearray, memoryview)) else data, dtype=np.uint8)
    if step < width * 3 or buf.size < height * step:
        raise ValueError(f"malformed image: {width}x{height} {encoding}, step {step}, {buf.size} bytes")
    rows = buf[:height * step].reshape(height, step)[:, :width * 3].reshape(height, width, 3)
    if encoding == "bgr8":
        rows = rows[:, :, ::-1]
    return np.ascontiguousarray(rows)


def stamp_to_float(sec: int, nanosec: int) -> float:
    return float(sec) + float(nanosec) * 1e-9


def frame_id_from_header(frame_id: str, sec: int, nanosec: int) -> int:
    """A strictly increasing per-camera sensor id: the Go2 time_frame when the header
    carries ``tf:<n>`` (ego_stream), otherwise the stamp in nanoseconds (D435i)."""
    if frame_id.startswith("tf:"):
        try:
            return int(frame_id[3:])
        except ValueError:
            pass
    return int(sec) * 1_000_000_000 + int(nanosec)


def bounds_from_params(params: Mapping) -> Bounds:
    return Bounds(vx=float(params["bound_vx"]), vy=float(params["bound_vy"]), wz=float(params["bound_wz"]),
                  ee_x=tuple(float(v) for v in params["bound_ee_x"]),
                  ee_y=tuple(float(v) for v in params["bound_ee_y"]),
                  ee_z=tuple(float(v) for v in params["bound_ee_z"]))


def tolerances_from_params(params: Mapping) -> Tolerances:
    return Tolerances(ego_s=float(params["ego_tolerance_s"]), d435i_s=float(params["d435i_tolerance_s"]),
                      state_s=float(params["state_tolerance_s"]))


def episode_identity(*, session_id: str, episode_id: str, reset_epoch: int, now: float) -> Tuple[str, str, int]:
    """Protocol identities: a fixed session, and per run an episode id / reset epoch that
    advance on their own (the policy server refuses an epoch that does not advance within a session)."""
    if not session_id:
        raise ValueError("session_id must not be empty")
    epoch = int(reset_epoch) if int(reset_epoch) > 0 else int(now)
    episode = episode_id or f"{session_id}:{epoch}"
    return session_id, episode, epoch


def frame_size_problem(width: int, height: int, camera: str) -> str:
    """'' when a ``camera`` frame of width x height is the profile size (640x360), else a line saying what is wrong."""
    if (int(height), int(width)) == PROFILE_HW:
        return ""
    want = f"{PROFILE_HW[1]}x{PROFILE_HW[0]}"
    hint = (f" (scripts/x5/d435i_sim_camera.py gives {want} at its default --scale 0.5; "
            "--scale 1.0 is the full 1280x720 export)"
            if camera == "task_centric" and (int(height), int(width)) == (720, 1280) else "")
    return f"{camera} frames are {int(width)}x{int(height)}, the {RGB_360} profile needs {want}{hint}"


def start_gate(*, newest: Mapping[str, Optional[float]], mode: Optional[str], tripped: bool, now: float,
               policy_since: Optional[float], settle_s: float, other_cmd_vel_publishers: int,
               max_age_s: float = 1.0, size_problems: Optional[Mapping[str, str]] = None) -> str:
    """'' when the bridge may take over, else one line saying what it is waiting for.

    In order: every source (ego, task_centric, state) delivered within ``max_age_s``;
    the controller in POLICY and not latched; no other ``cmd_vel`` publisher (the
    entry sequence that brought the controller to POLICY still running); and the
    controller in POLICY for ``settle_s`` (``policy_since`` = first POLICY report)."""
    missing = [k for k, v in newest.items() if v is None or now - v > max_age_s]
    if missing:
        problems = [size_problems[k] for k in missing if size_problems and size_problems.get(k)]
        return "no recent " + ", ".join(missing) + (": " + "; ".join(problems) if problems else "")
    if mode != "POLICY":
        return f"controller not in POLICY ({mode})"
    if tripped:
        return f"controller safety latched ({mode})"
    if other_cmd_vel_publishers > 0:
        return f"{other_cmd_vel_publishers} other cmd_vel publisher(s) (the entry sequence still running?)"
    if settle_s > 0.0 and (policy_since is None or now - policy_since < settle_s):
        return f"controller in POLICY, settling ({settle_s:.1f} s)"
    return ""
