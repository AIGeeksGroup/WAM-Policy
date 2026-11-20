"""Buffered camera frames and controller states -> protocol snapshots.

A feedback sample is "both RGB views and the root state at tick T". The real
cameras are free-running (Go2 ego ~14 fps from /frontvideostream through
ego_stream, the sim D435i stream 60 fps) and the controller state arrives at
100 Hz, so for a sample instant the bridge takes, per source, the record whose
capture time is nearest the instant within a tolerance. The chosen frame ids
must increase per camera (the session rejects stale or reused sensor frames),
both views are 640x360 RGB and pass unchanged (the ``dual-rgb-360p-v1``
profile of the released WAM-Policy models, the only one this deployment
serves), and anything that cannot be resolved is an ``ObservationGap``:
nothing is filled in or reused. Times are laptop CLOCK_REALTIME seconds; the
ego capture time comes from the stream's time model, the D435i's from its
(chrony-synced) header stamp.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import rootlink
from .timeline import nearest

CAMERA_ALIASES = ("ego", "task_centric")      # QLM snapshot keys -> go2_ego / d435i_rgb_task
RGB_360 = "dual-rgb-360p-v1"                   # the QLM spatial profile (qlm_bench.contracts.rgb_profiles)
PROFILE_HW = (360, 640)                        # its image size, both views (the policy's input)


class ObservationGap(RuntimeError):
    """A sample could not be resolved; ``reason`` is a stable token for records."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class FrameRecord:
    capture_time: float          # laptop clock [s]
    frame_id: int                # strictly increasing per camera
    image: np.ndarray            # uint8 HxWx3 RGB
    receive_time: float


@dataclass
class StateRecord:
    time: float                  # laptop clock [s] of the LowState this state was computed from
    height: float
    quat_wxyz: Sequence[float]
    lin_vel_b: Sequence[float]
    ang_vel_b: Sequence[float]
    mode: str
    fresh: bool
    action18: Optional[Sequence[float]] = None


@dataclass(frozen=True)
class Tolerances:
    ego_s: float = 0.040         # over half the ~14 fps ego period (~35 ms)
    d435i_s: float = 0.020       # at 60 fps one lost frame is bridged
    state_s: float = 0.015       # over one 10 ms controller tick


@dataclass
class ResolvedSample:
    physics_step: int
    simulation_time_ns: int
    reset_epoch: int
    grid_wall: float
    ego: FrameRecord
    d435i: FrameRecord
    state: StateRecord
    state_fields: dict
    ego_offset_s: float          # frame capture - grid instant
    d435i_offset_s: float
    state_offset_s: float

    @property
    def ego_frame_id(self) -> int:
        return self.ego.frame_id

    @property
    def d435i_frame_id(self) -> int:
        return self.d435i.frame_id

    def snapshot(self) -> dict:
        """The runtime snapshot QLM's ``make_observation`` consumes.

        The measured root state is always part of it (QLM derives each frame's camera pose
        from it); whether it is also sent as policy input is the episode's ``root_state``."""
        snap = {"physics_step": self.physics_step, "simulation_time_ns": self.simulation_time_ns,
                "reset_epoch": self.reset_epoch, "state": self.state_fields,
                "sensor_frame_ids": {"ego": self.ego.frame_id, "task_centric": self.d435i.frame_id},
                "rgb": {"ego": self.ego.image, "task_centric": self.d435i.image},
                "rgb_spatial_profile": RGB_360}
        return snap

    def receipt(self) -> dict:
        return {"tick": self.physics_step, "grid_wall": self.grid_wall,
                "ego": {"frame_id": self.ego.frame_id, "capture_time": self.ego.capture_time,
                        "offset_s": self.ego_offset_s, "receive_time": self.ego.receive_time},
                "d435i": {"frame_id": self.d435i.frame_id, "capture_time": self.d435i.capture_time,
                          "offset_s": self.d435i_offset_s, "receive_time": self.d435i.receive_time},
                "state": {"time": self.state.time, "offset_s": self.state_offset_s,
                          "height": self.state.height, "mode": self.state.mode},
                "pair_skew_s": self.d435i.capture_time - self.ego.capture_time}


class ObservationBuffers:
    """Recent frames per camera and recent controller states, trimmed to ``keep_s``."""

    def __init__(self, keep_s: float = 2.0):
        if not keep_s > 0.0:
            raise ValueError("keep_s must be positive")
        self.keep_s = float(keep_s)
        self.frames: Dict[str, List[FrameRecord]] = {alias: [] for alias in CAMERA_ALIASES}
        self.states: List[StateRecord] = []

    # inputs -----------------------------------------------------------------
    def add_frame(self, camera: str, record: FrameRecord) -> None:
        if camera not in CAMERA_ALIASES:
            raise ValueError(f"unknown camera alias {camera!r}")
        img = record.image
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3 \
                or img.shape[:2] != PROFILE_HW:
            raise ValueError(f"{camera}: expected a uint8 HxWx3 image of size {PROFILE_HW} "
                             f"({RGB_360}), got {getattr(img, 'shape', None)}")
        buf = self.frames[camera]
        buf.append(record)
        newest = max(f.capture_time for f in buf)
        self.frames[camera] = [f for f in buf if f.capture_time >= newest - self.keep_s]

    def add_state(self, record: StateRecord) -> None:
        self.states.append(record)
        newest = max(s.time for s in self.states)
        self.states = [s for s in self.states if s.time >= newest - self.keep_s]

    # queries ------------------------------------------------------------------
    def newest_times(self) -> dict:
        return {"ego": max((f.capture_time for f in self.frames["ego"]), default=None),
                "task_centric": max((f.capture_time for f in self.frames["task_centric"]), default=None),
                "state": max((s.time for s in self.states), default=None)}

    def latest_ego_time(self) -> Optional[float]:
        return self.newest_times()["ego"]

    def ready(self, grid_wall: float, tolerances: Tolerances) -> bool:
        """True once no record closer to ``grid_wall`` can still arrive (all sources past it)."""
        newest = self.newest_times()
        need = {"ego": tolerances.ego_s, "task_centric": tolerances.d435i_s, "state": tolerances.state_s}
        return all(newest[k] is not None and newest[k] >= grid_wall + need[k] for k in need)

    def resolve(self, *, tick: int, sim_time_ns: int, reset_epoch: int, grid_wall: float,
                tolerances: Tolerances, last_ids: Optional[dict] = None,
                ground_z: float = rootlink.GROUND_Z) -> ResolvedSample:
        last_ids = last_ids or {}
        ego = self._pick("ego", grid_wall, tolerances.ego_s, last_ids.get("ego"), "ego")
        d435i = self._pick("task_centric", grid_wall, tolerances.d435i_s, last_ids.get("task_centric"), "d435i")
        hit = nearest([(s.time, s) for s in self.states], grid_wall, tolerances.state_s)
        if hit is None:
            raise ObservationGap("state_missing", f"no controller state within {tolerances.state_s * 1e3:.0f} ms of {grid_wall:.3f}")
        state = hit[1]
        if state.mode != "POLICY":
            raise ObservationGap("controller_not_in_policy", f"rambo/mode {state.mode!r} at {state.time:.3f}")
        if not state.fresh:
            raise ObservationGap("state_stale", f"controller reported a stale LowState at {state.time:.3f}")
        fields = rootlink.root_state(height=float(state.height), quat_wxyz=state.quat_wxyz,
                                     lin_vel_b=state.lin_vel_b, ang_vel_b=state.ang_vel_b, ground_z=ground_z)
        return ResolvedSample(physics_step=int(tick), simulation_time_ns=int(sim_time_ns),
                              reset_epoch=int(reset_epoch), grid_wall=float(grid_wall), ego=ego, d435i=d435i,
                              state=state, state_fields=fields,
                              ego_offset_s=ego.capture_time - grid_wall,
                              d435i_offset_s=d435i.capture_time - grid_wall,
                              state_offset_s=state.time - grid_wall)

    def _pick(self, camera: str, grid_wall: float, tolerance: float, last_id, label: str) -> FrameRecord:
        hit = nearest([(f.capture_time, f) for f in self.frames[camera]], grid_wall, tolerance)
        if hit is None:
            raise ObservationGap(f"{label}_frame_missing",
                                 f"no {label} frame within {tolerance * 1e3:.0f} ms of {grid_wall:.3f}")
        frame = hit[1]
        if last_id is not None and frame.frame_id <= last_id:
            raise ObservationGap(f"{label}_frame_reused",
                                 f"{label} frame {frame.frame_id} already used (last {last_id})")
        return frame
