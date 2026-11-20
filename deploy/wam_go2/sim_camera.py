"""Real D435i colour image -> the simulation D435i camera (Real = Sim).

The simulation export in ``config/d435i_rgb_simulation/`` (the QLM-Bench robot-dual-v3 task
camera; provenance in its README.txt) IS the D435i model: 1280x720 ideal pinhole, fx = fy 924.28,
principal point (640, 360), no distortion, mounted at (0.30, 0, 0.34) m in the base frame pitched
65 deg down.

A real frame is resampled so that every output pixel holds the ray the simulated camera sees
there: output = sim size and sim K, input = the real camera_info (K, distortion), rotation
identity (the bracket is taken as verified, so the optical frame is the same; only the lens and
sensor differ). Output pixels whose ray misses the real sensor are BLACK and marked invalid in
``valid`` / ``mask_u8()`` - nothing is filled in. With a typical real camera (principal point
+18 / +28 px off centre at 1280x720) that is the sim frame's right 9 columns and bottom 23 rows,
3.9 % of the image.

A pixel is valid when its sample point lies on the real sensor, i.e. within [-0.5, W-0.5] x
[-0.5, H-0.5] in real pixel coordinates (pixel centres at integers); the half-pixel rim takes the
edge pixel (BORDER_REPLICATE), which is that pixel's own footprint, not an extrapolation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIM_JSON = Path(__file__).resolve().parents[1] / "config" / "d435i_rgb_simulation" / "d435i_rgb_simulation.json"

# OpenCV's own model: (k1, k2, p1, p2[, k3[, k4, k5, k6]]); ROS names it plumb_bob / rational_polynomial
_OPENCV_MODELS = ("plumb_bob", "rational_polynomial")
# Kannala-Brandt is not a pinhole even with zero coefficients (r_d = atan(r))
_FISHEYE_MODELS = ("equidistant", "fisheye", "kannala_brandt", "kannala_brandt4")


@dataclass(frozen=True, eq=False)        # identity equality; compare calibrations with key()
class PinholeCamera:
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    distortion_model: str = "plumb_bob"

    def key(self):
        """Hashable identity of the calibration (for change detection)."""
        return (self.width, self.height, tuple(np.round(self.K.ravel(), 9)), tuple(np.round(self.D, 12)),
                self.distortion_model)


def _check_k(K, width, height):
    if width <= 0 or height <= 0:
        raise ValueError(f"camera size must be positive, got {width}x{height}")
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError(f"K must be a finite 3x3 matrix, got {K!r}")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or K[2, 2] != 1.0 or K[0, 1] != 0 or np.any(K[2, :2] != 0) or K[1, 0] != 0:
        raise ValueError(f"K is not a pinhole matrix [[fx,0,cx],[0,fy,cy],[0,0,1]] with fx, fy > 0: {K.tolist()}")


def camera_from_info(*, width, height, k, d, distortion_model, allow_distortion=False):
    """PinholeCamera from sensor_msgs/CameraInfo fields (k row-major 9, d any length).

    Nonzero distortion coefficients are refused unless `allow_distortion`: realsense-ros
    labels every librealsense model except Kannala-Brandt "plumb_bob", and the D4xx colour
    intrinsics are inverse Brown-Conrady, so the direction of nonzero coefficients is not
    known from the message (the D435i colour stream reports zeros). Pass it only for OpenCV
    forward coefficients."""
    K = np.asarray(k, float).reshape(3, 3)
    width, height = int(width), int(height)
    _check_k(K, width, height)
    D = np.asarray(d if d is not None else [], float).ravel()
    if not np.all(np.isfinite(D)):
        raise ValueError(f"distortion coefficients must be finite, got {D.tolist()}")
    model = (distortion_model or "").strip()
    if model.lower() in _FISHEYE_MODELS:
        raise ValueError(f"distortion model {model!r} is not a pinhole camera, even with zero "
                         "coefficients (Kannala-Brandt maps r to atan(r)); not supported")
    if model not in _OPENCV_MODELS:
        if np.any(D):
            raise ValueError(f"distortion model {model!r} with nonzero coefficients {D.tolist()} is not supported "
                             f"(supported: {', '.join(_OPENCV_MODELS)}, or all-zero coefficients)")
        D = np.zeros(5)
        model = "plumb_bob"
    if D.size == 0:
        D = np.zeros(5)
    if D.size not in (4, 5, 8):
        raise ValueError(f"{model}: expected 4, 5 or 8 distortion coefficients, got {D.size}")
    if np.any(D) and not allow_distortion:
        raise ValueError(f"nonzero distortion {D.tolist()} labelled {model!r}: the coefficient direction "
                         "is ambiguous for realsense colour intrinsics; refusing (allow_distortion=True "
                         "only for OpenCV forward coefficients)")
    return PinholeCamera(width, height, K, D, model)


def load_sim_camera(path=None):
    """The simulation D435i colour camera from the stored export (default: this repo's copy)."""
    data = json.loads(Path(path or SIM_JSON).read_text())
    intr = data["intrinsics"]
    D = np.asarray(intr.get("opencv_distortion_coefficients", []), float)
    if np.any(D):
        raise ValueError(f"the simulation camera must be an ideal pinhole; distortion {D.tolist()} in {path or SIM_JSON}")
    K = np.asarray(intr["K"], float)
    _check_k(K, int(intr["image_width_px"]), int(intr["image_height_px"]))
    return PinholeCamera(int(intr["image_width_px"]), int(intr["image_height_px"]), K, np.zeros(5), "plumb_bob")


def scaled(cam, factor):
    """The same ideal camera at ``factor`` times the resolution: width / height scaled, the first two rows
    of K scaled (QLM's ``camera_profile`` derivation, e.g. its 640x360 ``dual-rgb-360p-v1`` profile from the
    1280x720 one: field of view unchanged)."""
    factor = float(factor)
    w, h = cam.width * factor, cam.height * factor
    if not factor > 0.0 or abs(w - round(w)) > 1e-9 or abs(h - round(h)) > 1e-9:
        raise ValueError(f"scale {factor} of {cam.width}x{cam.height} does not give whole pixels")
    K = np.array(cam.K, float)
    K[:2] *= factor
    _check_k(K, int(round(w)), int(round(h)))
    return PinholeCamera(int(round(w)), int(round(h)), K, np.array(cam.D, float), cam.distortion_model)


def camera_info_fields(cam):
    """CameraInfo fields of an ideal pinhole camera (width, height, distortion_model, d, k, r, p)."""
    K = cam.K
    return {
        "width": cam.width, "height": cam.height, "distortion_model": "plumb_bob",
        "d": [0.0] * 5, "k": [float(x) for x in K.ravel()], "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "p": [float(K[0, 0]), 0.0, float(K[0, 2]), 0.0, 0.0, float(K[1, 1]), float(K[1, 2]), 0.0, 0.0, 0.0, 1.0, 0.0],
    }


def build_remap(real, sim):
    """(map_x, map_y, valid): for every sim pixel, where its ray falls in the real image."""
    import cv2
    map_x, map_y = cv2.initUndistortRectifyMap(real.K, real.D, np.eye(3), sim.K, (sim.width, sim.height),
                                               cv2.CV_32FC1)
    valid = ((map_x >= -0.5) & (map_x <= real.width - 0.5) & (map_y >= -0.5) & (map_y <= real.height - 0.5))
    return map_x, map_y, valid


class SimRemapper:
    """Resample real frames into the simulation camera; rebuilt when the real calibration changes."""

    def __init__(self, sim):
        self.sim = sim
        self.real = None
        self.version = 0
        self.valid = None
        self._key = None
        self._maps = None

    def update(self, real):
        """Use calibration `real`; returns True when the remap was (re)built."""
        if real.key() == self._key:
            return False
        import cv2
        map_x, map_y, valid = build_remap(real, self.sim)
        self._maps = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
        self.valid = valid
        self._invalid = ~valid
        self.real = real
        self._key = real.key()
        self.version += 1
        return True

    def apply(self, image):
        """The sim-camera image of real frame `image` (H x W or H x W x C, uint8); unseen pixels = 0."""
        if self._maps is None:
            raise RuntimeError("no real calibration yet: call update() with the real camera_info first")
        h, w = image.shape[:2]
        if (w, h) != (self.real.width, self.real.height):
            raise ValueError(f"frame is {w}x{h} but the real camera_info is {self.real.width}x{self.real.height}")
        import cv2
        out = cv2.remap(image, self._maps[0], self._maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out[self._invalid] = 0
        return out

    def mask_u8(self):
        """Validity mask in the sim image: 255 = seen by the real sensor, 0 = black fill."""
        return np.where(self.valid, 255, 0).astype(np.uint8)

    @property
    def coverage(self):
        """How much of the sim frame the real sensor covers, and where the gaps are."""
        if self.valid is None:
            return None
        inv = ~self.valid

        def run(flags):                     # length of the all-invalid run at the start of a 1-D bool array
            idx = np.nonzero(~flags)[0]
            return int(idx[0]) if idx.size else int(flags.size)

        cols = inv.all(axis=0)
        rows = inv.all(axis=1)
        # band widths measured away from the opposite band (a full-height column band also fills corners)
        col_any = inv[~rows].any(axis=0) if (~rows).any() else cols
        row_any = inv[:, ~cols].any(axis=1) if (~cols).any() else rows
        return {
            "valid_fraction": float(self.valid.mean()),
            "left_cols": run(col_any), "right_cols": run(col_any[::-1]),
            "top_rows": run(row_any), "bottom_rows": run(row_any[::-1]),
            "scale": float(self.real.K[0, 0] / self.sim.K[0, 0]),
            "upsampled": bool(self.real.width < self.sim.width or self.real.height < self.sim.height),
            "real_size": [self.real.width, self.real.height],
            "sim_size": [self.sim.width, self.sim.height],
        }
