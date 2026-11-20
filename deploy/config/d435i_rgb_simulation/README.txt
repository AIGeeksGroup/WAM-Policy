QLM-Bench D435i RGB simulation intrinsics and extrinsics

Version: 66605a83180609f610e0dfd23b4440fe91ac689f
Configuration: robot-dual-v3 / task; camera pose relative to Robot/base.
Source: computed offline from the pinned repository configuration; no live simulator readback.
Scope: the simulated D435i RGB camera only; no stereo depth, RGB-depth or IMU calibration.
The export keeps the simulation's 65 degree downward pitch (not the 30.5 degrees of the STEP model).

Files
- d435i_rgb_simulation.json / .yaml: full intrinsics, extrinsics and provenance.
- d435i_rgb_camera_info.yaml: intrinsics in the ROS CameraInfo format.
- K.txt: 3x3 intrinsic matrix in pixels; image 1280 wide, 720 high.
- T_base_from_camera_optical.txt: optical camera frame -> Go2 base.
- T_camera_optical_from_base.txt: Go2 base -> optical camera frame (for computer-vision projection).
- T_base_from_camera_world.txt: the repository's camera-axis convention -> Go2 base.
- P_base_to_image.txt: K[R|t], projects 3D points in base coordinates to the image.
- validation.json: offline numerical checks of the rotation, the inverse and the optical-axis projection.
- SHA256SUMS.txt: checksums; it also lists the generator script and the pinned source copies of
  the original export, which are not shipped here (sha256sum -c --ignore-missing SHA256SUMS.txt).

Frames and units
base: X forward, Y left, Z up; optical camera: X right, Y down, Z forward.
Column vectors, left multiplication; matrices in JSON/YAML/text are stored row by row.
Extrinsic translations are in metres, intrinsics in pixels.
Isaac Lab's camera convention="world" is only a camera-axis convention, not the global world frame.
The repository quaternion (XYZW order) is the pose of that camera-axis convention relative to base;
do not use it directly as the quaternion of the OpenCV/ROS optical frame relative to base.

Projection
import json, numpy as np
d = json.load(open("d435i_rgb_simulation.json"))
K = np.array(d["intrinsics"]["K"])
T = np.array(d["extrinsics"]["T_camera_optical_from_base"])
p_base = np.array([0.6, 0.0, 0.0, 1.0])  # example point, metres
p_camera = (T @ p_base)[:3]
if p_camera[2] > 0:
    uvh = K @ p_camera
    uv = uvh[:2] / uvh[2]

World extrinsics change as the robot moves and must be computed per frame:
T_world_from_camera_optical(t) = T_world_from_base(t) @ T_base_from_camera_optical
The fixed base extrinsics here are not a static world extrinsic.

Note: the all-zero distortion coefficients describe the ideal simulated pinhole, not the measured
distortion of a real D435i. rectification_matrix=I in the ROS YAML is the rectification matrix,
not the camera mounting extrinsic.
