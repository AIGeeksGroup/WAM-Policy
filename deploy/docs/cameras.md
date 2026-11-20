# Cameras and time synchronisation

WAM-Policy takes two 640x360 RGB views: the Go2's front camera (`go2_ego`) and a task camera
(`d435i_rgb_task`), an Intel RealSense D435i mounted on the robot as in the QLM-Bench
simulation (0.30 / 0 / 0.34 m in the base frame, pitched 65 deg down).

## The Go2 front camera

`ego_stream` (`wam_go2/nodes/ego_stream_node.py`) subscribes to the robot's `/frontvideostream`
(360p H.264, ~14 fps), decodes it and publishes `rgb8` frames on `/camera/go2/ego/image_raw`.
The stream carries a frame counter (`time_frame`, in 1/15 s steps), not a capture clock, so
`ego_stream` fits a clock to the arrival times of many frames (lower envelope,
`wam_go2/timesync.py`) and subtracts the pipeline latency when it is calibrated
(`go2_latency_ms`); uncalibrated stamps are receive-time estimates. The frame id is
`tf:<counter>`. The bridge launch starts `ego_stream` together with the bridge; a task run passes it
the latency as `WAM_GO2_LATENCY_MS` (ms) or `WAM_CALIB` (a `sync_check.py` / `pair_report.py
--write-calib` JSON).

## The D435i on an RDK X5

The D435i is plugged into an RDK X5 board on the robot, on the same switch as the robot and the
laptop. The X5 runs one script, `scripts/x5/go2_camera.sh`, under systemd (`x5_install.sh`
deploys it from the laptop to `X5_IP` of `config/deploy.env`; key login as root on the X5). Its
settings live in the X5's `camera.conf` (`scripts/x5/camera.conf.example`): colour 640x360 at 60
fps, depth 640x360 at 15 fps, IMU on. With the camera streaming, `/lowstate` still arrives at
500 Hz (maximum gap ~3-4 ms).

The X5 clock follows the laptop through chrony (`laptop_chrony_install.sh` on the laptop,
`x5_chrony_setup.sh` for the X5); the camera node restarts after every clock step, because the
librealsense device-to-host time fit rejects large offsets.

`scripts/x5/d435i_sim_camera.py` resamples every real colour frame into the simulation camera of
`config/d435i_rgb_simulation/` (exported at 1280x720, fx = fy = 924.28, principal point at the
centre, no distortion) at half its resolution: 640x360, fx = fy = 462.14, principal point
320 / 180 (`dual-rgb-360p-v1`), so the policy sees the camera it was trained with. Pixels the real
sensor does not cover are black and flagged in `valid_mask`. Nonzero distortion coefficients in
the real camera_info are refused unless `--allow-opencv-distortion` says they are OpenCV forward
coefficients (the D435i reports zeros). `sha256sum -c --ignore-missing SHA256SUMS.txt` in that
directory checks the export.

```bash
source scripts/x5/laptop_camera_env.bash                 # wired-only DDS (DEMO_LAPTOP_IP) for camera tools
python3 scripts/x5/camera_sub_probe.py                   # D435i topics: rates, gaps, payload
python3 scripts/x5/sync_check.py --duration 60           # host clocks, camera stamps, robot/camera lag
python3 scripts/x5/ego_d435i_view.py                     # two windows Ego / D435i, one video each
CAMERA_DDS_SHM=1 source scripts/x5/laptop_camera_env.bash
python3 scripts/x5/d435i_sim_camera.py                   # /camera/d435i/sim/color/*
```

## Operator view and records

During a run `scripts/x5/ego_d435i_view.py` shows the two cameras in two windows and records one
video per camera on a common 50 Hz grid of capture times (`wam_go2/grid_sync.py`): per 20 ms tick
each camera gives the frame nearest the tick (a frame never serves two D435i ticks) or the tick
is marked `missing`; nothing is interpolated. A task run writes into its `view/` directory:
`camera_ego.mp4`, `camera_d435i.mp4`, the log, a summary JSON and three CSV files:

| File | One row per | Content |
|---|---|---|
| `camera_frames.csv` | received frame | stream, frame id, raw time as sent, its source, laptop receive time |
| `camera_pairs.csv` | Go2 frame | pair id, status (`matched` / `unmatched:<reason>`), both streams' ids and times, the difference, sync method, calibration version |
| `camera_grid.csv` | grid tick | each camera's frame, offset, new / repeat / missing |

Pairing (`wam_go2/camera_pairing.py`) takes each Go2 frame as the anchor and the nearest unused
D435i frame within a bounded window (`--slop-ms`, 20 ms); frames outside it are marked unmatched
with the reason. Until the Go2 latency is calibrated the pairs are labelled
`receive_time_approx`; with a calibration (`--calib`, `--go2-latency-ms`, or the `c` key in the
view) they are `capture_time`.

To calibrate from a few runs with visible body motion (both cameras see the same event):

```bash
python3 scripts/x5/pair_report.py result/*/*/view/camera.json
python3 scripts/x5/pair_report.py result/*/*/view/camera.json --write-calib calib_relative.json
DEMO_VIEW_CALIB=calib_relative.json tasks/push_box.sh
```

After calibrating, the common-event offset (`pairing_offset_ms`) of new runs should be near zero.

Limits: the two cameras cannot expose at the same instant (the Go2 camera has no trigger), and a
flash test cannot show millisecond accuracy at ~14 fps (one frame = 70 ms). The relative
calibration written by `pair_report.py` corrects only the pairing; absolute D435i capture times
also need the exposure offset that `sync_check.py` estimates from the D435i's own gyro.
