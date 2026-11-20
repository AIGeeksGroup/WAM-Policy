#!/usr/bin/env python3
"""Time-sync check of the Go2 and the RDK X5 / D435i (laptop side, read-only:
subscribes only, publishes nothing, never touches robot control).

Run it before task runs when the camera setup changed, with the robot and the
X5 camera service up:

    source scripts/x5/laptop_camera_env.bash
    python3 scripts/x5/sync_check.py --duration 60 --out-dir data/logs/camera_checks_$(date +%F)

Four measurements, one JSON + one text report + raw arrays (.npz):
  1. host clocks   - laptop `chronyc tracking`, X5 `chronyc tracking` (ssh), and an
                     SNTP query to the X5 (offset +- rtt/2 from a raw NTP packet). chronyd
                     serves its CORRECTED time, so SNTP = X5 served time minus laptop raw
                     time and chrony's offset = X5 served minus X5 raw; the frame stamps
                     depend on X5 raw minus laptop raw = SNTP - chrony offset
                     (host_clock_assessment, before AND after the capture)
  2. /lowstate     - Go2 MCU `tick` fitted onto the laptop wall clock
                     (wam_go2.timesync.fit_tick_clock): tick rate error and delay jitter
  3. camera stamps - receive time minus header.stamp for the D435i IMU / colour / depth
                     (= transport latency + X5 clock offset; a wrong X5 clock shows up here)
  4. gyro lag      - cross-correlation of |omega| from the Go2 IMU (on the fitted tick clock)
                     and the D435i gyro (header.stamp): the residual robot-vs-camera time shift
                     that a dataset should correct for. NEEDS MOTION: rock / tilt the robot
                     body by hand during the window (camera rigidly mounted on the body);
                     with the robot still it reports "insufficient motion" and no number.

Verdict thresholds (project targets): X5 chrony offset < 1 ms, X5 raw minus laptop
raw clock (SNTP - chrony offset) < 5 ms before and after, camera IMU recv-stamp
median 1.5 +- 5 ms (IMU_RECV_STAMP_MS / _TOL_MS), tick-fit residual p99 < 5 ms,
|gyro lag| < 5 ms.

  5. MODEL-INPUT view (WAM-Policy itself is NOT imported or wired in). The policy
     consumes exactly two RGB streams - the Go2 ego camera (/frontvideostream, the
     stream the ego_stream node decodes) and the D435i colour - paired at the
     SAME capture instant on an 80 ms grid (12.5 Hz), plus the executed action
     history as 20 ms holds on the laptop clock; timestamps must be measured
     capture times, never frame indices. So this section measures, on the
     laptop clock, the CAPTURE time of both RGB streams and what the grid costs:
       a. /frontvideostream (raw wire layout, see ego_d435i_view.py): frame rate,
          proof that `time_frame` is a counter, arrival-envelope time model;
       b. Go2 ego pipeline latency  = lag(ego frame-difference energy vs Go2 IMU |w|)
       c. D435i colour stamp minus exposure midpoint = lag(colour energy vs D435i gyro |w|)
          (the X5 driver publishes no exposure metadata, so this is the only way)
       d. ego-vs-colour capture skew = lag(ego energy vs colour energy)  (closure check: b - c)
       e. the 80 ms grid: per slot, how stale the chosen ego frame is and how far
          the nearest D435i frame is from it (wam_go2.timesync.model_grid_pairing).
          This is the CAPTURE-grid offset (offline, transport ignored); an online
          consumer's ego age is L + this offset (+ decode)
       f. X5-side RealSense metadata sample (ssh): fields present, SDK hold time.
     b-d need the ROBOT BODY to be rocked / tilted by hand for ~20 s with both
     cameras rigidly mounted (frame-difference energy of a static scene follows
     |omega|); they are stamped at the interval midpoint (timesync.midpoint_times).
     The constants are written to JSON `model_input.calibration` for
     ego_d435i_view.py --calib, the ego_stream node's `calib` parameter and the
     recording session.
"""
import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from wam_go2 import timesync as ts  # noqa: E402

NTP_EPOCH_OFFSET = 2208988800  # 1900 -> 1970


# ----------------------------------------------------------------- host clocks
def chronyc_tracking(cmd_prefix=()):
    """Parse `chronyc -c tracking` (local or via ssh prefix) into a dict, or {'error': ...}."""
    try:
        out = subprocess.run([*cmd_prefix, "chronyc", "-c", "tracking"], capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}
    if out.returncode != 0 or not out.stdout.strip():
        return {"error": (out.stderr or out.stdout or "no output").strip()}
    f = out.stdout.strip().split(",")
    if len(f) < 14:
        return {"error": f"unexpected chronyc output: {out.stdout.strip()}"}
    return {"ref_id": f[0], "ref_ip": f[1], "stratum": int(f[2]), "offset_s": float(f[4]),
            "last_offset_s": float(f[5]), "rms_offset_s": float(f[6]), "freq_ppm": float(f[7]),
            "skew_ppm": float(f[9]), "root_delay_s": float(f[10]), "root_dispersion_s": float(f[11]),
            "leap": f[13], "synced": f[13] == "Normal" and 0 < int(f[2]) < 16}


def sntp_offset(host, port=123, tries=8, timeout=1.0):
    """Raw SNTP exchange: offset of the remote's SERVED time relative to our raw
    clock (+ = remote ahead) and round-trip delay; the sample with the smallest
    delay is returned. chronyd answers with its corrected time (system clock plus
    the correction still being slewed), so against the X5 this is X5 served time
    minus laptop raw time - combine it with the X5's chrony offset to get the raw
    clock difference (host_clock_assessment)."""
    if not host:
        return {"error": "no X5 address (set X5_IP in config/deploy.env or pass --x5-ip)"}
    best = None
    leap = stratum = None
    pkt = bytearray(48); pkt[0] = (0 << 6) | (4 << 3) | 3   # LI 0, VN 4, mode 3 (client)
    for _ in range(tries):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(timeout)
        try:
            t1 = time.time()
            s.sendto(bytes(pkt), (host, port))
            data, _ = s.recvfrom(512)
            t4 = time.time()
        except OSError:
            continue
        finally:
            s.close()
        if len(data) < 48:
            continue
        li_vn_mode, stratum_b = data[0], data[1]
        rx_s, rx_f, tx_s, tx_f = struct.unpack("!IIII", data[32:48])
        t2 = rx_s - NTP_EPOCH_OFFSET + rx_f / 2**32
        t3 = tx_s - NTP_EPOCH_OFFSET + tx_f / 2**32
        delay = (t4 - t1) - (t3 - t2)
        offset = ((t2 - t1) + (t3 - t4)) / 2
        leap, stratum = li_vn_mode >> 6, stratum_b
        if best is None or delay < best["rtt_s"]:
            best = {"offset_s": offset, "rtt_s": delay}
        time.sleep(0.1)
    if best is None:
        return {"error": f"no SNTP reply from {host}:{port} (chrony not running / not allowed / port closed)"}
    best.update({"leap": leap, "stratum": stratum, "remote_synced": leap != 3 and 0 < stratum < 16,
                 "note": "offset = X5 served (chrony-corrected) time minus laptop raw clock, from the lowest-rtt "
                         "sample; X5 raw minus laptop raw = this offset_s - the X5 chrony offset_s"})
    return best


# The frame stamps depend on X5_raw - laptop_raw. The two host probes are the two
# complementary halves of it, not redundant checks:
#   SNTP offset_s   = X5_served - laptop_raw   (sntp_offset: chronyd serves corrected time)
#   chrony offset_s = X5_served - X5_raw       (chronyc tracking 'System time', + = X5 raw slow)
#   X5_raw - laptop_raw = SNTP.offset_s - chrony.offset_s
# X5 chrony alone cannot see the LAPTOP's raw clock still slewing towards its own
# served time (internet reconnect, resume); SNTP alone cannot see the X5's slew.
RAW_OFFSET_MAX_MS = 5.0          # ~30 % of the 16.7 ms same-instant pairing budget; an internet-synced laptop sits at 1-2 ms
SNTP_RTT_MAX_MS = 5.0            # the SNTP offset is only known to +- rtt/2
IMU_RECV_STAMP_MS = 1.5          # D435i gyro recv - stamp median: 1.34 ms gyro / 1.93 ms imu (measured at 640x480@30)
IMU_RECV_STAMP_TOL_MS = 5.0      # measured at 640x480@30; the rig runs 640x360@60 (scripts/x5/camera_check.sh)


def _imu_recv_minus_stamp_ms(arr):
    """(stream, median recv - stamp in ms) of the D435i gyro (else imu) samples, or (None, None)."""
    for k in ("gyro", "imu"):
        rk, sk = f"{k}_recv_ns", f"{k}_stamp_ns"
        if rk in arr and sk in arr and arr[rk].size and arr[rk].size == arr[sk].size:
            d = np.asarray(arr[rk], dtype=np.int64) - np.asarray(arr[sk], dtype=np.int64)
            return k, float(np.median(d)) / 1e6
    return None, None


def host_clock_assessment(hosts, arr=None, max_offset_ms=RAW_OFFSET_MAX_MS, max_rtt_ms=SNTP_RTT_MAX_MS,
                          imu_center_ms=IMU_RECV_STAMP_MS, imu_tol_ms=IMU_RECV_STAMP_TOL_MS):
    """Do the X5 frame stamps and the laptop raw CLOCK_REALTIME agree? Fails closed.

    Per probe pair - "before" (x5, x5_sntp) and "after" (x5_after, x5_sntp_after):
    X5_raw - laptop_raw = SNTP.offset_s - chrony.offset_s must be < max_offset_ms;
    a missing probe, an 'error', chrony not synced, SNTP remote not synced or SNTP
    rtt >= max_rtt_ms fails the pair. With `arr` (the sync_check arrays) an
    in-band item is added: the median recv - stamp of the D435i gyro (else imu)
    samples must lie within imu_center_ms +- imu_tol_ms (no IMU samples fails).
    It is the only guard that sees wrong librealsense stamps (e.g. its device->host
    fit after an X5 clock step) while every host number reads ~0.

    Returns {"ok", "items": [{"name", "ok", "detail", ...}], "limits"}."""
    items = []
    for name, ck, sk in (("before", "x5", "x5_sntp"), ("after", "x5_after", "x5_sntp_after")):
        c, s = hosts.get(ck), hosts.get(sk)
        why = []
        if not isinstance(c, dict) or not c:
            why.append(f"{ck} missing")
        elif "error" in c:
            why.append(f"{ck}: {c['error']}")
        elif not c.get("synced"):
            why.append(f"{ck}: chrony NOT synced (stratum {c.get('stratum')}, leap {c.get('leap')})")
        if not isinstance(s, dict) or not s:
            why.append(f"{sk} missing")
        elif "error" in s:
            why.append(f"{sk}: {s['error']}")
        else:
            if not s.get("remote_synced"):
                why.append(f"{sk}: X5 serves unsynced time (stratum {s.get('stratum')}, leap {s.get('leap')})")
            rtt = s.get("rtt_s")
            if not (isinstance(rtt, (int, float)) and rtt * 1e3 < max_rtt_ms):
                why.append(f"{sk}: rtt {rtt * 1e3 if isinstance(rtt, (int, float)) else float('nan'):.2f} ms >= {max_rtt_ms:.0f} ms")
        d_ms = None
        if (isinstance(c, dict) and isinstance(c.get("offset_s"), (int, float))
                and isinstance(s, dict) and isinstance(s.get("offset_s"), (int, float))):
            d_ms = (s["offset_s"] - c["offset_s"]) * 1e3
            if not abs(d_ms) < max_offset_ms:
                why.append(f"exceeds {max_offset_ms:.0f} ms")
        elif not why:
            why.append("offset_s missing")
        detail = (f"{name} {d_ms:+.3f} ms (SNTP {s['offset_s'] * 1e3:+.3f} - chrony {c['offset_s'] * 1e3:+.3f})"
                  if d_ms is not None else name)
        if why:
            detail += (" " if d_ms is not None else ": ") + "; ".join(why)
        items.append({"name": name, "ok": not why, "chrony": ck, "sntp": sk,
                      "x5_raw_minus_laptop_raw_ms": d_ms, "detail": detail})
    if arr is not None:
        key, med = _imu_recv_minus_stamp_ms(arr)
        if key is None:
            items.append({"name": "in_band", "ok": False, "stream": None, "median_ms": None,
                          "detail": "no D435i IMU samples: the X5 stamps cannot be checked against laptop receive time"})
        else:
            items.append({"name": "in_band", "ok": abs(med - imu_center_ms) <= imu_tol_ms, "stream": key, "median_ms": med,
                          "detail": f"D435i {key} recv-stamp median {med:+.2f} ms (expected {imu_center_ms:.1f} +- {imu_tol_ms:.1f} ms)"})
    return {"ok": all(i["ok"] for i in items), "items": items,
            "limits": {"max_raw_offset_ms": max_offset_ms, "max_sntp_rtt_ms": max_rtt_ms,
                       "imu_recv_stamp_ms": imu_center_ms, "imu_recv_stamp_tol_ms": imu_tol_ms},
            "quantity": "X5_raw - laptop_raw = x5_sntp.offset_s - x5.offset_s (ms)"}


def host_verdict_lines(hosts, max_offset_ms=RAW_OFFSET_MAX_MS):
    """The host-clock lines at the top of the report and the combined assessment:
    laptop chrony (info), X5 chrony (OK/BAD, target < 1 ms), SNTP (info: only one
    half of the answer) and the X5-raw minus laptop-raw verdict (before + after)."""
    lines = []
    lp = hosts.get("laptop") or {"error": "not probed"}
    lines.append(f"--  laptop chrony: {'offset %+.3f ms, stratum %d, %s' % (lp['offset_s'] * 1e3, lp['stratum'], lp['leap']) if 'error' not in lp else lp['error']}")
    x5 = hosts.get("x5", {})
    if "error" in x5:
        lines.append(f"--  X5 chrony: {x5['error']}")
    elif x5.get("synced"):
        ok = abs(x5["offset_s"]) < 1e-3
        lines.append(f"{'OK ' if ok else 'BAD'} X5 chrony offset {x5['offset_s'] * 1e3:+.3f} ms (rms {x5['rms_offset_s'] * 1e3:.3f} ms, stratum {x5['stratum']}); target < 1 ms")
    else:
        lines.append(f"BAD X5 chrony NOT synced (stratum {x5.get('stratum')}, leap {x5.get('leap')}) - is the laptop serving (laptop_chrony_install.sh)?")
    sn = hosts.get("x5_sntp") or {"error": "not probed"}
    if "error" in sn:
        lines.append(f"--  SNTP to X5: {sn['error']}")
    else:
        lines.append(f"--  SNTP X5 served (corrected) time minus laptop raw time {sn['offset_s'] * 1e3:+.3f} ms (rtt {sn['rtt_s'] * 1e3:.2f} ms, "
                     f"remote {'synced' if sn['remote_synced'] else 'UNSYNCED'}); combine with the X5 chrony offset (line above); the combined result is on the next line")
    a = host_clock_assessment(hosts, max_offset_ms=max_offset_ms)
    lines.append(f"{'OK ' if a['ok'] else 'BAD'} X5 raw clock minus laptop raw clock (= SNTP - X5 chrony offset; the frame stamps depend on it): "
                 + ", ".join(i["detail"] for i in a["items"]) + f"; target < {max_offset_ms:.0f} ms before and after")
    return lines, a


def res_seen_array(counts):
    """{res code: messages} -> (N, 2) int64 [[res, n], ...] sorted by res (npz key ego_res_seen)."""
    return np.asarray(sorted((int(r), int(n)) for r, n in counts.items()), dtype=np.int64).reshape(-1, 2)


def x5_metadata_sample(ssh_target, topic="/camera/d435i/color/metadata", seconds=4.0,
                       profile="/root/go2_camera_deploy/config/fastdds_x5_wired.xml"):
    """Sample the RealSense per-frame metadata ON THE X5 (the laptop has no
    realsense2_camera_msgs): which fields the driver publishes, frame-number
    gaps, and how long a frame sits inside the SDK before the ROS node sees it
    (time_of_arrival - frame_timestamp)."""
    if not ssh_target:
        return {"error": "skipped (no ssh target)"}
    cmd = (f"source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=0 FASTRTPS_DEFAULT_PROFILES_FILE={profile}; "
           f"timeout {seconds:g} ros2 topic echo --full-length --field json_data "
           f"--qos-reliability best_effort --qos-durability volatile {topic} 2>/dev/null")
    try:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", ssh_target, cmd],
                             capture_output=True, text=True, timeout=seconds + 20)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}
    rows = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if len(rows) < 5:
        return {"error": f"no metadata rows ({(out.stderr or out.stdout).strip()[:200]})"}
    keys = sorted(set().union(*(r.keys() for r in rows)))
    rep = {"topic": topic, "n": len(rows), "fields": keys,
           "has_exposure_metadata": all(k in keys for k in ("sensor_timestamp", "actual_exposure")),
           "clock_domain": rows[0].get("clock_domain")}
    def col(k):
        return np.asarray([float(r[k]) for r in rows if k in r], dtype=np.float64)
    if "frame_timestamp" in keys:
        ft = col("frame_timestamp")                                   # ms, global time
        rep["fps"] = float(1e3 / np.median(np.diff(ft))) if ft.size > 2 else None
        if "time_of_arrival" in keys:
            rep["sdk_hold_ms"] = ts.offset_stats_ms(ft / 1e3, col("time_of_arrival") / 1e3)
        if "backend_timestamp" in keys:
            rep["backend_minus_frame_ms"] = ts.offset_stats_ms(ft / 1e3, col("backend_timestamp") / 1e3)
    if "frame_number" in keys:
        fn = col("frame_number")
        rep["frame_number_gaps"] = int(np.count_nonzero(np.diff(fn) != 1))
    if rep["has_exposure_metadata"]:
        exp = col("actual_exposure")
        rep["actual_exposure_us"] = {"median": float(np.median(exp)), "max": float(exp.max())}
        if "hw_timestamp" in keys:
            d = (col("sensor_timestamp") - col("hw_timestamp") + 2**31) % 2**32 - 2**31   # signed wrap, us
            rep["sensor_minus_frame_us_median"] = float(np.median(d))
    return rep


# ---------------------------------------------------------------- ROS capture
def _thumb_energy(cv2, img, prev):
    """Frame-difference energy of a 160-px grey thumbnail (same as ego_d435i_view)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (160, max(1, int(160 * g.shape[0] / g.shape[1]))), interpolation=cv2.INTER_AREA).astype(np.int16)
    e = None if prev is None or prev.shape != small.shape else float(np.abs(small - prev).mean())
    return e, small


def capture(duration, ns, with_images, with_ego=True, go2_res=360):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, Imu
    from unitree_go.msg import LowState

    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=50,
                     reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
    try:
        import cv2
    except ImportError:
        cv2 = None
    rclpy.init()
    node = Node("sync_check_probe")
    ls = {"recv": [], "tick": [], "gyro": []}
    cams = {k: {"recv": [], "stamp": [], "gyro": []} for k in ("gyro", "imu", "color", "depth")}
    color_energy = []                                     # per colour frame: frame-difference energy (nan for the first)
    prev_small = {"color": None, "go2": None}
    ego = {"tf": [], "recv": [], "e_tf_prev": [], "e_tf": [], "e": [], "decode_errors": 0, "note": "",
           "res_seen": {}}                                # res code -> messages, counted before the resolution filter

    def on_low(msg):
        ls["recv"].append(time.time_ns()); ls["tick"].append(int(msg.tick))
        g = msg.imu_state.gyroscope; ls["gyro"].append((float(g[0]), float(g[1]), float(g[2])))

    def on_cam(key, has_gyro):
        def cb(msg):
            st = cams[key]
            st["recv"].append(time.time_ns())
            st["stamp"].append(int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec))
            if has_gyro:
                a = msg.angular_velocity; st["gyro"].append((a.x, a.y, a.z))
            elif key == "color" and cv2 is not None:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                e, prev_small["color"] = _thumb_energy(cv2, np.ascontiguousarray(img[:, :, :3]), prev_small["color"])
                color_energy.append(np.nan if e is None else e)
        return cb

    # Go2 ego camera: raw /frontvideostream -> decode on a worker thread -> energy
    dec_q, dec_ev = [], threading.Event()
    dec_lock = threading.Lock()
    decoder = None
    if with_ego and cv2 is not None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from ego_d435i_view import Go2Decoder, parse_frontvideo
            decoder = Go2Decoder()
        except Exception as e:                      # PyAV missing etc.: keep the rest of the check
            ego["note"] = f"ego decode unavailable: {e}"
    if decoder is not None:
        from unitree_go.msg import Go2FrontVideoData

        def on_video(raw):
            arrival = time.time_ns()
            try:
                tf, res, data = parse_frontvideo(bytes(raw))
            except struct.error:
                return
            ego["res_seen"][int(res)] = ego["res_seen"].get(int(res), 0) + 1
            if res != go2_res:
                return
            ego["tf"].append(int(tf)); ego["recv"].append(arrival)
            with dec_lock:
                if len(dec_q) < 8:
                    dec_q.append((tf, data))
            dec_ev.set()

        def decode_loop():
            prev_tf = None
            while not stop.is_set():
                dec_ev.wait(0.5)
                while True:
                    with dec_lock:
                        if not dec_q:
                            break
                        tf, data = dec_q.pop(0)
                    for ftf, img in decoder.decode(data, tf):
                        e, prev_small["go2"] = _thumb_energy(cv2, img, prev_small["go2"])
                        if e is not None and prev_tf is not None:
                            ego["e_tf_prev"].append(prev_tf); ego["e_tf"].append(ftf); ego["e"].append(e)
                        prev_tf = ftf
                dec_ev.clear()

        stop = threading.Event()
        node.create_subscription(Go2FrontVideoData, "/frontvideostream", on_video, qos, raw=True)
        threading.Thread(target=decode_loop, daemon=True).start()
    else:
        stop = threading.Event()

    node.create_subscription(LowState, "/lowstate", on_low, qos)
    node.create_subscription(Imu, f"{ns}/gyro/sample", on_cam("gyro", True), qos)
    node.create_subscription(Imu, f"{ns}/imu", on_cam("imu", True), qos)
    if with_images:
        node.create_subscription(Image, f"{ns}/color/image_raw", on_cam("color", False), qos)
        node.create_subscription(Image, f"{ns}/depth/image_rect_raw", on_cam("depth", False), qos)
    t0 = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic() - t0 < duration:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    observed = time.monotonic() - t0
    stop.set(); time.sleep(0.2)
    node.destroy_node(); rclpy.shutdown()
    arr = {"lowstate_recv_ns": np.asarray(ls["recv"], dtype=np.int64),
           "lowstate_tick": np.asarray(ls["tick"], dtype=np.int64),
           "lowstate_gyro": np.asarray(ls["gyro"], dtype=np.float64).reshape(-1, 3)}
    for k, st in cams.items():
        arr[f"{k}_recv_ns"] = np.asarray(st["recv"], dtype=np.int64)
        arr[f"{k}_stamp_ns"] = np.asarray(st["stamp"], dtype=np.int64)
        if st["gyro"]:
            arr[f"{k}_gyro"] = np.asarray(st["gyro"], dtype=np.float64).reshape(-1, 3)
    n_col = arr["color_stamp_ns"].size
    arr["color_energy"] = np.asarray(color_energy[:n_col] + [np.nan] * max(0, n_col - len(color_energy)), dtype=np.float64)
    arr["ego_tf"] = np.asarray(ego["tf"], dtype=np.int64); arr["ego_recv_ns"] = np.asarray(ego["recv"], dtype=np.int64)
    arr["ego_energy_tf_prev"] = np.asarray(ego["e_tf_prev"], dtype=np.int64)
    arr["ego_energy_tf"] = np.asarray(ego["e_tf"], dtype=np.int64); arr["ego_energy"] = np.asarray(ego["e"], dtype=np.float64)
    arr["ego_res"] = np.asarray([go2_res if decoder is not None else 0])
    arr["ego_res_seen"] = res_seen_array(ego["res_seen"])     # every /frontvideostream resolution, not only the decoded one
    arr["ego_decode_errors"] = np.asarray([decoder.errors if decoder is not None else -1])
    if ego["note"]:
        print(f"[sync_check] {ego['note']}")
    return observed, arr


# ------------------------------------------------------------------ analysis
def analyse(observed, arr, max_lag_s):
    rep = {"observed_s": observed}
    verdict = []
    # 2. tick fit
    n_ls = int(arr["lowstate_recv_ns"].size)
    rep["lowstate"] = {"n": n_ls, "hz": n_ls / observed if observed else 0.0}
    fit = None
    if n_ls >= 100:
        recv_s = arr["lowstate_recv_ns"] / 1e9
        fit = ts.fit_tick_clock(arr["lowstate_tick"], recv_s)
        rep["lowstate"]["tick_fit"] = fit.as_dict()
        ok = fit.resid_ms["p99"] < 5.0
        verdict.append(f"{'OK ' if ok else 'BAD'} tick->laptop fit: rate {fit.rate_ppm:+.1f} ppm, delay jitter p99 {fit.resid_ms['p99']:.2f} ms max {fit.resid_ms['max']:.2f} ms (target p99 < 5)")
    else:
        verdict.append("BAD /lowstate: too few samples - robot off or DDS not on the wired NIC?")
    # 3. camera stamps
    rep["camera"] = {}
    for k in ("gyro", "imu", "color", "depth"):
        n = int(arr[f"{k}_recv_ns"].size)
        if n == 0:
            rep["camera"][k] = {"n": 0}; continue
        st = ts.offset_stats_ms(arr[f"{k}_stamp_ns"] / 1e9, arr[f"{k}_recv_ns"] / 1e9)
        st["hz"] = n / observed
        rep["camera"][k] = st
    imu_key = "gyro" if rep["camera"]["gyro"].get("n") else ("imu" if rep["camera"]["imu"].get("n") else None)
    if imu_key:
        med = rep["camera"][imu_key]["median"]
        ok = abs(med - IMU_RECV_STAMP_MS) <= IMU_RECV_STAMP_TOL_MS
        verdict.append(f"{'OK ' if ok else 'BAD'} camera IMU recv-stamp median {med:+.2f} ms (p99 {rep['camera'][imu_key]['p99']:+.2f}); "
                       f"target {IMU_RECV_STAMP_MS:.1f} +- {IMU_RECV_STAMP_TOL_MS:.1f} ms = librealsense stamps consistent with the laptop clock")
    else:
        verdict.append("BAD camera IMU: no messages (CAMERA_IMU=0 on the X5, or camera down)")
    for k in ("color", "depth"):
        if rep["camera"].get(k, {}).get("n"):
            verdict.append(f"    {k}: {rep['camera'][k]['hz']:.2f} Hz, recv-stamp median {rep['camera'][k]['median']:+.1f} ms p99 {rep['camera'][k]['p99']:+.1f} ms")
    # 4. gyro lag
    rep["gyro_lag"] = None
    if fit is not None and imu_key and arr[f"{imu_key}_recv_ns"].size >= 100:
        t_a = fit.time_of(arr["lowstate_tick"])
        t_b = arr[f"{imu_key}_stamp_ns"] / 1e9
        res = ts.gyro_lag(t_a, arr["lowstate_gyro"], t_b, arr[f"{imu_key}_gyro"], max_lag_s=max_lag_s)
        rep["gyro_lag"] = res.as_dict()
        if res.ok:
            ok = abs(res.lag_s) < 0.005
            verdict.append(f"{'OK ' if ok else 'NOTE'} gyro lag camera-minus-robot {res.lag_s * 1e3:+.2f} ms (corr {res.peak_corr:.2f}, motion rms {res.motion_rms_a:.2f}/{res.motion_rms_b:.2f} rad/s); {'within 5 ms' if ok else 'apply as a constant correction to the camera stamps'}")
        else:
            verdict.append(f"--  gyro lag not measured: {res.reason}")
    return rep, verdict


# ------------------------------------------------------- model-input analysis
TF_UNIT_S = 1e-5            # nominal unit of the front camera's time_frame (10 us)


def _lag_line(name, res, unit_note=""):
    if res is None:
        return f"--  {name}: no data"
    if res.ok:
        return f"MODEL {name} = {res.lag_s * 1e3:+.1f} ms  (corr {res.peak_corr:.2f}, motion rms {res.motion_rms_a:.2f}/{res.motion_rms_b:.2f}{unit_note}, window {res.overlap_s:.0f} s)"
    return f"--  {name}: not measured - {res.reason}"


def analyse_model_input(observed, arr, fit, go2_latency_ms=None, d435i_offset_ms=None,
                        grid_ms=80.0, max_lag_s=0.6):
    """Section 5 of the report: capture times of the two RGB streams the policy
    consumes, their calibration constants and the cost of the 80 ms grid."""
    rep = {"reference_clock": "laptop CLOCK_REALTIME (chrony server for the X5)", "grid_ms": grid_ms}
    lines = []
    # a. ego stream and its time model
    tf, recv = arr.get("ego_tf", np.empty(0, np.int64)), arr.get("ego_recv_ns", np.empty(0, np.int64))
    ego_fit = None
    if tf.size >= 30:
        recv_s = recv / 1e9
        step = np.diff(tf)
        counter_like = float(np.mean((step >= 6660) & (step <= 6674))) if step.size else 0.0
        ego_fit = ts.fit_tick_clock(tf, recv_s, iters=2, nominal_s_per_tick=TF_UNIT_S)
        fps = (tf.size - 1) / (recv_s[-1] - recv_s[0]) if recv_s[-1] > recv_s[0] else 0.0
        gaps = np.diff(recv_s) * 1e3
        rep["go2_ego"] = {"res": int(arr["ego_res"][0]), "msgs": int(tf.size), "fps": fps,
                          "arrival_gap_ms": {"median": float(np.median(gaps)), "p99": float(np.percentile(gaps, 99)), "max": float(gaps.max())},
                          "time_frame_step_is_1_15s": counter_like, "arrival_fit": ego_fit.as_dict(),
                          "decode_errors": int(arr["ego_decode_errors"][0]), "energy_samples": int(arr["ego_energy"].size),
                          "time_source": "arrival-envelope fit of time_frame (a frame counter, not a clock) minus pipeline latency"}
        lines.append(f"MODEL ego /frontvideostream {arr['ego_res'][0]}p: {fps:.2f} fps, arrival gap p99 {np.percentile(gaps, 99):.0f} / max {gaps.max():.0f} ms; "
                     f"time_frame steps of exactly 1/15 s: {counter_like * 100:.0f} % (a counter, not a clock); arrival-envelope fit residual p99 {ego_fit.resid_ms['p99']:.1f} ms")
    else:
        rep["go2_ego"] = {"msgs": int(tf.size)}
        lines.append("--  ego /frontvideostream: too few messages (robot off / --no-ego / PyAV missing)")
    seen = arr.get("ego_res_seen")
    if seen is not None:                                  # messages per res code, before the --go2-res filter
        rep["go2_ego"]["res_seen"] = {str(int(r)): int(n) for r, n in np.asarray(seen).reshape(-1, 2)}
    # energy signals at interval midpoints
    e_go2 = arr.get("ego_energy", np.empty(0)); lag_ego_imu = lag_col_gyro = lag_ego_col = None
    t_e_go2 = None
    if ego_fit is not None and e_go2.size >= 60:
        t_e_go2 = 0.5 * (ego_fit.time_of(arr["ego_energy_tf_prev"]) + ego_fit.time_of(arr["ego_energy_tf"]))   # uncalibrated arrival model
    col_stamp = arr["color_stamp_ns"] / 1e9
    e_col = arr.get("color_energy", np.empty(0))
    t_e_col = None
    if e_col.size >= 120 and col_stamp.size == e_col.size:
        ok = np.isfinite(e_col); ok[0] = False
        t_e_col = ts.midpoint_times(col_stamp)[ok[1:]]; e_col = e_col[ok]
    # b. ego energy vs Go2 IMU  -> ego pipeline latency
    if t_e_go2 is not None and fit is not None and arr["lowstate_tick"].size >= 100:
        lag_ego_imu = ts.gyro_lag(fit.time_of(arr["lowstate_tick"]), arr["lowstate_gyro"], t_e_go2, e_go2,
                                  max_lag_s=max_lag_s, grid_hz=1000.0, highpass_s=1.5, min_motion_rms=0.05,
                                  min_corr=0.5, min_motion_rms_b=0.8)
    # c. colour energy vs D435i gyro -> colour stamp minus exposure midpoint
    imu_key = "gyro" if arr["gyro_stamp_ns"].size else ("imu" if arr["imu_stamp_ns"].size else None)
    if t_e_col is not None and imu_key and arr[f"{imu_key}_stamp_ns"].size >= 100:
        lag_col_gyro = ts.gyro_lag(arr[f"{imu_key}_stamp_ns"] / 1e9, arr[f"{imu_key}_gyro"], t_e_col, e_col,
                                   max_lag_s=max_lag_s, grid_hz=1000.0, highpass_s=1.5, min_motion_rms=0.05,
                                   min_corr=0.5, min_motion_rms_b=0.8)
    # d. ego energy vs colour energy -> direct skew (= b - c)
    if t_e_go2 is not None and t_e_col is not None:
        lag_ego_col = ts.gyro_lag(t_e_col, e_col, t_e_go2, e_go2, max_lag_s=max_lag_s, grid_hz=1000.0,
                                  highpass_s=1.5, min_motion_rms=0.8, min_corr=0.5)
    rep["lags"] = {"ego_energy_vs_go2_imu": lag_ego_imu.as_dict() if lag_ego_imu else None,
                   "color_energy_vs_d435i_gyro": lag_col_gyro.as_dict() if lag_col_gyro else None,
                   "ego_energy_vs_color_energy": lag_ego_col.as_dict() if lag_ego_col else None,
                   "sign": "lag > 0: the second stream is stamped LATER than the first for the same motion"}
    lines.append(_lag_line("Go2 ego pipeline latency (ego image motion vs Go2 IMU; = arrival model - exposure midpoint)",
                           lag_ego_imu, " rad/s|grey"))
    lines.append(_lag_line("D435i colour stamp - exposure midpoint (colour image motion vs D435i gyro)", lag_col_gyro,
                           " rad/s|grey"))
    lines.append(_lag_line("ego arrival model - D435i stamp, same motion (the two image motions cross-correlated)",
                           lag_ego_col, " grey"))
    if lag_ego_imu and lag_col_gyro and lag_ego_col and lag_ego_imu.ok and lag_col_gyro.ok and lag_ego_col.ok:
        closure = (lag_ego_imu.lag_s - lag_col_gyro.lag_s - lag_ego_col.lag_s) * 1e3
        rep["lags"]["closure_ms"] = closure
        lines.append(f"MODEL closure (b - c - d) = {closure:+.1f} ms  (three independent measurements should agree: |closure| < 10 ms)")
    # calibration constants for the recording session
    cal = {"go2_ego_latency_ms": None, "go2_ego_latency_source": "uncalibrated (0)",
           "d435i_stamp_minus_exposure_ms": None, "d435i_stamp_source": "uncalibrated (0)"}
    if lag_ego_imu and lag_ego_imu.ok:
        cal.update(go2_ego_latency_ms=lag_ego_imu.lag_s * 1e3, go2_ego_latency_source="this run: ego energy vs Go2 IMU")
    elif go2_latency_ms is not None:
        cal.update(go2_ego_latency_ms=go2_latency_ms, go2_ego_latency_source="given (--go2-latency-ms / --calib)")
    if lag_col_gyro and lag_col_gyro.ok:
        cal.update(d435i_stamp_minus_exposure_ms=lag_col_gyro.lag_s * 1e3, d435i_stamp_source="this run: colour energy vs D435i gyro")
    elif d435i_offset_ms is not None:
        cal.update(d435i_stamp_minus_exposure_ms=d435i_offset_ms, d435i_stamp_source="given (--d435i-stamp-offset-ms / --calib)")
    rep["calibration"] = cal
    L = (cal["go2_ego_latency_ms"] or 0.0) * 1e-3; eD = (cal["d435i_stamp_minus_exposure_ms"] or 0.0) * 1e-3
    calibrated = cal["go2_ego_latency_ms"] is not None and cal["d435i_stamp_minus_exposure_ms"] is not None
    lines.append(f"MODEL capture-time constants: Go2 latency {L * 1e3:.0f} ms [{cal['go2_ego_latency_source']}], "
                 f"D435i stamp - exposure midpoint {eD * 1e3:.0f} ms [{cal['d435i_stamp_source']}]"
                 + ("" if calibrated else "  <- uncalibrated: rock the robot body during the window (both cameras "
                                          "fixed on it) and run again"))
    # e. the model grid
    if ego_fit is not None and col_stamp.size >= 10:
        ego_cap = ego_fit.time_of(tf) - L
        grid = ts.model_grid_pairing(ego_cap, col_stamp - eD, period_s=grid_ms * 1e-3)
        grid["calibrated"] = calibrated
        rep["grid"] = grid
        if grid.get("ok"):
            c, n = grid["causal"], grid["nearest"]
            lines.append(f"MODEL {grid_ms:.0f} ms grid ({grid['slots']} slots, ego {grid['ego_period_ms']:.1f} ms/frame, "
                         f"D435i {grid['task_period_ms']:.1f} ms/frame){'' if calibrated else ' [constants uncalibrated, structure only]'}: "
                         f"capture grid (offline), latest ego frame captured before the slot -> staleness median "
                         f"{c['ego_staleness_ms']['median']:.0f} / max {c['ego_staleness_ms']['max']:.0f} ms "
                         f"(an online consumer's ego age = L {L * 1e3:.0f} ms + this offset (+ decode) = "
                         f"{L * 1e3 + c['ego_staleness_ms']['median']:.0f} / {L * 1e3 + c['ego_staleness_ms']['max']:.0f} ms), "
                         f"nearest D435i frame vs ego capture |median| {c['abs_skew_ms']['median']:.0f} / max "
                         f"{c['abs_skew_ms']['max']:.0f} ms, duplicates {c['duplicates']}, gaps {c['ego_gaps']}; "
                         f"offline nearest ego frame -> offset max {n['ego_staleness_ms']['max']:.0f} ms")
        else:
            lines.append(f"--  {grid_ms:.0f} ms grid: {grid.get('reason')}")
    return rep, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--ns", default="/camera/d435i")
    x5_ip = os.environ.get("X5_IP", "")                      # config/deploy.env
    ap.add_argument("--x5", default=f"root@{x5_ip}" if x5_ip else "",
                    help="ssh target for the X5 chronyc query (default root@$X5_IP; '' to skip)")
    ap.add_argument("--x5-ip", default=x5_ip, help="the X5's address for the SNTP check (default $X5_IP)")
    ap.add_argument("--label", default="sync_check")
    ap.add_argument("--out-dir", default=None, help="default data/logs/camera_checks_<today>/")
    ap.add_argument("--no-images", action="store_true", help="skip colour/depth (saves ~1 subscriber of bandwidth)")
    ap.add_argument("--max-lag", type=float, default=0.2, help="gyro cross-correlation search window, s")
    ap.add_argument("--no-ego", action="store_true", help="skip the Go2 ego camera (/frontvideostream) section")
    ap.add_argument("--go2-res", type=int, default=360, choices=(360, 720), help="ego stream to decode for the motion signal")
    ap.add_argument("--go2-latency-ms", type=float, default=None, help="Go2 ego latency to use when this run cannot measure it")
    ap.add_argument("--d435i-stamp-offset-ms", type=float, default=None, help="D435i colour stamp minus exposure midpoint when not measurable here")
    ap.add_argument("--calib", default=None, help="earlier sync_check JSON: take model_input.calibration from it")
    ap.add_argument("--grid-ms", type=float, default=80.0, help="model RGB grid period (the policy: 80 ms = 12.5 Hz)")
    ap.add_argument("--no-metadata", action="store_true", help="skip the X5-side RealSense metadata sample")
    args = ap.parse_args()
    if args.calib:
        with open(args.calib) as f:
            cal = json.load(f).get("model_input", {}).get("calibration", {})
        if args.go2_latency_ms is None and cal.get("go2_ego_latency_ms") is not None:
            args.go2_latency_ms = float(cal["go2_ego_latency_ms"])
        if args.d435i_stamp_offset_ms is None and cal.get("d435i_stamp_minus_exposure_ms") is not None:
            args.d435i_stamp_offset_ms = float(cal["d435i_stamp_minus_exposure_ms"])
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir or os.path.join(here, "..", "..", "data", "logs", f"camera_checks_{time.strftime('%Y-%m-%d')}")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, f"{args.label}_{stamp}")

    report = {"label": args.label, "time": time.strftime("%Y-%m-%d %H:%M:%S %z"), "duration_request_s": args.duration}
    print("[sync_check] host clocks ...")
    hosts = {"laptop": chronyc_tracking(), "x5_sntp": sntp_offset(args.x5_ip)}
    if args.x5:
        hosts["x5"] = chronyc_tracking(("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", args.x5))
    report["hosts"] = hosts
    if not args.no_metadata and args.x5:
        print("[sync_check] sampling RealSense colour metadata on the X5 ...")
        report["x5_metadata"] = x5_metadata_sample(args.x5, f"{args.ns}/color/metadata")
    print(f"[sync_check] capturing {args.duration:.0f} s of /lowstate + camera + ego video - ROCK / TILT THE ROBOT BODY "
          f"(both cameras fixed on it) for the motion steps ...")
    observed, arr = capture(args.duration, args.ns, not args.no_images, with_ego=not args.no_ego, go2_res=args.go2_res)
    print("[sync_check] host clocks again ...")
    hosts["x5_sntp_after"] = sntp_offset(args.x5_ip)
    if args.x5:
        hosts["x5_after"] = chronyc_tracking(("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", args.x5))
    rep, verdict = analyse(observed, arr, args.max_lag)
    report.update(rep)
    fit = ts.TickFit(**rep["lowstate"]["tick_fit"]) if "tick_fit" in rep["lowstate"] else None
    model_rep, model_lines = analyse_model_input(observed, arr, fit, go2_latency_ms=args.go2_latency_ms,
                                                 d435i_offset_ms=args.d435i_stamp_offset_ms, grid_ms=args.grid_ms)
    md = report.get("x5_metadata")
    if md is not None:
        if "error" in md:
            model_lines.append(f"--  X5 RealSense colour metadata: {md['error']}")
        else:
            hold = md.get("sdk_hold_ms", {}).get("median")
            model_lines.append(f"MODEL X5 colour metadata ({md['n']} frames, {md.get('fps', 0):.1f} fps, frame-number gaps "
                               f"{md.get('frame_number_gaps', '?')}, {md.get('clock_domain')}): "
                               f"fields {','.join(md['fields'])}; "
                               + (f"exposure {md['actual_exposure_us']['median']:.0f} us, sensor-frame {md['sensor_minus_frame_us_median']:+.0f} us"
                                  if md["has_exposure_metadata"] else
                                  "no sensor_timestamp/actual_exposure -> the exposure midpoint is measurable only by "
                                  "the motion method (c)")
                               + (f"; frames wait {hold:.0f} ms in the SDK before the ROS node gets them" if hold is not None else ""))
    report["model_input"] = model_rep
    verdict.extend(model_lines)

    # host verdicts: laptop chrony, X5 chrony, SNTP (info), X5 raw minus laptop raw (before + after)
    host_lines, report["host_clock"] = host_verdict_lines(hosts)
    verdict[0:0] = host_lines
    report["verdict"] = verdict

    text = [f"# {report['time']} label={args.label} observed={observed:.1f}s", *verdict]
    print("\n".join(text))
    with open(base + ".txt", "w") as f:
        f.write("\n".join(text) + "\n")
    with open(base + ".json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    np.savez_compressed(base + ".npz", **arr)
    print(f"saved: {base}.{{txt,json,npz}}")
    return 0 if not any(v.startswith("BAD") for v in verdict) else 1


if __name__ == "__main__":
    sys.exit(main())
