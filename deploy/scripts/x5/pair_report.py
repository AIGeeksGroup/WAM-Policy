#!/usr/bin/env python3
"""Time-sync report over ego_d435i_view.py session summaries (every task run writes one:
result/<task>/<time>/view/camera.json) plus a long-term common-event re-test.

Every session in which both cameras saw the same motion (the robot standing up or walking - both cameras are on the body - or
a hand-wave) carries motion-lag windows, lag = Go2 arrival model - D435i stamp = L - eD. This tool
prints one line per session (pairs, unmatched ratio, |delta| p95, sync method, time-source
verdicts, lags) and the lags across sessions; --write-calib turns their median into a --calib file
with which the next sessions pair by capture time (sync_method capture_time):

    python3 scripts/x5/pair_report.py result/*/*/view/camera.json
    python3 scripts/x5/pair_report.py result/*/*/view/camera.json --write-calib calib_relative.json
    DEMO_VIEW_CALIB=calib_relative.json tasks/push_box.sh

The correlation windows overlap (25 s every 3 s): judge stability by the per-session medians and
their spread, not by the window count. Only L - eD is measured, so the file calibrates the PAIRING;
absolute capture times still need eD from sync_check.py. Read-only on its inputs, no ROS.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from wam_go2 import camera_pairing as cp  # noqa: E402


def session_lags(d):
    return [float(h["lag_ms"]) for h in (d.get("motion_lag_hist") or []) if h.get("lag_ms") is not None]


def session_line(name, d):
    parts = [name, str(d.get("time", "?")), f"{float(d.get('wall_s') or 0):.0f} s"]
    tsum = d.get("time_sync")
    if tsum:
        p = tsum.get("pairing", {})
        dl = p.get("delta_ms", {})
        if p.get("decided"):
            parts.append(f"pairs {p['matched']}/{p['decided']} ({100 * p['unmatched_ratio']:.1f} % unmatched)"
                         + (f", |delta| p95 {dl['p95_abs']:.1f} ms" if dl.get("n") else ""))
        else:
            parts.append("pairs 0/0")
        cals = tsum.get("calibrations") or []
        if len(cals) > 1:                       # the calibration changed mid-session: each part has its own label
            parts.append("mixed: " + " then ".join(f"{c.get('sync_method')} {c.get('matched')}/{c.get('decided')} "
                                                   f"[{c.get('calibration_version')}]" for c in cals))
        else:
            parts.append(f"{tsum.get('sync_method')} [{tsum.get('calibration_version')}]")
        src = tsum.get("time_sources", {})
        e, dd = src.get("ego", {}), src.get("d435i", {})
        parts.append(f"ego {e.get('verdict', '?')} {e.get('receive_rate_hz') or 0:.2f} fps lost {e.get('lost_frames', '?')}, "
                     f"d435i {dd.get('stamp_rate_hz') or 0:.2f} fps lost {dd.get('lost_frames', '?')}")
    else:
        msgs = d.get("msgs", {})
        parts.append(f"no time_sync block: msgs go2 {msgs.get('go2', '?')} / d435i {msgs.get('d435i', '?')}")
    lags = session_lags(d)
    parts.append(f"common-event lag n={len(lags)} median {np.median(lags):+.1f} ms [{min(lags):+.1f}, {max(lags):+.1f}]"
                 if lags else "no common-event lag")
    return " | ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="ego_d435i_view.py summary JSONs (e.g. result/*/*/view/camera.json)")
    ap.add_argument("--write-calib", default=None, help="write a relative --calib JSON from the lags' median")
    ap.add_argument("--ed-ms", type=float, default=0.0,
                    help="D435i stamp - exposure midpoint to keep in the file (sync_check.py); default 0 = unknown")
    args = ap.parse_args(argv)
    sessions = []
    for path in args.paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, ValueError) as e:
            print(f"{name} | unreadable: {e}")
            continue
        print(session_line(name, d))
        sessions.append((name, session_lags(d)))
    agg = cp.aggregate_motion_lags(sessions)
    if agg["n_windows"]:
        n_with = sum(1 for r in agg["sessions"] if r["n"])
        print(f"common-event lag across sessions: windows {agg['n_windows']} in {n_with} session(s), "
              f"median {agg['median_ms']:+.1f} ms, IQR [{agg['p25_ms']:+.1f}, {agg['p75_ms']:+.1f}], "
              f"range [{agg['min_ms']:+.1f}, {agg['max_ms']:+.1f}], session medians spread {agg['session_median_spread_ms']:.1f} ms")
        print("  lag = L - eD (Go2 arrival model minus D435i stamp for the same motion): uncalibrated "
              "(receive_time_approx) pairs are this far apart in capture time, the D435i frame being the later one")
    else:
        print("common-event lag across sessions: none (no session saw the same motion in both cameras)")
    if args.write_calib:
        try:
            cal = cp.relative_calibration(agg, ed_cfg_ms=args.ed_ms)
        except ValueError as e:
            print(f"--write-calib: {e}")
            return 2
        cal.update({"created": time.strftime("%Y-%m-%d %H:%M:%S %z"), "tool": "scripts/x5/pair_report.py",
                    "inputs": list(args.paths)})
        with open(args.write_calib, "w") as f:
            json.dump(cal, f, indent=2)
        print(f"wrote {args.write_calib}: go2_ego_latency_ms {cal['model_input']['calibration']['go2_ego_latency_ms']:.1f} "
              f"(relative, n={cal['n_windows']} windows). Use: DEMO_VIEW_CALIB={args.write_calib} scripts/demo/<demo>.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
