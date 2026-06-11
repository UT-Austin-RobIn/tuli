#!/usr/bin/env python3
"""Convert a Qualisys TSV (3D mocap export, single marker) into a ROS1 bag with
geometry_msgs/PointStamped on /marker_1, time-aligned to the original recording.

The TSV format is:
    NO_OF_FRAMES   <int>
    NO_OF_CAMERAS  <int>
    NO_OF_MARKERS  <int>          (this script handles 1 marker)
    FREQUENCY      <Hz>
    NO_OF_ANALOG   0
    ANALOG_FREQUENCY 0
    DESCRIPTION    --
    TIME_STAMP     <YYYY-MM-DD>, <HH:MM:SS.fff>  <internal>
    DATA_INCLUDED  3D
    MARKER_NAMES   <name>
    TRAJECTORY_TYPES Measured
    <frame_idx>\t<elapsed_sec>\t<X>\t<Y>\t<Z>\r\n
    ...

Coordinates in the TSV are millimeters; output is meters.
Frames where (X,Y,Z) == (0,0,0) are treated as "marker not tracked" and skipped.

Run inside `conda activate ros1`.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import rosbag
import rospy
from geometry_msgs.msg import PointStamped


def parse_header(lines):
    """Return (n_frames, freq_hz, t0_unix, header_end_idx)."""
    meta = {}
    end = None
    for i, line in enumerate(lines):
        if "\t" in line:
            key, _, rest = line.partition("\t")
            meta[key.strip()] = rest.strip()
        if line.startswith("TRAJECTORY_TYPES"):
            end = i + 1
            break
    if end is None:
        sys.exit("TSV header missing TRAJECTORY_TYPES line")
    n_frames = int(meta["NO_OF_FRAMES"])
    freq = float(meta["FREQUENCY"])
    n_markers = int(meta["NO_OF_MARKERS"])
    if n_markers != 1:
        sys.exit(f"This converter only handles NO_OF_MARKERS=1, got {n_markers}")
    # TIME_STAMP looks like: "2026-04-15, 22:18:58.600\t76636.70220850"
    ts_str = meta["TIME_STAMP"].split("\t", 1)[0].strip()
    m = re.match(r"(\d{4})-(\d{2})-(\d{2}),\s*(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?", ts_str)
    if not m:
        sys.exit(f"Could not parse TIME_STAMP {ts_str!r}")
    yr, mo, dy, hh, mm, ss, frac = m.groups()
    micro = int((frac or "0").ljust(6, "0")[:6])
    dt = datetime(int(yr), int(mo), int(dy), int(hh), int(mm), int(ss), micro)
    t0 = dt.timestamp()  # naive interpretation = local TZ; matches the bag's stamps
    return n_frames, freq, t0, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", type=Path, help="path to Qualisys TSV file")
    ap.add_argument("out", type=Path, help="output .bag path")
    ap.add_argument("--topic", default="/marker_1", help="output topic name")
    ap.add_argument("--frame-id", default="qualisys_ref",
                    help="header.frame_id for each PointStamped (default: qualisys_ref)")
    ap.add_argument("--t0", type=float, default=None,
                    help="override Unix epoch of TSV frame 1 (default: parsed from TIME_STAMP)")
    ap.add_argument("--keep-zeros", action="store_true",
                    help="emit frames where (X,Y,Z)==(0,0,0) (default: skip)")
    args = ap.parse_args()

    raw = args.tsv.read_text(encoding="utf-8", errors="replace")
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    n_frames, freq, t0_parsed, body_start = parse_header(lines)
    t0 = args.t0 if args.t0 is not None else t0_parsed
    print(f"TSV: {args.tsv}")
    print(f"  n_frames = {n_frames}, freq = {freq} Hz, duration = {n_frames / freq:.2f} s")
    print(f"  t0 (Unix) = {t0:.6f}  (parsed: {t0_parsed:.6f})")

    written = 0
    skipped = 0
    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with rosbag.Bag(str(args.out), "w") as bag:
        for line in lines[body_start:]:
            if not line.strip():
                continue
            cols = line.split("\t")
            if len(cols) < 5:
                continue
            try:
                idx = int(cols[0])
                # cols[1] is the elapsed time in the file but we recompute from idx
                # for exact alignment with `t0 + (idx-1)/freq`.
                x_mm = float(cols[2])
                y_mm = float(cols[3])
                z_mm = float(cols[4])
            except ValueError:
                continue
            if (not args.keep_zeros) and x_mm == 0.0 and y_mm == 0.0 and z_mm == 0.0:
                skipped += 1
                continue
            stamp_sec = t0 + (idx - 1) / freq
            stamp = rospy.Time.from_sec(stamp_sec)
            msg = PointStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = args.frame_id
            msg.point.x = x_mm / 1000.0
            msg.point.y = y_mm / 1000.0
            msg.point.z = z_mm / 1000.0
            bag.write(args.topic, msg, t=stamp)
            written += 1
    print(f"  wrote {written} messages to {args.out} (skipped {skipped} zero rows)")


if __name__ == "__main__":
    main()
