#!/usr/bin/env python3
"""Convert a Qualisys marker TSV to a rosbag and optionally merge with a trial bag.

Examples:
  # Marker bag only (next to the TSV):
  python infants/scripts/process_marker_tsv.py --file_path trial.tsv

  # TSV + trial_ros.bag -> combined bag for run_trial_viz.py --markers:
  python infants/scripts/process_marker_tsv.py \\
      --file_path trial.tsv \\
      --camera-bag data/.../trial_001/trial_ros.bag
"""
import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import pytz
import rosbag
import rospy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Int32

DEFAULT_TZ = "America/Chicago"
DATA_START_ROW = 12


def parse_tsv_header(rows, tz_name=DEFAULT_TZ):
    """Read marker count, rate, and wall-clock start time from Qualisys TSV header."""
    num_markers = int(rows[2][1])
    marker_rate = int(rows[3][1])
    start_timestamp_str = rows[7][1]
    local_tz = pytz.timezone(tz_name)
    dt = datetime.strptime(start_timestamp_str, "%Y-%m-%d, %H:%M:%S.%f")
    start_timestamp = local_tz.localize(dt).timestamp()
    return num_markers, marker_rate, start_timestamp


def load_start_timestamp_from_tsv(tsv_path, tz_name=DEFAULT_TZ):
    """Return Unix start time from the Qualisys TSV header (row 7)."""
    tsv_path = Path(tsv_path).expanduser().resolve()
    if not tsv_path.is_file():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")
    with open(tsv_path, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    _, _, start_timestamp = parse_tsv_header(rows, tz_name=tz_name)
    return start_timestamp


def tsv_to_marker_bag(tsv_path, output_bag_path=None):
    """Write /marker_N PointStamped messages from a Qualisys TSV export."""
    tsv_path = Path(tsv_path).expanduser().resolve()
    if not tsv_path.is_file():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")

    if output_bag_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_bag_path = tsv_path.parent / f"tsv_{stamp}.bag"
    else:
        output_bag_path = Path(output_bag_path).expanduser().resolve()

    output_bag_path.parent.mkdir(parents=True, exist_ok=True)
    if output_bag_path.exists():
        output_bag_path.unlink()

    with open(tsv_path, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))

    num_markers, marker_rate, start_timestamp = parse_tsv_header(rows)
    marker_rows = rows[DATA_START_ROW:]

    with rosbag.Bag(str(output_bag_path), "w") as bag:
        num_markers_msg = Int32(data=num_markers)
        bag.write(
            "/metadata/num_markers",
            num_markers_msg,
            rospy.Time.from_sec(start_timestamp),
        )

        for row_idx, row in enumerate(marker_rows):
            t = start_timestamp + row_idx / marker_rate
            stamp = rospy.Time.from_sec(t)

            for m in range(num_markers):
                try:
                    x = float(row[m * 3 + 0])
                    y = float(row[m * 3 + 1])
                    z = float(row[m * 3 + 2])
                except (IndexError, ValueError):
                    x = y = z = float("nan")

                msg = PointStamped()
                msg.header.stamp = stamp
                msg.header.frame_id = f"marker_{m + 1}"
                msg.point.x = x
                msg.point.y = y
                msg.point.z = z
                bag.write(f"/marker_{m + 1}", msg, stamp)

    return output_bag_path, {
        "num_markers": num_markers,
        "marker_rate": marker_rate,
        "start_timestamp": start_timestamp,
        "num_rows": len(marker_rows),
    }


def merge_camera_and_marker_bags(camera_bag_path, marker_bag_path, output_path, pad=0.05):
    """Copy camera bag as-is; add marker topics trimmed to the camera time window."""
    camera_bag_path = Path(camera_bag_path).expanduser().resolve()
    marker_bag_path = Path(marker_bag_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if not camera_bag_path.is_file():
        raise FileNotFoundError(f"Camera bag not found: {camera_bag_path}")
    if not marker_bag_path.is_file():
        raise FileNotFoundError(f"Marker bag not found: {marker_bag_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with rosbag.Bag(str(camera_bag_path)) as cb:
        cam_start = cb.get_start_time()
        cam_end = cb.get_end_time()
    with rosbag.Bag(str(marker_bag_path)) as mb:
        marker_start = mb.get_start_time()
        marker_end = mb.get_end_time()

    t_lo = cam_start - pad
    t_hi = cam_end + pad

    print(f"\nCamera bag:  {cam_start:.3f} -> {cam_end:.3f}")
    print(f"Marker bag:  {marker_start:.3f} -> {marker_end:.3f}")
    print(f"Real offset (cam_start - marker_start): {cam_start - marker_start:+.3f}s")
    print(f"Keeping mocap in window [{t_lo:.3f}, {t_hi:.3f}] (pad={pad}s)")
    print(f"\nMerging -> {output_path}\n")

    kept = dropped_early = dropped_late = 0
    cam_count = 0
    with rosbag.Bag(str(output_path), "w") as outbag:
        with rosbag.Bag(str(camera_bag_path)) as inbag:
            for topic, msg, t in inbag.read_messages():
                outbag.write(topic, msg, t)
                cam_count += 1
        print(f"  {cam_count} camera messages copied as-is")

        with rosbag.Bag(str(marker_bag_path)) as inbag:
            for topic, msg, t in inbag.read_messages():
                ts = t.to_sec()
                if ts < t_lo:
                    dropped_early += 1
                    continue
                if ts > t_hi:
                    dropped_late += 1
                    continue
                outbag.write(topic, msg, t)
                kept += 1

    print(
        f"  marker kept: {kept}  "
        f"dropped_early: {dropped_early}  dropped_late: {dropped_late}"
    )
    return output_path


def default_combined_output(camera_bag_path):
    camera_bag_path = Path(camera_bag_path)
    return camera_bag_path.with_name(f"{camera_bag_path.stem}_combined.bag")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Qualisys TSV to marker rosbag; optionally merge with trial bag."
    )
    parser.add_argument(
        "--file_path",
        required=True,
        help="Path to Qualisys marker TSV file",
    )
    parser.add_argument(
        "--camera-bag",
        type=Path,
        help="Trial camera rosbag (e.g. trial_ros.bag). If set, writes a combined bag.",
    )
    parser.add_argument(
        "--marker-bag",
        type=Path,
        help="Output path for intermediate marker bag (default: <tsv_dir>/tsv_<timestamp>.bag)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Combined output bag when --camera-bag is set "
        "(default: <camera_bag_stem>_combined.bag next to camera bag)",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.05,
        help="Seconds of mocap to keep outside camera window when merging (default 0.05)",
    )
    parser.add_argument(
        "--keep-marker-bag",
        action="store_true",
        default=True,
        help="Keep the intermediate marker bag after merging (default: true)",
    )
    parser.add_argument(
        "--no-keep-marker-bag",
        dest="keep_marker_bag",
        action="store_false",
        help="Delete intermediate marker bag after merging",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    marker_bag_path, info = tsv_to_marker_bag(args.file_path, args.marker_bag)
    print(f"Marker bag saved to {marker_bag_path}")
    print(
        f"  markers={info['num_markers']}  rate={info['marker_rate']} Hz  "
        f"rows={info['num_rows']}"
    )

    if args.camera_bag is None:
        return 0

    output_path = args.output or default_combined_output(args.camera_bag)
    merge_camera_and_marker_bags(
        args.camera_bag, marker_bag_path, output_path, pad=args.pad
    )
    print(f"\nCombined bag saved to {output_path}")

    if not args.keep_marker_bag and marker_bag_path != output_path:
        marker_bag_path.unlink(missing_ok=True)
        print(f"Removed intermediate marker bag {marker_bag_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
