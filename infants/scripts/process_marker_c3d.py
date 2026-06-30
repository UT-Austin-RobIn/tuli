#!/usr/bin/env python3
"""Convert a Qualisys marker C3D to a rosbag and optionally merge with a trial bag.

C3D is Qualisys's binary motion-capture format (see --inspect). Unlike TSV exports,
it usually does not embed wall-clock start time — pass --tsv (recommended),
--trial-metadata, or --start-time so marker timestamps align with trial_ros.bag.

Requires: pip install c3d

Examples:
  # Inspect structure (labels, rate, units, frame count):
  python infants/scripts/process_marker_c3d.py --file_path trial.c3d --inspect

  # Marker bag using wall-clock start from a Qualisys TSV export of the same trial:
  python infants/scripts/process_marker_c3d.py \\
      --file_path trial.c3d \\
      --tsv trial.tsv

  # C3D + trial_ros.bag -> combined bag for run_trial_viz.py --markers:
  python infants/scripts/process_marker_c3d.py \\
      --file_path trial.c3d \\
      --tsv trial.tsv \\
      --camera-bag data/.../trial_001/trial_ros.bag
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pytz
import rosbag
import rospy
import yaml
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Int32

try:
    import c3d
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'c3d'. Install with: pip install c3d"
    ) from exc

from process_marker_tsv import (
    default_combined_output,
    load_start_timestamp_from_tsv,
    merge_camera_and_marker_bags,
)

DEFAULT_TZ = "America/Chicago"
UNUSED_LABELS = {"", "*", "undefined"}


def _param_float(reader, key, default=None):
    param = reader.get(key)
    if param is None or not param.dimensions:
        return default
    return float(param.float_array[0])


def _param_strings(reader, key):
    param = reader.get(key)
    if param is None or not param.dimensions:
        return []
    return [str(s).strip() for s in param.string_array]


def parse_start_timestamp(start_time_str, tz_name=DEFAULT_TZ):
    """Parse ISO or Qualisys-style wall-clock strings to Unix seconds."""
    local_tz = pytz.timezone(tz_name)
    text = start_time_str.strip()
    for fmt in ("%Y-%m-%d, %H:%M:%S.%f", "%Y-%m-%d, %H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt)
            return local_tz.localize(dt).timestamp()
        except ValueError:
            continue
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = local_tz.localize(dt)
    return dt.timestamp()


def load_start_timestamp_from_metadata(metadata_path, tz_name=DEFAULT_TZ):
    metadata_path = Path(metadata_path).expanduser().resolve()
    with open(metadata_path) as f:
        metadata = yaml.safe_load(f) or {}
    start_time = metadata.get("start_time")
    if not start_time:
        raise ValueError(f"No start_time in {metadata_path}")
    return parse_start_timestamp(str(start_time), tz_name=tz_name)


def marker_labels(reader, num_markers):
    """Return num_markers labels, padding with marker_N if the file has fewer names."""
    try:
        raw = [str(label).strip() for label in reader.point_labels]
    except AttributeError:
        raw = _param_strings(reader, "POINT:LABELS")
    labels = []
    for i in range(num_markers):
        if i < len(raw) and raw[i].lower() not in UNUSED_LABELS:
            labels.append(raw[i])
        else:
            labels.append(f"marker_{i + 1}")
    return labels


def active_marker_labels(reader):
    """Return non-empty Qualisys marker names for all points in the file."""
    count = reader.header.point_count
    if count == 0:
        return []
    return marker_labels(reader, count)


def inspect_c3d(c3d_path):
    """Print a human-readable summary of a C3D file."""
    c3d_path = Path(c3d_path).expanduser().resolve()
    with open(c3d_path, "rb") as f:
        reader = c3d.Reader(f)
        header = reader.header
        labels = active_marker_labels(reader)
        rate = _param_float(reader, "POINT:RATE", default=float(header.frame_rate))
        units = _param_strings(reader, "POINT:UNITS")
        units_str = units[0] if units else "mm (assumed)"

        print(f"File: {c3d_path}")
        print(f"  frames (header): {reader.frame_count}  "
              f"(first={reader.first_frame}, last={reader.last_frame})")
        print(f"  point_count (header): {header.point_count}")
        print(f"  rate: {rate} Hz")
        print(f"  units: {units_str}")
        print(f"  markers ({len(labels)}): {labels or '(none — file may be header-only)'}")

        event_times = _param_strings(reader, "EVENT:LABELS")
        if event_times:
            print(f"  events: {event_times}")

        if header.point_count == 0 or not labels:
            print(
                "\n  WARNING: No 3-D point data in this file. "
                "Re-export from Qualisys (File -> Export -> C3D) and ensure "
                "the trial was captured with visible markers."
            )
        elif reader.frame_count > 0:
            duration = reader.frame_count / rate if rate else 0.0
            print(f"  duration: {duration:.2f} s")


def c3d_to_marker_bag(
    c3d_path,
    output_bag_path=None,
    start_timestamp=None,
    num_markers=None,
    tz_name=DEFAULT_TZ,
):
    """Write /marker_N PointStamped messages from a Qualisys C3D export."""
    c3d_path = Path(c3d_path).expanduser().resolve()
    if not c3d_path.is_file():
        raise FileNotFoundError(f"C3D not found: {c3d_path}")
    if start_timestamp is None:
        raise ValueError(
            "C3D files rarely include wall-clock time. "
            "Pass --tsv, --start-time, or --trial-metadata."
        )

    if output_bag_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_bag_path = c3d_path.parent / f"c3d_{stamp}.bag"
    else:
        output_bag_path = Path(output_bag_path).expanduser().resolve()

    output_bag_path.parent.mkdir(parents=True, exist_ok=True)
    if output_bag_path.exists():
        output_bag_path.unlink()

    with open(c3d_path, "rb") as f:
        reader = c3d.Reader(f)
        if reader.header.point_count == 0:
            raise ValueError(
                f"No marker trajectories in {c3d_path.name} "
                f"(point_count=0). "
                "The file looks like metadata only — re-export from Qualisys."
            )

        point_count = reader.header.point_count
        if num_markers is None:
            num_markers = point_count
        elif num_markers > point_count:
            raise ValueError(
                f"--num-markers {num_markers} exceeds C3D point_count {point_count}"
            )

        labels = marker_labels(reader, num_markers)
        rate = _param_float(reader, "POINT:RATE", default=float(reader.header.frame_rate))
        if rate <= 0:
            raise ValueError(f"Invalid marker rate in C3D: {rate}")
        frames_written = 0

        with rosbag.Bag(str(output_bag_path), "w") as bag:
            num_markers_msg = Int32(data=num_markers)
            bag.write(
                "/metadata/num_markers",
                num_markers_msg,
                rospy.Time.from_sec(start_timestamp),
            )

            for frame_idx, (_frame_num, points, _analog) in enumerate(
                reader.read_frames()
            ):
                t = start_timestamp + frame_idx / rate
                stamp = rospy.Time.from_sec(t)
                frames_written += 1

                if points.shape[0] < num_markers:
                    raise ValueError(
                        f"Frame {frame_idx}: expected >= {num_markers} points, "
                        f"got shape {points.shape}"
                    )
                xyz = points[:num_markers, :3]

                for m in range(num_markers):
                    x, y, z = (float(xyz[m, 0]), float(xyz[m, 1]), float(xyz[m, 2]))
                    msg = PointStamped()
                    msg.header.stamp = stamp
                    msg.header.frame_id = labels[m]
                    msg.point.x = x
                    msg.point.y = y
                    msg.point.z = z
                    bag.write(f"/marker_{m + 1}", msg, stamp)

    return output_bag_path, {
        "num_markers": num_markers,
        "marker_labels": labels,
        "marker_rate": rate,
        "start_timestamp": start_timestamp,
        "num_frames": frames_written,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Qualisys C3D to marker rosbag; optionally merge with trial bag."
    )
    parser.add_argument(
        "--file_path",
        required=True,
        help="Path to Qualisys marker C3D file",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print C3D structure and exit (no bag written)",
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        help="Qualisys TSV export of the same trial; wall-clock start read from header",
    )
    parser.add_argument(
        "--start-time",
        help="Wall-clock trial start (ISO or 'YYYY-MM-DD, HH:MM:SS.ffffff'). "
        "Defaults to America/Chicago if no timezone.",
    )
    parser.add_argument(
        "--trial-metadata",
        type=Path,
        help="trial_metadata.yaml from experiment_driver (uses start_time field)",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TZ,
        help=f"Timezone for naive timestamps (default: {DEFAULT_TZ})",
    )
    parser.add_argument(
        "--num-markers",
        type=int,
        help="How many C3D markers to export as /marker_1..N (default: all in file). "
        "Use 1 for single-marker infant trials when Qualisys exports many unlabeled points.",
    )
    parser.add_argument(
        "--camera-bag",
        type=Path,
        help="Trial camera rosbag (e.g. trial_ros.bag). If set, writes a combined bag.",
    )
    parser.add_argument(
        "--marker-bag",
        type=Path,
        help="Output path for intermediate marker bag (default: <c3d_dir>/c3d_<timestamp>.bag)",
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

    if args.inspect:
        inspect_c3d(args.file_path)
        return 0

    start_timestamp = None
    time_source = None
    if args.tsv:
        start_timestamp = load_start_timestamp_from_tsv(args.tsv, tz_name=args.timezone)
        time_source = args.tsv
    elif args.start_time:
        start_timestamp = parse_start_timestamp(args.start_time, tz_name=args.timezone)
        time_source = "--start-time"
    elif args.trial_metadata:
        start_timestamp = load_start_timestamp_from_metadata(
            args.trial_metadata, tz_name=args.timezone
        )
        time_source = args.trial_metadata
    else:
        print(
            "Error: C3D has no embedded wall-clock time. "
            "Use --tsv, --trial-metadata, or --start-time.",
            file=sys.stderr,
        )
        return 1

    print(f"Wall-clock start from {time_source}: {start_timestamp:.6f}")

    marker_bag_path, info = c3d_to_marker_bag(
        args.file_path,
        args.marker_bag,
        start_timestamp=start_timestamp,
        num_markers=args.num_markers,
        tz_name=args.timezone,
    )
    print(f"Marker bag saved to {marker_bag_path}")
    print(
        f"  markers={info['num_markers']} {info['marker_labels']}  "
        f"rate={info['marker_rate']} Hz  frames={info['num_frames']}"
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
