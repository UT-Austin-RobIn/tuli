#!/usr/bin/env python3
"""Merge camera + marker rosbags preserving real wall-clock timestamps.

Both bags already carry Unix-epoch timestamps (RS driver uses /clock, the
mocap tsv->bag uses the TSV header wall-clock). We keep them AS-IS and just
drop mocap samples outside the camera bag's time window so the combined bag
doesn't contain mocap data from before/after the RS recording.

A small symmetric pad (default 50 ms) is kept on both ends so nearest-in-time
lookup at the very first/last camera frame still sees a marker.
"""
import argparse
import os
from datetime import datetime

import rosbag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-bag")
    ap.add_argument("--marker-bag")
    ap.add_argument("--output", default=None)
    ap.add_argument("--pad", type=float, default=0.05,
                    help="seconds of mocap to keep outside camera window "
                         "(default 0.05s on each side)")
    args = ap.parse_args()

    camera_bag_path = args.camera_bag or input("Path to camera rosbag: ").strip()
    marker_bag_path = args.marker_bag or input("Path to marker rosbag: ").strip()

    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.expanduser("~/infants/data")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"combined_{timestamp}.bag")

    with rosbag.Bag(camera_bag_path) as cb:
        cam_start = cb.get_start_time()
        cam_end = cb.get_end_time()
    with rosbag.Bag(marker_bag_path) as mb:
        marker_start = mb.get_start_time()
        marker_end = mb.get_end_time()

    t_lo = cam_start - args.pad
    t_hi = cam_end + args.pad

    print(f"\nCamera bag:  {cam_start:.3f} -> {cam_end:.3f}")
    print(f"Marker bag:  {marker_start:.3f} -> {marker_end:.3f}")
    print(f"Real offset (cam_start - marker_start): {cam_start - marker_start:+.3f}s")
    print(f"Keeping mocap in window [{t_lo:.3f}, {t_hi:.3f}] (pad={args.pad}s)")
    print(f"\nMerging -> {output_path}\n")

    kept = dropped_early = dropped_late = 0
    with rosbag.Bag(output_path, 'w') as outbag:
        with rosbag.Bag(camera_bag_path) as inbag:
            cam_count = 0
            for topic, msg, t in inbag.read_messages():
                outbag.write(topic, msg, t)
                cam_count += 1
            print(f"  {cam_count} camera messages copied as-is")

        with rosbag.Bag(marker_bag_path) as inbag:
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

    print(f"  marker kept: {kept}  dropped_early: {dropped_early}  dropped_late: {dropped_late}")
    print(f"\nCombined bag saved to {output_path}")


if __name__ == "__main__":
    main()
