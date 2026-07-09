#!/usr/bin/env python3
"""Overlay Qualisys markers onto RealSense color images from a trial bag.

Uses the same transform chain as marker_transformer / run_trial_viz:
  p_cam = T_mcR_to_cam @ p_mcR

Marker positions come from /marker_N topics in the bag (mm, Qualisys mcR).
Intrinsics come from /cam_L/color/camera_info in the bag.

Examples:
  python infants/scripts/overlay_markers_on_image.py \\
      --bag data/2026-06-29_15-03-28/trial_001/trial_ros_combined.bag \\
      --calib-config data/calibration_data/26_06_29_infant_017/calibration_markers.yaml

  # Save frames instead of interactive playback:
  python infants/scripts/overlay_markers_on_image.py \\
      --bag .../trial_ros_combined.bag \\
      --calib-config .../calibration_markers.yaml \\
      --save-dir /tmp/marker_overlay
"""
import argparse
import bisect
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rosbag

from calibration_chain import (
    build_camera_transforms,
    load_config,
    resolve_config_path,
)

CAMERA_NAMES = {
    "L": "cam_L",
    "M": "cam_M",
    "R": "cam_R",
}

MARKER_COLORS_BGR = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
]


def imgmsg_to_bgr8(msg):
    """Decode sensor_msgs/Image from a bag without rospy / roscore."""
    if msg.encoding not in ("bgr8", "rgb8", "8UC3"):
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(img)


def overlay_point_on_image(image, pixel, color=(0, 0, 255), radius=8, thickness=-1):
    img_out = image.copy()
    u, v = int(round(pixel[0])), int(round(pixel[1]))
    h, w = img_out.shape[:2]
    if 0 <= u < w and 0 <= v < h:
        cv2.circle(img_out, (u, v), radius, color, thickness)
    return img_out


def project_point_to_image(point_3d, K, dist_coeffs=None):
    points_3d = np.asarray(point_3d, dtype=np.float32).reshape(-1, 1, 3)
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    dist = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float32)
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist)
    return points_2d.reshape(-1, 2)


def find_closest_msg(times, msgs, target_time):
    idx = bisect.bisect_left(times, target_time)
    if idx == 0:
        return msgs[0], times[0]
    if idx >= len(times):
        return msgs[-1], times[-1]
    before_t, after_t = times[idx - 1], times[idx]
    if abs(before_t - target_time) <= abs(after_t - target_time):
        return msgs[idx - 1], before_t
    return msgs[idx], after_t


def marker_is_valid(msg):
    vals = (msg.point.x, msg.point.y, msg.point.z)
    if any(np.isnan(v) for v in vals):
        return False
    if vals == (0.0, 0.0, 0.0):
        return False
    return True


def load_intrinsics_from_bag(bag, camera_name):
    info_topic = f"/{camera_name}/color/camera_info"
    for _topic, msg, _t in bag.read_messages(topics=[info_topic]):
        K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        dist = np.array(msg.D, dtype=np.float64) if msg.D else None
        return K, dist
    raise ValueError(f"No CameraInfo on {info_topic}")


def load_bag_streams(bag, camera_name, num_markers):
    """Read camera images and marker topics in a single bag pass."""
    color_topic = f"/{camera_name}/color/image_raw"
    info_topic = f"/{camera_name}/color/camera_info"
    marker_topics = {f"/marker_{m}" for m in range(1, num_markers + 1)}
    topics = [color_topic, info_topic, "/metadata/num_markers", *sorted(marker_topics)]

    color_times = []
    color_msgs = []
    marker_streams = defaultdict(lambda: ([], []))
    K = None
    dist = None
    bag_num_markers = None

    for topic, msg, t in bag.read_messages(topics=topics):
        if topic == info_topic and K is None:
            K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
            dist = np.array(msg.D, dtype=np.float64) if msg.D else None
        elif topic == "/metadata/num_markers":
            bag_num_markers = int(msg.data)
        elif topic == color_topic:
            color_times.append(t.to_sec())
            color_msgs.append(msg)
        elif topic in marker_topics:
            marker_streams[topic][0].append(t.to_sec())
            marker_streams[topic][1].append(msg)

    if K is None:
        raise ValueError(f"No CameraInfo on {info_topic}")
    if not color_msgs:
        raise ValueError(f"No images on {color_topic}")
    if not marker_streams:
        raise ValueError(
            "No /marker_N topics in bag. "
            "Use a combined bag from process_marker_c3d.py or process_marker_tsv.py."
        )

    effective_markers = bag_num_markers or len(marker_streams)
    effective_markers = min(effective_markers, num_markers)
    marker_streams = {
        k: v for k, v in sorted(marker_streams.items())
        if int(k.rsplit("_", 1)[-1]) <= effective_markers
    }
    return K, dist, color_times, color_msgs, marker_streams, effective_markers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay Qualisys markers on RealSense color images."
    )
    parser.add_argument("--bag", type=Path, required=True, help="Trial or combined rosbag")
    parser.add_argument(
        "--calib-config",
        type=Path,
        required=True,
        help="calibration_markers.yaml (stereo + T_mc_wrt_mcR for cam_L)",
    )
    parser.add_argument(
        "--camera",
        choices=tuple(CAMERA_NAMES),
        default="L",
        help="RealSense camera to draw on (default: L)",
    )
    parser.add_argument(
        "--num-markers",
        type=int,
        help="Override num_markers from calib YAML (default: YAML, capped by bag)",
    )
    parser.add_argument(
        "--max-time-diff",
        type=float,
        default=0.02,
        help="Max |color_time - marker_time| in seconds (default 0.02)",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=8,
        help="Overlay circle radius in pixels",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Write overlay frames here instead of cv2.imshow playback",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth color frame (default 1)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Stop after this many processed frames",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bag_path = args.bag.expanduser().resolve()
    if not bag_path.is_file():
        raise SystemExit(f"Bag not found: {bag_path}")

    config_path = resolve_config_path(args.calib_config)
    config = load_config(config_path)
    num_markers = args.num_markers or int(config.get("num_markers", 1))
    cam_key = args.camera
    cam_name = CAMERA_NAMES[cam_key]

    transforms, _frame_ids, _colors = build_camera_transforms(config, enabled_keys=[cam_key])
    if cam_key not in transforms:
        raise SystemExit(
            f"No complete calibration for {cam_name} in {config_path}. "
            "Need stereo_R/T and T_mc_wrt_mcR."
        )
    T_mcR_to_cam = transforms[cam_key]

    print(f"Reading bag (single pass): {bag_path}", flush=True)
    with rosbag.Bag(str(bag_path)) as bag:
        K, dist, color_times, color_msgs, marker_streams, num_markers = load_bag_streams(
            bag, cam_name, num_markers
        )

    print(f"Bag:        {bag_path}")
    print(f"Camera:     {cam_name}")
    print(f"Calib:      {config_path}")
    print(f"Markers:    {num_markers} ({', '.join(marker_streams)})")
    print(f"Color fr.:  {len(color_msgs)}")
    print(f"K:\n{K}")
    if dist is not None:
        print(f"dist:       {dist.ravel()[:5]}...")

    save_dir = None
    if args.save_dir:
        save_dir = args.save_dir.expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    shown = 0
    breakpoint()
    for i, (color_time, color_msg) in enumerate(zip(color_times, color_msgs)):
        if i % args.stride != 0:
            continue

        img = imgmsg_to_bgr8(color_msg)
        img_overlay = img.copy()
        matched = 0

        for marker_idx, (topic, (m_times, m_msgs)) in enumerate(
            sorted(marker_streams.items()), start=1
        ):
            marker_msg, marker_time = find_closest_msg(m_times, m_msgs, color_time)
            if abs(color_time - marker_time) > args.max_time_diff:
                continue
            if not marker_is_valid(marker_msg):
                continue

            pt_mcR = np.array(
                [
                    marker_msg.point.x / 1000.0,
                    marker_msg.point.y / 1000.0,
                    marker_msg.point.z / 1000.0,
                    1.0,
                ]
            )
            pt_cam = T_mcR_to_cam @ pt_mcR
            if pt_cam[2] <= 0:
                continue

            pixel = project_point_to_image(pt_cam[:3], K, dist)[0]
            color = MARKER_COLORS_BGR[(marker_idx - 1) % len(MARKER_COLORS_BGR)]
            img_overlay = overlay_point_on_image(
                img_overlay, pixel, color=color, radius=args.radius
            )
            matched += 1

        processed += 1
        if matched == 0:
            continue
        shown += 1

        label = f"t={color_time:.3f}  markers={matched}"
        cv2.putText(
            img_overlay,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if save_dir:
            out_path = save_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(out_path), img_overlay)
        else:
            cv2.imshow(f"Marker overlay ({cam_name})", img_overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        if args.max_frames and shown >= args.max_frames:
            break

    if not save_dir:
        cv2.destroyAllWindows()

    print(f"Processed {processed} color frames, drew markers on {shown}.")
    if save_dir:
        print(f"Saved frames to {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
