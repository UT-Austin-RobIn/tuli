#!/usr/bin/env python3
"""render_marker_video.py

Read a camera rosbag + marker rosbag, project the mocap marker onto each
camera's RGB image using the full calibration chain, and write a video.

For each camera frame, finds the nearest-in-time marker message,
transforms to camera frame, projects to pixel coordinates, and draws a circle.

Usage:
    python examples/render_marker_video.py \
        --camera-bag data/trial_ros.bag \
        --marker-bag data/tsv_20260415.bag \
        --config examples/fullchain_config_example.yaml \
        --camera cam_L \
        --output output_video.mp4
"""
import argparse
import json
import yaml
import numpy as np
import cv2
import rosbag
from cv_bridge import CvBridge
from bisect import bisect_left


AXIS_FLIP = np.eye(4)
AXIS_FLIP[1, 1] = -1
AXIS_FLIP[2, 2] = -1


def load_intrinsics(calib_json_path, side):
    """Return (K, dist) from a stereo calibration_results.json.

    side: "left" or "right" picking left_/right_camera_calibration_data.
    dist returned as float64 1D array (rational model: 14 coeffs).
    """
    with open(calib_json_path) as f:
        d = json.load(f)
    key = f"{side}_camera_calibration_data"
    cam = d[key]
    K = np.array(cam["camera_matrix"], dtype=np.float64)
    dist = np.array(cam["dist_coeffs"], dtype=np.float64).ravel()
    return K, dist


def build_4x4(R, T):
    M = np.eye(4)
    M[:3, :3] = np.array(R, dtype=np.float64)
    M[:3, 3] = np.array(T, dtype=np.float64).ravel()
    return M


def build_transform_chain(cam_cfg):
    T_mc_wrt_mcR = np.array(cam_cfg["T_mc_wrt_mcR"], dtype=np.float64)
    stereo_RT = build_4x4(cam_cfg["stereo_R"], cam_cfg["stereo_T"])
    return stereo_RT @ AXIS_FLIP @ np.linalg.inv(T_mc_wrt_mcR)


def build_chained_transform(cam_cfg, parent_transform):
    stereo_RT = build_4x4(cam_cfg["stereo_R"], cam_cfg["stereo_T"])
    return stereo_RT @ parent_transform


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_markers(bag_path, marker_topic="/marker_1"):
    """Load all marker messages as (timestamp_sec, xyz_meters) pairs."""
    markers = []
    bag = rosbag.Bag(bag_path)
    for topic, msg, t in bag.read_messages(topics=[marker_topic]):
        ts = msg.header.stamp.to_sec()
        x = msg.point.x / 1000.0  # mm -> m
        y = msg.point.y / 1000.0
        z = msg.point.z / 1000.0
        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
            markers.append((ts, np.array([x, y, z, 1.0])))
    bag.close()
    print(f"Loaded {len(markers)} marker messages from {marker_topic}")
    return markers


def find_nearest_marker(markers, timestamps, query_time):
    """Find nearest marker by timestamp."""
    idx = bisect_left(timestamps, query_time)
    if idx == 0:
        return markers[0]
    if idx >= len(markers):
        return markers[-1]
    before = markers[idx - 1]
    after = markers[idx]
    if (query_time - before[0]) <= (after[0] - query_time):
        return before
    return after


def project_to_pixel(pt_cam, K, dist=None):
    """Project 3D point (already in camera frame) to distorted pixel coords.

    Uses cv2.projectPoints with rvec=tvec=0 so pt_cam is treated as the
    camera-frame 3D position. Applies full dist model (rational k1..k6, p1, p2).
    """
    if pt_cam[2] <= 0:
        return None
    obj = np.asarray(pt_cam[:3], dtype=np.float64).reshape(1, 1, 3)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    dc = None if dist is None else np.asarray(dist, dtype=np.float64).ravel()
    uv, _ = cv2.projectPoints(obj, rvec, tvec, K, dc)
    u, v = uv.ravel()
    return int(round(u)), int(round(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-bag", required=True)
    ap.add_argument("--marker-bag", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--camera", default="cam_L",
                    help="Which camera to render (cam_L, cam_M, cam_R)")
    ap.add_argument("--marker-topic", default="/marker_1")
    ap.add_argument("--output", default=None,
                    help="Output video path (default: marker_overlay_<camera>.mp4)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Stop after N frames (for quick testing)")
    ap.add_argument("--time-offset", type=float, default=0.0,
                    help="Seconds added to camera stamp before nearest-marker "
                         "lookup. Positive = sample mocap AFTER cam frame "
                         "(use if predicted dot lags moving marker). "
                         "Negative = sample mocap BEFORE cam frame (use if "
                         "predicted dot leads moving marker). Sweep to find "
                         "the value that kills drift, e.g. -0.3 .. +0.3 in "
                         "50 ms steps.")
    args = ap.parse_args()

    config = load_config(args.config)
    cam_name = args.camera

    # Build transform for selected camera
    transforms = {}
    for cname, cam_cfg in config["cameras"].items():
        if "parent" in cam_cfg:
            transforms[cname] = build_chained_transform(cam_cfg, transforms[cam_cfg["parent"]])
        else:
            transforms[cname] = build_transform_chain(cam_cfg)

    if cam_name not in transforms:
        print(f"Camera '{cam_name}' not in config. Available: {list(transforms.keys())}")
        return

    T_mcR_to_cam = transforms[cam_name]
    print(f"Transform mcR -> {cam_name}:\n{T_mcR_to_cam}")

    # Load intrinsics from the stereo calibration json (NOT from camera_info).
    # stereo_R/T were solved jointly with this K+dist; using driver/factory K
    # introduces a constant pixel offset that looks like drift.
    cam_cfg = config["cameras"][cam_name]
    if "calib_json" not in cam_cfg:
        raise KeyError(
            f"'{cam_name}' is missing 'calib_json' in config. Add the path to "
            "the calibration_results.json that produced its stereo_R/stereo_T."
        )
    side = cam_cfg.get("calib_side", "right")
    K, dist = load_intrinsics(cam_cfg["calib_json"], side)
    print(f"Loaded {cam_name} K from {cam_cfg['calib_json']} ({side}):\n{K}")
    print(f"dist: {dist}")

    # Load markers
    markers = load_markers(args.marker_bag, args.marker_topic)
    marker_timestamps = [m[0] for m in markers]

    # Read camera images from bag
    color_topic = f"/{cam_name}/color/image_raw"

    bridge = CvBridge()
    bag = rosbag.Bag(args.camera_bag)

    # Count frames for progress
    frame_count = 0
    for topic, msg, t in bag.read_messages(topics=[color_topic]):
        frame_count += 1
    bag.close()
    if args.max_frames:
        frame_count = min(frame_count, args.max_frames)
    print(f"Total frames to process: {frame_count}")

    # Process frames
    output_path = args.output or f"marker_overlay_{cam_name}.mp4"
    writer = None
    bag = rosbag.Bag(args.camera_bag)
    processed = 0

    for topic, msg, t in bag.read_messages(topics=[color_topic]):
        if args.max_frames and processed >= args.max_frames:
            break

        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cam_time = msg.header.stamp.to_sec()
        query_time = cam_time + args.time_offset

        # Find nearest marker
        marker_time, pt_mcR = find_nearest_marker(markers, marker_timestamps, query_time)
        gap_ms = (marker_time - cam_time) * 1000.0

        # Transform to camera frame
        pt_cam = T_mcR_to_cam @ pt_mcR

        # Project to pixel (with full distortion model)
        px = project_to_pixel(pt_cam[:3], K, dist)
        if px is not None:
            u, v = px
            h, w = frame.shape[:2]
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(frame, (u, v), 12, (0, 0, 255), -1)  # red filled
                cv2.circle(frame, (u, v), 14, (255, 255, 255), 2)  # white outline
            else:
                # Draw at edge with arrow indicating direction
                cv2.putText(frame, f"MARKER OOB ({u},{v})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Timestamp overlay: cam time, chosen marker time gap, applied offset
        cv2.putText(frame,
                    f"cam={cam_time:.3f}  gap(m-c)={gap_ms:+.1f}ms  "
                    f"offset={args.time_offset*1000:+.0f}ms  frame={processed}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, args.fps, (w, h))

        writer.write(frame)
        processed += 1
        if processed % 100 == 0:
            print(f"  {processed}/{frame_count}")

    bag.close()
    if writer:
        writer.release()
    print(f"\nVideo saved to {output_path} ({processed} frames)")


if __name__ == "__main__":
    main()
