#!/usr/bin/env python3
"""debug_offset_sweep.py

Extract ONE camera frame from a combined rosbag and overlay the mocap
marker projected at many different --time-offset values. Produces a
single mosaic PNG where you can visually pick the offset that lands on
the real marker/hand.

Usage:
    /usr/bin/python3 examples/debug_offset_sweep.py \\
        --bag /home/robotlearning2/infants/data/combined_fixed.bag \\
        --config examples/fullchain_config_example.yaml \\
        --camera cam_L \\
        --frame 900 \\
        --offsets -0.5,-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3,0.5 \\
        --output offset_sweep_cam_L_f900.png
"""
import argparse
import json
from bisect import bisect_left

import numpy as np
import yaml
import cv2
import rosbag
from cv_bridge import CvBridge


AXIS_FLIP = np.eye(4)
AXIS_FLIP[1, 1] = -1
AXIS_FLIP[2, 2] = -1


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


def load_intrinsics(calib_json_path, side):
    with open(calib_json_path) as f:
        d = json.load(f)
    cam = d[f"{side}_camera_calibration_data"]
    K = np.array(cam["camera_matrix"], dtype=np.float64)
    dist = np.array(cam["dist_coeffs"], dtype=np.float64).ravel()
    return K, dist


def find_nearest(sorted_list, times, q):
    idx = bisect_left(times, q)
    if idx == 0:
        return sorted_list[0]
    if idx >= len(sorted_list):
        return sorted_list[-1]
    b = sorted_list[idx - 1]
    a = sorted_list[idx]
    return b if (q - b[0]) <= (a[0] - q) else a


def project(pt_cam, K, dist):
    if pt_cam[2] <= 0:
        return None
    obj = np.asarray(pt_cam[:3], dtype=np.float64).reshape(1, 1, 3)
    uv, _ = cv2.projectPoints(obj,
                              np.zeros((3, 1)), np.zeros((3, 1)),
                              K, np.asarray(dist).ravel())
    u, v = uv.ravel()
    return int(round(u)), int(round(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--camera", default="cam_L")
    ap.add_argument("--frame", type=int, default=None,
                    help="Which camera frame index to use (default: middle frame)")
    ap.add_argument("--marker-topic", default="/marker_1")
    ap.add_argument("--offsets",
                    default="-0.5,-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3,0.5",
                    help="Comma-separated seconds to sweep.")
    ap.add_argument("--output", default=None)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="Downscale each cell by this factor (mosaic size).")
    args = ap.parse_args()

    offsets = [float(x) for x in args.offsets.split(",")]
    print(f"Sweeping {len(offsets)} offsets: {offsets}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Build chain
    transforms = {}
    for cname, cam_cfg in config["cameras"].items():
        if "parent" in cam_cfg:
            transforms[cname] = build_chained_transform(cam_cfg, transforms[cam_cfg["parent"]])
        else:
            transforms[cname] = build_transform_chain(cam_cfg)
    T = transforms[args.camera]

    cam_cfg = config["cameras"][args.camera]
    K, dist = load_intrinsics(cam_cfg["calib_json"],
                              cam_cfg.get("calib_side", "right"))

    # Load markers
    markers = []
    with rosbag.Bag(args.bag) as bag:
        for _, msg, _ in bag.read_messages(topics=[args.marker_topic]):
            ts = msg.header.stamp.to_sec()
            x, y, z = msg.point.x / 1000.0, msg.point.y / 1000.0, msg.point.z / 1000.0
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                markers.append((ts, np.array([x, y, z, 1.0])))
    marker_ts = [m[0] for m in markers]

    # Load one camera frame
    color_topic = f"/{args.camera}/color/image_raw"
    frames = []
    with rosbag.Bag(args.bag) as bag:
        total = 0
        for _, msg, _ in bag.read_messages(topics=[color_topic]):
            total += 1
        target = args.frame if args.frame is not None else total // 2
        print(f"Total frames: {total}. Using frame {target}.")

        idx = 0
        bridge = CvBridge()
        for _, msg, _ in bag.read_messages(topics=[color_topic]):
            if idx == target:
                frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                cam_time = msg.header.stamp.to_sec()
                break
            idx += 1

    print(f"cam_time = {cam_time:.6f}")

    # Render each offset
    cells = []
    for off in offsets:
        q = cam_time + off
        m_ts, pt_mcR = find_nearest(markers, marker_ts, q)
        pt_cam = T @ pt_mcR
        px = project(pt_cam, K, dist)

        img = frame.copy()
        label_color = (0, 255, 0)  # green text
        if px is not None:
            u, v = px
            h, w = img.shape[:2]
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(img, (u, v), 14, (0, 0, 255), -1)
                cv2.circle(img, (u, v), 16, (255, 255, 255), 2)
            else:
                cv2.putText(img, f"OOB ({u},{v})", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                label_color = (0, 0, 255)

        gap_ms = (m_ts - cam_time) * 1000
        cv2.putText(img, f"offset={off*1000:+.0f} ms",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2)
        cv2.putText(img, f"picked marker gap={gap_ms:+.1f} ms",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

        if args.scale != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * args.scale), int(h * args.scale)))
        cells.append(img)

    # Mosaic: grid rows × cols ~ sqrt(N)
    n = len(cells)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    ch, cw = cells[0].shape[:2]
    mosaic = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)
    for i, c in enumerate(cells):
        r, k = divmod(i, cols)
        mosaic[r * ch:(r + 1) * ch, k * cw:(k + 1) * cw] = c

    output = args.output or f"offset_sweep_{args.camera}.png"
    cv2.imwrite(output, mosaic)
    print(f"Saved {output}  ({rows}x{cols} grid, cell {cw}x{ch})")


if __name__ == "__main__":
    main()
