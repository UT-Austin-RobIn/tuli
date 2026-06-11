#!/usr/bin/env python3
"""render_pointcloud_video.py

For a selected camera in a combined rosbag (RS + mocap), render per-frame:
  - Open3D RGBD point cloud (color + aligned_depth, built with the bag's
    camera_info K so the cloud's *shape* matches what the driver aligned).
  - Red sphere at the transformed mocap marker position in the camera's
    frame using the full extrinsic chain from fullchain_config.

Output: an mp4 of the scene from a fixed viewpoint. Since the marker is
placed via extrinsics only (no K, no distortion), its 3D alignment with
the cloud tells you whether the calibration chain is right, independent
of the projection/intrinsics path that render_marker_video.py uses.

Usage:
    /usr/bin/python3 examples/render_pointcloud_video.py \\
        --bag /home/robotlearning2/infants/data/combined_fixed.bag \\
        --config examples/fullchain_config_example.yaml \\
        --camera cam_L \\
        --output cam_L_cloud.mp4
"""
import argparse
import os
import sys
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


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_markers(bag_path, marker_topic):
    markers = []
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, t in bag.read_messages(topics=[marker_topic]):
            ts = msg.header.stamp.to_sec()
            x = msg.point.x / 1000.0
            y = msg.point.y / 1000.0
            z = msg.point.z / 1000.0
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                markers.append((ts, np.array([x, y, z, 1.0])))
    print(f"Loaded {len(markers)} markers from {marker_topic}")
    return markers


def find_nearest(sorted_list, timestamps, query_time):
    idx = bisect_left(timestamps, query_time)
    if idx == 0:
        return sorted_list[0]
    if idx >= len(sorted_list):
        return sorted_list[-1]
    before = sorted_list[idx - 1]
    after = sorted_list[idx]
    if (query_time - before[0]) <= (after[0] - query_time):
        return before
    return after


def index_messages(bag_path, topics):
    """Return {topic: [(ts, msg), ...] sorted by ts}."""
    out = {t: [] for t in topics}
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, t in bag.read_messages(topics=topics):
            ts = msg.header.stamp.to_sec() if hasattr(msg, "header") else t.to_sec()
            out[topic].append((ts, msg))
    for t in topics:
        out[t].sort(key=lambda x: x[0])
        print(f"  {t}: {len(out[t])} messages")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--camera", default="cam_L")
    ap.add_argument("--marker-topic", default="/marker_1")
    ap.add_argument("--output", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--time-offset", type=float, default=0.0,
                    help="Seconds added to cam stamp before nearest-marker lookup.")
    ap.add_argument("--sphere-radius", type=float, default=0.03)
    ap.add_argument("--depth-scale", type=float, default=1000.0,
                    help="Divide uint16 depth by this to get meters.")
    ap.add_argument("--max-depth", type=float, default=4.0,
                    help="Crop points farther than this (meters).")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="Voxel downsample size in meters (0 = off).")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--view-json", default=None,
                    help="Open3D pinhole camera trajectory json. If omitted, "
                         "a default head-on view is used.")
    ap.add_argument("--headless", action="store_true",
                    help="Try Open3D rendering.OffscreenRenderer (no display).")
    args = ap.parse_args()

    # Import open3d late so --help works without it.
    import open3d as o3d

    config = load_config(args.config)
    cam_name = args.camera

    # Build chain to selected camera
    transforms = {}
    for cname, cam_cfg in config["cameras"].items():
        if "parent" in cam_cfg:
            transforms[cname] = build_chained_transform(cam_cfg, transforms[cam_cfg["parent"]])
        else:
            transforms[cname] = build_transform_chain(cam_cfg)
    if cam_name not in transforms:
        print(f"Camera '{cam_name}' not in config.")
        sys.exit(1)
    T_mcR_to_cam = transforms[cam_name]
    print(f"T_mcR->{cam_name}:\n{T_mcR_to_cam}")

    # Topics
    color_topic = f"/{cam_name}/color/image_raw"
    depth_topic = f"/{cam_name}/aligned_depth_to_color/image_raw"
    info_topic = f"/{cam_name}/color/camera_info"

    # Index depth and info by timestamp (color drives the loop)
    print("Indexing messages...")
    idx = index_messages(args.bag, [depth_topic, info_topic])
    depth_times = [ts for ts, _ in idx[depth_topic]]
    if not idx[info_topic]:
        print(f"No camera_info on {info_topic}")
        sys.exit(1)
    info_msg = idx[info_topic][0][1]  # first one is enough
    fx = info_msg.K[0]; fy = info_msg.K[4]
    cx = info_msg.K[2]; cy = info_msg.K[5]
    w_img = info_msg.width; h_img = info_msg.height
    print(f"  K: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} {w_img}x{h_img}")
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w_img, h_img, fx, fy, cx, cy)

    # Markers
    markers = load_markers(args.bag, args.marker_topic)
    marker_ts = [m[0] for m in markers]

    # Output writer
    output_path = args.output or f"pointcloud_{cam_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, args.fps, (args.width, args.height))

    # Visualizer setup
    vis = o3d.visualization.Visualizer()
    # visible=False still opens a hidden window on most Linux Open3D builds;
    # it'll fail on truly headless boxes. Use --headless for OffscreenRenderer.
    created = vis.create_window(visible=not args.headless,
                                width=args.width, height=args.height)
    if not created:
        print("Failed to create Open3D window. Try --headless or ensure a display.")
        sys.exit(1)

    pcd = o3d.geometry.PointCloud()
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=args.sphere_radius)
    sphere.paint_uniform_color([1, 0, 0])
    sphere.compute_vertex_normals()
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)

    vis.add_geometry(pcd)
    vis.add_geometry(sphere)
    vis.add_geometry(coord)

    ctr = vis.get_view_control()
    if args.view_json and os.path.exists(args.view_json):
        params = o3d.io.read_pinhole_camera_parameters(args.view_json)
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    else:
        # default: look at ~2 m in front of camera origin
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0])
        ctr.set_lookat([0, 0, 1.5])
        ctr.set_zoom(0.6)

    bridge = CvBridge()
    processed = 0

    # Iterate color frames from the bag (streamed, not pre-indexed, to save RAM)
    with rosbag.Bag(args.bag) as bag:
        for topic, msg, t in bag.read_messages(topics=[color_topic]):
            if args.max_frames and processed >= args.max_frames:
                break

            color_ts = msg.header.stamp.to_sec()
            color_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Nearest-in-time depth (should be ~ same stamp for aligned_depth)
            if not depth_times:
                continue
            d_ts, d_msg = find_nearest(idx[depth_topic], depth_times, color_ts)
            depth = bridge.imgmsg_to_cv2(d_msg, desired_encoding="passthrough")
            if depth.dtype == np.uint16:
                depth_m = depth.astype(np.float32) / args.depth_scale
            else:
                depth_m = depth.astype(np.float32)
            # crop far range
            depth_m[depth_m > args.max_depth] = 0.0
            depth_m[depth_m <= 0.0] = 0.0

            # Build RGBD -> PointCloud
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            color_o3d = o3d.geometry.Image(color_rgb.copy())
            depth_o3d = o3d.geometry.Image(depth_m)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1.0, depth_trunc=args.max_depth,
                convert_rgb_to_intensity=False)
            new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            if args.voxel > 0:
                new_pcd = new_pcd.voxel_down_sample(args.voxel)

            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors

            # Marker -> camera frame
            q_time = color_ts + args.time_offset
            _, pt_mcR = find_nearest(markers, marker_ts, q_time)
            pt_cam = T_mcR_to_cam @ pt_mcR
            # Move the sphere to the new position: recreate instead of
            # incremental translate (avoids cumulative drift on the mesh).
            sphere_new = o3d.geometry.TriangleMesh.create_sphere(radius=args.sphere_radius)
            sphere_new.translate(pt_cam[:3])
            sphere_new.paint_uniform_color([1, 0, 0])
            sphere_new.compute_vertex_normals()
            sphere.vertices = sphere_new.vertices
            sphere.triangles = sphere_new.triangles
            sphere.vertex_colors = sphere_new.vertex_colors
            sphere.vertex_normals = sphere_new.vertex_normals

            vis.update_geometry(pcd)
            vis.update_geometry(sphere)
            vis.poll_events()
            vis.update_renderer()

            buf = vis.capture_screen_float_buffer(do_render=True)
            img = (np.asarray(buf) * 255.0).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if img_bgr.shape[1] != args.width or img_bgr.shape[0] != args.height:
                img_bgr = cv2.resize(img_bgr, (args.width, args.height))

            # hud
            cv2.putText(img_bgr,
                        f"cam={color_ts:.3f}  offset={args.time_offset*1000:+.0f}ms  "
                        f"frame={processed}  pt_cam=({pt_cam[0]:+.2f},{pt_cam[1]:+.2f},{pt_cam[2]:+.2f})",
                        (10, args.height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            writer.write(img_bgr)
            processed += 1
            if processed % 50 == 0:
                print(f"  {processed} frames")

    vis.destroy_window()
    writer.release()
    print(f"\nSaved {output_path} ({processed} frames)")


if __name__ == "__main__":
    main()
