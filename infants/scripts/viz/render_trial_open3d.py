#!/usr/bin/env python3
"""Headless Open3D render of trial clouds + Qualisys markers (no X11 / RViz).

Replays a combined bag, unprojects RGB-D into qualisys_mcR, and writes an MP4
with the same slow 180deg back-and-forth orbit used for RViz recordings.

Runs in the background like overlay_realsense_markers.py (no unlocked desktop).
Requires OPEN3D_CPU_RENDERING=1 (set automatically).

Example:
  python infants/scripts/viz/render_trial_open3d.py \\
      --bag data/2026-07-16_11-03-27/trial_002/trial_ros_combined.bag \\
      --cameras L --markers \\
      --calib-config data/calibration_data/26_07_16_infant_019/calibration_markers.yaml \\
      --output /tmp/o3d_L.mp4
"""
from __future__ import annotations

import argparse
import bisect
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import rosbag

# CPU/EGL offscreen before importing Open3D (pip wheel otherwise segfaults).
os.environ.setdefault("OPEN3D_CPU_RENDERING", "1")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import open3d as o3d  # noqa: E402

VIZ_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VIZ_DIR))
from calibration_chain import (  # noqa: E402
    build_T_cam_to_mcR,
    build_camera_transforms,
    load_config,
)
from marker_transformer import _load_marker_timeline  # noqa: E402
from orbit_view import (  # noqa: E402
    look_at_from_markers,
    pick_recording_camera,
    recording_camera_view,
    yaw_eye_around_look_at,
)

CAMERA_ROS = {"L": "cam_L", "M": "cam_M", "R": "cam_R"}


def imgmsg_to_rgb8(msg) -> np.ndarray:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == "bgr8":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif msg.encoding not in ("rgb8", "8UC3"):
        raise ValueError(f"Unsupported color encoding: {msg.encoding}")
    return np.ascontiguousarray(img)


def imgmsg_to_depth_m(msg) -> np.ndarray:
    if msg.encoding not in ("16UC1", "mono16"):
        raise ValueError(f"Unsupported depth encoding: {msg.encoding}")
    depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    return depth.astype(np.float32) / 1000.0


def unproject_rgb_d(color: np.ndarray, depth_m: np.ndarray, K: np.ndarray, step: int):
    step = max(1, int(step))
    color_s = color[::step, ::step]
    depth_s = depth_m[::step, ::step]
    fx, fy = K[0, 0] / step, K[1, 1] / step
    cx, cy = K[0, 2] / step, K[1, 2] / step
    h, w = depth_s.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth_s
    valid = z > 0.05
    if not np.any(valid):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    z_v = z[valid]
    xyz = np.column_stack(
        (
            (uu[valid] - cx) * z_v / fx,
            (vv[valid] - cy) * z_v / fy,
            z_v,
        )
    )
    rgb = color_s[valid].astype(np.float32) / 255.0
    return xyz.astype(np.float32), rgb


def transform_xyz(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    if len(xyz) == 0:
        return xyz
    hom = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    return (T.astype(np.float32) @ hom.T).T[:, :3]


def ping_pong_angle(t_frac: float) -> float:
    """0 → π → 0 over t_frac in [0, 1] (one 180deg out-and-back)."""
    t_frac = min(1.0, max(0.0, float(t_frac)))
    tri = 1.0 - abs(2.0 * t_frac - 1.0)
    return tri * np.pi


def make_grid_lineset(half: float = 1.5, step: float = 0.1, y: float = 0.0):
    """Horizontal grid in Qualisys Y-up (XZ plane)."""
    pts = []
    lines = []
    n = int(round(2 * half / step))
    for i in range(n + 1):
        x = -half + i * step
        pts.append([x, y, -half])
        pts.append([x, y, half])
        lines.append([2 * i, 2 * i + 1])
    offset = 2 * (n + 1)
    for j in range(n + 1):
        z = -half + j * step
        pts.append([-half, y, z])
        pts.append([half, y, z])
        lines.append([offset + 2 * j, offset + 2 * j + 1])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.55, 0.55, 0.57]]), (len(lines), 1))
    )
    return ls


def mux_audio(silent_mp4: Path, bag_path: Path, output_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        silent_mp4.replace(output_path)
        return
    chunks = bytearray()
    with rosbag.Bag(str(bag_path), "r") as bag:
        topics = bag.get_type_and_topic_info().topics
        if "/audio/audio" not in topics:
            silent_mp4.replace(output_path)
            return
        for _, msg, _t in bag.read_messages(topics=["/audio/audio"]):
            if msg.data:
                chunks.extend(msg.data)
    if not chunks:
        silent_mp4.replace(output_path)
        return
    tmp = Path(tempfile.mkstemp(prefix="o3d_audio_", suffix=".mp3")[1])
    tmp.write_bytes(bytes(chunks))
    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(silent_mp4), "-i", str(tmp),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart", str(output_path),
        ]
        subprocess.run(cmd, check=True)
        silent_mp4.unlink(missing_ok=True)
        print(f"[o3d] Muxed audio -> {output_path}")
    except subprocess.CalledProcessError:
        print("[o3d] Audio mux failed; keeping silent video")
        silent_mp4.replace(output_path)
    finally:
        tmp.unlink(missing_ok=True)


def markers_at(times, frames, t_sec: float) -> np.ndarray:
    if not times:
        return np.zeros((0, 3), np.float32)
    i = bisect.bisect_right(times, t_sec) - 1
    if i < 0:
        return np.zeros((0, 3), np.float32)
    pts = [xyz for xyz in frames[i].values() if xyz is not None]
    if not pts:
        return np.zeros((0, 3), np.float32)
    return np.asarray(pts, dtype=np.float32)


def start_ffmpeg(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", f"{fps:.4f}",
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def markers_mesh(xyz: np.ndarray, radius: float):
    """Merged icospheres in Qualisys meters (not screen-space points)."""
    mesh = o3d.geometry.TriangleMesh()
    if len(xyz) == 0 or radius <= 0:
        return mesh
    proto = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius), resolution=5)
    proto.paint_uniform_color([1.0, 0.12, 0.12])
    proto.compute_vertex_normals()
    for p in xyz:
        s = o3d.geometry.TriangleMesh(proto)
        s.translate(p.astype(np.float64))
        mesh += s
    mesh.compute_vertex_normals()
    return mesh


def parse_args():
    p = argparse.ArgumentParser(description="Headless Open3D trial video (clouds + markers).")
    p.add_argument("--bag", required=True, type=Path)
    p.add_argument("--calib-config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--cameras", default="L")
    p.add_argument("--markers", action="store_true", default=True)
    p.add_argument("--no-markers", dest="markers", action="store_false")
    p.add_argument("--max-markers", type=int, default=0)
    p.add_argument("--subsample", type=int, default=3)
    p.add_argument("--frame-stride", type=int, default=2, help="Keep every Nth color frame")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--look-depth", type=float, default=1.1)
    p.add_argument(
        "--dolly",
        type=float,
        default=0.58,
        help="Keep this fraction of camera-to-infant distance (1=exact RS pose)",
    )
    p.add_argument("--vfov", type=float, default=44.0, help="Vertical FOV degrees")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--max-frames", type=int, default=0, help="0 = whole bag (debug cap)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bag_path = args.bag.expanduser().resolve()
    calib_path = args.calib_config.expanduser().resolve()
    out_path = args.output.expanduser().resolve()
    cameras = [c.strip().upper() for c in args.cameras.split(",") if c.strip()]
    cameras = [c for c in ("L", "M", "R") if c in cameras]
    if not cameras:
        raise SystemExit("Need at least one of L,M,R")
    if not bag_path.is_file():
        raise SystemExit(f"Bag not found: {bag_path}")
    if not calib_path.is_file():
        raise SystemExit(f"Calib not found: {calib_path}")

    config = load_config(calib_path)
    transforms, _, _ = build_camera_transforms(config, enabled_keys=cameras)
    T_cam_to_mcR = {
        cam: build_T_cam_to_mcR(transforms[cam])
        for cam in cameras
        if cam in transforms
    }
    if not T_cam_to_mcR:
        raise SystemExit(f"No mcR transforms for cameras {cameras} in {calib_path}")
    missing = [c for c in cameras if c not in T_cam_to_mcR]
    if missing:
        print(f"[o3d] Skipping cameras without calib: {missing}")
        cameras = [c for c in cameras if c in T_cam_to_mcR]

    view_cam = pick_recording_camera(cameras)
    if view_cam not in T_cam_to_mcR:
        view_cam = cameras[0]
    view = recording_camera_view(T_cam_to_mcR[view_cam], args.look_depth)
    eye0 = view["eye"]
    look_at = view["look_at"]
    # YAML sphere_radius is ROS Marker diameter; infant Qualisys balls are ~10mm.
    marker_diameter = min(0.012, max(0.006, float(config.get("sphere_radius", 0.02)) * 0.5))
    marker_radius = 0.5 * marker_diameter

    num_markers = int(config.get("num_markers", 700))
    if args.max_markers > 0:
        num_markers = min(num_markers, args.max_markers)

    marker_times, marker_frames = [], []
    if args.markers:
        print(f"[o3d] Loading marker timeline ({num_markers} ids) from bag...")
        marker_times, marker_frames = _load_marker_timeline(bag_path, num_markers)
        print(f"[o3d] {len(marker_times)} marker frames")
        look_at = look_at_from_markers(marker_frames, look_at)

    # Same viewing direction as the recording camera, slightly closer so the
    # infant fills the frame (RS images often have the baby at the edge).
    dolly = min(1.0, max(0.45, float(args.dolly)))
    eye0 = look_at + dolly * (eye0 - look_at)
    world_up = np.array([0.0, 1.0, 0.0])

    color_topics = {f"/{CAMERA_ROS[c]}/color/image_raw": c for c in cameras}
    depth_topics = {f"/{CAMERA_ROS[c]}/aligned_depth_to_color/image_raw": c for c in cameras}
    info_topics = {f"/{CAMERA_ROS[c]}/color/camera_info": c for c in cameras}
    want = set(color_topics) | set(depth_topics) | set(info_topics)

    with rosbag.Bag(str(bag_path), "r") as bag:
        t0 = bag.get_start_time()
        t1 = bag.get_end_time()
    duration = max(t1 - t0, 1e-3)
    fps = 15.0 if args.frame_stride >= 2 else 30.0
    print(
        f"[o3d] {bag_path.name} cams={','.join(cameras)} view={view_cam} "
        f"{duration:.1f}s stride={args.frame_stride} vfov={args.vfov:.0f} "
        f"marker_d={marker_diameter*100:.1f}cm -> {out_path}"
    )
    print(
        f"[o3d] start eye={np.round(eye0, 3)} look={np.round(look_at, 3)} "
        f"dist={np.linalg.norm(eye0 - look_at):.2f}m",
        flush=True,
    )

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    scene = renderer.scene
    scene.set_background([0.28, 0.28, 0.30, 1.0])
    scene.set_lighting(scene.LightingProfile.NO_SHADOWS, (0.5, 0.8, 0.5))
    grid = make_grid_lineset(y=float(look_at[1]) - 0.25)
    grid_mat = o3d.visualization.rendering.MaterialRecord()
    grid_mat.shader = "unlitLine"
    grid_mat.line_width = 1.0
    scene.add_geometry("grid", grid, grid_mat)

    cloud_mat = o3d.visualization.rendering.MaterialRecord()
    cloud_mat.shader = "defaultUnlit"
    cloud_mat.point_size = 3.0
    marker_mat = o3d.visualization.rendering.MaterialRecord()
    marker_mat.shader = "defaultLit"
    marker_mat.base_color = [1.0, 0.12, 0.12, 1.0]

    silent = out_path.with_name(f"{out_path.stem}_silent{out_path.suffix}")
    ff = start_ffmpeg(silent, args.width, args.height, fps)

    Ks = {}
    latest_depth = {}
    latest_color = {}
    n_written = 0
    color_idx = {c: 0 for c in cameras}
    primary = cameras[0]

    try:
        with rosbag.Bag(str(bag_path), "r") as bag:
            for topic, msg, t in bag.read_messages(topics=list(want)):
                if topic in info_topics and info_topics[topic] not in Ks:
                    Ks[info_topics[topic]] = np.array(msg.K, dtype=np.float64).reshape(3, 3)
                    continue
                if topic in depth_topics:
                    latest_depth[depth_topics[topic]] = (t.to_sec(), msg)
                    continue
                if topic not in color_topics:
                    continue
                cam = color_topics[topic]
                color_idx[cam] += 1
                if color_idx[cam] % max(1, args.frame_stride) != 0:
                    continue
                latest_color[cam] = (t.to_sec(), msg)
                if cam != primary:
                    continue
                if primary not in Ks or primary not in latest_depth:
                    continue

                t_sec = t.to_sec()
                xyz_parts = []
                rgb_parts = []
                for c in cameras:
                    if c not in Ks or c not in latest_color or c not in latest_depth:
                        continue
                    tc, cmsg = latest_color[c]
                    td, dmsg = latest_depth[c]
                    if abs(tc - td) > 0.08:
                        continue
                    try:
                        rgb = imgmsg_to_rgb8(cmsg)
                        depth = imgmsg_to_depth_m(dmsg)
                    except Exception:
                        continue
                    xyz, cols = unproject_rgb_d(rgb, depth, Ks[c], args.subsample)
                    xyz_parts.append(transform_xyz(xyz, T_cam_to_mcR[c]))
                    rgb_parts.append(cols)
                if not xyz_parts:
                    continue
                xyz_w = np.concatenate(xyz_parts, axis=0)
                rgb_w = np.concatenate(rgb_parts, axis=0)
                if len(xyz_w) > 220_000:
                    keep = np.linspace(0, len(xyz_w) - 1, 220_000).astype(np.int32)
                    xyz_w, rgb_w = xyz_w[keep], rgb_w[keep]

                t_frac = (t_sec - t0) / duration
                ang = ping_pong_angle(t_frac)
                eye = yaw_eye_around_look_at(eye0, look_at, ang)
                renderer.setup_camera(float(args.vfov), look_at, eye, world_up)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_w.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(rgb_w.astype(np.float64))
                if scene.has_geometry("cloud"):
                    scene.remove_geometry("cloud")
                scene.add_geometry("cloud", pcd, cloud_mat)

                n_mark = 0
                if args.markers:
                    mxyz = markers_at(marker_times, marker_frames, t_sec)
                    n_mark = len(mxyz)
                    if scene.has_geometry("markers"):
                        scene.remove_geometry("markers")
                    if n_mark:
                        scene.add_geometry(
                            "markers", markers_mesh(mxyz, marker_radius), marker_mat
                        )

                img = np.asarray(renderer.render_to_image())
                if img.shape[1] != args.width or img.shape[0] != args.height:
                    img = cv2.resize(img, (args.width, args.height))
                ff.stdin.write(np.ascontiguousarray(img).tobytes())
                n_written += 1
                if n_written == 1 or n_written % 30 == 0:
                    print(
                        f"[o3d] frame {n_written} t={t_sec - t0:.1f}s "
                        f"pts={len(xyz_w)} markers={n_mark}",
                        flush=True,
                    )
                if args.max_frames and n_written >= args.max_frames:
                    break
    except KeyboardInterrupt:
        print("\n[o3d] Interrupted", flush=True)
        try:
            ff.stdin.close()
        except Exception:
            pass
        ff.kill()
        silent.unlink(missing_ok=True)
        return 130
    finally:
        if ff.poll() is None and ff.stdin is not None:
            try:
                ff.stdin.close()
            except Exception:
                pass
        err = b""
        if ff.stderr is not None:
            err = ff.stderr.read() or b""
        code = ff.wait(timeout=30) if ff.poll() is None else ff.returncode
        if code not in (0, None) and n_written == 0:
            print(err.decode("utf-8", errors="ignore")[-800:])
            raise SystemExit(f"ffmpeg failed ({code})")

    if n_written == 0:
        silent.unlink(missing_ok=True)
        raise SystemExit("No frames rendered (missing depth/color sync?)")
    print(f"[o3d] Wrote {n_written} frames to {silent}")
    if args.no_audio:
        silent.replace(out_path)
    else:
        mux_audio(silent, bag_path, out_path)
    print(f"[o3d] Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
