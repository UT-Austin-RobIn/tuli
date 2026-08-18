#!/usr/bin/env python3
"""Overlay Qualisys markers onto Miqus AVI videos and mux bag audio.

Projection matches visualize_data_on_qualisys_image.py:
  p_mcR (m) -> inv(T_mc_wrt_mcR) -> AXIS_FLIP -> project with Miqus K

Camera pose + intrinsics come from the session mocap calibration XML
(matched by serial in the AVI name, e.g. Miqus_1_31039 -> serial 31039).
Marker trajectories come from a combined trial bag (/marker_N).
Video t=0 is the Qualisys TSV TIME_STAMP (same as mux_miqus_with_bag_audio).

Examples:
  python infants/scripts/viz/overlay_miqus_markers.py \\
      --trial-dir data/2026-07-16_11-03-27/trial_001 \\
      --calibration-dir data/calibration_data/26_07_16_infant_019
"""
from __future__ import annotations

import argparse
import bisect
import csv
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytz
import rosbag

_VIZ_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _VIZ_DIR.parent
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from viz_layout import vis_dir
from export_calibration_config import DATA_ROOT as CALIB_DATA_ROOT, find_mocap_calibration

from calibration_chain import AXIS_FLIP

DEFAULT_AUDIO_TOPIC = "/audio/audio"
DEFAULT_TZ = "America/Chicago"
MIQUS_GLOB = "*Miqus*.avi"
MIQUS_RE = re.compile(r"Miqus_(\d+)_(\d+)", re.I)

MARKER_COLORS_BGR = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
]


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("[ERROR] ffmpeg not found. Install with: sudo apt install ffmpeg")


def parse_tsv_start(tsv_path: Path, tz_name: str = DEFAULT_TZ) -> float:
    with open(tsv_path, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    if len(rows) < 8:
        raise ValueError(f"TSV header too short: {tsv_path}")
    local_tz = pytz.timezone(tz_name)
    dt = datetime.strptime(rows[7][1], "%Y-%m-%d, %H:%M:%S.%f")
    return local_tz.localize(dt).timestamp()


def find_trial_tsv(trial_dir: Path) -> Optional[Path]:
    cands = sorted(trial_dir.glob("*.tsv"))
    preferred = [p for p in cands if "qualisys" not in p.name.lower()]
    pool = preferred or cands
    return pool[0] if pool else None


def find_combined_bag(trial_dir: Path) -> Path:
    for name in ("trial_ros_combined.bag", "trial_ros.bag"):
        p = trial_dir / name
        if p.is_file():
            return p
    bags = sorted(trial_dir.glob("*.bag"))
    if not bags:
        raise FileNotFoundError(f"No .bag in {trial_dir}")
    return bags[0]


def find_audio_bag(trial_dir: Path) -> Path:
    for name in ("trial_ros.bag", "trial_ros_combined.bag"):
        p = trial_dir / name
        if p.is_file():
            return p
    return find_combined_bag(trial_dir)


def serial_from_avi(path: Path) -> Tuple[str, str]:
    m = MIQUS_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse Miqus serial from {path.name}")
    return m.group(1), m.group(2)


def mocap_camera_from_serial(xml_path: Path, serial: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return (T_mc_wrt_mcR 4x4 meters, K 3x3 for full FoV video, dist or None)."""
    root = ET.parse(str(xml_path)).getroot()
    for cam in root.iter("camera"):
        if cam.get("serial") != serial:
            continue
        tf = cam.find("transform")
        intrinsic = cam.find("intrinsic")
        if tf is None or intrinsic is None:
            raise ValueError(f"Camera {serial}: missing transform/intrinsic in {xml_path}")

        T = np.array(
            [
                [
                    float(tf.get("r11")),
                    float(tf.get("r21")),
                    float(tf.get("r31")),
                    float(tf.get("x")) / 1000.0,
                ],
                [
                    float(tf.get("r12")),
                    float(tf.get("r22")),
                    float(tf.get("r32")),
                    float(tf.get("y")) / 1000.0,
                ],
                [
                    float(tf.get("r13")),
                    float(tf.get("r23")),
                    float(tf.get("r33")),
                    float(tf.get("z")) / 1000.0,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Same pixel scaling as visualize_data_on_qualisys_image.py / test_qualisys_calibration.py
        sensor_max_u = float(intrinsic.get("sensorMaxU"))
        sensor_max_v = float(intrinsic.get("sensorMaxV"))
        fov = cam.find("fov_video")
        if fov is not None:
            u0 = float(fov.get("left"))
            v0 = float(fov.get("top"))
            u1 = float(fov.get("right"))
            v1 = float(fov.get("bottom"))
        else:
            u0, v0, u1, v1 = 0.0, 0.0, 1919.0, 1087.0
        width_m1 = u1 - u0
        height_m1 = v1 - v0
        scale_u = width_m1 / sensor_max_u
        scale_v = height_m1 / sensor_max_v
        fx = float(intrinsic.get("focalLengthU")) * scale_u
        fy = float(intrinsic.get("focalLengthV")) * scale_v
        cx = float(intrinsic.get("centerPointU")) * scale_u - u0
        cy = float(intrinsic.get("centerPointV")) * scale_v - v0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        dist = np.array(
            [
                float(intrinsic.get("radialDistortion1") or 0.0),
                float(intrinsic.get("radialDistortion2") or 0.0),
                float(intrinsic.get("tangentalDistortion1") or 0.0),
                float(intrinsic.get("tangentalDistortion2") or 0.0),
                float(intrinsic.get("radialDistortion3") or 0.0),
            ],
            dtype=np.float64,
        )
        return T, K, dist

    raise KeyError(f"Serial {serial} not found in {xml_path}")


def project_point_to_image(point_3d, K, dist_coeffs=None):
    points_3d = np.asarray(point_3d, dtype=np.float32).reshape(-1, 1, 3)
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    dist = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float32)
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist)
    return points_2d.reshape(-1, 2)


def overlay_point_on_image(image, pixel, color=(0, 0, 255), radius=10, thickness=-1):
    img_out = image
    u, v = int(round(pixel[0])), int(round(pixel[1]))
    h, w = img_out.shape[:2]
    if 0 <= u < w and 0 <= v < h:
        cv2.circle(img_out, (u, v), radius, color, thickness)
    return img_out


def marker_is_valid(msg) -> bool:
    vals = (msg.point.x, msg.point.y, msg.point.z)
    if any(np.isnan(v) for v in vals):
        return False
    if vals == (0.0, 0.0, 0.0):
        return False
    return True


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


def load_marker_streams(bag_path: Path, num_markers: int):
    topics = {f"/marker_{m}" for m in range(1, num_markers + 1)}
    topics.add("/metadata/num_markers")
    streams = defaultdict(lambda: ([], []))
    bag_n = None
    with rosbag.Bag(str(bag_path)) as bag:
        for topic, msg, t in bag.read_messages(topics=sorted(topics)):
            if topic == "/metadata/num_markers":
                bag_n = int(msg.data)
                continue
            streams[topic][0].append(t.to_sec())
            streams[topic][1].append(msg)
    if not streams:
        raise SystemExit(
            f"No /marker_N topics in {bag_path}. "
            "Run process_marker_c3d.py to build trial_ros_combined.bag first."
        )
    effective = bag_n or max(int(k.rsplit("_", 1)[-1]) for k in streams)
    effective = min(effective, num_markers)
    return {
        k: v
        for k, v in sorted(streams.items())
        if int(k.rsplit("_", 1)[-1]) <= effective
    }, effective


def load_audio_mp3(bag_path: Path, topic: str = DEFAULT_AUDIO_TOPIC):
    chunks = bytearray()
    first_t = None
    with rosbag.Bag(str(bag_path), "r") as bag:
        if topic not in bag.get_type_and_topic_info().topics:
            return None, None
        for _, msg, t in bag.read_messages(topics=[topic]):
            if not msg.data:
                continue
            if first_t is None:
                first_t = t.to_sec()
            chunks.extend(msg.data)
    if not chunks:
        return None, None
    return bytes(chunks), first_t


def mux_mp4(video_path: Path, audio_path: Optional[Path], output_path: Path, audio_offset_sec: float):
    require_ffmpeg()
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path)]
    if audio_path is not None:
        if audio_offset_sec > 0:
            cmd += ["-itsoffset", f"{audio_offset_sec:.6f}", "-i", str(audio_path)]
        elif audio_offset_sec < 0:
            cmd += ["-ss", f"{-audio_offset_sec:.6f}", "-i", str(audio_path)]
        else:
            cmd += ["-i", str(audio_path)]
        cmd += [
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            str(output_path),
        ]
    else:
        cmd += [
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-an", "-movflags", "+faststart",
            str(output_path),
        ]
    subprocess.run(cmd, check=True)


def project_marker_to_miqus(msg, T_mc_wrt_mcR: np.ndarray, K: np.ndarray, dist) -> Optional[np.ndarray]:
    if not marker_is_valid(msg):
        return None
    # mm -> meters (homogeneous w=1)
    p_mcR = np.array(
        [msg.point.x / 1000.0, msg.point.y / 1000.0, msg.point.z / 1000.0, 1.0],
        dtype=np.float64,
    )
    p_mc = np.linalg.inv(T_mc_wrt_mcR) @ p_mcR
    p_img = AXIS_FLIP[:3, :3] @ p_mc[:3]
    if p_img[2] <= 1e-6:
        return None
    uv = project_point_to_image(p_img, K, dist)
    return uv[0]


def overlay_one_avi(
    avi_path: Path,
    marker_streams: Dict[str, Tuple[List[float], list]],
    T: np.ndarray,
    K: np.ndarray,
    dist,
    video_t0: float,
    max_time_diff: float,
    radius: int,
    output_path: Path,
    audio_bytes: Optional[bytes],
    audio_t0: Optional[float],
    audio_offset_extra: float,
    max_frames: Optional[int] = None,
) -> None:
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open AVI: {avi_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale K if AVI resolution differs from FoV used to build K (usually 1920x1088)
    # K was built for (u1-u0+1) style via (right-left); fov uses inclusive right/bottom.
    # If capture size differs, scale proportionally.
    # Default FoV for Miqus Video is 1920x1088.
    k_w, k_h = 1920.0, 1088.0
    if abs(width - k_w) > 1 or abs(height - k_h) > 1:
        sx, sy = width / k_w, height / k_h
        K = K.copy()
        K[0, :] *= sx
        K[1, :] *= sy

    tmp_avi = Path(tempfile.mkstemp(prefix="miqus_overlay_", suffix=".avi")[1])
    writer = cv2.VideoWriter(
        str(tmp_avi),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise SystemExit(f"Cannot open VideoWriter for {tmp_avi}")

    frame_i = 0
    drawn = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_i >= max_frames:
            break
        t_wall = video_t0 + frame_i / float(fps)
        for mi, (topic, (times, msgs)) in enumerate(marker_streams.items()):
            if not times:
                continue
            msg, mt = find_closest_msg(times, msgs, t_wall)
            if abs(mt - t_wall) > max_time_diff:
                continue
            uv = project_marker_to_miqus(msg, T, K, dist)
            if uv is None:
                continue
            color = MARKER_COLORS_BGR[mi % len(MARKER_COLORS_BGR)]
            overlay_point_on_image(frame, uv, color=color, radius=radius)
            drawn += 1
        writer.write(frame)
        frame_i += 1

    cap.release()
    writer.release()

    audio_offset = 0.0
    tmp_mp3 = None
    try:
        if audio_bytes is not None and audio_t0 is not None:
            audio_offset = (audio_t0 - video_t0) + float(audio_offset_extra)
            tmp_mp3 = Path(tempfile.mkstemp(prefix="miqus_audio_", suffix=".mp3")[1])
            tmp_mp3.write_bytes(audio_bytes)
            mux_mp4(tmp_avi, tmp_mp3, output_path, audio_offset)
        else:
            mux_mp4(tmp_avi, None, output_path, 0.0)
    finally:
        tmp_avi.unlink(missing_ok=True)
        if tmp_mp3 is not None:
            tmp_mp3.unlink(missing_ok=True)

    print(
        f"  frames={frame_i} marker_draws={drawn} "
        f"audio_offset={audio_offset:+.3f}s -> {output_path.name}"
    )


def resolve_calibration_dir(value: Path) -> Path:
    path = value.expanduser()
    candidates = []
    if not path.is_absolute():
        candidates.append(CALIB_DATA_ROOT / path)
        candidates.append(path)
    else:
        candidates.append(path)
    for cand in candidates:
        if cand.is_dir():
            return cand.resolve()
    raise SystemExit(f"Calibration folder not found: {value}")


def resolve_mocap_calibration(calib_dir: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Mocap calibration not found: {path}")
        return path
    found = find_mocap_calibration(calib_dir, None)
    if found is None:
        raise SystemExit(
            f"No mocap calibration XML in {calib_dir} "
            f"(expected {calib_dir.name}_mocap_calibration.txt or *.qca)"
        )
    return found


def parse_args():
    p = argparse.ArgumentParser(
        description="Overlay markers on Miqus AVIs and mux bag audio."
    )
    p.add_argument("--trial-dir", type=Path, required=True)
    p.add_argument(
        "--calibration-dir",
        type=Path,
        required=True,
        help="Calibration session folder (e.g. data/calibration_data/26_07_16_infant_019). "
        "The mocap XML is found inside it.",
    )
    p.add_argument(
        "--mocap-calibration",
        type=Path,
        help="Override: path to Qualisys mocap calibration XML (.txt/.qca)",
    )
    p.add_argument("--bag", type=Path, help="Combined bag with /marker_N (default: trial_ros_combined.bag)")
    p.add_argument("--audio-bag", type=Path, help="Bag with /audio/audio (default: trial_ros.bag)")
    p.add_argument("--tsv", type=Path, help="Trial TSV for video wall-clock start")
    p.add_argument("--num-markers", type=int, default=700)
    p.add_argument("--max-time-diff", type=float, default=0.02)
    p.add_argument("--radius", type=int, default=10)
    p.add_argument("--max-frames", type=int, help="Stop after N video frames (debug)")
    p.add_argument("--tz", default=DEFAULT_TZ)
    p.add_argument("--offset-sec", type=float, default=0.0)
    p.add_argument("--audio-topic", default=DEFAULT_AUDIO_TOPIC)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    require_ffmpeg()

    trial_dir = args.trial_dir.expanduser().resolve()
    if not trial_dir.is_dir():
        raise SystemExit(f"Trial dir not found: {trial_dir}")

    calib_dir = resolve_calibration_dir(args.calibration_dir)
    mocap_path = resolve_mocap_calibration(calib_dir, args.mocap_calibration)

    avis = sorted(trial_dir.glob(MIQUS_GLOB))
    # Skip already-processed overlays if someone named them *.avi by mistake
    avis = [a for a in avis if "overlay" not in a.name.lower()]
    if not avis:
        raise SystemExit(f"No {MIQUS_GLOB} in {trial_dir}")

    bag_path = (
        args.bag.expanduser().resolve()
        if args.bag
        else find_combined_bag(trial_dir)
    )
    audio_bag = (
        args.audio_bag.expanduser().resolve()
        if args.audio_bag
        else find_audio_bag(trial_dir)
    )
    tsv_path = (
        args.tsv.expanduser().resolve()
        if args.tsv
        else find_trial_tsv(trial_dir)
    )
    if tsv_path is None or not tsv_path.is_file():
        raise SystemExit(f"TSV required for Miqus timing (not found in {trial_dir})")

    video_t0 = parse_tsv_start(tsv_path, args.tz)
    marker_streams, n_markers = load_marker_streams(bag_path, args.num_markers)
    audio_bytes, audio_t0 = load_audio_mp3(audio_bag, args.audio_topic)

    print(f"Trial:   {trial_dir}")
    print(f"Bag:     {bag_path}")
    print(f"Audio:   {audio_bag} ({'yes' if audio_bytes else 'none'})")
    print(f"TSV:     {tsv_path} (t0={video_t0:.3f})")
    print(f"Mocap:   {mocap_path}")
    print(f"Markers: {n_markers}")
    print(f"AVIs:    {len(avis)}")

    for avi in avis:
        _cam_id, serial = serial_from_avi(avi)
        T, K, dist = mocap_camera_from_serial(mocap_path, serial)
        out = vis_dir(trial_dir, "mocap") / f"mocap_marker_overlay_Miqus_{_cam_id}_{serial}.mp4"
        print(f"[miqus] {avi.name} (serial={serial}) -> {out.name}")
        overlay_one_avi(
            avi,
            marker_streams,
            T,
            K,
            dist,
            video_t0,
            args.max_time_diff,
            args.radius,
            out,
            audio_bytes,
            audio_t0,
            args.offset_sec,
            max_frames=args.max_frames,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
