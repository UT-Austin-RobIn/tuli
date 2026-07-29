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

  # Live audio + write MP4 (overlay + bag audio):
  python infants/scripts/overlay_markers_on_image.py \\
      --bag .../trial_ros_combined.bag --calib-config ... \\
      --audio --save-mp4

  # Save frames instead of interactive playback:
  python infants/scripts/overlay_markers_on_image.py \\
      --bag .../trial_ros_combined.bag --calib-config ... \\
      --save-dir /tmp/marker_overlay
"""
from __future__ import annotations

import argparse
import bisect
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

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

DEFAULT_AUDIO_TOPIC = "/audio/audio"


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


def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise SystemExit("[ERROR] ffmpeg not found. Install with: sudo apt install ffmpeg")


def load_audio_mp3(bag_path: Path, topic: str = DEFAULT_AUDIO_TOPIC) -> Tuple[Optional[bytes], Optional[float]]:
    chunks = bytearray()
    first_t = None
    with rosbag.Bag(str(bag_path), "r") as bag:
        topics = bag.get_type_and_topic_info().topics
        if topic not in topics:
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


def start_ffplay(mp3_path: Path) -> subprocess.Popen:
    if shutil.which("ffplay") is None:
        raise SystemExit("[ERROR] ffplay not found (install ffmpeg). Needed for --audio.")
    return subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(mp3_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def estimate_fps(times):
    if len(times) < 2:
        return 30.0
    dts = np.diff(np.asarray(times, dtype=np.float64))
    dts = dts[dts > 1e-6]
    if len(dts) == 0:
        return 30.0
    return float(1.0 / np.median(dts))


def mux_mp4(video_path: Path, audio_path: Optional[Path], output_path: Path, audio_offset_sec: float):
    require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ]
    else:
        cmd += [
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-an",
            "-movflags", "+faststart",
            str(output_path),
        ]
    subprocess.run(cmd, check=True)


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
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Play /audio/audio from the bag (ffplay), paced to bag time",
    )
    parser.add_argument(
        "--audio-topic",
        default=DEFAULT_AUDIO_TOPIC,
        help=f"Audio topic in bag (default: {DEFAULT_AUDIO_TOPIC})",
    )
    parser.add_argument(
        "--save-mp4",
        action="store_true",
        help="Write overlay MP4 next to the bag (<bag_stem>_overlay_<cam>.mp4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="MP4 output path (implies saving; overrides default --save-mp4 name)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip OpenCV window (useful with --save-mp4 / --output)",
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

    output_path = None
    if args.output:
        output_path = args.output.expanduser().resolve()
    elif args.save_mp4:
        output_path = bag_path.parent / f"{bag_path.stem}_overlay_{cam_key}.mp4"

    # Continuous timeline when audio or MP4 export is on; classic mode skips empty frames.
    continuous = bool(args.audio or output_path is not None)

    play_audio = args.audio
    mux_audio = output_path is not None
    audio_bytes, audio_t0 = (None, None)
    if play_audio or mux_audio:
        audio_bytes, audio_t0 = load_audio_mp3(bag_path, args.audio_topic)
        if audio_bytes is None:
            if play_audio:
                print(f"[WARN] No audio on {args.audio_topic}; continuing without live sound.")
            play_audio = False
            mux_audio = False

    print(f"Bag:        {bag_path}")
    print(f"Camera:     {cam_name}")
    print(f"Calib:      {config_path}")
    print(f"Markers:    {num_markers} ({', '.join(marker_streams)})")
    print(f"Color fr.:  {len(color_msgs)}")
    print(f"Audio:      {play_audio}")
    print(f"Save MP4:   {output_path if output_path else '(off)'}")
    print(f"K:\n{K}")
    if dist is not None:
        print(f"dist:       {dist.ravel()[:5]}...")

    save_dir = None
    if args.save_dir:
        save_dir = args.save_dir.expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

    show_window = not args.no_display and save_dir is None
    audio_proc = None
    tmp_mp3 = None
    tmp_video = None
    writer = None
    processed = 0
    shown = 0
    written_frames = 0
    t0 = color_times[0]
    audio_offset = (audio_t0 - t0) if (mux_audio and audio_t0 is not None) else 0.0
    wall0 = time.perf_counter()

    try:
        if play_audio:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp_mp3 = Path(tmp.name)
            tmp.write(audio_bytes)
            tmp.close()
            audio_delay = max(0.0, audio_offset)
            if audio_delay > 0:
                print(f"[audio] Delaying ffplay by {audio_delay:.3f}s to match bag")
                time.sleep(audio_delay)
            print(f"[audio] Starting ffplay: {tmp_mp3}")
            audio_proc = start_ffplay(tmp_mp3)
            wall0 = time.perf_counter() - audio_delay

        for i, (color_time, color_msg) in enumerate(zip(color_times, color_msgs)):
            if i % args.stride != 0:
                continue

            if play_audio or (continuous and show_window):
                target = wall0 + (color_time - t0)
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

            img = imgmsg_to_bgr8(color_msg)
            img_overlay = img.copy()
            matched = 0

            for marker_idx, (_topic, (m_times, m_msgs)) in enumerate(
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
            if not continuous and matched == 0:
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

            if output_path is not None:
                if writer is None:
                    h, w = img_overlay.shape[:2]
                    fps = estimate_fps(color_times[:: max(args.stride, 1)])
                    tmp_video = Path(tempfile.mkstemp(prefix="overlay_vid_", suffix=".avi")[1])
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (w, h))
                    if not writer.isOpened():
                        raise SystemExit(f"Failed to open VideoWriter for {tmp_video}")
                    print(f"[video] Recording temp AVI at ~{fps:.2f} fps ({w}x{h})")
                writer.write(img_overlay)
                written_frames += 1

            if save_dir:
                out_path = save_dir / f"frame_{i:06d}.jpg"
                cv2.imwrite(str(out_path), img_overlay)
            elif show_window:
                cv2.imshow(f"Marker overlay ({cam_name})", img_overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

            if args.max_frames and shown >= args.max_frames:
                break
    finally:
        if show_window:
            cv2.destroyAllWindows()
        if writer is not None:
            writer.release()
        if audio_proc is not None and audio_proc.poll() is None:
            audio_proc.terminate()
            try:
                audio_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                audio_proc.kill()

        try:
            if output_path is not None and tmp_video is not None and tmp_video.is_file():
                mux_mp3 = tmp_mp3
                if mux_audio and mux_mp3 is None and audio_bytes is not None:
                    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                    mux_mp3 = Path(tmp.name)
                    tmp.write(audio_bytes)
                    tmp.close()
                    tmp_mp3 = mux_mp3
                print(f"[video] Muxing MP4 -> {output_path}")
                mux_mp4(
                    tmp_video,
                    mux_mp3 if mux_audio else None,
                    output_path,
                    audio_offset if mux_audio else 0.0,
                )
                print(f"[OK] Wrote {output_path} ({written_frames} frames)")
        finally:
            if tmp_video is not None:
                tmp_video.unlink(missing_ok=True)
            if tmp_mp3 is not None:
                tmp_mp3.unlink(missing_ok=True)

    print(f"Processed {processed} color frames, drew markers on {shown}.")
    if save_dir:
        print(f"Saved frames to {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
