#!/usr/bin/env python3
"""Launch trial bag visualization with 1-3 point clouds and optional markers.

Point clouds are published in each camera's optical frame. When calibration is
enabled (default), we also publish TF so RViz can display them in Qualisys mcR:

  p_cam  = T_mcR_to_cam @ p_mcR          (see calibration_chain.py)
  p_mcR  = inv(T_mcR_to_cam) @ p_cam     (what RViz applies via TF)

This matches stereo-calib/examples/visualize_fullchain.py marker projection,
but applied in reverse so depth clouds land in the mocap reference frame.

Examples:
  python run_trial_viz.py --bag data/.../trial_ros.bag --cameras L
  python run_trial_viz.py --bag data/.../trial_ros.bag --cameras L,M,R
  python run_trial_viz.py --bag data/.../trial_ros.bag --cameras L,M,R --raw
  python run_trial_viz.py --bag data/.../trial_ros.bag --cameras L --markers \\
      --calib-config data/calibration_data/26_06_29_infant_017/calibration_markers.yaml
  python run_trial_viz.py --bag data/.../trial_ros_combined.bag --cameras L --markers --audio \\
      --calib-config data/calibration_data/26_06_29_infant_017/calibration_markers.yaml
  python run_trial_viz.py --bag data/.../trial_ros_combined.bag --cameras L --markers --audio --record \\
      --calib-config data/calibration_data/26_06_29_infant_017/calibration_markers.yaml
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from calibration_chain import MOCAP_REF_FRAME
from generate_trial_viz_rviz import write_rviz_config

INFANTS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIB_CONFIG = INFANTS_ROOT / "config" / "calibration_markers_example.yaml"
LAUNCH_FILE = INFANTS_ROOT / "launch" / "trial_viz.launch"
RECORD_SCRIPT = SCRIPTS_DIR / "record_rviz_screen.sh"
DEFAULT_RECORDINGS_DIR = INFANTS_ROOT / "recordings"

VALID_CAMERAS = ("L", "M", "R")


def parse_cameras(raw):
    cameras = [part.strip().upper() for part in raw.split(",") if part.strip()]
    if not cameras:
        raise ValueError("At least one camera is required.")
    for cam in cameras:
        if cam not in VALID_CAMERAS:
            raise ValueError(f"Unknown camera '{cam}'. Choose from L, M, R.")
    # Stable canonical order.
    return [cam for cam in VALID_CAMERAS if cam in cameras]


def fixed_camera(cameras):
    return "L" if "L" in cameras else cameras[0]


def default_record_path(bag_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RECORDINGS_DIR / f"rviz_{bag_path.stem}_{stamp}.mp4"


def stop_proc(proc, name, timeout=5):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[{name}] SIGINT timed out; killing")
        proc.kill()
        proc.wait()


def extract_bag_audio_mp3(bag_path: Path, topic: str = "/audio/audio"):
    """Return concatenated MP3 bytes from bag, or None if missing."""
    try:
        import rosbag
    except ImportError:
        print("[record] rosbag not available; skipping audio mux")
        return None

    chunks = bytearray()
    with rosbag.Bag(str(bag_path), "r") as bag:
        topics = bag.get_type_and_topic_info().topics
        if topic not in topics:
            return None
        for _, msg, _t in bag.read_messages(topics=[topic]):
            if msg.data:
                chunks.extend(msg.data)
    return bytes(chunks) if chunks else None


def mux_recording_with_bag_audio(
    silent_video: Path,
    bag_path: Path,
    output_path: Path,
    audio_skip_sec: float,
):
    """Mux X11 recording with /audio/audio from the bag.

    audio_skip_sec > 0: recording started after bag play → skip that much audio.
    audio_skip_sec < 0: recording started before bag play → delay audio.
    """
    if shutil.which("ffmpeg") is None:
        print("[record] ffmpeg not found; leaving silent video as-is")
        if silent_video != output_path:
            silent_video.replace(output_path)
        return False

    audio_bytes = extract_bag_audio_mp3(bag_path)
    if audio_bytes is None:
        print("[record] No /audio/audio in bag; leaving silent video")
        if silent_video != output_path:
            silent_video.replace(output_path)
        return False

    tmp_mp3 = Path(tempfile.mkstemp(prefix="rviz_audio_", suffix=".mp3")[1])
    tmp_mp3.write_bytes(audio_bytes)
    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(silent_video),
        ]
        if audio_skip_sec > 0:
            cmd += ["-ss", f"{audio_skip_sec:.6f}", "-i", str(tmp_mp3)]
        elif audio_skip_sec < 0:
            cmd += ["-itsoffset", f"{-audio_skip_sec:.6f}", "-i", str(tmp_mp3)]
        else:
            cmd += ["-i", str(tmp_mp3)]
        cmd += [
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ]
        print(
            f"[record] Muxing bag audio into MP4 "
            f"(audio_skip={audio_skip_sec:+.3f}s) -> {output_path}"
        )
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[record] Audio mux failed ({exc}); keeping silent video")
        if silent_video != output_path:
            silent_video.replace(output_path)
        return False
    finally:
        tmp_mp3.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trial rosbags with 1-3 calibrated point clouds."
    )
    parser.add_argument("--bag", required=True, type=Path, help="Path to trial .bag")
    parser.add_argument(
        "--cameras",
        default="L",
        help="Comma-separated cameras to show (L, M, R). Example: L,M,R",
    )
    parser.add_argument(
        "--markers",
        action="store_true",
        help="Overlay calibrated Qualisys markers (/marker_N in bag)",
    )
    parser.add_argument(
        "--calib-config",
        type=Path,
        default=DEFAULT_CALIB_CONFIG,
        help="Calibration YAML for markers and multi-camera alignment",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Side-by-side layout in camera frames (no mcR / calibration alignment)",
    )
    parser.add_argument(
        "--mcr-frame",
        dest="mcr_frame",
        action="store_true",
        default=None,
        help=f"View clouds in Qualisys mcR frame ({MOCAP_REF_FRAME}); default when calib is used",
    )
    parser.add_argument(
        "--no-mcr-frame",
        dest="mcr_frame",
        action="store_false",
        help="Keep RViz fixed frame on a camera optical frame instead of mcR",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop bag playback (default: play once)",
    )
    parser.add_argument(
        "--bag-delay",
        type=float,
        default=1.0,
        help="Seconds to wait before starting bag playback",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=2,
        help="Point cloud pixel subsample factor (higher = faster)",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Also start audio_play so /audio/audio from the bag is heard",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the X11 display (RViz + your camera moves) and mux bag /audio/audio into the MP4",
    )
    parser.add_argument(
        "--record-output",
        type=Path,
        help="MP4 path for --record (default: recordings/rviz_<bag>_<timestamp>.mp4)",
    )
    parser.add_argument(
        "--record-delay",
        type=float,
        default=2.0,
        help="Seconds to wait after launching RViz before starting screen record (default: 2)",
    )
    args = parser.parse_args()

    bag_path = args.bag.expanduser().resolve()
    if not bag_path.is_file():
        raise SystemExit(f"Bag not found: {bag_path}")
    if not LAUNCH_FILE.is_file():
        raise SystemExit(f"Launch file not found: {LAUNCH_FILE}")

    record_path = None
    silent_record_path = None
    if args.record:
        if not RECORD_SCRIPT.is_file():
            raise SystemExit(f"Record script not found: {RECORD_SCRIPT}")
        record_path = (
            args.record_output.expanduser().resolve()
            if args.record_output
            else default_record_path(bag_path)
        )
        record_path.parent.mkdir(parents=True, exist_ok=True)
        silent_record_path = record_path.with_name(
            f"{record_path.stem}_silent{record_path.suffix}"
        )

    cameras = parse_cameras(args.cameras)
    calib_config = args.calib_config.expanduser().resolve()
    needs_calib = (len(cameras) > 1 and not args.raw) or args.markers
    if needs_calib and not calib_config.is_file():
        raise SystemExit(
            f"Calibration config required for multi-camera or markers: {calib_config}"
        )

    use_debug_layout = args.raw and len(cameras) > 1
    # Default: express clouds in mcR whenever we have a calibration chain.
    if args.mcr_frame is None:
        use_mcr_frame = needs_calib and not use_debug_layout
    else:
        use_mcr_frame = args.mcr_frame and not use_debug_layout

    # TF broadcaster needed for multi-camera alignment OR mcR reprojection.
    show_calib_tf = (len(cameras) > 1 and not args.raw) or use_mcr_frame

    rviz_fd, rviz_name = tempfile.mkstemp(prefix="trial_viz_", suffix=".rviz")
    os.close(rviz_fd)
    rviz_path = Path(rviz_name)
    write_rviz_config(
        rviz_path,
        cameras,
        args.markers,
        fixed_camera=fixed_camera(cameras),
        use_mcr_frame=use_mcr_frame,
    )

    camera_csv = ",".join(cameras)
    launch_cmd = [
        "roslaunch",
        str(LAUNCH_FILE),
        f"bag_file:={bag_path}",
        f"rviz_config:={rviz_path}",
        f"loop_bag:={'true' if args.loop else 'false'}",
        f"bag_delay:={args.bag_delay}",
        f"subsample:={args.subsample}",
        f"cameras:={camera_csv}",
        f"show_markers:={'true' if args.markers else 'false'}",
        f"calib_config:={calib_config if needs_calib else ''}",
        f"show_calib_tf:={'true' if show_calib_tf else 'false'}",
        f"use_mcr_frame:={'true' if use_mcr_frame else 'false'}",
        f"debug_layout:={'true' if use_debug_layout else 'false'}",
        f"fixed_camera:={fixed_camera(cameras)}",
        f"show_cam_L:={'true' if 'L' in cameras else 'false'}",
        f"show_cam_M:={'true' if 'M' in cameras else 'false'}",
        f"show_cam_R:={'true' if 'R' in cameras else 'false'}",
    ]

    print("=== Trial visualization ===")
    print(f"Bag:       {bag_path}")
    print(f"Cameras:   {camera_csv}")
    print(f"Markers:   {args.markers}")
    print(f"Audio:     {args.audio}")
    print(f"Record:    {record_path if args.record else False}")
    print(f"Raw layout:{use_debug_layout}")
    print(f"mcR frame: {use_mcr_frame} (RViz Fixed Frame = {MOCAP_REF_FRAME if use_mcr_frame else 'camera'})")
    print(f"Loop:      {args.loop}")
    if len(cameras) > 1:
        print("Colors:    L=red, M=green, R=blue (toggle each under Displays)")
    if use_mcr_frame:
        print(
            "Transform: p_mcR = inv(T_mcR_to_cam) @ p_cam  "
            "(see calib_tf_broadcaster logs for matrices)"
        )
    print(f"RViz cfg:  {rviz_path}")
    print(f"Command:   {' '.join(launch_cmd)}\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SCRIPTS_DIR}:{env.get('PYTHONPATH', '')}"

    audio_proc = None
    record_proc = None
    proc = None
    try:
        if args.audio:
            audio_cmd = ["roslaunch", "audio_play", "play.launch"]
            print(f"[audio] Starting: {' '.join(audio_cmd)}")
            audio_proc = subprocess.Popen(audio_cmd, env=env)

        proc = subprocess.Popen(launch_cmd, env=env)

        if args.record:
            if args.record_delay > 0:
                print(f"[record] Waiting {args.record_delay:.1f}s for RViz to come up...")
                time.sleep(args.record_delay)
            display_id = env.get("DISPLAY", ":0")
            record_cmd = [str(RECORD_SCRIPT), str(silent_record_path), display_id]
            print(f"[record] Starting: {' '.join(record_cmd)}")
            record_proc = subprocess.Popen(record_cmd, env=env)

        return proc.wait()
    except KeyboardInterrupt:
        if proc is not None:
            proc.send_signal(signal.SIGINT)
            return proc.wait()
        return 130
    finally:
        stop_proc(record_proc, "record")
        stop_proc(audio_proc, "audio")
        if (
            args.record
            and silent_record_path is not None
            and record_path is not None
            and silent_record_path.is_file()
        ):
            # Bag play starts at bag_delay; screen capture at record_delay.
            audio_skip = float(args.record_delay) - float(args.bag_delay)
            mux_recording_with_bag_audio(
                silent_record_path,
                bag_path,
                record_path,
                audio_skip_sec=audio_skip,
            )
            silent_record_path.unlink(missing_ok=True)
            if record_path.is_file():
                print(f"[record] Saved {record_path}")
        rviz_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
