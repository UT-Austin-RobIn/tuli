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
import atexit
import os
import shutil
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
from viz_lifecycle import (
    install_stop_signals,
    kill_stale_viz,
    popen_session,
    raise_keyboard,
    restore_wall_clock_time,
    set_use_sim_time,
    sleep_interruptible,
    stop_proc,
)

# .../infants/infants/scripts/viz -> repo root is parents[3]
INFANTS_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CALIB_CONFIG = INFANTS_ROOT / "config" / "calibration_markers_example.yaml"
LAUNCH_FILE = INFANTS_ROOT / "launch" / "trial_viz.launch"
RECORD_SCRIPT = SCRIPTS_DIR / "record_rviz_screen.sh"
ORBIT_DRAG_SCRIPT = SCRIPTS_DIR / "orbit_drag_yaw.py"
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


_RESTORE_DONE = False


def _restore_wall_clock_time_once():
    """Force /use_sim_time=false so live cameras use wall clock after bag viz."""
    global _RESTORE_DONE
    if _RESTORE_DONE:
        return
    _RESTORE_DONE = True
    restore_wall_clock_time()


def wait_for_rviz_window(display_id: str, timeout_sec: float = 45.0) -> bool:
    """Block until an RViz window is mapped on DISPLAY (avoids recording pure black)."""
    env = os.environ.copy()
    env["DISPLAY"] = display_id
    deadline = time.time() + float(timeout_sec)
    print(f"[record] Waiting for RViz window on {display_id}...")
    while time.time() < deadline:
        try:
            tree = subprocess.run(
                ["xwininfo", "-root", "-tree"],
                capture_output=True,
                text=True,
                timeout=3,
                env=env,
            )
            text = (tree.stdout or "") + (tree.stderr or "")
            if "rviz" in text.lower():
                # Give the first paint / fullscreen a moment.
                time.sleep(1.0)
                print("[record] RViz window found")
                return True
        except Exception:
            pass
        time.sleep(0.4)
    print("[WARN] Timed out waiting for RViz window; recording anyway")
    return False


# Track launch proc so SIGTERM/SIGINT handlers can stop it before restoring time.
_ACTIVE = {"proc": None, "record": None, "audio": None, "orbit_drag": None}


def bag_play_duration_sec(bag_path: Path) -> float:
    """Wall-clock span of messages in the bag (end - start)."""
    try:
        import rosbag
    except ImportError:
        return 0.0
    with rosbag.Bag(str(bag_path), "r") as bag:
        start = bag.get_start_time()
        end = bag.get_end_time()
    return max(0.0, float(end) - float(start))


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
        dest="loop",
        action="store_true",
        default=None,
        help="Loop bag playback (default: on for viewing, off when --record)",
    )
    parser.add_argument(
        "--no-loop",
        dest="loop",
        action="store_false",
        help="Play the bag only once",
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
        default=5.0,
        help="Seconds to wait after launching RViz before starting screen record (default: 5)",
    )
    parser.add_argument(
        "--stop-pad",
        type=float,
        default=1.0,
        help="Extra seconds to keep recording after the bag ends before teardown (default: 1)",
    )
    parser.add_argument(
        "--look-depth",
        type=float,
        default=1.2,
        help="Meters along each RealSense optical axis used to seed the orbit focus",
    )
    parser.add_argument(
        "--orbit",
        dest="orbit",
        action="store_true",
        default=True,
        help="Slow 180deg back-and-forth RViz yaw through the trial (default: on with mcR)",
    )
    parser.add_argument(
        "--no-orbit",
        dest="orbit",
        action="store_false",
        help="Disable orbit; use a static camera-frame Orbit view",
    )
    parser.add_argument(
        "--orbit-period",
        type=float,
        default=None,
        help="Seconds to keep yawing (default: bag duration; one 180 out+back fills this)",
    )
    parser.add_argument(
        "--max-markers",
        type=int,
        default=0,
        help="Optional cap on /marker_N subscriptions (0=use full YAML num_markers)",
    )
    args = parser.parse_args()
    # Viewing: loop so RViz does not go blank after one pass.
    # Recording: play once unless the user passed --loop.
    if args.loop is None:
        args.loop = not args.record
    # Give marker_transformer / TF / RViz time before bag playback / recording.
    # Too-short delays + /use_sim_time=false mid-run produce pure-black MP4s.
    if args.markers:
        # Bag-backed marker index takes ~45s to build; wait so markers exist from t0.
        if args.bag_delay < 48.0:
            args.bag_delay = 50.0
        if args.record and args.record_delay < 52.0:
            args.record_delay = 55.0
    elif args.record and args.record_delay < 4.0:
        args.record_delay = 4.0

    bag_path = args.bag.expanduser().resolve()
    if not bag_path.is_file():
        raise SystemExit(f"Bag not found: {bag_path}")
    if not LAUNCH_FILE.is_file():
        raise SystemExit(f"Launch file not found: {LAUNCH_FILE}")

    bag_duration = bag_play_duration_sec(bag_path)
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
    use_debug_layout = args.raw and len(cameras) > 1
    # Calib TF is required for multi-cam alignment, markers, or explicit mcR view.
    needs_calib = ((len(cameras) > 1 and not args.raw) or args.markers) and not use_debug_layout
    # Default: express clouds in mcR whenever we have a calibration chain.
    if args.mcr_frame is None:
        use_mcr_frame = needs_calib and not use_debug_layout
    else:
        use_mcr_frame = bool(args.mcr_frame) and not use_debug_layout
    if use_mcr_frame:
        needs_calib = True
    if needs_calib and not calib_config.is_file():
        raise SystemExit(
            f"Calibration config required for multi-camera, markers, or --mcr-frame: {calib_config}"
        )

    # TF broadcaster needed for multi-camera alignment OR mcR reprojection.
    show_calib_tf = (len(cameras) > 1 and not args.raw) or use_mcr_frame
    animate_orbit = bool(args.orbit and use_mcr_frame)
    orbit_period = (
        float(args.orbit_period)
        if args.orbit_period is not None
        else max(bag_duration, 1.0)
    )

    rviz_fd, rviz_name = tempfile.mkstemp(prefix="trial_viz_", suffix=".rviz")
    os.close(rviz_fd)
    rviz_path = Path(rviz_name)
    write_rviz_config(
        rviz_path,
        cameras,
        args.markers,
        fixed_camera=fixed_camera(cameras),
        use_mcr_frame=use_mcr_frame,
        calib_config=calib_config if needs_calib else None,
        look_depth=args.look_depth,
        animate_orbit=animate_orbit,
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
        f"max_markers:={args.max_markers}",
        f"calib_config:={calib_config if needs_calib else ''}",
        f"show_calib_tf:={'true' if show_calib_tf else 'false'}",
        f"use_mcr_frame:={'true' if use_mcr_frame else 'false'}",
        f"debug_layout:={'true' if use_debug_layout else 'false'}",
        f"fixed_camera:={fixed_camera(cameras)}",
        f"show_cam_L:={'true' if 'L' in cameras else 'false'}",
        f"show_cam_M:={'true' if 'M' in cameras else 'false'}",
        f"show_cam_R:={'true' if 'R' in cameras else 'false'}",
        f"animate_orbit:={'false'}",
        f"orbit_period:={orbit_period:.3f}",
        f"look_depth:={args.look_depth}",
    ]

    print("=== Trial visualization ===")
    print(f"Bag:       {bag_path}")
    print(f"Cameras:   {camera_csv}")
    print(f"Markers:   {args.markers}")
    print(f"Audio:     {args.audio}")
    print(f"Record:    {record_path if args.record else False}")
    print(f"Raw layout:{use_debug_layout}")
    print(f"mcR frame: {use_mcr_frame} (RViz Fixed Frame = {MOCAP_REF_FRAME if use_mcr_frame else 'camera'})")
    print(f"Orbit:     {animate_orbit} (180deg back-forth, duration={orbit_period:.1f}s)")
    print(f"Loop:      {args.loop}")
    if not args.loop:
        auto_sec = float(args.bag_delay) + bag_duration + float(args.stop_pad)
        print(
            f"Auto-stop: after bag ends "
            f"(~{auto_sec:.1f}s = delay {args.bag_delay:.1f} + "
            f"bag {bag_duration:.1f} + pad {args.stop_pad:.1f}; no Ctrl+C needed)"
        )
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
    # Suppress RViz's modal "ROS 1 End-of-Life" OK dialog (blocks unattended recording).
    env["DISABLE_ROS1_EOL_WARNINGS"] = "1"
    env["PYTHONPATH"] = f"{SCRIPTS_DIR}:{env.get('PYTHONPATH', '')}"

    # Ctrl+C / SIGTERM both become KeyboardInterrupt so finally always runs.
    global _RESTORE_DONE
    _RESTORE_DONE = False
    atexit.register(_restore_wall_clock_time_once)
    install_stop_signals(raise_keyboard)

    kill_stale_viz()
    # Must be true before bag/--clock or RViz drops messages → black empty view.
    set_use_sim_time(True)

    audio_proc = None
    record_proc = None
    orbit_drag_proc = None
    proc = None
    exit_code = 0
    stopped = False
    try:
        if args.audio:
            audio_cmd = ["roslaunch", "audio_play", "play.launch"]
            print(f"[audio] Starting: {' '.join(audio_cmd)}")
            audio_proc = popen_session(audio_cmd, env=env)
            _ACTIVE["audio"] = audio_proc

        proc = popen_session(launch_cmd, env=env)
        _ACTIVE["proc"] = proc
        t_launch = time.time()

        display_id = env.get("DISPLAY", ":0")
        if args.record or animate_orbit:
            wait_for_rviz_window(
                display_id, timeout_sec=max(30.0, args.record_delay + 20.0)
            )

        if args.record:
            if args.record_delay > 0:
                sleep_interruptible(
                    args.record_delay,
                    procs=(proc,),
                    label="settle for TF/clouds/markers before record",
                )
            set_use_sim_time(True)
            record_cmd = [str(RECORD_SCRIPT), str(silent_record_path), display_id]
            print(f"[record] Starting: {' '.join(record_cmd)}")
            record_proc = popen_session(record_cmd, env=env)
            _ACTIVE["record"] = record_proc

        if animate_orbit and ORBIT_DRAG_SCRIPT.is_file():
            drag_cmd = [
                sys.executable,
                str(ORBIT_DRAG_SCRIPT),
                "--display", display_id,
                "--period", f"{orbit_period:.3f}",
                "--span-deg", "180",
            ]
            print(f"[orbit] Starting 180deg back-and-forth yaw: {' '.join(drag_cmd)}")
            orbit_drag_proc = popen_session(drag_cmd, env=env)
            _ACTIVE["orbit_drag"] = orbit_drag_proc

        if args.loop:
            exit_code = proc.wait()
        else:
            run_secs = float(args.bag_delay) + bag_duration + float(args.stop_pad)
            remaining = max(1.0, run_secs - (time.time() - t_launch))
            print(
                f"[auto-stop] Waiting {remaining:.1f}s more "
                f"(total window {run_secs:.1f}s from launch)..."
            )
            early = sleep_interruptible(remaining, procs=(proc,))
            if early is not None:
                print(f"[auto-stop] Launch exited early (code={early})")
                exit_code = early
            else:
                print("[auto-stop] Bag window done; stopping launch")
                stop_proc(proc, "roslaunch", timeout=15)
                exit_code = proc.returncode if proc.returncode is not None else 0
                proc = None
                _ACTIVE["proc"] = None
    except KeyboardInterrupt:
        stopped = True
        print("\n[stop] Interrupted — shutting down RViz/record/orbit...", flush=True)
        exit_code = 130
    finally:
        stop_proc(orbit_drag_proc, "orbit-drag")
        stop_proc(record_proc, "record", timeout=12)
        stop_proc(audio_proc, "audio")
        stop_proc(proc, "roslaunch")
        _ACTIVE["proc"] = _ACTIVE["record"] = _ACTIVE["audio"] = _ACTIVE["orbit_drag"] = None
        kill_stale_viz()
        if (
            not stopped
            and args.record
            and silent_record_path is not None
            and record_path is not None
            and silent_record_path.is_file()
        ):
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
        elif stopped and silent_record_path is not None:
            silent_record_path.unlink(missing_ok=True)
        rviz_path.unlink(missing_ok=True)
        _restore_wall_clock_time_once()

    return exit_code


if __name__ == "__main__":
    sys.exit(main() or 0)
