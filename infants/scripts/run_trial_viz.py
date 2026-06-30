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
"""
import argparse
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from calibration_chain import MOCAP_REF_FRAME
from generate_trial_viz_rviz import write_rviz_config

INFANTS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIB_CONFIG = INFANTS_ROOT / "config" / "calibration_markers_example.yaml"
LAUNCH_FILE = INFANTS_ROOT / "launch" / "trial_viz.launch"

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
    args = parser.parse_args()

    bag_path = args.bag.expanduser().resolve()
    if not bag_path.is_file():
        raise SystemExit(f"Bag not found: {bag_path}")
    if not LAUNCH_FILE.is_file():
        raise SystemExit(f"Launch file not found: {LAUNCH_FILE}")

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

    proc = subprocess.Popen(launch_cmd, env=env)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        return proc.wait()
    finally:
        rviz_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
