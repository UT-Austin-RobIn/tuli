#!/usr/bin/env python3
"""Generate the 6 marker-overlay videos for one or more trials.

Outputs (per trial folder):
  visualizations/realsense/realsense_marker_overlay_{L,M,R}.mp4
  visualizations/mocap/mocap_marker_overlay_Miqus_*_*.mp4

Examples:
  # All 9 trials for infant 019:
  python infants/scripts/viz/make_trial_overlay_videos.py \\
      --session data/2026-07-16_11-03-27 \\
      --calibration-dir data/calibration_data/26_07_16_infant_019

  # One trial:
  python infants/scripts/viz/make_trial_overlay_videos.py \\
      --trial-dir data/2026-07-16_11-03-27/trial_001 \\
      --calibration-dir data/calibration_data/26_07_16_infant_019
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

VIZ_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VIZ_DIR))
from viz_layout import vis_dir
RS_SCRIPT = VIZ_DIR / "overlay_realsense_markers.py"
MIQUS_SCRIPT = VIZ_DIR / "overlay_miqus_markers.py"


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def process_trial(
    trial_dir: Path,
    calibration_dir: Path,
    cameras: list[str],
    num_markers: int | None,
    skip_existing: bool,
) -> None:
    bag = trial_dir / "trial_ros_combined.bag"
    if not bag.is_file():
        raise SystemExit(
            f"Missing {bag}. Run process_marker_c3d.py first."
        )

    for cam in cameras:
        out = vis_dir(trial_dir, "realsense") / f"realsense_marker_overlay_{cam}.mp4"
        if skip_existing and out.is_file():
            print(f"skip {out.name}")
            continue
        cmd = [
            sys.executable,
            str(RS_SCRIPT),
            "--trial-dir",
            str(trial_dir),
            "--calibration-dir",
            str(calibration_dir),
            "--camera",
            cam,
            "--save-mp4",
            "--no-display",
            "--output",
            str(out),
        ]
        if num_markers is not None:
            cmd += ["--num-markers", str(num_markers)]
        run(cmd)

    # Miqus overlays (script writes all AVIs in the trial)
    existing = list(vis_dir(trial_dir, "mocap").glob("mocap_marker_overlay_Miqus_*.mp4"))
    avis = list(trial_dir.glob("*Miqus*.avi"))
    if skip_existing and existing and len(existing) >= len(avis):
        print(f"skip mocap overlays ({len(existing)} already present)")
    else:
        cmd = [
            sys.executable,
            str(MIQUS_SCRIPT),
            "--trial-dir",
            str(trial_dir),
            "--calibration-dir",
            str(calibration_dir),
        ]
        if num_markers is not None:
            cmd += ["--num-markers", str(num_markers)]
        run(cmd)


def parse_args():
    p = argparse.ArgumentParser(description="Make RealSense + Miqus marker overlay MP4s.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", type=Path, help="Session folder containing trial_*/")
    g.add_argument("--trial-dir", type=Path, help="Single trial folder")
    p.add_argument(
        "--calibration-dir",
        type=Path,
        required=True,
        help="Calibration session folder (calibration_markers.yaml and mocap XML)",
    )
    p.add_argument("--cameras", default="L,M,R", help="RealSense cameras (default L,M,R)")
    p.add_argument(
        "--num-markers",
        type=int,
        help="Draw only the first N markers (default: all markers in the bag)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not overwrite existing overlay MP4s",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    calib_dir = args.calibration_dir.expanduser().resolve()
    cameras = [c.strip().upper() for c in args.cameras.split(",") if c.strip()]

    if args.trial_dir:
        trials = [args.trial_dir.expanduser().resolve()]
    else:
        session = args.session.expanduser().resolve()
        trials = sorted(session.glob("trial_*"))
        trials = [t for t in trials if t.is_dir()]
        if not trials:
            raise SystemExit(f"No trial_* under {session}")

    for trial in trials:
        print(f"\n===== {trial} =====", flush=True)
        process_trial(
            trial,
            calib_dir,
            cameras,
            args.num_markers,
            args.skip_existing,
        )
    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
