#!/usr/bin/env python3
"""Batch-record RViz marker videos for a session (hands-off; Ctrl+C stops cleanly).

For each trial writes under visualizations/rviz/:
  rviz_L.mp4  rviz_M.mp4  rviz_R.mp4  rviz_LMR.mp4

Each clip uses Qualisys markers + a slow 180deg back-and-forth orbit.

Default --backend open3d renders offscreen (no desktop / RViz window). Re-run
the same command to continue (complete files are skipped). Ctrl+C stops the
current clip.

  python infants/scripts/viz/make_trial_rviz_videos.py \\
      --session data/2026-07-16_11-03-27 \\
      --calib-config data/calibration_data/26_07_16_infant_019/calibration_markers.yaml
"""
from __future__ import annotations

import argparse
import atexit
import os
import sys
import time
from pathlib import Path

VIZ_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VIZ_DIR))
from viz_layout import vis_dir  # noqa: E402
from viz_lifecycle import (  # noqa: E402
    install_stop_signals,
    kill_stale_viz,
    list_stale_viz,
    popen_session,
    raise_keyboard,
    restore_wall_clock_time,
    stop_proc,
)

RUN_VIZ = VIZ_DIR / "run_trial_viz.py"
RENDER_O3D = VIZ_DIR / "render_trial_open3d.py"

VARIANTS = (
    ("L", "rviz_L.mp4"),
    ("M", "rviz_M.mp4"),
    ("R", "rviz_R.mp4"),
    ("L,M,R", "rviz_LMR.mp4"),
)


def parse_args():
    p = argparse.ArgumentParser(description="Batch RViz recordings for trials.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", type=Path)
    g.add_argument("--trial-dir", type=Path)
    p.add_argument("--calib-config", type=Path, required=True)
    p.add_argument(
        "--variants",
        default="L,M,R,LMR",
        help="Comma list of L, M, R, LMR (default: all four)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Max videos to record this run (default: 8). 0 = remaining jobs.",
    )
    p.add_argument(
        "--min-size-mb",
        type=float,
        default=30.0,
        help="Treat existing MP4s smaller than this as incomplete and redo them (default: 30)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Redo even complete existing videos (still respects --limit)",
    )
    p.add_argument("--stop-pad", type=float, default=1.0)
    p.add_argument(
        "--max-markers",
        type=int,
        default=0,
        help="Optional cap on /marker_N (0=full YAML num_markers; default uncapped)",
    )
    p.add_argument(
        "--backend",
        choices=("open3d", "rviz"),
        default="open3d",
        help="open3d = headless (default, no GUI). rviz = screen-record a real display.",
    )
    p.add_argument(
        "--keep-stale",
        action="store_true",
        help="Do not kill leftover RViz/ffmpeg jobs (rviz backend only)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.backend == "rviz":
        atexit.register(restore_wall_clock_time)
    install_stop_signals(raise_keyboard)

    env = os.environ.copy()
    env["DISABLE_ROS1_EOL_WARNINGS"] = "1"
    env["PYTHONPATH"] = f"{VIZ_DIR}:{env.get('PYTHONPATH', '')}"

    calib = args.calib_config.expanduser().resolve()
    if not calib.is_file():
        raise SystemExit(f"Calibration config not found: {calib}")
    wanted = {v.strip().upper() for v in args.variants.split(",") if v.strip()}
    variants = []
    for cams, name in VARIANTS:
        key = "LMR" if cams == "L,M,R" else cams
        if key in wanted:
            variants.append((cams, name))
    if not variants:
        raise SystemExit(f"No variants selected from {args.variants}")

    if args.trial_dir:
        trials = [args.trial_dir.expanduser().resolve()]
    else:
        session = args.session.expanduser().resolve()
        trials = sorted(t for t in session.glob("trial_*") if t.is_dir())

    min_bytes = max(0.0, float(args.min_size_mb)) * 1024 * 1024
    remaining: list[tuple[Path, str, str, Path]] = []
    skipped = 0
    missing_bag = 0
    for trial in trials:
        bag = trial / "trial_ros_combined.bag"
        if not bag.is_file():
            missing_bag += 1
            print(f"[skip] no combined bag: {trial}")
            continue
        for cams, out_name in variants:
            out = vis_dir(trial, "rviz") / out_name
            complete = (
                not args.force
                and out.is_file()
                and out.stat().st_size >= min_bytes
            )
            if complete:
                skipped += 1
                continue
            remaining.append((trial, cams, out_name, out))

    limit = len(remaining) if int(args.limit) <= 0 else min(int(args.limit), len(remaining))
    jobs = remaining[:limit]
    leftover_after = len(remaining) - len(jobs)
    print(
        f"=== batch backend={args.backend}: {len(jobs)} this run / {len(remaining)} still needed "
        f"({skipped} already complete, {missing_bag} trials missing combined bag) ===",
        flush=True,
    )
    if not jobs:
        print("Nothing to record.")
        return 0

    if args.backend == "rviz":
        if not args.keep_stale:
            kill_stale_viz()
        leftover = list_stale_viz()
        if leftover:
            print("[ERROR] Other RViz/ffmpeg viz processes are still running:")
            for pid, args_line in leftover[:12]:
                print(f"  pid {pid}: {args_line[:140]}")
            print("Re-run without --keep-stale, or stop them manually.")
            return 1

    child = None
    try:
        for i, (trial, cams, out_name, out) in enumerate(jobs, start=1):
            print(f"\n===== [{i}/{len(jobs)}] {trial.name} {out_name} =====", flush=True)
            silent = out.with_name(f"{out.stem}_silent{out.suffix}")
            if silent.is_file():
                silent.unlink()
            if args.backend == "open3d":
                cmd = [
                    sys.executable,
                    str(RENDER_O3D),
                    "--bag", str(trial / "trial_ros_combined.bag"),
                    "--cameras", cams,
                    "--markers",
                    "--calib-config", str(calib),
                    "--output", str(out),
                    "--max-markers", str(args.max_markers),
                ]
            else:
                cmd = [
                    sys.executable,
                    str(RUN_VIZ),
                    "--bag", str(trial / "trial_ros_combined.bag"),
                    "--cameras", cams,
                    "--markers",
                    "--mcr-frame",
                    "--calib-config", str(calib),
                    "--audio",
                    "--record",
                    "--record-output", str(out),
                    "--stop-pad", str(args.stop_pad),
                    "--max-markers", str(args.max_markers),
                    "--no-loop",
                ]
            print("+", " ".join(cmd), flush=True)
            child_env = env.copy()
            child_env["OPEN3D_CPU_RENDERING"] = "1"
            child = popen_session(cmd, env=child_env)
            code = child.wait()
            child = None
            if args.backend == "rviz":
                kill_stale_viz()
                restore_wall_clock_time()
            if code not in (0, None):
                print(f"[WARN] {trial.name}/{out_name} exited {code}; continuing")
            time.sleep(0.3)
        still = leftover_after
        if still > 0:
            print(
                f"\nBatch done ({len(jobs)} recorded). {still} left. "
                "Re-run the same command to continue."
            )
        else:
            print("\nAll done. No RViz videos left in this session.")
        return 0
    except KeyboardInterrupt:
        print("\n[stop] Ctrl+C — stopping current render...", flush=True)
        stop_proc(child, "render")
        if args.backend == "rviz":
            kill_stale_viz()
            restore_wall_clock_time()
        return 130
    finally:
        stop_proc(child, "render")
        if args.backend == "rviz":
            kill_stale_viz()
            restore_wall_clock_time()


if __name__ == "__main__":
    raise SystemExit(main())
