#!/usr/bin/env python3
"""Mux ROS bag /audio/audio onto Qualisys Miqus AVI videos.

The Miqus AVIs are silent. Experiment audio lives in trial_ros.bag.
This script extracts that audio and writes one MP4 per AVI.

Alignment (wall-clock, NTP-synced machines):
  Prefer Qualisys TSV header TIME_STAMP as video t=0.
  Else align AVI start with the first /audio/audio message (offset 0).
  Fine-tune with --offset-sec (positive delays audio).

Examples:
  # All Miqus AVIs in a trial folder:
  python infants/scripts/mux_miqus_with_bag_audio.py \\
      --trial-dir data/2026-07-16_11-03-27/trial_001

  # One file:
  python infants/scripts/mux_miqus_with_bag_audio.py \\
      --avi data/.../trial_001/26_07_16_019_1_Miqus_1_31039.avi \\
      --bag data/.../trial_001/trial_ros.bag \\
      --tsv data/.../trial_001/26_07_16_019_1.tsv
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pytz

try:
    import rosbag
except ImportError as exc:
    raise SystemExit(
        "rosbag not available. Activate the infants env: "
        "source ~/envs/infants/bin/activate"
    ) from exc

DEFAULT_AUDIO_TOPIC = "/audio/audio"
DEFAULT_TZ = "America/Chicago"
MIQUS_GLOB = "*Miqus*.avi"


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("[ERROR] ffmpeg not found. Install with: sudo apt install ffmpeg")


def load_audio_mp3(
    bag_path: Path, topic: str = DEFAULT_AUDIO_TOPIC
) -> Tuple[Optional[bytes], Optional[float]]:
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


def parse_tsv_start(tsv_path: Path, tz_name: str = DEFAULT_TZ) -> float:
    """Unix time of Qualisys recording start from TSV header row 7."""
    with open(tsv_path, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    if len(rows) < 8:
        raise ValueError(f"TSV header too short: {tsv_path}")
    start_str = rows[7][1]
    local_tz = pytz.timezone(tz_name)
    dt = datetime.strptime(start_str, "%Y-%m-%d, %H:%M:%S.%f")
    return local_tz.localize(dt).timestamp()


def find_trial_tsv(trial_dir: Path) -> Optional[Path]:
    cands = sorted(trial_dir.glob("*.tsv"))
    # Prefer trial marker TSVs over anything odd.
    preferred = [p for p in cands if "qualisys" not in p.name.lower()]
    pool = preferred or cands
    return pool[0] if pool else None


def find_bag(trial_dir: Path) -> Path:
    for name in ("trial_ros.bag", "trial_ros_combined.bag"):
        p = trial_dir / name
        if p.is_file():
            return p
    bags = sorted(trial_dir.glob("*.bag"))
    if not bags:
        raise FileNotFoundError(f"No .bag in {trial_dir}")
    return bags[0]


def mux_one(
    avi_path: Path,
    mp3_path: Path,
    output_path: Path,
    audio_offset_sec: float,
) -> None:
    """AVI t=0 + audio with offset (same convention as overlay_markers_on_image)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(avi_path),
    ]
    if audio_offset_sec > 0:
        # Delay audio relative to video.
        cmd += ["-itsoffset", f"{audio_offset_sec:.6f}", "-i", str(mp3_path)]
    elif audio_offset_sec < 0:
        # Trim leading audio so it starts with video.
        cmd += ["-ss", f"{-audio_offset_sec:.6f}", "-i", str(mp3_path)]
    else:
        cmd += ["-i", str(mp3_path)]
    cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Mux bag /audio/audio onto Qualisys Miqus AVI videos."
    )
    p.add_argument(
        "--trial-dir",
        type=Path,
        help="Trial folder containing *Miqus*.avi and trial_ros.bag",
    )
    p.add_argument("--avi", type=Path, action="append", help="One Miqus AVI (repeatable)")
    p.add_argument("--bag", type=Path, help="ROS bag with /audio/audio")
    p.add_argument(
        "--tsv",
        type=Path,
        help="Qualisys TSV (uses header TIME_STAMP to align video start)",
    )
    p.add_argument(
        "--audio-topic",
        default=DEFAULT_AUDIO_TOPIC,
        help=f"Bag audio topic (default: {DEFAULT_AUDIO_TOPIC})",
    )
    p.add_argument(
        "--tz",
        default=DEFAULT_TZ,
        help=f"Timezone for TSV TIME_STAMP (default: {DEFAULT_TZ})",
    )
    p.add_argument(
        "--offset-sec",
        type=float,
        default=0.0,
        help="Extra audio delay in seconds (positive = audio later). Added on top of TSV align.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for MP4s (default: same folder as each AVI)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    require_ffmpeg()

    avis: list[Path] = []
    bag_path: Optional[Path] = None
    tsv_path: Optional[Path] = args.tsv.expanduser().resolve() if args.tsv else None

    if args.trial_dir:
        trial_dir = args.trial_dir.expanduser().resolve()
        if not trial_dir.is_dir():
            raise SystemExit(f"Trial dir not found: {trial_dir}")
        avis = sorted(trial_dir.glob(MIQUS_GLOB))
        if not avis:
            raise SystemExit(f"No {MIQUS_GLOB} in {trial_dir}")
        bag_path = args.bag.expanduser().resolve() if args.bag else find_bag(trial_dir)
        if tsv_path is None:
            tsv_path = find_trial_tsv(trial_dir)
    else:
        if not args.avi or not args.bag:
            raise SystemExit("Provide --trial-dir, or both --avi and --bag")
        avis = [p.expanduser().resolve() for p in args.avi]
        bag_path = args.bag.expanduser().resolve()

    for avi in avis:
        if not avi.is_file():
            raise SystemExit(f"AVI not found: {avi}")
    if bag_path is None or not bag_path.is_file():
        raise SystemExit(f"Bag not found: {bag_path}")

    print(f"Bag:   {bag_path}")
    print(f"AVIs:   {len(avis)}")
    for avi in avis:
        print(f"  - {avi.name}")

    audio_bytes, audio_t0 = load_audio_mp3(bag_path, args.audio_topic)
    if audio_bytes is None or audio_t0 is None:
        raise SystemExit(f"No audio on {args.audio_topic} in {bag_path}")

    # audio_offset = when audio should start on the video timeline (t=0 = AVI start).
    if tsv_path is not None and tsv_path.is_file():
        video_t0 = parse_tsv_start(tsv_path, args.tz)
        audio_offset = (audio_t0 - video_t0) + float(args.offset_sec)
        print(f"TSV:    {tsv_path}")
        print(f"Align:  TSV start → audio_t0 (offset={audio_offset:+.3f}s)")
    else:
        audio_offset = float(args.offset_sec)
        if tsv_path is None:
            print("TSV:    (none) — aligning AVI start with first bag audio message")
        else:
            print(f"TSV:    missing ({tsv_path}) — aligning AVI start with first bag audio")
        if args.offset_sec:
            print(f"Align:  offset-sec={audio_offset:+.3f}s")
        else:
            print("Align:  0.000s (use --tsv or --offset-sec to refine)")

    tmp_mp3 = Path(tempfile.mkstemp(prefix="miqus_audio_", suffix=".mp3")[1])
    try:
        tmp_mp3.write_bytes(audio_bytes)
        for avi in avis:
            out_dir = (
                args.output_dir.expanduser().resolve()
                if args.output_dir
                else avi.parent
            )
            out_path = out_dir / f"{avi.stem}_with_audio.mp4"
            print(f"[mux] {avi.name} -> {out_path.name}")
            mux_one(avi, tmp_mp3, out_path, audio_offset)
            print(f"  saved {out_path}")
    finally:
        tmp_mp3.unlink(missing_ok=True)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
