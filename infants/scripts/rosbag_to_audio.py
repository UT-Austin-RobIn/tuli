#!/usr/bin/env python3
"""Extract /audio/audio from trial rosbags and write an MP4 audio file.

Concatenates MP3 chunks recorded by audio_capture, then transcodes to AAC in
an MP4 container. Applies a volume gain by default because bag playback is
often quiet.

Usage:
    python rosbag_to_audio.py /path/to/trial_ros.bag
    python rosbag_to_audio.py /path/to/subject_folder
    python rosbag_to_audio.py trial_ros.bag --gain-db 18 --output trial_audio.mp4
"""
import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import rosbag

DEFAULT_TOPIC = "/audio/audio"
DEFAULT_BAG_NAME = "trial_ros.bag"
DEFAULT_OUTPUT_NAME = "trial_audio.mp4"
TRIAL_RE = re.compile(r"^trial_\d{3}$")


def log(msg):
    print(msg, flush=True)


def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise SystemExit("[ERROR] ffmpeg not found. Install with: sudo apt install ffmpeg")


def resolve_bag_path(path: Path, bag_name: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        direct = path / bag_name
        if direct.is_file():
            return direct
    raise FileNotFoundError(f"Rosbag not found: {path} (expected file or {bag_name})")


def find_trial_dirs(root: Path, bag_name: str):
    if (root / bag_name).is_file():
        return [root]

    trials = sorted(
        p for p in root.iterdir()
        if p.is_dir() and TRIAL_RE.match(p.name) and (p / bag_name).is_file()
    )
    if not trials:
        raise FileNotFoundError(f"No {bag_name} found under {root}")
    return trials


def read_mp3_chunks(bag_path: Path, topic: str) -> bytes:
    chunks = bytearray()
    msg_count = 0
    with rosbag.Bag(str(bag_path), "r") as bag:
        if topic not in bag.get_type_and_topic_info().topics:
            raise ValueError(f"Topic {topic} not found in {bag_path}")

        for _, msg, _ in bag.read_messages(topics=[topic]):
            if msg.data:
                chunks.extend(msg.data)
                msg_count += 1

    if msg_count == 0:
        raise ValueError(f"No messages on {topic} in {bag_path}")
    return bytes(chunks), msg_count


def write_audio_mp4(mp3_bytes: bytes, output_path: Path, gain_db: float):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        mp3_path = Path(tmp.name)
        mp3_path.write_bytes(mp3_bytes)

    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(mp3_path),
            "-af",
            f"volume={gain_db}dB",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
    finally:
        mp3_path.unlink(missing_ok=True)


def process_bag(bag_path: Path, output_path: Path, topic: str, gain_db: float):
    log(f"Reading {bag_path}")
    mp3_bytes, msg_count = read_mp3_chunks(bag_path, topic)
    log(f"  {msg_count} audio messages ({len(mp3_bytes)} bytes MP3)")

    write_audio_mp4(mp3_bytes, output_path, gain_db)
    log(f"  wrote {output_path} (gain {gain_db:+.1f} dB)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export /audio/audio from a rosbag to an MP4 file."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a .bag file, trial folder, or subject folder with trial_XXX dirs",
    )
    parser.add_argument(
        "--bag-name",
        default=DEFAULT_BAG_NAME,
        help=f"Rosbag filename inside trial folders (default: {DEFAULT_BAG_NAME})",
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help=f"Audio topic to export (default: {DEFAULT_TOPIC})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4 path (default: trial_audio.mp4 next to each bag)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory when processing multiple trials",
    )
    parser.add_argument(
        "--gain-db",
        type=float,
        default=15.0,
        help="Volume boost applied during export in dB (default: 15)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    require_ffmpeg()

    path = args.path.expanduser().resolve()
    if path.is_file():
        bag_paths = [path]
    elif path.is_dir() and (path / args.bag_name).is_file():
        bag_paths = [path]
    elif path.is_dir():
        bag_paths = [trial / args.bag_name for trial in find_trial_dirs(path, args.bag_name)]
        log(f"Found {len(bag_paths)} trial bag(s) under {path}")
    else:
        raise SystemExit(f"Not found: {path}")

    for bag_path in bag_paths:
        if args.output and len(bag_paths) == 1:
            output_path = args.output.expanduser().resolve()
        elif args.output_dir:
            output_path = args.output_dir.expanduser().resolve() / f"{bag_path.parent.name}_{DEFAULT_OUTPUT_NAME}"
        else:
            output_path = bag_path.parent / DEFAULT_OUTPUT_NAME

        process_bag(bag_path, output_path, args.topic, args.gain_db)


if __name__ == "__main__":
    main()
