#!/usr/bin/env python3
"""Extract image topics from trial rosbags and write MP4 videos.

Walks a subject folder for trial_XXX subdirectories, opens each trial_ros.bag,
and writes one MP4 per image topic.

Usage:
    python rosbag_to_videos.py /path/to/subject_folder
    python rosbag_to_videos.py /path/to/subject_folder --topics /cam_L/color/image_raw

NFS note: opening large bags on ~/synology-tuli can take minutes. Use
--local-cache to copy each bag to /tmp first, or write outputs locally with
--output-dir ~/infants/recordings/videos.
"""
import argparse
import re
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import rosbag
from cv_bridge import CvBridge


DEFAULT_TOPICS = [
    "/cam_L/color/image_raw",
    "/cam_M/color/image_raw",
    "/cam_R/color/image_raw",
]

TRIAL_RE = re.compile(r"^trial_\d{3}$")


def log(msg):
    print(msg, flush=True)


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


def topic_to_filename(topic: str) -> str:
    return topic.strip("/").replace("/", "_") + ".mp4"


def image_msg_to_bgr(msg, bridge: CvBridge):
    if msg.encoding == "16UC1":
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(img.astype("uint8"), cv2.COLOR_GRAY2BGR)
    return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


def open_bag(bag_path: Path):
    """Open a rosbag, with a visible progress message for slow NFS paths."""
    size_gb = bag_path.stat().st_size / (1024 ** 3)
    log(f"  opening {bag_path.name} ({size_gb:.1f} GB)...")
    t0 = time.time()
    bag = rosbag.Bag(str(bag_path), "r")
    log(f"  bag open took {time.time() - t0:.1f}s")
    return bag


def list_image_topics(bag_path: Path):
    with open_bag(bag_path) as bag:
        return [
            topic
            for topic, info in bag.get_type_and_topic_info().topics.items()
            if info.msg_type == "sensor_msgs/Image"
        ]


def resolve_bag_path(bag_path: Path, local_cache: bool):
    if not local_cache:
        return bag_path, None

    tmp_dir = Path(tempfile.mkdtemp(prefix="rosbag_cache_"))
    cached = tmp_dir / bag_path.name
    size_gb = bag_path.stat().st_size / (1024 ** 3)
    log(f"  copying bag to {cached} ({size_gb:.1f} GB)...")
    t0 = time.time()
    shutil.copy2(bag_path, cached)
    log(f"  copy took {time.time() - t0:.1f}s")
    return cached, tmp_dir


def write_trial_videos(bag_path, topics, output_dir, fps, max_frames, bridge):
    writers = {}
    frame_counts = {topic: 0 for topic in topics}

    with open_bag(bag_path) as bag:
        for topic, msg, _ in bag.read_messages(topics=topics):
            count = frame_counts[topic]
            if max_frames is not None and count >= max_frames:
                continue

            frame = image_msg_to_bgr(msg, bridge)
            writer = writers.get(topic)
            if writer is None:
                h, w = frame.shape[:2]
                output_path = output_dir / topic_to_filename(topic)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {output_path}")
                writers[topic] = writer

            writer.write(frame)
            frame_counts[topic] += 1

            if max_frames is not None and all(
                frame_counts[t] >= max_frames for t in topics
            ):
                break

    for writer in writers.values():
        writer.release()

    return frame_counts


def process_trial(trial_dir, bag_name, topics, output_dir, fps, max_frames, local_cache):
    bag_path = trial_dir / bag_name
    out_dir = output_dir or trial_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bridge = CvBridge()
    log(f"\n{trial_dir.name}: {bag_path}")

    cached_bag, tmp_dir = resolve_bag_path(bag_path, local_cache)
    try:
        frame_counts = write_trial_videos(
            cached_bag, topics, out_dir, fps, max_frames, bridge
        )
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    for topic in topics:
        output_path = out_dir / topic_to_filename(topic)
        n = frame_counts.get(topic, 0)
        if n == 0:
            log(f"  SKIP {topic}: no frames")
            if output_path.exists():
                output_path.unlink()
            continue
        log(f"  wrote {output_path} ({n} frames)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MP4 videos from image topics in trial rosbags."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Subject folder containing trial_XXX subfolders (or a single trial folder)",
    )
    parser.add_argument(
        "--bag-name",
        default="trial_ros.bag",
        help="Rosbag filename inside each trial folder (default: trial_ros.bag)",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="Image topics to export (default: cam L/M/R color image_raw)",
    )
    parser.add_argument(
        "--all-image-topics",
        action="store_true",
        help="Export every sensor_msgs/Image topic found in the first trial bag",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video frame rate (default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for MP4 files (default: write into each trial folder)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N frames per topic (useful for quick tests)",
    )
    parser.add_argument(
        "--local-cache",
        action="store_true",
        help="Copy each bag to /tmp before reading (recommended for NFS bags)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.folder.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    trials = find_trial_dirs(root, args.bag_name)
    log(f"Found {len(trials)} trial folder(s) under {root}")

    if args.all_image_topics:
        topics = sorted(list_image_topics(trials[0] / args.bag_name))
        if not topics:
            raise SystemExit(f"No image topics found in {trials[0] / args.bag_name}")
        log(f"Image topics: {topics}")
    else:
        topics = args.topics or DEFAULT_TOPICS

    for trial_dir in trials:
        process_trial(
            trial_dir,
            args.bag_name,
            topics,
            args.output_dir,
            args.fps,
            args.max_frames,
            args.local_cache,
        )


if __name__ == "__main__":
    main()
