#!/usr/bin/env python3
"""Record a calibration rosbag for one camera/Qualisys pair.

Usage:
    record_for_calibration.py --left_to_qualisys  --folder_name yy_mm_dd_infant_XXX
    record_for_calibration.py --right_to_qualisys --folder_name yy_mm_dd_infant_XXX
    record_for_calibration.py --right_to_left     --folder_name yy_mm_dd_infant_XXX

Press SPACE to start, then SPACE again to stop.
Output: data/calibration_data/{folder_name}/{pair}/ros.bag
"""
import argparse
import os
import signal
import subprocess
import sys
import select
import termios
import threading
import tty
from pathlib import Path

try:
    from pynput import keyboard
except Exception as e:
    keyboard = None
    _PYNPUT_IMPORT_ERROR = e

DATA_ROOT = Path("/home/robotlearning2/infants/data/calibration_data")

PAIR_CAMS = {
    "left_to_qualisys": ["cam_L"],
    "right_to_qualisys": ["cam_R"],
    "right_to_left": ["cam_L", "cam_M"],
}


def flush_stdin():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def wait_for_keypress(target_key, action="continue"):
    if keyboard is None:
        raise RuntimeError(f"pynput not available: {_PYNPUT_IMPORT_ERROR}")
    print(f"Waiting for {target_key.name.upper()} press to {action}...")
    key_pressed = threading.Event()

    def on_press(key):
        if key == target_key:
            key_pressed.set()
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        key_pressed.wait()


def build_topics(cams):
    topics = []
    for cam in cams:
        topics += [
            f"/{cam}/color/image_raw",
            f"/{cam}/aligned_depth_to_color/image_raw",
            f"/{cam}/color/camera_info",
        ]
    topics += ["/tf", "/tf_static", "/clock"]
    return topics


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--left_to_qualisys", action="store_true")
    group.add_argument("--right_to_qualisys", action="store_true")
    group.add_argument("--right_to_left", action="store_true")
    parser.add_argument("--folder_name", required=True,
                        help="Session folder, e.g. 26_05_09_infant_010")
    args = parser.parse_args()

    if keyboard is None:
        print(f"[ERROR] pynput not available: {_PYNPUT_IMPORT_ERROR}")
        print("[ERROR] This script requires an X display for keyboard input.")
        return

    if args.left_to_qualisys:
        pair = "left_to_qualisys"
    elif args.right_to_qualisys:
        pair = "right_to_qualisys"
    else:
        pair = "right_to_left"

    out_dir = DATA_ROOT / args.folder_name / pair
    out_dir.mkdir(parents=True, exist_ok=True)
    bag_path = out_dir / "ros"
    topics = build_topics(PAIR_CAMS[pair])

    print("=" * 60)
    print(f"[INFO] Calibration pair: {pair}")
    print(f"[INFO] Cameras: {PAIR_CAMS[pair]}")
    print(f"[INFO] Output:  {bag_path}.bag")
    print(f"[INFO] Topics:  {topics}")
    print("=" * 60)

    wait_for_keypress(keyboard.Key.space, action="start recording")

    print("[INFO] Starting rosbag...")
    rosbag_proc = subprocess.Popen(
        ["rosbag", "record", "-b", "0", "-O", str(bag_path), "-q", *topics],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print("[INFO] START QUALISYS NOW")
    wait_for_keypress(keyboard.Key.space, action="stop recording")

    print("[INFO] Stopping rosbag (SIGINT to flush index)...")
    try:
        os.killpg(os.getpgid(rosbag_proc.pid), signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        rosbag_proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print("[WARN] rosbag did not exit in time. Killing forcibly...")
        rosbag_proc.kill()
        rosbag_proc.wait()

    flush_stdin()
    print(f"[OK] Saved {bag_path}.bag")


if __name__ == "__main__":
    main()
