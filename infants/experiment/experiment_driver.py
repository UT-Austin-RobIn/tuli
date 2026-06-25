import argparse
import subprocess
import time
import yaml
import csv
import shutil
import threading
import re
import sys
import termios
import tty
import select
import rospy
import paramiko

from datetime import datetime
from pathlib import Path
try:
    from pynput import keyboard
except Exception as e:
    keyboard = None
    _PYNPUT_IMPORT_ERROR = e
from std_msgs.msg import String

DATA_ROOT = Path("/home/robotlearning2/infants/data")
AUDIO_RATE = 44100

CONDITIONS = {
    'bang': """
    1. Soft Board - Sponge Cube			(low haptics,  low audio )
    2. Soft Board - Soft Cube 		(low haptics,  high audio)
    3. Soft Board - Hard Cube 			(high haptics, low audio )
    4. Hard Board - Sponge Cube 		(high haptics, high audio)
    5. Hard Board - Soft Cube			(high haptics, low audio )
    6. Hard Board - Hard cube 		(high haptics, high audio)
    """,
    'slide': """
    1. washboard-sphere         (high haptic, high audio)
    2. soft-sphere              (high haptic, low audio)
    3. soft-rattle              (low haptic, high audio)
    4. washboard-sphere-muffled (high haptic, low audio)
    """,
    'hammer': ''
}

NAS_PATH = Path("/home/robotlearning2/synology-tuli")  # Adjust if needed

WINDOWS_HOSTNAME = "192.168.253.101"
WINDOWS_USERNAME = "ut austin"
WINDOWS_PASSWORD = "1234"

WindowsClient = paramiko.SSHClient()
WindowsClient.set_missing_host_key_policy(paramiko.AutoAddPolicy())

def get_ntp_offset(retries=3, timeout_s=5):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            proc = subprocess.run(
                ["chronyc", "tracking"],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=True,
            )
            output = proc.stdout
            match = re.search(r"Last offset\\s*:\\s*([+-]?[0-9.]+)\\s*(seconds|s|ms|us|ns)", output)
            if not match:
                raise RuntimeError(f"chronyc tracking parse failed; output was: {output!r}")
            value = float(match.group(1))
            unit = match.group(2)
            if unit == "seconds" or unit == "s":
                pass
            elif unit == "ms":
                value /= 1_000.0
            elif unit == "us":
                value /= 1_000_000.0
            elif unit == "ns":
                value /= 1_000_000_000.0
            return value
        except Exception as e:
            last_error = e
            print(f"[WARN] NTP offset attempt {attempt}/{retries} failed: {e}")
            time.sleep(0.5)
    raise RuntimeError(f"NTP offset failed after {retries} attempts: {last_error}")

def copy_and_cleanup_background(subject_path: Path, delete_local: bool = False):
    """Spawn a detached background process to copy data to NAS."""
    dest_path = NAS_PATH / subject_path.name
    log_file = subject_path.parent / f".transfer_{subject_path.name}.log"

    script = f"""
import subprocess, shutil, sys
from pathlib import Path

subject = Path("{subject_path}")
dest = Path("{dest_path}")
delete_local = {delete_local}
log = open("{log_file}", "a")
sys.stdout = log
sys.stderr = log

print("[BG] Starting rsync...")
result = subprocess.run(
    ["rsync", "-av", "--no-group", str(subject) + "/", str(dest) + "/"],
    capture_output=True, text=True, timeout=3600
)
if result.returncode != 0:
    print(f"[BG ERROR] rsync failed (code {{result.returncode}}): {{result.stderr}}")
    sys.exit(1)

print("[BG] rsync complete. Verifying sizes...")
local_size = sum(f.stat().st_size for f in subject.rglob("*") if f.is_file())
remote_size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())

if remote_size < local_size * 0.99:
    print(f"[BG ERROR] Size mismatch: local={{local_size}}, NAS={{remote_size}}. Keeping local data.")
    sys.exit(1)

print(f"[BG] Verified (local={{local_size}}, NAS={{remote_size}}).")
if delete_local:
    print("[BG] Removing local data...")
    shutil.rmtree(subject)
    print("[BG] Done. Local data removed.")
else:
    print("[BG] Done. Local data kept.")
"""
    subprocess.Popen(
        [sys.executable, "-c", script],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"[INFO] Background transfer started for {subject_path.name}.")
    if delete_local:
        print("[INFO] Local data will be removed after successful transfer.")
    else:
        print("[INFO] Local data will be kept after transfer.")
    print(f"[INFO] Check progress: cat {log_file}")

def flush_stdin():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def get_next_trial_number(subject_path: Path):
    trial_dirs = [
        p for p in subject_path.iterdir()
        if p.is_dir() and re.match(r"trial_\d{3}", p.name)
    ]
    if not trial_dirs:
        return 1
    trial_numbers = [int(p.name.split('_')[1]) for p in trial_dirs]
    return max(trial_numbers) + 1


import json

class ButtonLogger:
    def __init__(self):
        if keyboard is None:
            raise RuntimeError(f"pynput not available: {_PYNPUT_IMPORT_ERROR}")
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.active = False

        if not rospy.core.is_initialized():
            rospy.init_node('keyboard_logger', anonymous=True)

        self.keydown_pub = rospy.Publisher('/keydown', String, queue_size=10)
        self.keyup_pub = rospy.Publisher('/keyup', String, queue_size=10)

    def _on_press(self, key):
        if not self.active:
            return
        try:
            msg = json.dumps({
                "action": "keydown",
                "key": self._key_to_str(key),
                "timestamp": datetime.now().isoformat()
            })
            print(f"[KeyDown] {msg}")
            self.keydown_pub.publish(msg)
        except Exception as e:
            print(f"[ERROR] Exception in _on_press: {e}")

    def _on_release(self, key):
        if not self.active:
            return
        try:
            msg = json.dumps({
                "action": "keyup",
                "key": self._key_to_str(key),
                "timestamp": datetime.now().isoformat()
            })
            print(f"[KeyUp] {msg}")
            self.keyup_pub.publish(msg)
        except Exception as e:
            print(f"[ERROR] Exception in _on_release: {e}")

    def _key_to_str(self, key):
        if isinstance(key, keyboard.Key):
            return key.name
        else:
            return str(key).strip("'")

    def start(self):
        self.active = True
        self.listener.start()

    def stop(self):
        self.active = False
        self.listener.stop()
        self.listener.join(timeout=2)
            

def make_subject_dir(subject_id: str) -> Path:
    subject_path = DATA_ROOT / subject_id
    subject_path.mkdir(parents=True, exist_ok=True)
    return subject_path

def make_timestamp_subject_id() -> str:
    """
    Generate a human-readable, filesystem-safe timestamp for subject/session folders.
    Example: 2026-05-01_09-58-00
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def wait_for_keypress(target_key):
    if keyboard is None:
        raise RuntimeError(f"pynput not available: {_PYNPUT_IMPORT_ERROR}")
    print(f"Waiting for {target_key.name.upper()} press to continue...")
    key_pressed = threading.Event()

    def on_press(key):
        if key == target_key:
            key_pressed.set()
            return False  # Stop listener

    with keyboard.Listener(on_press=on_press) as listener:
        key_pressed.wait()


def run_trial(subject_path: Path, trial_number: int, condition_name: str, task_name: str, use_button_log=False):
    trial_name = f"trial_{trial_number:03d}"
    trial_path = subject_path / trial_name
    trial_path.mkdir(parents=True, exist_ok=True)

    rosbag_file = trial_path / "trial_ros"
    metadata_file = trial_path / "trial_metadata.yaml"

    metadata = {
        "trial_name": trial_name,
        "condition": condition_name,
        "start_time": None,
        "end_time": None,
        "task": task_name,
    }

    print("[DEBUG] Waiting for SPACE to start trial...")
    wait_for_keypress(keyboard.Key.space)
    metadata["start_time"] = datetime.now().isoformat()
    # Obtain the NTP offset
    print("[INFO] Obtaining NTP offset...")
    try:
        ntp_offset = get_ntp_offset()
        metadata["ntp_offset"] = ntp_offset
        print(f"[DEBUG] NTP offset: {ntp_offset}")
    except Exception as e:
        metadata["ntp_offset"] = None
        print(f"[WARN] Failed to obtain NTP offset: {e}")
    
    print(f"[DEBUG] Trial started at {metadata['start_time']}")

    # Start rosbag
    print("[INFO] Starting rosbag...")
    rosbag_proc = subprocess.Popen([
        "rosbag", "record", "-b", "0",
        "/cam_L/color/image_raw",
        "/cam_L/aligned_depth_to_color/image_raw",
        "/cam_L/color/camera_info",
        "/cam_R/color/image_raw",
        "/cam_R/aligned_depth_to_color/image_raw",
        "/cam_R/color/camera_info",
        "/cam_M/color/image_raw",
        "/cam_M/aligned_depth_to_color/image_raw",
        "/cam_M/color/camera_info",
        "/tf", "/tf_static", "/clock",
        "/audio/audio",
	"/keydown", "/keyup",
	"-O", str(rosbag_file),
	"-q"
    ], start_new_session=True)
    print("[DEBUG] rosbag process started")

    button_logger = ButtonLogger() if use_button_log else None
    if button_logger:
        print("[DEBUG] Starting button logger...")
        button_logger.start()
        print("[DEBUG] Button logger started")

    print("[INFO] START QUALISYS NOW")
    print("[DEBUG] Waiting for SPACE to end trial...")
    wait_for_keypress(keyboard.Key.space)
    metadata["end_time"] = datetime.now().isoformat()
    print(f"[DEBUG] Trial ended at {metadata['end_time']}")

    print("[INFO] Stopping rosbag...")
    rosbag_proc.terminate()
    try:
        rosbag_proc.wait(timeout=5)
        print("[DEBUG] rosbag terminated cleanly")
    except subprocess.TimeoutExpired:
        print("[WARN] rosbag did not terminate in time. Killing forcibly...")
        rosbag_proc.kill()
        rosbag_proc.wait()
        print("[DEBUG] rosbag process killed")

    if button_logger:
        print("[DEBUG] Stopping button logger...")
        button_logger.stop()
        print("[DEBUG] Button logger stopped")

    print("[DEBUG] Writing trial metadata to YAML...")
    with open(metadata_file, "w") as f:
        yaml.dump(metadata, f)
    print("[DEBUG] Metadata written")

    print()  # Ensure newline before returning
    return metadata

def append_to_csv(subject_path: Path, metadata_dict):
    csv_path = subject_path / "metadata.csv"
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metadata_dict.keys())
        if is_new:
            writer.writeheader()
        writer.writerow(metadata_dict)


def parse_args():
    parser = argparse.ArgumentParser(description="Run infant experiment trials.")
    parser.add_argument(
        "--delete-local-after-transfer",
        action="store_true",
        help="Remove the local subject folder after a successful NAS transfer",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rospy.init_node("experiment_driver", anonymous=True, disable_signals=True)
    if keyboard is None:
        print(f"[ERROR] pynput not available: {_PYNPUT_IMPORT_ERROR}")
        print("[ERROR] This script requires an X display for keyboard input.")
        return
    subject_id = make_timestamp_subject_id()
    print(f"[INFO] Subject ID (timestamp): {subject_id}")
    subject_path = make_subject_dir(subject_id)
    task = input("Enter task name [bang/slide/hammer]: ").strip()
    if task not in CONDITIONS:
        print(f"[WARN] Unknown task '{task}', defaulting to 'bang'")
        task = "bang"

    print("\nReady to begin trials. Press Ctrl+C to exit.\n")

    trial_number = get_next_trial_number(subject_path)
    try:
        while True:
            print('Conditions:')
            print(CONDITIONS[task])
            try:
                condition = input(f"[Trial {trial_number}] Enter condition number: ").strip()
            except KeyboardInterrupt:
                print("\n[INFO] Experiment interrupted during condition input.")
                break  # Exit trials loop gracefully

            metadata = run_trial(
                subject_path,
                trial_number,
                condition,
                task,
                use_button_log=condition.strip() in ["7", "8"]
            )

            flush_stdin()
            keep = input("\nKeep trial? (y/n): ").strip().lower()
            if keep == "y":
                append_to_csv(subject_path, metadata)
                print("[INFO] Trial saved.\n")
            else:
                trial_path = subject_path / f"trial_{trial_number:03d}"
                if trial_path.exists():
                    shutil.rmtree(trial_path)
                    print(f"[INFO] Deleted trial folder: {trial_path}\n")
                else:
                    print(f"[WARN] Trial path not found: {trial_path}\n")

            # New input to decide whether to continue or end experiment
            cont = input("Record another condition? (y/n): ").strip().lower()
            if cont != 'y':
                print("[INFO] Ending experiment as per user request.")
                break

            trial_number += 1

    except KeyboardInterrupt:
        print("\n[INFO] Experiment ended by user.")

    # Copy to NAS in background so user can start next experiment immediately
    copy_and_cleanup_background(
        subject_path,
        delete_local=args.delete_local_after_transfer,
    )


if __name__ == "__main__":
    main()
