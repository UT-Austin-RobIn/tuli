import csv
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

import pytz
import rosbag


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_BAG = "/home/robotlearning2/synology-tuli/030/trial_001/trial_ros.bag"
TOPIC = "/cam_L/color/image_raw"
QUALISYS_START_TIME = "2026-03-30, 18:03:44.964"
QUALISYS_FPS = 30.0
QUALISYS_EXTRA_FRAMES = 500
LOCAL_TZ = pytz.timezone("America/Chicago")

#ROS_CSV = SCRIPT_DIR / "ros_frame_timestamps.csv"
#QUALISYS_CSV = SCRIPT_DIR / "qualisys_frame_timestamps.csv"
MATCHED_CSV = SCRIPT_DIR / "matched_timestamp_5_minutes.csv"


def parse_local_time(time_str):
    dt = datetime.strptime(time_str, "%Y-%m-%d, %H:%M:%S.%f")
    return LOCAL_TZ.localize(dt).timestamp()


def format_local_time(unix_ts):
    return datetime.fromtimestamp(unix_ts, LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S.%f %Z")


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_ros_frames():
    rows = []
    with rosbag.Bag(INPUT_BAG, "r") as bag:
        for frame_idx, (_, _, ros_time) in enumerate(bag.read_messages(topics=[TOPIC])):
            unix_ts = ros_time.to_sec()
            rows.append(
                {
                    "ros_frame": frame_idx,
                    "unix_timestamp": f"{unix_ts:.6f}",
                    "local_time": format_local_time(unix_ts),
                }
            )
    return rows


def load_qualisys_frames(num_frames, start_unix_time):
    rows = []
    frame_period = 1.0 / QUALISYS_FPS
    for frame_idx in range(num_frames):
        unix_ts = start_unix_time + (frame_idx * frame_period)
        rows.append(
            {
                "qualisys_frame": frame_idx,
                "unix_timestamp": f"{unix_ts:.6f}",
                "local_time": format_local_time(unix_ts),
            }
        )
    return rows


def closest_index(sorted_values, target):
    insert_idx = bisect_left(sorted_values, target)
    if insert_idx == 0:
        return 0
    if insert_idx >= len(sorted_values):
        return len(sorted_values) - 1

    before_idx = insert_idx - 1
    after_idx = insert_idx
    before_diff = abs(sorted_values[before_idx] - target)
    after_diff = abs(sorted_values[after_idx] - target)
    if after_diff < before_diff:
        return after_idx
    return before_idx


def build_matches(ros_rows, qualisys_rows):
    qualisys_times = [float(row["unix_timestamp"]) for row in qualisys_rows]
    matched_rows = []
    for ros_row in ros_rows:
        ros_time = float(ros_row["unix_timestamp"])
        ros_frame = int(ros_row["ros_frame"])
        qual_idx = closest_index(qualisys_times, ros_time)
        qual_frame = int(qualisys_rows[qual_idx]["qualisys_frame"])
        time_diff = qualisys_times[qual_idx] - ros_time
        frame_diff = qual_frame - ros_frame
        matched_rows.append(
            {
                "ros_frame": ros_frame,
                "time_difference_sec": f"{time_diff:.6f}",
                "qualisys_frame": qual_frame,
                "frame_difference": frame_diff,
            }
        )
    return matched_rows


def main():
    qualisys_start_unix = parse_local_time(QUALISYS_START_TIME)
    ros_rows = load_ros_frames()
    qualisys_rows = load_qualisys_frames(len(ros_rows) + QUALISYS_EXTRA_FRAMES, qualisys_start_unix)
    matched_rows = build_matches(ros_rows, qualisys_rows)

    #write_csv(ROS_CSV, ["ros_frame", "unix_timestamp", "local_time"], ros_rows)
    #write_csv(QUALISYS_CSV, ["qualisys_frame", "unix_timestamp", "local_time"], qualisys_rows)
    write_csv(
        MATCHED_CSV,
        ["ros_frame", "time_difference_sec", "qualisys_frame", "frame_difference"],
        matched_rows,
    )

    #print(f"Wrote {len(ros_rows)} ROS frame timestamps to {ROS_CSV}")
    #print(f"Wrote {len(qualisys_rows)} Qualisys frame timestamps to {QUALISYS_CSV}")
    print(f"Wrote {len(matched_rows)} matched frame rows to {MATCHED_CSV}")


if __name__ == "__main__":
    main()
