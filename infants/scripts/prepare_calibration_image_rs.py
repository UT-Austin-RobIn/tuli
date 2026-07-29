from pathlib import Path
import argparse

from misc_utils import extract_images_from_ros


def prompt_with_default(label, default_value):
    value = input(f"{label} [{default_value}]: ").strip()
    return value or default_value


def clear_jpgs(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for image_path in folder.glob("*.jpg"):
        image_path.unlink()


def trim_to_equal_count(left_folder: Path, right_folder: Path):
    """Drop trailing frames so left/right counts match (stereo-calib requires equal N)."""
    left = sorted(left_folder.glob("*.jpg"))
    right = sorted(right_folder.glob("*.jpg"))
    n = min(len(left), len(right))
    dropped = left[n:] + right[n:]
    for path in dropped:
        path.unlink()
    if dropped:
        print(
            f"[INFO] Trimmed to {n} pairs "
            f"(was L={len(left)}, R={len(right)}; dropped {len(dropped)})"
        )


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--folder_name", type=str, required=True,
                        help="Session folder, e.g. 26_05_09_infant_010")
    args.add_argument("--left_topic", type=str, default="/cam_L/color/image_raw",
                        help="Left ROS topic")
    args.add_argument("--right_topic", type=str, default="/cam_M/color/image_raw",
                        help="Right ROS topic")
    return args.parse_args()

def main():
    args = parse_args()
    root_folder_path = "/home/robotlearning2/infants/data/calibration_data"
    folder_name = args.folder_name
    rosbag_path = f"{root_folder_path}/{folder_name}/left_to_mid/ros.bag"

    left_topic = args.left_topic
    right_topic = args.right_topic

    left_output = Path(f"{root_folder_path}/{folder_name}/left_to_mid/left_images")
    right_output = Path(f"{root_folder_path}/{folder_name}/left_to_mid/right_images")

    clear_jpgs(left_output)
    clear_jpgs(right_output)

    extract_images_from_ros(
        time_str=None,
        topic_name=left_topic,
        rosbag_path=str(rosbag_path),
        output_folder=str(left_output),
    )
    extract_images_from_ros(
        time_str=None,
        topic_name=right_topic,
        rosbag_path=str(rosbag_path),
        output_folder=str(right_output),
    )
    trim_to_equal_count(left_output, right_output)

    print(
        f"[INFO] Extracted RealSense calibration images to {left_output} and {right_output}."
    )


if __name__ == "__main__":
    main()
