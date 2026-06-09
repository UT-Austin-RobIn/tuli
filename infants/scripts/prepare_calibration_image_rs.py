from pathlib import Path

from misc_utils import extract_images_from_ros


def prompt_with_default(label, default_value):
    value = input(f"{label} [{default_value}]: ").strip()
    return value or default_value


def clear_jpgs(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for image_path in folder.glob("*.jpg"):
        image_path.unlink()


def main():
    rosbag_path = Path(
        prompt_with_default(
            "ROS bag path",
            "/home/robotlearning2/infants/data/0/trial_001/trial_ros.bag",
        )
    ).expanduser()

    left_topic = prompt_with_default("Left ROS topic", "/cam_L/color/image_raw")
    right_topic = prompt_with_default("Right ROS topic", "/cam_M/color/image_raw")

    left_output = Path(
        prompt_with_default(
            "Left output folder",
            "/home/robotlearning2/stereo-calib/dataset/left",
        )
    ).expanduser()
    right_output = Path(
        prompt_with_default(
            "Right output folder",
            "/home/robotlearning2/stereo-calib/dataset/right",
        )
    ).expanduser()

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

    print(
        f"[INFO] Extracted RealSense calibration images to {left_output} and {right_output}."
    )


if __name__ == "__main__":
    main()
