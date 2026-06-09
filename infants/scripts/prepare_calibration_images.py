from pathlib import Path

from misc_utils import build_paired_dataset, extract_images_from_ros, extract_images_from_video


def prompt_with_default(label, default_value):
    value = input(f"{label} [{default_value}]: ").strip()
    return value or default_value


def main():
    video_path = Path(prompt_with_default("Qualisys video path", "qualisys_video.avi")).expanduser()
    rosbag_path = Path(
        prompt_with_default(
            "ROS bag path",
            "/home/robotlearning2/infants/data/0/trial_001/trial_ros.bag",
        )
    ).expanduser()

    qualisys_output = Path(prompt_with_default("Qualisys output folder", "qualisys_camera_images")).expanduser()
    ros_output = Path(prompt_with_default("ROS output folder", "rs_images")).expanduser()
    topic_name = prompt_with_default("ROS topic", "/cam_L/color/image_raw")
    resize_qualisys = prompt_with_default("Crop Qualisys image to match the size of Realsense image?", "y").lower() in {"y", "yes"}

    extract_images_from_video(
        video_path=str(video_path),
        output_folder=str(qualisys_output),
        resize_frame=resize_qualisys,
    )

    match_frames = prompt_with_default("Match frames? (y/n)", "n").lower() in {"y", "yes"}
    if match_frames:
        qualisys_start_time = input("Qualisys start time [YYYY-MM-DD, HH:MM:SS.mmm]: ").strip()
        if not qualisys_start_time:
            raise ValueError("Qualisys start time is required for frame matching")

        match = extract_images_from_ros(
            time_str=qualisys_start_time,
            topic_name=topic_name,
            rosbag_path=str(rosbag_path),
            output_folder=str(ros_output),
        )
        left_output = Path(prompt_with_default("Paired left output folder", "/home/robotlearning2/stereo-calib/dataset/left")).expanduser()
        right_output = Path(prompt_with_default("Paired right output folder", "/home/robotlearning2/stereo-calib/dataset/right")).expanduser()
        build_paired_dataset(
            qualisys_source_folder=str(qualisys_output),
            ros_source_folder=str(ros_output),
            ros_timestamps=match["saved_timestamps"],
            qualisys_start_time=qualisys_start_time,
            left_output_folder=str(left_output),
            right_output_folder=str(right_output),
            qualisys_fps=30.0,
        )
        print(
            "Alignment summary: "
            f"closest ROS frame {match['frame_idx']}, "
            f"timestamp diff {match['diff_sec']:.6f}s"
        )
    else:
        extract_images_from_ros(
            time_str=None,
            topic_name=topic_name,
            rosbag_path=str(rosbag_path),
            output_folder=str(ros_output),
        )
        print(f"[INFO] Images extracted to {qualisys_output} and {ros_output}. Skipping frame matching.")


if __name__ == "__main__":
    main()
