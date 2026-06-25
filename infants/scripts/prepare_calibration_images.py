from pathlib import Path
import argparse
from misc_utils import build_paired_dataset, extract_images_from_ros, extract_images_from_video


def prompt_with_default(label, default_value):
    value = input(f"{label} [{default_value}]: ").strip()
    return value or default_value

def read_qualisys_start_time(tsv_path):
    for line in Path(tsv_path).read_text().splitlines():
        if line.startswith("TIME_STAMP\t"):
            return line.split("\t", 2)[1].strip()
    raise ValueError(f"TIME_STAMP not found in {tsv_path}")

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--folder_name", type=str, required=True,
                        help="Session folder, e.g. 26_05_09_infant_010")
    args.add_argument("--type", type=str, required=True,
                        choices=["left_to_qualisys", "right_to_qualisys"],
                        help="Calibration type")
    args.add_argument("--left_topic", type=str, default="/cam_L/color/image_raw",
                        help="Left ROS topic")
    args.add_argument("--right_topic", type=str, default="/cam_R/color/image_raw",
                        help="Right ROS topic")
    return args.parse_args()

def main():
    args = parse_args()
    root_folder_path = "/home/robotlearning2/infants/data/calibration_data"
    folder_name = args.folder_name
    video_path = f"{root_folder_path}/{folder_name}/{args.type}/{folder_name}_{args.type}_Miqus_1_31039.avi"
    rosbag_path = f"{root_folder_path}/{folder_name}/{args.type}/ros.bag"

    qualisys_output = f"{root_folder_path}/{folder_name}/{args.type}/qualisys_images"
    ros_output = f"{root_folder_path}/{folder_name}/{args.type}/rs_images"
    topic_name = args.left_topic if args.type == "left_to_qualisys" else args.right_topic
    resize_qualisys = True

    extract_images_from_video(
        video_path=str(video_path),
        output_folder=str(qualisys_output),
        resize_frame=resize_qualisys,
    )

    match_frames = True
    if match_frames:
        tsv_path = f"{root_folder_path}/{folder_name}/{args.type}/{folder_name}_{args.type}.tsv"
        qualisys_start_time = read_qualisys_start_time(tsv_path)
        print(f"Qualisys start time: {qualisys_start_time}")

        match = extract_images_from_ros(
            time_str=qualisys_start_time,
            topic_name=topic_name,
            rosbag_path=str(rosbag_path),
            output_folder=str(ros_output),
        )
        left_output = f"{root_folder_path}/{folder_name}/{args.type}/left_images"
        right_output = f"{root_folder_path}/{folder_name}/{args.type}/right_images"
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
