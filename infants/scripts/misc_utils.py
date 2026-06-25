import cv2
import os
import rosbag
import pytz
import shutil
from bisect import bisect_left
from pathlib import Path
from cv_bridge import CvBridge
from datetime import datetime

# def resize_qualisys_frame(img):
#     h, w = img.shape[:2]   # h=544, w=736

#     # Compute target 4:3 crop
#     target_w = int(h * 4 / 3)  # 544 * 4/3 ≈ 725
#     target_h = h               # keep full height

#     # Center crop width to target_w
#     start_x = (w - target_w) // 2
#     end_x = start_x + target_w
#     img_cropped = img[:, start_x:end_x]   # shape ~ (544, 724, 3)

#     # Resize to 640x480
#     img_resized = cv2.resize(img_cropped, (640, 480), interpolation=cv2.INTER_AREA)
    
#     return img_resized

def resize_qualisys_frame(img):
    h, w = img.shape[:2]   # h=544, w=736

    # Compute target 4:3 crop
    target_w = w  # 544 * 4/3 ≈ 725
    target_h = int(w * 9 / 16)               # keep full height

    # Center crop width to target_w
    start_x = (h - target_h) // 2
    end_x = start_x + target_h
    img_cropped = img[start_x:end_x, :]   # shape ~ (544, 724, 3)
    img_resized = cv2.resize(img_cropped, (1280, 720), interpolation=cv2.INTER_AREA)
    return img_resized

def extract_images_from_video(video_path="qualisys_video.avi", output_folder="qualisys_camera_images", resize_frame=True):
    # === Configuration ===
    frame_interval = 1                      # Save every frame (use higher value to skip frames)

    # === Create output folder if it doesn't exist ===
    os.makedirs(output_folder, exist_ok=True)

    # === Open the video ===
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break  # End of video

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f'{saved_count:04d}.jpg')
            if resize_frame:
                frame = resize_qualisys_frame(frame)
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames to '{output_folder}'")


def save_frame(msg, count, output_folder):
    bridge = CvBridge()
    # Convert ROS Image message to OpenCV image
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    # Build filename like 0000.jpg, 0001.jpg ...
    filename = os.path.join(output_folder, f"{count:04d}.jpg")

    # Save image
    cv2.imwrite(filename, cv_img)
    # print(f"Saved {filename}")

def qualisys_time_to_unix(time_str, timezone_name="America/Chicago"):
    local_tz = pytz.timezone(timezone_name)
    dt = datetime.strptime(time_str, "%Y-%m-%d, %H:%M:%S.%f")
    localized_dt = local_tz.localize(dt)
    return localized_dt.timestamp()

def find_closest_ros_frame_start(target_unix_time,
                                 topic_name="/cam_L/color/image_raw",
                                 rosbag_path="/home/robotlearning2/infants/data/0/trial_001/trial_ros.bag"):
    best_diff = None
    best_frame_idx = None
    best_ros_time = None

    with rosbag.Bag(rosbag_path) as bag:
        frame_idx = 0
        for topic, _, t in bag.read_messages(topics=[topic_name]):
            ros_time = t.to_sec()
            diff = abs(target_unix_time - ros_time)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_frame_idx = frame_idx
                best_ros_time = ros_time
            frame_idx += 1

    if best_frame_idx is None:
        raise RuntimeError(f"No frames found for topic {topic_name} in {rosbag_path}")

    return {
        "frame_idx": best_frame_idx,
        "ros_time": best_ros_time,
        "diff_sec": best_diff,
    }

def extract_images_from_ros(time_str=None,
                            topic_name="/cam_L/color/image_raw",
                            rosbag_path="/home/robotlearning2/infants/data/0/trial_001/trial_ros.bag",
                            output_folder="rs_images",
                            timezone_name="America/Chicago"):
    os.makedirs(output_folder, exist_ok=True)

    if time_str is not None:
        target_unix_time = qualisys_time_to_unix(time_str, timezone_name=timezone_name)
        match = find_closest_ros_frame_start(
            target_unix_time,
            topic_name=topic_name,
            rosbag_path=rosbag_path,
        )
        start_frame = match["frame_idx"]
        print(f"Qualisys video Unix time: {target_unix_time}")
        print(
            f"Starting ROS extraction at frame {start_frame} "
            f"(timestamp {match['ros_time']:.6f}, diff {match['diff_sec']:.6f}s)"
        )
        # breakpoint()
    else:
        match = {}
        start_frame = 0

    with rosbag.Bag(rosbag_path) as bag:
        count = 0
        frame_idx = 0
        saved_timestamps = []
        for topic, msg, ros_time in bag.read_messages(topics=[topic_name]):
            if frame_idx >= start_frame:
                save_frame(msg, count, output_folder)
                saved_timestamps.append(ros_time.to_sec())
                count += 1
            frame_idx += 1

    # print(f"Saved {count} ROS frames to '{output_folder}'")
    match["saved_timestamps"] = saved_timestamps
    return match


def generate_qualisys_timestamps(start_time_str,
                                 num_frames,
                                 fps=30.0,
                                 timezone_name="America/Chicago"):
    start_unix = qualisys_time_to_unix(start_time_str, timezone_name=timezone_name)
    frame_period = 1.0 / fps
    return [start_unix + (frame_idx * frame_period) for frame_idx in range(num_frames)]


def _closest_index(sorted_values, target):
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
        print("after diff: ", after_diff)
        return after_idx
    print("before diff: ", before_diff)
    return before_idx


def build_paired_dataset(qualisys_source_folder,
                         ros_source_folder,
                         ros_timestamps,
                         qualisys_start_time,
                         left_output_folder,
                         right_output_folder,
                         qualisys_fps=30.0,
                         timezone_name="America/Chicago"):
    qualisys_source = Path(qualisys_source_folder)
    ros_source = Path(ros_source_folder)
    left_output = Path(left_output_folder)
    right_output = Path(right_output_folder)

    qualisys_files = sorted(qualisys_source.glob("*.jpg"))
    ros_files = sorted(ros_source.glob("*.jpg"))
    if not qualisys_files:
        raise RuntimeError(f"No Qualisys frames found in {qualisys_source}")
    if not ros_files:
        raise RuntimeError(f"No ROS frames found in {ros_source}")

    left_output.mkdir(parents=True, exist_ok=True)
    right_output.mkdir(parents=True, exist_ok=True)
    for folder in (left_output, right_output):
        for old_file in folder.glob("*.jpg"):
            old_file.unlink()

    qualisys_timestamps = generate_qualisys_timestamps(
        qualisys_start_time,
        len(qualisys_files),
        fps=qualisys_fps,
        timezone_name=timezone_name,
    )

    paired_count = min(len(ros_files), len(ros_timestamps))
    for ros_idx in range(paired_count):
        ros_time = ros_timestamps[ros_idx]
        qual_idx = _closest_index(qualisys_timestamps, ros_time)
        target_name = f"{ros_idx:04d}.jpg"
        shutil.copy2(qualisys_files[qual_idx], left_output / target_name)
        shutil.copy2(ros_files[ros_idx], right_output / target_name)

    print(f"Built paired dataset with {paired_count} image pairs")

def clean_and_rename_images(folder_path, rs_offset_to_qualisys):
    # Get all JPG files sorted by name
    images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])

    # Delete the first N images
    for i in range(min(rs_offset_to_qualisys, len(images))):
        os.remove(os.path.join(folder_path, images[i]))
    print(f"Deleted {min(rs_offset_to_qualisys, len(images))} images.")

    # Get remaining images after deletion
    remaining_images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])

    # Rename remaining images to 0000.jpg, 0001.jpg, ...
    for idx, filename in enumerate(remaining_images):
        new_name = f"{idx:04d}.jpg"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
    print(f"Renamed {len(remaining_images)} images starting from 0000.jpg.")

def transfer_images(source_folder, destination_folder, start_index, end_index, filename_digits=4):
    # === Ensure destination exists ===
    destination_folder.mkdir(parents=True, exist_ok=True)

    # === Get and sort all .jpg files ===
    image_files = sorted(source_folder.glob("*.jpg"))

    # === Filter and copy files within the index range ===
    counter = 0
    for img_path in image_files:
        stem = img_path.stem  # e.g., '0001'
        
        try:
            index = int(stem)
        except ValueError:
            print(f"Skipping {img_path.name}: filename does not contain a valid number.")
            continue

        if start_index <= index <= end_index:
            new_name = f"img_{str(counter).zfill(filename_digits)}.jpg"
            destination_path = destination_folder / new_name
            shutil.copy(img_path, destination_path)
            print(f"Copied: {img_path.name} → {new_name}")
            counter += 1


def extract_iamges_from_ros2(topic_name="/cam_L/color/image_raw",
                             rosbag_path="/home/robotlearning2/infants/data/0/trial_001/trial_ros.bag",
                             output_folder="rs_images"):
    # ======== Inpsect rosbg ============
    bag = rosbag.Bag(rosbag_path)
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all messages
    count = 0
    check_diff = True
    for topic, msg, t in bag.read_messages():
        # print(f"Topic: {topic}, Time: {t}")
        # breakpoint()
        if topic == topic_name:
            save_frame(msg, count, output_folder)
            count += 1
    bag.close()
    # ==================================
