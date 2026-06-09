import pytz
import rosbag

from datetime import datetime


INPUT_BAG = "/home/robotlearning2/infants/data/055/trial_001/trial_ros.bag"
TOPIC = "/cam_L/color/image_raw"
LOCAL_TZ = pytz.timezone("America/Chicago")


def main():
    with rosbag.Bag(INPUT_BAG, "r") as bag:
        for frame_idx, (_, _, ros_time) in enumerate(bag.read_messages(topics=[TOPIC])):
            unix_ts = ros_time.to_sec()
            local_time = datetime.fromtimestamp(unix_ts, LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S.%f %Z")
            print(f"{frame_idx:04d}  {unix_ts:.6f}  {local_time}")


if __name__ == "__main__":
    main()
