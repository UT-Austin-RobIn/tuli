import csv
import rospy
import rosbag
from geometry_msgs.msg import PointStamped
import os
from std_msgs.msg import Int32, String

import pytz
from datetime import datetime

file_path = input("Path to Qualisys TSV file: ").strip()

output_dir = os.path.join(os.path.expanduser("~"), "infants", "data")
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_bag_path = os.path.join(output_dir, f"tsv_{timestamp}.bag")

all_rows = []
with open(file_path, newline='') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        all_rows.append(row)

start_row = 12
start_timestamp_str = all_rows[7][1]
local_tz = pytz.timezone("America/Chicago")
dt = datetime.strptime(start_timestamp_str, "%Y-%m-%d, %H:%M:%S.%f")
localized_dt = local_tz.localize(dt)
start_timestamp = localized_dt.timestamp()

marker_rate = int(all_rows[3][1])
frame_rate = 30
skip_frames = int(marker_rate / frame_rate)
num_markers = int(all_rows[2][1])
# num_markers = 2

marker_rows = all_rows[start_row:]
# Create rosbag
if os.path.exists(output_bag_path):
    os.remove(output_bag_path)  # overwrite if exists
bag = rosbag.Bag(output_bag_path, 'w')

num_markers_msg = Int32()
num_markers_msg.data = num_markers
bag.write("/metadata/num_markers", num_markers_msg, rospy.Time.from_sec(start_timestamp))

try:
    for row_idx, row in enumerate(marker_rows):
        # if row_idx % skip_frames != 0:
        #     continue
        # Compute timestamp
        t = start_timestamp + row_idx / marker_rate
        stamp = rospy.Time.from_sec(t)

        for m in range(num_markers):
            # Extract x, y, z for this marker
            try:
                x = float(row[m*3 + 0])
                y = float(row[m*3 + 1])
                z = float(row[m*3 + 2])
            except (IndexError, ValueError):
                x = y = z = float('nan')  # handle missing values

            # breakpoint()
            msg = PointStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = f"marker_{m+1}"
            msg.point.x = x
            msg.point.y = y
            msg.point.z = z

            # Write to topic /marker_1, /marker_2, ...
            topic_name = f"/marker_{m+1}"
            bag.write(topic_name, msg, stamp)
finally:
    bag.close()

print(f"Rosbag saved to {output_bag_path}")