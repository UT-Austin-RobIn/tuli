import rosbag
import rospy
import pytz
import os
import cv2
from cv_bridge import CvBridge
from datetime import datetime
import rosbag
from datetime import datetime

input_bag = "/home/robotlearning2/infants/data/055/trial_001/trial_ros.bag"  # path to your rosbag

# Topics to synchronize
topics_to_sync = [
    '/cam_L/aligned_depth_to_color/image_raw',
    '/cam_L/color/camera_info',
    '/cam_L/color/image_raw'
]

# Step 1: Read messages from each topic into a list
topic_msgs = {topic: [] for topic in topics_to_sync}

with rosbag.Bag(input_bag, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=topics_to_sync):
        if topic == '/cam_L/color/image_raw':
            breakpoint()
        # topic_msgs[topic].append((msg, t))

# Step 2: Determine the minimum number of messages
min_len = min(len(lst) for lst in topic_msgs.values())
print(f"Minimum number of messages among the topics: {min_len}")

# Step 3: Optionally, select first N messages, or pick closest timestamps
# Here, we just take the first min_len messages for simplicity
for topic in topics_to_sync:
    topic_msgs[topic] = topic_msgs[topic][:min_len]

# Step 4: Write a new bag
with rosbag.Bag(output_bag, 'w') as out_bag:
    for i in range(min_len):
        for topic in topics_to_sync:
            msg, t = topic_msgs[topic][i]
            out_bag.write(topic, msg, t)

print(f"Resampled bag saved to {output_bag}")
