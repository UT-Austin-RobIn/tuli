#!/usr/bin/env python3
"""Subscribe to cam_L RGB-D and publish a colored PointCloud2 for RViz."""
import message_filters
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField


class DepthToPointCloud:
    def __init__(self):
        self.bridge = CvBridge()
        self.subsample = int(rospy.get_param("~subsample", 2))
        color_topic = rospy.get_param("~color_topic", "/cam_L/color/image_raw")
        depth_topic = rospy.get_param(
            "~depth_topic", "/cam_L/aligned_depth_to_color/image_raw"
        )
        info_topic = rospy.get_param("~camera_info_topic", "/cam_L/color/camera_info")
        output_topic = rospy.get_param("~output_topic", "/cam_L/points")

        self.pub = rospy.Publisher(output_topic, PointCloud2, queue_size=2)
        self._frame_idx = 0

        color_sub = message_filters.Subscriber(color_topic, Image, queue_size=10)
        depth_sub = message_filters.Subscriber(depth_topic, Image, queue_size=10)
        info_sub = message_filters.Subscriber(info_topic, CameraInfo, queue_size=10)
        sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub],
            queue_size=10,
            slop=0.05,
        )
        sync.registerCallback(self.callback)
        rospy.loginfo(
            "Publishing %s from %s + %s (subsample=%d)",
            output_topic,
            color_topic,
            depth_topic,
            self.subsample,
        )

    def callback(self, color_msg, depth_msg, info_msg):
        color = self.bridge.imgmsg_to_cv2(color_msg, "rgb8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

        step = max(1, self.subsample)
        color = color[::step, ::step]
        depth = depth[::step, ::step]

        fx = info_msg.K[0] / step
        fy = info_msg.K[4] / step
        cx = info_msg.K[2] / step
        cy = info_msg.K[5] / step

        h, w = depth.shape
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)

        z = depth.astype(np.float32) / 1000.0
        valid = z > 0
        if not np.any(valid):
            return

        z_valid = z[valid]
        u_valid = uu[valid]
        v_valid = vv[valid]
        colors = color[valid]

        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy

        rgb_uint32 = (
            colors[:, 0].astype(np.uint32) << 16
            | colors[:, 1].astype(np.uint32) << 8
            | colors[:, 2].astype(np.uint32)
        )
        rgb_float = rgb_uint32.view(np.float32)

        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
        ]
        points = np.column_stack([x, y, z_valid, rgb_float])
        cloud = pc2.create_cloud(color_msg.header, fields, points)
        self.pub.publish(cloud)
        self._frame_idx += 1
        if self._frame_idx == 1 or self._frame_idx % 30 == 0:
            rospy.loginfo(
                "Published cloud frame %d (%d points, stamp=%.3f)",
                self._frame_idx,
                len(x),
                color_msg.header.stamp.to_sec(),
            )


def main():
    rospy.init_node("depth_to_pointcloud")
    DepthToPointCloud()
    rospy.spin()


if __name__ == "__main__":
    main()
