#!/usr/bin/env python3
"""visualize_fullchain.py

Subscribes to /marker_N topics (Qualisys mocap markers in mm),
transforms into each camera frame using calibration chain,
publishes as visualization_msgs/Marker in each camera's optical frame.

RViz Camera display then overlays the markers onto the camera image.

Usage:
    python visualize_fullchain.py --config config.yaml
"""
import argparse
import yaml
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray


AXIS_FLIP = np.eye(4)
AXIS_FLIP[1, 1] = -1
AXIS_FLIP[2, 2] = -1


def build_4x4(R, T):
    M = np.eye(4)
    M[:3, :3] = np.array(R, dtype=np.float64)
    M[:3, 3] = np.array(T, dtype=np.float64).ravel()
    return M


def build_transform_chain(cam_cfg):
    T_mc_wrt_mcR = np.array(cam_cfg["T_mc_wrt_mcR"], dtype=np.float64)
    stereo_RT = build_4x4(cam_cfg["stereo_R"], cam_cfg["stereo_T"])
    return stereo_RT @ AXIS_FLIP @ np.linalg.inv(T_mc_wrt_mcR)


def build_chained_transform(cam_cfg, parent_transform):
    stereo_RT = build_4x4(cam_cfg["stereo_R"], cam_cfg["stereo_T"])
    return stereo_RT @ parent_transform


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class MarkerTransformer:
    def __init__(self, config):
        self.transforms = {}
        self.frame_ids = {}
        self.colors = {}

        for cam_name, cam_cfg in config["cameras"].items():
            if "parent" in cam_cfg:
                parent_T = self.transforms[cam_cfg["parent"]]
                self.transforms[cam_name] = build_chained_transform(cam_cfg, parent_T)
            else:
                self.transforms[cam_name] = build_transform_chain(cam_cfg)

            self.frame_ids[cam_name] = cam_cfg["frame_id"]
            self.colors[cam_name] = cam_cfg.get("color", [1.0, 0.0, 0.0])
            rospy.loginfo(f"  {cam_name}: frame={self.frame_ids[cam_name]}")
            rospy.loginfo(f"    T_mcR_to_{cam_name}:\n{self.transforms[cam_name]}")

        self.marker_pub = rospy.Publisher(
            "/calibration_markers", MarkerArray, queue_size=10)

        self.num_markers = config.get("num_markers", 1)
        self.sphere_radius = config.get("sphere_radius", 0.03)

        for m in range(1, self.num_markers + 1):
            topic = f"/marker_{m}"
            rospy.Subscriber(topic, PointStamped, self.marker_cb, callback_args=m)
            rospy.loginfo(f"  Subscribed to {topic}")

    def marker_cb(self, msg, marker_idx):
        # Qualisys TSV values are in mm
        pt_mcR = np.array([
            msg.point.x / 1000.0,
            msg.point.y / 1000.0,
            msg.point.z / 1000.0,
            1.0
        ])

        if np.any(np.isnan(pt_mcR[:3])):
            return

        ma = MarkerArray()
        marker_id = 0

        for cam_name, T in self.transforms.items():
            pt_cam = T @ pt_mcR

            m = Marker()
            m.header.stamp = msg.header.stamp
            m.header.frame_id = self.frame_ids[cam_name]
            m.ns = f"marker_{marker_idx}_{cam_name}"
            m.id = marker_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = pt_cam[0]
            m.pose.position.y = pt_cam[1]
            m.pose.position.z = pt_cam[2]
            m.pose.orientation.w = 1.0
            m.scale.x = self.sphere_radius
            m.scale.y = self.sphere_radius
            m.scale.z = self.sphere_radius
            c = self.colors[cam_name]
            m.color.r = c[0]
            m.color.g = c[1]
            m.color.b = c[2]
            m.color.a = 1.0
            m.lifetime = rospy.Duration(0.1)

            ma.markers.append(m)
            marker_id += 1

        self.marker_pub.publish(ma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args, _ = ap.parse_known_args()

    config = load_config(args.config)

    rospy.init_node("marker_transformer", anonymous=True)
    rospy.loginfo("Starting marker transformer...")

    transformer = MarkerTransformer(config)
    rospy.spin()


if __name__ == "__main__":
    main()
