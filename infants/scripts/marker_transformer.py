#!/usr/bin/env python3
"""Publish Qualisys markers for RViz alongside calibrated point clouds.

reference_frame=qualisys_mcR (default when trial viz uses --mcr-frame):
  Markers stay in mcR — same frame as the aligned point clouds.

reference_frame=cameras (legacy / image-overlay style):
  Project each marker into every enabled camera optical frame using T_mcR_to_cam.
"""
import rospy
import numpy as np
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray

from calibration_chain import (
    MOCAP_REF_FRAME,
    build_camera_transforms,
    load_config,
    resolve_config_path,
)


class MarkerTransformer:
    def __init__(self, config, enabled_cameras, reference_frame):
        self.reference_frame = reference_frame
        transforms, frame_ids, colors = build_camera_transforms(
            config, enabled_keys=enabled_cameras
        )
        self.transforms = {
            k: transforms[k] for k in enabled_cameras if k in transforms
        }
        self.frame_ids = {k: frame_ids[k] for k in self.transforms}
        self.colors = {k: colors[k] for k in self.transforms}

        self.marker_pub = rospy.Publisher(
            "/calibration_markers", MarkerArray, queue_size=10
        )
        self.num_markers = int(config.get("num_markers", 1))
        self.sphere_radius = float(config.get("sphere_radius", 0.02))

        if reference_frame == MOCAP_REF_FRAME:
            rospy.loginfo(
                "Markers published in %s (raw Qualisys /marker_N, mm->m)",
                MOCAP_REF_FRAME,
            )
        else:
            for cam_key, frame_id in self.frame_ids.items():
                rospy.loginfo(
                    "  marker overlay: mcR -> cam_%s (%s) via T_mcR_to_cam",
                    cam_key,
                    frame_id,
                )

        for m in range(1, self.num_markers + 1):
            topic = f"/marker_{m}"
            rospy.Subscriber(topic, PointStamped, self.marker_cb, callback_args=m)
            rospy.loginfo("  subscribed to %s", topic)

    def marker_cb(self, msg, marker_idx):
        # Qualisys bag topics carry marker positions in millimeters wrt mcR.
        pt_mcR = np.array(
            [
                msg.point.x / 1000.0,
                msg.point.y / 1000.0,
                msg.point.z / 1000.0,
                1.0,
            ]
        )
        if np.any(np.isnan(pt_mcR[:3])):
            return

        ma = MarkerArray()
        if self.reference_frame == MOCAP_REF_FRAME:
            m = Marker()
            m.header.stamp = msg.header.stamp
            m.header.frame_id = MOCAP_REF_FRAME
            m.ns = f"marker_{marker_idx}"
            m.id = 0
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = pt_mcR[0]
            m.pose.position.y = pt_mcR[1]
            m.pose.position.z = pt_mcR[2]
            m.pose.orientation.w = 1.0
            m.scale.x = self.sphere_radius
            m.scale.y = self.sphere_radius
            m.scale.z = self.sphere_radius
            m.color.r = 1.0
            m.color.g = 0.2
            m.color.b = 0.2
            m.color.a = 1.0
            m.lifetime = rospy.Duration(0.2)
            ma.markers.append(m)
        else:
            marker_id = 0
            for cam_key, T_mcR_to_cam in self.transforms.items():
                # Same as visualize_fullchain.py: p_cam = T_mcR_to_cam @ p_mcR
                pt_cam = T_mcR_to_cam @ pt_mcR
                m = Marker()
                m.header.stamp = msg.header.stamp
                m.header.frame_id = self.frame_ids[cam_key]
                m.ns = f"marker_{marker_idx}_cam_{cam_key}"
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
                c = self.colors[cam_key]
                m.color.r = c[0]
                m.color.g = c[1]
                m.color.b = c[2]
                m.color.a = 1.0
                m.lifetime = rospy.Duration(0.2)
                ma.markers.append(m)
                marker_id += 1

        self.marker_pub.publish(ma)


def parse_camera_list(raw):
    if not raw:
        return ["L"]
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def main():
    rospy.init_node("marker_transformer")
    config_path = resolve_config_path(rospy.get_param("~config"))
    enabled = parse_camera_list(rospy.get_param("~cameras", "L"))
    reference_frame = rospy.get_param("~reference_frame", MOCAP_REF_FRAME)
    config = load_config(config_path)
    rospy.loginfo(
        "Starting marker transformer (cameras=%s, reference_frame=%s)",
        ",".join(enabled),
        reference_frame,
    )
    MarkerTransformer(config, enabled, reference_frame)
    rospy.spin()


if __name__ == "__main__":
    main()
