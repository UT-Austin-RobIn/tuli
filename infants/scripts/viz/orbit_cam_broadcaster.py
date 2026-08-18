#!/usr/bin/env python3
"""Broadcast a camera TF that slowly circles the workspace for RViz.

Publishes parent_frame -> rviz_orbit_cam. Position moves on a horizontal
circle; orientation looks at the center (robot +X forward). RViz FPS view
tracks this frame.

Important: bag /clock often repeats the same timestamp under load. TF2 drops
updates with a repeated stamp (TF_REPEATED_DATA), which freezes the orbit.
We therefore publish with a monotonically increasing stamp.
"""
from __future__ import annotations

import math
import time

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

from orbit_view import (
    ORBIT_CAM_FRAME,
    camera_rotation_looking_at,
    eye_on_orbit,
    orbit_params_from_calib,
    rotation_to_quaternion,
)
from calibration_chain import MOCAP_REF_FRAME
from pathlib import Path


def main():
    rospy.init_node("orbit_cam_broadcaster")
    parent = rospy.get_param("~parent_frame", MOCAP_REF_FRAME)
    child = rospy.get_param("~child_frame", ORBIT_CAM_FRAME)
    calib = rospy.get_param("~config", "")
    cameras_raw = rospy.get_param("~cameras", "L")
    cameras = [c.strip().upper() for c in str(cameras_raw).split(",") if c.strip()]
    look_depth = float(rospy.get_param("~look_depth", 1.2))
    period = max(1.0, float(rospy.get_param("~period", 60.0)))
    rate_hz = float(rospy.get_param("~rate", 20.0))

    params = orbit_params_from_calib(
        Path(calib) if calib else None,
        cameras,
        use_mcr_frame=True,
        look_depth=look_depth,
    )
    center = np.array(params["center"], dtype=np.float64)
    radius = float(params["radius"])
    height = float(params["height"])
    yaw0 = float(params["yaw0"])

    broadcaster = tf2_ros.TransformBroadcaster()
    t0_wall = None
    last_stamp = None
    published_zero = False
    last_angle_bucket = -1
    dt = 1.0 / max(rate_hz, 1.0)

    rospy.loginfo(
        "Orbit cam %s -> %s: center=(%.3f,%.3f,%.3f) r=%.2f h=%.2f yaw0=%.2f period=%.1fs",
        parent,
        child,
        center[0],
        center[1],
        center[2],
        radius,
        height,
        yaw0,
        period,
    )

    while not rospy.is_shutdown():
        now = rospy.Time.now()

        if now.to_sec() <= 0.0:
            if published_zero:
                time.sleep(dt)
                continue
            stamp = rospy.Time(0)
            angle = 0.0
            published_zero = True
        else:
            if t0_wall is None:
                t0_wall = time.time()
                rospy.loginfo("Orbit cam wall-clock started (sim=%.3f)", now.to_sec())
            elapsed = max(0.0, time.time() - t0_wall)
            angle = 2.0 * math.pi * (elapsed / period)
            # Monotonic stamp so TF2 accepts updates even when /clock repeats.
            stamp = now
            if last_stamp is not None and stamp <= last_stamp:
                stamp = last_stamp + rospy.Duration.from_sec(dt)
            last_stamp = stamp

        eye = eye_on_orbit(center, radius, height, yaw0, angle)
        R = camera_rotation_looking_at(eye, center)
        qx, qy, qz, qw = rotation_to_quaternion(R)

        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = parent
        msg.child_frame_id = child
        msg.transform.translation.x = float(eye[0])
        msg.transform.translation.y = float(eye[1])
        msg.transform.translation.z = float(eye[2])
        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw
        broadcaster.sendTransform(msg)

        bucket = int(math.degrees(angle) / 45.0)
        if now.to_sec() > 0.0 and bucket != last_angle_bucket:
            last_angle_bucket = bucket
            rospy.loginfo(
                "Orbit cam angle=%.1fdeg eye=(%.2f,%.2f,%.2f)",
                math.degrees(angle) % 360.0,
                eye[0],
                eye[1],
                eye[2],
            )

        time.sleep(dt)


if __name__ == "__main__":
    main()
