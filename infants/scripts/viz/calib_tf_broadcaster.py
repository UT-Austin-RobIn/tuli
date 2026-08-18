#!/usr/bin/env python3
"""Publish static TF so point clouds can be viewed in Qualisys mcR or aligned across cameras.

Two modes (selected via ~use_mcr_frame):

  mcR frame (use_mcr_frame=true):
    RViz Fixed Frame = qualisys_mcR.
    Each cloud stays in its camera optical frame; we publish TF:
      qualisys_mcR -> cam_X_color_optical_frame
    with matrix T_cam_to_mcR = inv(T_mcR_to_cam).
    RViz applies that TF so the cloud appears in mcR.

  Camera-aligned (use_mcr_frame=false, multi-camera):
    RViz Fixed Frame = fixed camera optical frame.
    Publish TF fixed_cam -> other_cam using inter-camera alignment.
"""
import rospy
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped

import tf.transformations as tft
from calibration_chain import (
    MOCAP_REF_FRAME,
    build_T_cam_to_mcR,
    build_camera_transforms,
    load_config,
    resolve_config_path,
    transform_child_to_parent,
)


def matrix_to_transform(stamp, parent_frame, child_frame, matrix):
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame
    t.transform.translation.x = matrix[0, 3]
    t.transform.translation.y = matrix[1, 3]
    t.transform.translation.z = matrix[2, 3]
    q = tft.quaternion_from_matrix(matrix)
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    return t


def parse_camera_list(raw):
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


DEBUG_LAYOUT_OFFSETS = {
    "M": np.array([0.9, 0.0, 0.0, 1.0]),
    "R": np.array([1.8, 0.0, 0.0, 1.0]),
}


def build_debug_transforms(enabled, fixed_key, frame_ids):
    """Side-by-side layout so each cloud stays visible without real calibration."""
    fixed_frame = frame_ids[fixed_key]
    msgs = []
    for cam_key in enabled:
        if cam_key == fixed_key:
            continue
        offset = DEBUG_LAYOUT_OFFSETS.get(cam_key)
        if offset is None:
            continue
        matrix = np.eye(4)
        matrix[:3, 3] = offset[:3]
        msgs.append(
            matrix_to_transform(
                rospy.Time(0),
                fixed_frame,
                frame_ids[cam_key],
                matrix,
            )
        )
    return msgs


def build_mcr_transforms(enabled, T_mcR_to_cam, frame_ids):
    """Publish qualisys_mcR -> cam_X TF so camera-frame clouds display in mcR.

    calibration_chain gives T_mcR_to_cam where p_cam = T_mcR_to_cam @ p_mcR.
    ROS TF parent=qualisys_mcR, child=cam needs p_mcR = TF @ p_cam, hence inv().
    """
    msgs = []
    for cam_key in enabled:
        if cam_key not in T_mcR_to_cam:
            continue
        T_mcR_to_cam_mat = T_mcR_to_cam[cam_key]
        T_cam_to_mcR = build_T_cam_to_mcR(T_mcR_to_cam_mat)
        child_frame = frame_ids[cam_key]
        rospy.loginfo(
            "[mcR] cam_%s: T_mcR_to_cam (p_cam = T @ p_mcR):\n%s",
            cam_key,
            T_mcR_to_cam_mat,
        )
        rospy.loginfo(
            "[mcR] TF %s -> %s uses inv(T_mcR_to_cam):\n%s",
            MOCAP_REF_FRAME,
            child_frame,
            T_cam_to_mcR,
        )
        msgs.append(
            matrix_to_transform(
                rospy.Time(0),
                MOCAP_REF_FRAME,
                child_frame,
                T_cam_to_mcR,
            )
        )
    return msgs


def build_calibrated_transforms(
    enabled, fixed_key, transforms, frame_ids, default_frame_ids
):
    """Align secondary cameras into the fixed camera optical frame.

    Cameras without complete calibration get a debug side-by-side offset so
    their clouds are still visible while calibration is being filled in.
    """
    fixed_frame = frame_ids.get(fixed_key, default_frame_ids[fixed_key])
    if fixed_key not in transforms:
        rospy.logerr(
            "Cannot align cameras: fixed camera %s has no calibration",
            fixed_key,
        )
        return []

    T_fixed = transforms[fixed_key]
    msgs = []
    for cam_key in enabled:
        if cam_key == fixed_key:
            continue
        child_frame = frame_ids.get(cam_key, default_frame_ids[cam_key])

        if cam_key not in transforms:
            offset = DEBUG_LAYOUT_OFFSETS.get(cam_key)
            if offset is None:
                rospy.logwarn(
                    "cam_%s: no calibration and no debug offset; cloud hidden",
                    cam_key,
                )
                continue
            rospy.logwarn(
                "cam_%s: no calibration in YAML; using debug offset (not aligned)",
                cam_key,
            )
            matrix = np.eye(4)
            matrix[:3, 3] = offset[:3]
            msgs.append(
                matrix_to_transform(
                    rospy.Time(0),
                    fixed_frame,
                    child_frame,
                    matrix,
                )
            )
            continue

        T_child = transforms[cam_key]
        # Both T map mcR->cam; compose so p_fixed = parent_T_child @ p_child.
        parent_T_child = transform_child_to_parent(T_fixed, T_child)
        rospy.loginfo(
            "[align] TF %s -> %s (child into fixed camera):\n%s",
            fixed_frame,
            child_frame,
            parent_T_child,
        )
        msgs.append(
            matrix_to_transform(
                rospy.Time(0),
                fixed_frame,
                child_frame,
                parent_T_child,
            )
        )
    return msgs


def main():
    rospy.init_node("calib_tf_broadcaster")
    enabled = parse_camera_list(rospy.get_param("~cameras", "L"))
    fixed_key = rospy.get_param("~fixed_camera", enabled[0])
    debug_layout = rospy.get_param("~debug_layout", False)
    use_mcr_frame = rospy.get_param("~use_mcr_frame", False)

    default_frame_ids = {
        "L": "cam_L_color_optical_frame",
        "M": "cam_M_color_optical_frame",
        "R": "cam_R_color_optical_frame",
    }
    if fixed_key not in enabled:
        raise RuntimeError(f"Fixed camera {fixed_key} is not in enabled set {enabled}")

    if debug_layout:
        rospy.loginfo("Using debug side-by-side TF layout (not calibrated alignment)")
        msgs = build_debug_transforms(enabled, fixed_key, default_frame_ids)
    else:
        config_path = resolve_config_path(rospy.get_param("~config"))
        config = load_config(config_path)
        T_mcR_to_cam, frame_ids, _ = build_camera_transforms(
            config, enabled_keys=enabled
        )
        if use_mcr_frame:
            missing = [k for k in enabled if k not in T_mcR_to_cam]
            if missing:
                rospy.logwarn(
                    "No complete calibration for camera(s) %s; "
                    "those clouds will not appear in %s",
                    ",".join(missing),
                    MOCAP_REF_FRAME,
                )
            if not T_mcR_to_cam:
                rospy.logerr(
                    "Cannot publish mcR TF: no complete camera calibrations in %s",
                    config_path,
                )
            rospy.loginfo(
                "Publishing point clouds into %s via qualisys_mcR -> cam_X TF",
                MOCAP_REF_FRAME,
            )
            msgs = build_mcr_transforms(enabled, T_mcR_to_cam, frame_ids)
        else:
            msgs = build_calibrated_transforms(
                enabled, fixed_key, T_mcR_to_cam, frame_ids, default_frame_ids
            )

    broadcaster = tf2_ros.StaticTransformBroadcaster()

    def publish_transforms(_event=None):
        stamp = rospy.Time(0)
        stamped = []
        for msg in msgs:
            out = TransformStamped()
            out.header.stamp = stamp
            out.header.frame_id = msg.header.frame_id
            out.child_frame_id = msg.child_frame_id
            out.transform = msg.transform
            stamped.append(out)
        if stamped:
            broadcaster.sendTransform(stamped)

    if msgs:
        for msg in msgs:
            rospy.loginfo(
                "Static TF %s -> %s",
                msg.header.frame_id,
                msg.child_frame_id,
            )
        publish_transforms()
        rospy.Timer(rospy.Duration(1.0), publish_transforms)
    else:
        rospy.loginfo("Single-camera view; no inter-camera TF needed.")

    rospy.spin()


if __name__ == "__main__":
    main()
