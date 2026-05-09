"""Pure-numpy transform composition for the qualisys -> realsense calibration chain.

Kept ROS-free so it can be unit tested without rospy. Used by calib_tf_publisher.py.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# diag(1, -1, -1) maps from Qualisys (Y-up, Z-forward-from-camera-back)
# to OpenCV image convention (Y-down, Z-forward). It is its own inverse.
AXIS_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def rt_to_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 from a 3x3 rotation and a 3-vector / 3x1 translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def parse_qtm_calibration(xml_path: Path) -> Dict[str, np.ndarray]:
    """Parse a QTM calibration XML and return {camera_serial: T_mc_in_qualisysRef (4x4, meters)}.

    The XML's translation x,y,z is millimeters; it is converted to meters here.

    Rotation convention: QTM stores the rotation in *column-major* order in the
    transform attributes. That is, attribute rIJ holds R[J-1, I-1] in the matrix
    that maps camera-local coordinates to global. We confirmed this by direct
    comparison against calib_data/test_calibration.py.txt:147-152 where the
    cam_R anchor (serial 31041) translation matched exactly while the rotation
    matched only after transposing. Build R column-by-column accordingly.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    out: Dict[str, np.ndarray] = {}
    for cam in root.iter("camera"):
        serial = cam.get("serial")
        tf = cam.find("transform")
        if serial is None or tf is None:
            continue
        R = np.array([
            [float(tf.get("r11")), float(tf.get("r21")), float(tf.get("r31"))],
            [float(tf.get("r12")), float(tf.get("r22")), float(tf.get("r32"))],
            [float(tf.get("r13")), float(tf.get("r23")), float(tf.get("r33"))],
        ])
        t = np.array([float(tf.get("x")), float(tf.get("y")), float(tf.get("z"))]) / 1000.0
        out[serial] = rt_to_homogeneous(R, t)
    return out


def load_calib_rt(json_path: Path) -> np.ndarray:
    """Load 'rot' and 'trans' from a calibration JSON and return as a 4x4 transform.

    Per the convention in examples/test_calibration_via_triangulation.py and
    calib_data/test_calibration.py.txt, this matrix maps points from the LEFT
    camera frame (mocap-cam in image convention, for qualisys<->cam_X JSONs;
    cam_L for the L<->M JSON) to the RIGHT camera frame (realsense, or cam_M).
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    R = np.array(data["rot"])
    t = np.array(data["trans"]).reshape(3)
    return rt_to_homogeneous(R, t)


def qualisys_to_realsense_pose(
    T_mc_in_qualisysRef: np.ndarray,
    T_realsense_from_mc_imgconv: np.ndarray,
) -> np.ndarray:
    """Pose of the realsense camera expressed in the qualisys reference frame.

    Math derivation (matches calib_data/test_calibration.py.txt:147-191):
        p_mc        = inv(T_mc_in_qualisysRef) @ p_qualisysRef
        p_mc_img    = AXIS_FLIP @ p_mc
        p_realsense = T_realsense_from_mc_imgconv @ p_mc_img

    So the realsense-from-qualisys point transform is:
        T_rs_from_q = T_realsense_from_mc_imgconv @ AXIS_FLIP @ inv(T_mc_in_qualisysRef)

    The TF "pose of realsense in qualisys" is the inverse of the point transform:
        T_q_from_rs = T_mc_in_qualisysRef @ AXIS_FLIP @ inv(T_realsense_from_mc_imgconv)
    """
    return (
        T_mc_in_qualisysRef
        @ AXIS_FLIP
        @ np.linalg.inv(T_realsense_from_mc_imgconv)
    )


def chain_camL_to_camM_pose(
    T_q_from_camL: np.ndarray,
    T_camM_from_camL: np.ndarray,
) -> np.ndarray:
    """Pose of cam_M in qualisys, given cam_L's pose in qualisys and the L->M calib."""
    return T_q_from_camL @ np.linalg.inv(T_camM_from_camL)


def matrix_to_translation_quaternion(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 4x4 transform to (translation [x,y,z], quaternion [x,y,z,w]).

    Quaternion conversion uses the standard Shepperd method to stay stable across
    all rotations. No external dependency on tf.transformations.
    """
    R = T[:3, :3]
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return T[:3, 3].copy(), np.array([qx, qy, qz, qw])
