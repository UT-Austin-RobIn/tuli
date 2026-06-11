<<<<<<< Updated upstream
#!/usr/bin/env python
=======
#!/usr/bin/env python3
>>>>>>> Stashed changes
"""ROS1 node: publish static TFs for the qualisys + 3 RealSense calibration tree.

Frames published:
    world  ->  qualisys_ref  (identity, or the chosen camera's frame as world)
    qualisys_ref  ->  cam_L_color_optical_frame
    qualisys_ref  ->  cam_R_color_optical_frame
    cam_L_color_optical_frame  ->  cam_M_color_optical_frame

The cam_*_color_optical_frame names match the frame_ids inside the bag's CameraInfo
and depth Image headers, so our calibration TFs slot directly into the RealSense
tf_static tree shipped in the bag without needing alias TFs.

Math chain (per calib_data/test_calibration.py.txt:147-191):
    p_realsense = T_calib_X @ AXIS_FLIP @ inv(T_mc_in_qualisysRef) @ p_qualisysRef

so the TF (pose-of-camera-in-qualisys) is the inverse of that point transform.

Required ROS params (private):
    ~mocap_anchor_L : str — Qualisys camera serial used as L's anchor (e.g. "31039")
    ~mocap_anchor_R : str — Qualisys camera serial used as R's anchor (e.g. "31041")
    ~world_frame    : str — one of {qualisys_ref, cam_L_optical, cam_R_optical, cam_M_optical}
    ~qtm_xml        : str — path to QTM calibration XML (default: calib_data/mocap_calib_0415.txt)
    ~calib_dir      : str — directory holding the 3 calibration JSONs
<<<<<<< Updated upstream
=======
    ~calib_L_path   : str — explicit path to qualisys -> cam_L calibration JSON
    ~calib_R_path   : str — explicit path to qualisys -> cam_R calibration JSON
    ~calib_LM_path  : str — explicit path to cam_L -> cam_M calibration JSON
>>>>>>> Stashed changes
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

# Allow running this file directly (e.g. via roslaunch's pkg/type) without the
# repo being on PYTHONPATH: prepend the repo root so `import stereo_calib.*` works.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# rospy on this conda env still calls Thread.isAlive (removed in Python 3.9).
# Restore the alias so SIGINT/SIGTERM shutdown doesn't AttributeError.
if not hasattr(threading.Thread, "isAlive"):
    threading.Thread.isAlive = threading.Thread.is_alive

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

from stereo_calib.ros.calib_chain import (
    chain_camL_to_camM_pose,
    load_calib_rt,
    matrix_to_translation_quaternion,
    parse_qtm_calibration,
    qualisys_to_realsense_pose,
)


def make_tf(parent: str, child: str, T: np.ndarray, stamp) -> TransformStamped:
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent
    msg.child_frame_id = child
    t, q = matrix_to_translation_quaternion(T)
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = t
    msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w = q
    return msg


CAM_L = "cam_L_color_optical_frame"
CAM_M = "cam_M_color_optical_frame"
CAM_R = "cam_R_color_optical_frame"
QUALISYS_REF = "qualisys_ref"
VALID_WORLD_FRAMES = {QUALISYS_REF, CAM_L, CAM_M, CAM_R}


def main():
    rospy.init_node("calib_tf_publisher")
    repo_root = Path(__file__).resolve().parents[2]
    default_calib_dir = repo_root / "calib_data"
    default_qtm_xml = default_calib_dir / "mocap_calib_0415.txt"

    qtm_xml = Path(rospy.get_param("~qtm_xml", str(default_qtm_xml)))
    calib_dir = Path(rospy.get_param("~calib_dir", str(default_calib_dir)))
    world_frame = rospy.get_param("~world_frame", QUALISYS_REF)
    anchor_L = str(rospy.get_param("~mocap_anchor_L", ""))
    anchor_R = str(rospy.get_param("~mocap_anchor_R", ""))

    if world_frame not in VALID_WORLD_FRAMES:
        rospy.logfatal(f"~world_frame={world_frame!r} not in {sorted(VALID_WORLD_FRAMES)}")
        return

    qtm_cams = parse_qtm_calibration(qtm_xml)
    if not anchor_L or not anchor_R:
        rospy.logfatal(
            "Both ~mocap_anchor_L and ~mocap_anchor_R are required.  "
            f"Available QTM camera serials in {qtm_xml.name}: {sorted(qtm_cams.keys())}"
        )
        return
    for label, anchor in (("L", anchor_L), ("R", anchor_R)):
        if anchor not in qtm_cams:
            rospy.logfatal(
                f"~mocap_anchor_{label}={anchor!r} not found in {qtm_xml.name}.  "
                f"Available: {sorted(qtm_cams.keys())}"
            )
            return

<<<<<<< Updated upstream
    calib_L_path = calib_dir / "calib_qualisys_cam_L.json"
    calib_R_path = calib_dir / "calib_qualisys_cam_R.json"
    calib_LM_path = calib_dir / "callib_calib_L_cam_M.json"
=======
    calib_L_path = Path(rospy.get_param("~calib_L_path", str(calib_dir / "calib_qualisys_cam_L.json")))
    calib_R_path = Path(rospy.get_param("~calib_R_path", str(calib_dir / "calib_qualisys_cam_R.json")))
    calib_LM_path = Path(rospy.get_param("~calib_LM_path", str(calib_dir / "callib_calib_L_cam_M.json")))
>>>>>>> Stashed changes
    for p in (calib_L_path, calib_R_path, calib_LM_path):
        if not p.exists():
            rospy.logfatal(f"Calibration file missing: {p}")
            return

    T_calib_L = load_calib_rt(calib_L_path)
    T_calib_R = load_calib_rt(calib_R_path)
    T_calib_LM = load_calib_rt(calib_LM_path)

    T_q_from_camL = qualisys_to_realsense_pose(qtm_cams[anchor_L], T_calib_L)
    T_q_from_camR = qualisys_to_realsense_pose(qtm_cams[anchor_R], T_calib_R)
    # T_calib_LM is a point-transform cam_L -> cam_M. The static TF cam_L -> cam_M
    # needs the pose of cam_M expressed in cam_L (the inverse).
    T_camL_from_camM = np.linalg.inv(T_calib_LM)

    # If world is something other than qualisys_ref, root the tree on that frame instead.
    # Easiest implementation: keep all the above edges, then add a single edge connecting
    # `world` to `qualisys_ref` (identity if world IS qualisys_ref, else the inverse of
    # the chosen camera's pose in qualisys_ref).
    if world_frame == QUALISYS_REF:
        T_world_from_qref = np.eye(4)
    elif world_frame == CAM_L:
        T_world_from_qref = np.linalg.inv(T_q_from_camL)
    elif world_frame == CAM_R:
        T_world_from_qref = np.linalg.inv(T_q_from_camR)
    else:  # CAM_M
        T_q_from_camM = chain_camL_to_camM_pose(T_q_from_camL, T_calib_LM)
        T_world_from_qref = np.linalg.inv(T_q_from_camM)

    bcaster = tf2_ros.StaticTransformBroadcaster()
    now = rospy.Time.now()
    tfs = [
        make_tf("world", QUALISYS_REF, T_world_from_qref, now),
        make_tf(QUALISYS_REF, CAM_L, T_q_from_camL, now),
        make_tf(QUALISYS_REF, CAM_R, T_q_from_camR, now),
        make_tf(CAM_L, CAM_M, T_camL_from_camM, now),
    ]
    bcaster.sendTransform(tfs)

    rospy.loginfo("calib_tf_publisher: published static TFs")
    rospy.loginfo(f"  world_frame    = {world_frame}")
    rospy.loginfo(f"  mocap_anchor_L = {anchor_L}")
    rospy.loginfo(f"  mocap_anchor_R = {anchor_R}")
    rospy.loginfo(f"  T (qualisys_ref -> {CAM_L}): t={T_q_from_camL[:3, 3]}")
    rospy.loginfo(f"  T (qualisys_ref -> {CAM_R}): t={T_q_from_camR[:3, 3]}")
    rospy.loginfo(f"  T ({CAM_L} -> {CAM_M}) [pose of cam_M in cam_L]: t={T_camL_from_camM[:3, 3]}")
    rospy.spin()


if __name__ == "__main__":
    main()
