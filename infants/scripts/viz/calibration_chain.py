"""Load full-chain calibration YAML and build camera transforms.

Transform chain (same as stereo-calib/examples/visualize_fullchain.py):

  For cameras calibrated against Qualisys (cam_L, cam_R):
    p_cam = T_mcR_to_cam @ p_mcR
    T_mcR_to_cam = stereo_RT @ AXIS_FLIP @ inv(T_mc_wrt_mcR)

    where:
      p_mcR          — homogeneous point in Qualisys global / mocap reference frame
      T_mc_wrt_mcR   — Qualisys mocap camera pose from calibration XML
      AXIS_FLIP      — diag(1,-1,-1), Qualisys Y-up -> OpenCV Y-down
      stereo_RT      — Charuco stereo calib (mocap/RS or RS/RS)

  For chained cameras (cam_M via cam_L):
    T_mcR_to_cam_M = stereo_RT_M @ T_mcR_to_cam_L

  To express a RealSense point cloud in mcR (what RViz needs when Fixed Frame
  is qualisys_mcR):
    p_mcR = inv(T_mcR_to_cam) @ p_cam
"""
from pathlib import Path

import numpy as np
import yaml

# Qualisys global frame used as RViz Fixed Frame when --mcr-frame is enabled.
MOCAP_REF_FRAME = "qualisys_mcR"

AXIS_FLIP = np.eye(4)
AXIS_FLIP[1, 1] = -1
AXIS_FLIP[2, 2] = -1

CAMERA_KEYS = {
    "L": "cam_L",
    "M": "cam_M",
    "R": "cam_R",
}


def build_4x4(R, T):
    M = np.eye(4)
    M[:3, :3] = np.array(R, dtype=np.float64)
    M[:3, 3] = np.array(T, dtype=np.float64).ravel()
    return M


def build_T_mcR_to_cam(cam_cfg):
    """Map a point from Qualisys mcR into this camera's optical frame."""
    T_mc_wrt_mcR = np.array(cam_cfg["T_mc_wrt_mcR"], dtype=np.float64)
    stereo_RT = build_4x4(cam_cfg["stereo_R"], cam_cfg["stereo_T"])
    # mocap cam frame -> RS optical frame, composed with mcR -> mocap cam.
    return stereo_RT @ AXIS_FLIP @ np.linalg.inv(T_mc_wrt_mcR)


def build_T_cam_to_mcR(T_mcR_to_cam):
    """Map a camera-frame point cloud into Qualisys mcR."""
    return np.linalg.inv(T_mcR_to_cam)


def build_transform_chain(cam_cfg):
    """Alias kept for compatibility; returns T_mcR_to_cam."""
    return build_T_mcR_to_cam(cam_cfg)


def build_chained_transform(cam_cfg, parent_T_mcR_to_cam):
    """Chain an RS-only stereo step onto a parent camera's mcR transform."""
    stereo_RT = build_4x4(cam_cfg["stereo_R"], cam_cfg["stereo_T"])
    return stereo_RT @ parent_T_mcR_to_cam


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_config_path(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Calibration config not found: {path}")
    return path


def _has_stereo_calib(cam_cfg):
    return cam_cfg.get("stereo_R") is not None and cam_cfg.get("stereo_T") is not None


def _has_mocap_anchor(cam_cfg):
    return cam_cfg.get("T_mc_wrt_mcR") is not None


def _can_build_transform(cam_cfg, transforms):
    if not _has_stereo_calib(cam_cfg):
        return False
    if "parent" in cam_cfg:
        parent_name = cam_cfg["parent"]
        parent_key = next(k for k, v in CAMERA_KEYS.items() if v == parent_name)
        return parent_key in transforms
    return _has_mocap_anchor(cam_cfg)


def _parent_key(cam_cfg):
    """Return CAMERA_KEYS letter for cam_cfg['parent'], or None."""
    if "parent" not in cam_cfg:
        return None
    parent_name = cam_cfg["parent"]
    return next(k for k, v in CAMERA_KEYS.items() if v == parent_name)


def _keys_with_parents(config, enabled_keys):
    """Enabled cameras plus any parents needed to chain their transforms.

    e.g. cameras=M alone still builds L first when cam_M.parent is cam_L.
    """
    cameras = config.get("cameras", {})
    ordered = []
    seen = set()

    def add(cam_key):
        if cam_key in seen or cam_key not in CAMERA_KEYS:
            return
        cam_name = CAMERA_KEYS[cam_key]
        if cam_name not in cameras:
            return
        parent = _parent_key(cameras[cam_name])
        if parent is not None:
            add(parent)
        seen.add(cam_key)
        ordered.append(cam_key)

    for cam_key in CAMERA_KEYS:
        if cam_key in enabled_keys:
            add(cam_key)
    return ordered


def build_camera_transforms(config, enabled_keys=None):
    """Return T_mcR_to_cam for each camera that has complete calibration data.

    Cameras with null stereo_R/T or null T_mc_wrt_mcR are skipped so partial YAML
    files (e.g. only cam_L calibrated) still work.

    Parent cameras are always built when a child is enabled, even if the parent
    itself is not in enabled_keys (needed for --cameras M alone).
    """
    transforms = {}
    frame_ids = {}
    colors = {}

    if enabled_keys is not None:
        keys = _keys_with_parents(config, enabled_keys)
    else:
        keys = _keys_with_parents(config, list(CAMERA_KEYS.keys()))

    for cam_key in keys:
        cam_name = CAMERA_KEYS[cam_key]
        if cam_name not in config.get("cameras", {}):
            continue
        cam_cfg = config["cameras"][cam_name]
        if not _can_build_transform(cam_cfg, transforms):
            continue

        if "parent" in cam_cfg:
            parent_key = _parent_key(cam_cfg)
            transforms[cam_key] = build_chained_transform(
                cam_cfg, transforms[parent_key]
            )
        else:
            transforms[cam_key] = build_T_mcR_to_cam(cam_cfg)

        frame_ids[cam_key] = cam_cfg["frame_id"]
        colors[cam_key] = cam_cfg.get("color", [1.0, 0.0, 0.0])

    # Only return transforms for cameras the caller asked to enable (not parents
    # pulled in solely for chaining), unless enabled_keys is None (all).
    if enabled_keys is not None:
        enabled = set(enabled_keys)
        transforms = {k: v for k, v in transforms.items() if k in enabled}
        frame_ids = {k: v for k, v in frame_ids.items() if k in enabled}
        colors = {k: v for k, v in colors.items() if k in enabled}

    return transforms, frame_ids, colors


def build_cam_to_mcR_transforms(config, enabled_keys):
    """Return T_cam_to_mcR per camera for debugging or direct point transforms."""
    T_mcR_to_cam, _, _ = build_camera_transforms(config, enabled_keys=enabled_keys)
    return {k: build_T_cam_to_mcR(T_mcR_to_cam[k]) for k in enabled_keys if k in T_mcR_to_cam}


def transform_child_to_parent(T_parent_from_ref, T_child_from_ref):
    """Map points from child camera frame into parent (fixed) camera frame."""
    return T_parent_from_ref @ np.linalg.inv(T_child_from_ref)
