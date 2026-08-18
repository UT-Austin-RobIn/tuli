#!/usr/bin/env python3
"""Shared helpers for RViz circular orbit camera (Qualisys Y-up / mcR)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from calibration_chain import (
    build_T_cam_to_mcR,
    build_camera_transforms,
    load_config,
)

ORBIT_CAM_FRAME = "rviz_orbit_cam"  # legacy name; prefer ORBIT_FOCUS_FRAME
ORBIT_FOCUS_FRAME = "rviz_orbit_focus"

# Slightly tighter than the old static overview so the subject fills more of the frame.
_ZOOM = 0.70
_MIN_RADIUS = 1.35
_HEIGHT_OFFSET = 0.42


def _default_orbit_params() -> dict:
    return {
        "center": (0.30, 0.05, 0.35),
        "radius": 2.4,
        "height": 0.05 + _HEIGHT_OFFSET,
        "yaw0": 0.9,
    }


def orbit_params_from_calib(
    calib_config: Optional[Path],
    cameras: Sequence[str],
    use_mcr_frame: bool,
    look_depth: float = 1.2,
) -> dict:
    """Return center/radius/height/yaw0 for a slow circular orbit in mcR."""
    if not use_mcr_frame or calib_config is None:
        return _default_orbit_params()

    try:
        config = load_config(calib_config)
        transforms, _, _ = build_camera_transforms(
            config, enabled_keys=list(cameras)
        )
    except Exception as exc:
        print(f"[orbit] Could not load calib ({exc}); using default")
        return _default_orbit_params()

    eyes = []
    focals = []
    for cam in cameras:
        if cam not in transforms:
            continue
        T_cam_to_mcR = build_T_cam_to_mcR(transforms[cam])
        eye = T_cam_to_mcR[:3, 3]
        forward = T_cam_to_mcR[:3, :3] @ np.array([0.0, 0.0, 1.0])
        n = np.linalg.norm(forward)
        if n < 1e-9:
            continue
        forward = forward / n
        eyes.append(eye)
        focals.append(eye + forward * float(look_depth))

    if not eyes:
        return _default_orbit_params()

    focal = np.mean(np.stack(focals, axis=0), axis=0)
    eye_mean = np.mean(np.stack(eyes, axis=0), axis=0)
    view_dir = focal - eye_mean
    n = np.linalg.norm(view_dir)
    if n < 1e-6:
        view_dir = np.array([1.0, 0.0, 0.0])
    else:
        view_dir = view_dir / n

    distance = max(
        float(look_depth) * (1.7 if len(eyes) == 1 else 1.45),
        float(np.linalg.norm(eye_mean - focal)) * 1.2,
        _MIN_RADIUS,
    )
    radius = max(_MIN_RADIUS, distance * _ZOOM)

    # Horizontal bearing of the initial overview direction (XZ, Y-up).
    yaw0 = float(np.arctan2(view_dir[2], view_dir[0]))
    height = float(focal[1] + _HEIGHT_OFFSET)
    return {
        "center": (float(focal[0]), float(focal[1]), float(focal[2])),
        "radius": float(radius),
        "height": height,
        "yaw0": yaw0,
    }


def recording_camera_view(
    T_cam_to_mcR: np.ndarray,
    look_depth: float = 1.2,
) -> dict:
    """Virtual camera that starts at a RealSense pose in Qualisys mcR.

    OpenCV optical frame: +Z forward, +Y down. World is Qualisys Y-up.
    """
    T = np.asarray(T_cam_to_mcR, dtype=np.float64)
    eye = T[:3, 3].copy()
    forward = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    n = np.linalg.norm(forward)
    if n < 1e-9:
        forward = np.array([1.0, 0.0, 0.0])
    else:
        forward = forward / n
    look_at = eye + forward * float(look_depth)
    up = T[:3, :3] @ np.array([0.0, -1.0, 0.0])
    un = np.linalg.norm(up)
    up = up / un if un > 1e-9 else np.array([0.0, 1.0, 0.0])
    return {"eye": eye, "forward": forward, "look_at": look_at, "up": up}


def pick_recording_camera(cameras: Sequence[str]) -> str:
    """Which RealSense pose to start from (LMR uses the middle camera)."""
    cams = [c.strip().upper() for c in cameras if c.strip()]
    if "M" in cams:
        return "M"
    return cams[0]


def yaw_vec_around_y(vec: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a direction around world +Y."""
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    c = np.cos(float(angle))
    s = np.sin(float(angle))
    x, y, z = v
    return np.array([c * x + s * z, y, -s * x + c * z], dtype=np.float64)


def yaw_eye_around_look_at(
    eye: np.ndarray,
    look_at: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Rotate the camera around world +Y through look_at (angle=0 keeps `eye`)."""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    look_at = np.asarray(look_at, dtype=np.float64).reshape(3)
    offset = eye - look_at
    c = np.cos(float(angle))
    s = np.sin(float(angle))
    x, y, z = offset
    return look_at + np.array([c * x + s * z, y, -s * x + c * z], dtype=np.float64)


def look_at_from_markers(frames, fallback: np.ndarray) -> np.ndarray:
    """Median Qualisys marker position (the infant), skipping empty startup frames."""
    pts = []
    n = len(frames)
    start = n // 5
    end = max(start + 1, (4 * n) // 5)
    for frame in frames[start:end]:
        for xyz in frame.values():
            if xyz is None:
                continue
            x, y, z = xyz
            if x == 0 and y == 0 and z == 0:
                continue
            pts.append(xyz)
    if len(pts) < 20:
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    return np.median(np.asarray(pts, dtype=np.float64), axis=0)


def eye_on_orbit(
    center: np.ndarray,
    radius: float,
    height: float,
    yaw0: float,
    angle: float,
) -> np.ndarray:
    """Camera position on a horizontal circle (Qualisys Y-up)."""
    center = np.asarray(center, dtype=np.float64).reshape(3)
    a = float(yaw0) + float(angle)
    return np.array(
        [
            center[0] + radius * np.cos(a),
            height,
            center[2] + radius * np.sin(a),
        ],
        dtype=np.float64,
    )


def camera_rotation_looking_at(
    eye: np.ndarray,
    target: np.ndarray,
    world_up: Optional[np.ndarray] = None,
) -> np.ndarray:
    """3x3 rotation for RViz FPS/Frame tracking (robot: +X forward, +Z up).

    RViz applies ROBOT_TO_CAMERA_ROTATION on top of this, so +X should point
    toward the look target.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = (
        np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if world_up is None
        else np.asarray(world_up, dtype=np.float64).reshape(3)
    )
    x_axis = target - eye  # forward (+X)
    n = np.linalg.norm(x_axis)
    if n < 1e-9:
        return np.eye(3)
    x_axis = x_axis / n
    y_axis = np.cross(up, x_axis)  # left (+Y) in right-handed robot frame? 
    # ROS robot: X fwd, Y left, Z up → Y = Z × X = up × forward
    yn = np.linalg.norm(y_axis)
    if yn < 1e-9:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = np.cross(up, x_axis)
        yn = np.linalg.norm(y_axis)
    y_axis = y_axis / yn
    z_axis = np.cross(x_axis, y_axis)  # up
    return np.column_stack([x_axis, y_axis, z_axis])


def rotation_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (x, y, z, w) from 3x3 rotation matrix."""
    R = np.asarray(R, dtype=np.float64)
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])
