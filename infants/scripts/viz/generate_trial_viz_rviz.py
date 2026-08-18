#!/usr/bin/env python3
"""Build an RViz config for trial visualization.

When Fixed Frame is qualisys_mcR, the view uses rviz/Orbit centered on the
workspace. Yaw is a 180deg back-and-forth driven by orbit_drag_yaw.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from calibration_chain import MOCAP_REF_FRAME
from orbit_view import ORBIT_CAM_FRAME, orbit_params_from_calib

CAMERA_TOPICS = {
    "L": "/cam_L/points",
    "M": "/cam_M/points",
    "R": "/cam_R/points",
}

CAMERA_FRAMES = {
    "L": "cam_L_color_optical_frame",
    "M": "cam_M_color_optical_frame",
    "R": "cam_R_color_optical_frame",
}

CAMERA_COLORS = {
    "L": "255; 64; 64",
    "M": "64; 255; 64",
    "R": "64; 128; 255",
}


def _pointcloud_display(name, topic, use_flat_color=False):
    if use_flat_color:
        color_block = f"""      Color: {CAMERA_COLORS[name]}
      Color Transformer: FlatColor"""
    else:
        color_block = """      Color: 255; 255; 255
      Color Transformer: RGB8
      Channel Name: rgb"""
    return f"""    - Alpha: 0.85
      Autocompute Intensity Bounds: true
      Autocompute Value Bounds:
        Max Value: 10
        Min Value: -10
        Value: true
      Axis: Z
      Class: rviz/PointCloud2
{color_block}
      Decay Time: 0
      Enabled: true
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Min Color: 0; 0; 0
      Name: cam_{name}
      Position Transformer: XYZ
      Queue Size: 10
      Selectable: true
      Size (Pixels): 5
      Size (m): 0.02
      Style: Points
      Topic: {topic}
      Unreliable: false
      Use Fixed Frame: true
      Value: true"""


def generate_rviz_config(
    cameras,
    show_markers,
    fixed_camera="L",
    use_mcr_frame=False,
    calib_config: Optional[Path] = None,
    look_depth: float = 1.2,
    animate_orbit: bool = True,
):
    fixed_frame = MOCAP_REF_FRAME if use_mcr_frame else CAMERA_FRAMES[fixed_camera]
    use_orbit = bool(animate_orbit and use_mcr_frame)
    orbit = None
    if use_orbit:
        orbit = orbit_params_from_calib(
            calib_config, cameras, use_mcr_frame, look_depth
        )
        print(
            "[rviz] Orbit view: "
            f"center={tuple(round(c, 3) for c in orbit['center'])} "
            f"distance={orbit['radius']:.2f}m (yaw animated while recording)"
        )

    displays = [
        """    - Alpha: 0.5
      Cell Size: 0.1
      Class: rviz/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 20
      Reference Frame: <Fixed Frame>
      Value: true""",
    ]

    if len(cameras) > 1 or use_mcr_frame:
        displays.append("""    - Class: rviz/TF
      Enabled: false
      Frame Timeout: 15
      Marker Scale: 0.15
      Name: TF
      Show Arrows: false
      Show Axes: true
      Show Names: false
      Update Interval: 0
      Value: false""")

    use_flat_color = False
    for cam in cameras:
        displays.append(_pointcloud_display(cam, CAMERA_TOPICS[cam], use_flat_color))

    if show_markers:
        displays.append("""    - Class: rviz/MarkerArray
      Enabled: true
      Marker Topic: /calibration_markers
      Name: CalibrationMarkers
      Queue Size: 100
      Value: true""")

    if use_orbit and orbit is not None:
        # Known-good Orbit view centered on the workspace. Yaw is animated during
        # recording by orbit_drag_yaw.py as a 180deg back-and-forth (no ROS yaw API).
        cx, cy, cz = orbit["center"]
        views = f"""    Current:
      Class: rviz/Orbit
      Distance: {orbit['radius']:.6f}
      Enable Stereo: false
      Eye Angle: 0
      Focal Point:
        X: {cx:.6f}
        Y: {cy:.6f}
        Z: {cz:.6f}
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.05
      Pitch: 0.620000
      Target Frame: <Fixed Frame>
      Yaw: {orbit['yaw0']:.6f}"""
    else:
        views = f"""    Current:
      Class: rviz/Orbit
      Distance: {look_depth * 1.15:.6f}
      Focal Point:
        X: 0.000000
        Y: 0.000000
        Z: {look_depth:.6f}
      Name: Current View
      Near Clip Distance: 0.05
      Pitch: 1.570796
      Target Frame: <Fixed Frame>
      Yaw: -1.570796"""

    body = f"""Panels:
  - Class: rviz/Displays
    Help Height: 0
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Grid1
      Splitter Ratio: 0.5
    Tree Height: 100
Visualization Manager:
  Class: ""
  Displays:
{chr(10).join(displays)}
  Enabled: true
  Global Options:
    Background Color: 72; 72; 76
    Default Light: true
    Fixed Frame: {fixed_frame}
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz/Interact
      Hide Inactive Objects: true
    - Class: rviz/MoveCamera
    - Class: rviz/Select
    - Class: rviz/FocusCamera
  Value: true
  Views:
{views}
    Saved: ~
Window Geometry:
  Displays:
    collapsed: true
  Height: 1080
  Hide Left Dock: true
  Hide Right Dock: true
  Width: 1920
  X: 0
  Y: 0
Preferences:
  PromptSaveOnExit: false
"""
    return body


def write_rviz_config(
    output_path,
    cameras,
    show_markers,
    fixed_camera="L",
    use_mcr_frame=False,
    calib_config: Optional[Path] = None,
    look_depth: float = 1.2,
    animate_orbit: bool = True,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        generate_rviz_config(
            cameras,
            show_markers,
            fixed_camera,
            use_mcr_frame,
            calib_config=calib_config,
            look_depth=look_depth,
            animate_orbit=animate_orbit,
        )
    )
    return output_path
