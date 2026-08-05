#!/usr/bin/env python3
"""Build an RViz config for trial visualization."""
from pathlib import Path

from calibration_chain import MOCAP_REF_FRAME

CAMERA_TOPICS = {
    "L": "/cam_L/points",
    "M": "/cam_M/points",
    "R": "/cam_R/points",
}

CAMERA_IMAGE_TOPICS = {
    "L": "/cam_L/color/image_raw",
    "M": "/cam_M/color/image_raw",
    "R": "/cam_R/color/image_raw",
}

CAMERA_FRAMES = {
    "L": "cam_L_color_optical_frame",
    "M": "cam_M_color_optical_frame",
    "R": "cam_R_color_optical_frame",
}

# Distinct flat colors so overlapping aligned clouds are still distinguishable.
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


def _image_display(name, topic):
    """2D RealSense RGB so the camera video is visible (not only the point cloud)."""
    return f"""    - Class: rviz/Image
      Enabled: true
      Image Topic: {topic}
      Max Value: 1
      Median window: 5
      Min Value: 0
      Name: cam_{name}_image
      Normalize Range: true
      Queue Size: 2
      Transport Hint: raw
      Unreliable: false
      Value: true"""


def generate_rviz_config(cameras, show_markers, fixed_camera="L", use_mcr_frame=False):
    # When use_mcr_frame, RViz Fixed Frame = qualisys_mcR; TF qualisys_mcR->cam_X
    # reprojects each camera-frame cloud into the mocap reference frame.
    fixed_frame = MOCAP_REF_FRAME if use_mcr_frame else CAMERA_FRAMES[fixed_camera]
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
      Enabled: true
      Frame Timeout: 15
      Marker Scale: 0.2
      Name: TF
      Show Arrows: true
      Show Axes: true
      Show Names: false
      Update Interval: 0
      Value: true""")

    # use_flat_color = len(cameras) > 1
    use_flat_color = False
    for cam in cameras:
        displays.append(_pointcloud_display(cam, CAMERA_TOPICS[cam], use_flat_color))
        displays.append(_image_display(cam, CAMERA_IMAGE_TOPICS[cam]))

    if show_markers:
        displays.append("""    - Class: rviz/MarkerArray
      Enabled: true
      Marker Topic: /calibration_markers
      Name: CalibrationMarkers
      Queue Size: 100
      Value: true""")

    expanded = ["        - /Global Options1", "        - /Grid1"]
    for cam in cameras:
        expanded.append(f"        - /cam_{cam}1")
        expanded.append(f"        - /cam_{cam}_image1")
    if show_markers:
        expanded.append("        - /CalibrationMarkers1")

    # Image panel(s) so RGB video is on-screen without hunting Displays
    image_panels = []
    for i, cam in enumerate(cameras):
        image_panels.append(
            f"""  - Class: rviz/Image
    Name: cam_{cam} RGB
    Topic: {CAMERA_IMAGE_TOPICS[cam]}
    Transport Hint: raw
    Unreliable: false
    Window Geometry:
      X: 800
      Y: {50 + i * 420}
      Width: 640
      Height: 400"""
        )

    body = f"""Panels:
  - Class: rviz/Displays
    Help Height: 70
    Name: Displays
    Property Tree Widget:
      Expanded:
{chr(10).join(expanded)}
      Splitter Ratio: 0.5
    Tree Height: 600
  - Class: rviz/Selection
    Name: Selection
  - Class: rviz/Views
    Expanded:
      - /Current View1
    Name: Views
{chr(10).join(image_panels)}
Visualization Manager:
  Class: ""
  Displays:
{chr(10).join(displays)}
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
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
    Current:
      Class: rviz/Orbit
      Distance: 3.0
      Focal Point:
        X: 0.3
        Y: 0.0
        Z: 0.4
      Name: Current View
      Near Clip Distance: 0.01
      Pitch: 0.6
      Target Frame: <Fixed Frame>
      Yaw: 0.9
    Saved: ~
Window Geometry:
  Height: 1000
  Width: 1600
  X: 40
  Y: 40
Preferences:
  PromptSaveOnExit: false
"""
    return body


def write_rviz_config(output_path, cameras, show_markers, fixed_camera="L", use_mcr_frame=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        generate_rviz_config(cameras, show_markers, fixed_camera, use_mcr_frame)
    )
    return output_path
