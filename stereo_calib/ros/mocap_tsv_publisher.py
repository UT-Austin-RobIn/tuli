#!/usr/bin/env python3
"""Publish a Qualisys multi-marker TSV as visualization_msgs/MarkerArray, time-aligned
to /clock (sim time from a rosbag).

TSV header has TIME_STAMP "<YYYY-MM-DD>, <HH:MM:SS.fff>  <internal>". Coords are mm,
converted to meters. Frame published in ~frame_id (default qualisys_ref).

Params (~private):
    ~tsv_path     : str (required)
    ~frame_id     : str (default qualisys_ref)
    ~topic        : str (default /mocap_markers)
    ~sphere_radius: float meters (default 0.02)
    ~rate_hz      : float publish rate (default 50)
"""
from __future__ import annotations

import sys
import threading
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if not hasattr(threading.Thread, "isAlive"):
    threading.Thread.isAlive = threading.Thread.is_alive

import numpy as np
import rospy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def parse_tsv(path: Path):
    with path.open("r") as f:
        lines = f.readlines()
    meta = {}
    end = None
    for i, line in enumerate(lines):
        if line.startswith("TRAJECTORY_TYPES"):
            end = i + 1
            break
        if "\t" in line:
            k, _, v = line.partition("\t")
            meta[k.strip()] = v.rstrip("\n").rstrip("\r")
    if end is None:
        raise RuntimeError("TSV missing TRAJECTORY_TYPES line")
    n_frames = int(meta["NO_OF_FRAMES"])
    n_markers = int(meta["NO_OF_MARKERS"])
    freq = float(meta["FREQUENCY"])
    ts_field = meta["TIME_STAMP"]
    # "2026-05-01, 11:55:04.077\t911440.55765990"
    date_time = ts_field.split("\t")[0].strip()
    t0 = datetime.strptime(date_time, "%Y-%m-%d, %H:%M:%S.%f").timestamp()

    data = np.zeros((n_frames, n_markers, 3), dtype=np.float64)
    for row, line in enumerate(lines[end:end + n_frames]):
        parts = line.rstrip().split("\t")
        # parts[0]=frame_idx, parts[1]=elapsed_sec, then X Y Z per marker
        coords = np.array(parts[2:2 + 3 * n_markers], dtype=np.float64)
        data[row] = coords.reshape(n_markers, 3)
    return data, freq, t0, n_markers


def color_for(idx: int, n: int) -> ColorRGBA:
    # cycle hue
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((idx / max(n, 1)) % 1.0, 0.85, 1.0)
    return ColorRGBA(r=r, g=g, b=b, a=1.0)


def main():
    rospy.init_node("mocap_tsv_publisher")
    tsv_path = Path(rospy.get_param("~tsv_path"))
    frame_id = rospy.get_param("~frame_id", "qualisys_ref")
    topic = rospy.get_param("~topic", "/mocap_markers")
    radius = float(rospy.get_param("~sphere_radius", 0.02))
    rate_hz = float(rospy.get_param("~rate_hz", 50.0))

    data, freq, t0, n_markers = parse_tsv(tsv_path)
    rospy.loginfo(
        f"mocap_tsv_publisher: loaded {data.shape[0]} frames, {n_markers} markers "
        f"from {tsv_path.name}; freq={freq} Hz; t0_unix={t0:.3f}"
    )
    colors = [color_for(i, n_markers) for i in range(n_markers)]
    pub = rospy.Publisher(topic, MarkerArray, queue_size=1)

    # Wait for /clock to become non-zero so we don't index before bag starts.
    while not rospy.is_shutdown() and rospy.get_rostime().to_sec() == 0:
        rospy.sleep(0.05)

    rate = rospy.Rate(rate_hz)
    n_frames = data.shape[0]
    while not rospy.is_shutdown():
        now = rospy.get_rostime()
        idx = int(round((now.to_sec() - t0) * freq))
        if 0 <= idx < n_frames:
            arr = MarkerArray()
            stamp = now
            for m_id in range(n_markers):
                xyz_mm = data[idx, m_id]
                if not np.any(xyz_mm):  # untracked
                    continue
                m = Marker()
                m.header.stamp = stamp
                m.header.frame_id = frame_id
                m.ns = "mocap"
                m.id = m_id
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = xyz_mm[0] / 1000.0
                m.pose.position.y = xyz_mm[1] / 1000.0
                m.pose.position.z = xyz_mm[2] / 1000.0
                m.pose.orientation.w = 1.0
                d = 2.0 * radius
                m.scale.x = m.scale.y = m.scale.z = d
                m.color = colors[m_id]
                m.lifetime = rospy.Duration(0.5)
                arr.markers.append(m)
            if arr.markers:
                pub.publish(arr)
        try:
            rate.sleep()
        except rospy.ROSTimeMovedBackwardsException:
            pass


if __name__ == "__main__":
    main()
