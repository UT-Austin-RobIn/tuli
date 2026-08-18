#!/usr/bin/env python3
"""Publish Qualisys markers for RViz alongside calibrated point clouds.

When ~bag_file is set (trial viz), markers are read from the bag and sampled
against /clock. That avoids hundreds of /marker_N subscribers, which stall
rosbag --clock and freeze the orbit view.

reference_frame=qualisys_mcR: markers stay in mcR (same as aligned clouds).
reference_frame=cameras: project into each enabled camera optical frame.
"""
from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray

from calibration_chain import (
    MOCAP_REF_FRAME,
    build_camera_transforms,
    load_config,
    resolve_config_path,
)


def _load_marker_timeline(bag_path: Path, num_markers: int):
    """Return (times_sec sorted, frames) where frames[i] is {mid: xyz_m}."""
    import rosbag

    topics = {f"/marker_{m}" for m in range(1, num_markers + 1)}
    # time_key (rounded) -> {mid: xyz}
    buckets = {}
    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, _t in bag.read_messages(topics=list(topics)):
            if topic not in topics:
                continue
            mid = int(topic.rsplit("_", 1)[-1])
            ts = float(msg.header.stamp.to_sec())
            # ~100 Hz mocap: 5 ms buckets keep one sample per marker per frame.
            key = round(ts * 200.0) / 200.0
            x, y, z = msg.point.x, msg.point.y, msg.point.z
            if np.isnan(x) or np.isnan(y) or np.isnan(z) or (x == 0 and y == 0 and z == 0):
                xyz = None
            else:
                xyz = (x / 1000.0, y / 1000.0, z / 1000.0)
            bucket = buckets.setdefault(key, {})
            bucket[mid] = xyz

    times = sorted(buckets.keys())
    frames = [buckets[t] for t in times]
    return times, frames


class MarkerTransformer:
    def __init__(self, config, enabled_cameras, reference_frame, max_markers, bag_file=""):
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
            "/calibration_markers", MarkerArray, queue_size=1, latch=True
        )
        cfg_n = int(config.get("num_markers", 1))
        if int(max_markers) <= 0:
            self.num_markers = max(1, cfg_n)
        else:
            self.num_markers = max(1, min(cfg_n, int(max_markers)))
        # visualization_msgs SPHERE scale is diameter (meters). Honor YAML; old
        # max(0.05) forced 5 cm balls that swallowed the infant.
        raw = float(config.get("sphere_radius", 0.02))
        self.sphere_radius = min(0.04, max(0.008, raw))
        self._latest = {}
        self._dirty = False
        self._shown = set()
        self._times = []
        self._frames = []
        self._bag_mode = False

        if reference_frame == MOCAP_REF_FRAME:
            rospy.loginfo(
                "Markers published in %s (mm->m)",
                MOCAP_REF_FRAME,
            )
        else:
            if not self.transforms:
                rospy.logwarn("No camera transforms; camera-frame markers empty")
            for cam_key, frame_id in self.frame_ids.items():
                rospy.loginfo(
                    "  marker overlay: mcR -> cam_%s (%s)",
                    cam_key,
                    frame_id,
                )

        bag_path = Path(bag_file).expanduser() if bag_file else None
        if bag_path is not None and bag_path.is_file():
            rospy.loginfo(
                "Loading markers from bag %s (/%d topics, background, no live subscribers)",
                bag_path,
                self.num_markers,
            )
            import threading

            def _load():
                try:
                    times, frames = _load_marker_timeline(bag_path, self.num_markers)
                    self._times = times
                    self._frames = frames
                    self._bag_mode = True
                    rospy.loginfo(
                        "Loaded %d marker frames (t0=%.3f t1=%.3f)",
                        len(times),
                        times[0] if times else -1.0,
                        times[-1] if times else -1.0,
                    )
                except Exception as exc:
                    rospy.logerr("Failed to load markers from bag: %s", exc)

            threading.Thread(target=_load, daemon=True).start()
            rospy.Timer(rospy.Duration(0.05), self._bag_sample_timer)
        else:
            if bag_file:
                rospy.logwarn(
                    "bag_file not found (%s); falling back to live /marker_N", bag_file
                )
            rospy.loginfo("Subscribing live to /marker_1..%d", self.num_markers)
            for m in range(1, self.num_markers + 1):
                rospy.Subscriber(
                    f"/marker_{m}",
                    PointStamped,
                    self.marker_cb,
                    callback_args=m,
                    queue_size=1,
                )

        rospy.Timer(rospy.Duration(0.05), self._publish_timer)

    def _bag_sample_timer(self, _event):
        now = rospy.Time.now().to_sec()
        if now <= 0.0 or not self._times:
            return
        # Latest frame at or before sim time.
        i = bisect.bisect_right(self._times, now) - 1
        if i < 0:
            return
        frame = self._frames[i]
        stamp = rospy.Time.from_sec(self._times[i])
        latest = {}
        for mid, xyz in frame.items():
            latest[mid] = (stamp, None if xyz is None else np.asarray(xyz, dtype=np.float64))
        self._latest = latest
        self._dirty = True

    def marker_cb(self, msg, marker_idx):
        pt = np.array(
            [msg.point.x / 1000.0, msg.point.y / 1000.0, msg.point.z / 1000.0],
            dtype=np.float64,
        )
        if np.any(np.isnan(pt)) or (pt == 0.0).all():
            self._latest[marker_idx] = (msg.header.stamp, None)
        else:
            self._latest[marker_idx] = (msg.header.stamp, pt)
        self._dirty = True

    def _publish_timer(self, _event):
        if not self._dirty and not self._latest:
            return
        self._dirty = False
        ma = MarkerArray()
        stamp = rospy.Time(0)
        if self.reference_frame == MOCAP_REF_FRAME:
            want_shown = set()
            for marker_idx, (_st, pt) in self._latest.items():
                if pt is None:
                    continue
                want_shown.add(int(marker_idx))
                m = Marker()
                m.header.stamp = stamp
                m.header.frame_id = MOCAP_REF_FRAME
                m.ns = "qualisys_markers"
                m.id = int(marker_idx)
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(pt[0])
                m.pose.position.y = float(pt[1])
                m.pose.position.z = float(pt[2])
                m.pose.orientation.w = 1.0
                m.scale.x = self.sphere_radius
                m.scale.y = self.sphere_radius
                m.scale.z = self.sphere_radius
                m.color.r = 1.0
                m.color.g = 0.05
                m.color.b = 0.05
                m.color.a = 1.0
                m.lifetime = rospy.Duration(0)
                ma.markers.append(m)
            for marker_idx in sorted(self._shown - want_shown):
                m = Marker()
                m.header.stamp = stamp
                m.header.frame_id = MOCAP_REF_FRAME
                m.ns = "qualisys_markers"
                m.id = int(marker_idx)
                m.action = Marker.DELETE
                ma.markers.append(m)
            self._shown = want_shown
        else:
            mid = 0
            for marker_idx, (_st, pt) in self._latest.items():
                if pt is None:
                    continue
                pt_h = np.array([pt[0], pt[1], pt[2], 1.0])
                for cam_key, T_mcR_to_cam in self.transforms.items():
                    pt_cam = T_mcR_to_cam @ pt_h
                    m = Marker()
                    m.header.stamp = stamp
                    m.header.frame_id = self.frame_ids[cam_key]
                    m.ns = f"qualisys_markers_cam_{cam_key}"
                    m.id = mid
                    mid += 1
                    m.type = Marker.SPHERE
                    m.action = Marker.ADD
                    m.pose.position.x = float(pt_cam[0])
                    m.pose.position.y = float(pt_cam[1])
                    m.pose.position.z = float(pt_cam[2])
                    m.pose.orientation.w = 1.0
                    m.scale.x = self.sphere_radius
                    m.scale.y = self.sphere_radius
                    m.scale.z = self.sphere_radius
                    c = self.colors[cam_key]
                    m.color.r = c[0]
                    m.color.g = c[1]
                    m.color.b = c[2]
                    m.color.a = 1.0
                    m.lifetime = rospy.Duration(0)
                    ma.markers.append(m)

        if not ma.markers:
            return
        try:
            self.marker_pub.publish(ma)
        except rospy.ROSException:
            return


def parse_camera_list(raw):
    if not raw:
        return ["L"]
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def main():
    rospy.init_node("marker_transformer")
    config_path = resolve_config_path(rospy.get_param("~config"))
    enabled = parse_camera_list(rospy.get_param("~cameras", "L"))
    reference_frame = rospy.get_param("~reference_frame", MOCAP_REF_FRAME)
    max_markers = int(rospy.get_param("~max_markers", 0))
    bag_file = rospy.get_param("~bag_file", "")
    config = load_config(config_path)
    rospy.loginfo(
        "Starting marker transformer (cameras=%s, reference_frame=%s, max_markers=%s, bag=%s)",
        ",".join(enabled),
        reference_frame,
        "uncapped" if max_markers <= 0 else str(max_markers),
        bag_file or "live",
    )
    MarkerTransformer(config, enabled, reference_frame, max_markers, bag_file=bag_file)
    rospy.spin()


if __name__ == "__main__":
    main()
