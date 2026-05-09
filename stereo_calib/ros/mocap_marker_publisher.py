#!/usr/bin/env python
"""ROS1 node: subscribe to a geometry_msgs/PointStamped mocap stream and
republish each sample as a single visualization_msgs/Marker (red sphere) so
RViz can display it on top of the calibrated point clouds.

Params (~private):
    ~input_topic    : str (default /marker_1)
    ~output_topic   : str (default /mocap_markers)
    ~frame_id       : str — overrides the inbound header.frame_id (default qualisys_ref)
    ~sphere_radius  : float — sphere diameter in meters (default 0.03)
    ~color_rgba     : 4 floats — marker color (default red: 1 0 0 1)
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# rospy on this conda env still calls Thread.isAlive (removed in Python 3.9).
# Restore the alias so SIGINT/SIGTERM shutdown doesn't AttributeError.
if not hasattr(threading.Thread, "isAlive"):
    threading.Thread.isAlive = threading.Thread.is_alive

import rospy
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker


class MocapMarkerRepublisher:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/marker_1")
        self.output_topic = rospy.get_param("~output_topic", "/mocap_markers")
        self.frame_id = rospy.get_param("~frame_id", "qualisys_ref")
        self.radius = float(rospy.get_param("~sphere_radius", 0.03))
        rgba = rospy.get_param("~color_rgba", [1.0, 0.0, 0.0, 1.0])
        if len(rgba) != 4:
            rospy.logwarn(f"~color_rgba expects 4 floats, got {rgba!r}; using red")
            rgba = [1.0, 0.0, 0.0, 1.0]
        self.r, self.g, self.b, self.a = (float(x) for x in rgba)

        self.pub = rospy.Publisher(self.output_topic, Marker, queue_size=10)
        self.sub = rospy.Subscriber(self.input_topic, PointStamped, self._cb, queue_size=50)
        rospy.loginfo(
            f"mocap_marker_publisher: {self.input_topic} -> {self.output_topic} "
            f"(frame_id={self.frame_id}, radius={self.radius})"
        )

    def _cb(self, msg: PointStamped):
        m = Marker()
        m.header.stamp = msg.header.stamp
        m.header.frame_id = self.frame_id
        m.ns = "mocap_marker"
        m.id = 0  # single marker, overwrite each frame
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = msg.point.x
        m.pose.position.y = msg.point.y
        m.pose.position.z = msg.point.z
        m.pose.orientation.w = 1.0
        d = 2.0 * self.radius
        m.scale.x = d
        m.scale.y = d
        m.scale.z = d
        m.color.r = self.r
        m.color.g = self.g
        m.color.b = self.b
        m.color.a = self.a
        # Hold the marker on screen ~half a sample interval at 100 Hz so brief
        # gaps don't make it disappear, but new samples always replace it.
        m.lifetime = rospy.Duration(0.2)
        self.pub.publish(m)


def main():
    rospy.init_node("mocap_marker_publisher")
    MocapMarkerRepublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
