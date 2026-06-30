#!/bin/bash
source /opt/ros/noetic/setup.bash
exec python3 "$(dirname "$0")/depth_to_pointcloud.py" "$@"
