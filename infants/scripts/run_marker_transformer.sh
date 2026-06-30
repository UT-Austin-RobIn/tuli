#!/bin/bash
source /opt/ros/noetic/setup.bash
export PYTHONPATH="$(dirname "$0"):${PYTHONPATH:-}"
exec python3 "$(dirname "$0")/marker_transformer.py" "$@"
