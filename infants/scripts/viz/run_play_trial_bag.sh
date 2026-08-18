#!/bin/bash
# launch-prefix wrapper: rewrite `rosbag play ... BAG` into play without /marker_*.
# Invoked as: run_play_trial_bag.sh <rosbag-play-path> play --clock ... BAG.bag
set -euo pipefail
source /opt/ros/noetic/setup.bash

# Drop the rosbag binary path; keep "play" and the rest.
shift || true
if [[ "${1:-}" == "play" ]]; then
  shift
fi

BAG=""
ARGS=()
for a in "$@"; do
  if [[ -z "$BAG" && -f "$a" && "$a" == *.bag ]]; then
    BAG="$a"
  else
    ARGS+=("$a")
  fi
done

if [[ -z "$BAG" ]]; then
  echo "[ERROR] no bag file in rosbag play args: $*" >&2
  exit 1
fi

mapfile -t TOPICS < <(python3 - "$BAG" <<'PY'
import sys
import rosbag
bag = rosbag.Bag(sys.argv[1], "r")
topics = sorted(
    t for t in bag.get_type_and_topic_info().topics
    if not t.startswith("/marker_")
)
print("\n".join(topics))
PY
)

if [[ ${#TOPICS[@]} -eq 0 ]]; then
  exec rosbag play "${ARGS[@]}" "$BAG"
fi

echo "[INFO] Playing bag without /marker_* (${#TOPICS[@]} topics)" >&2
exec rosbag play "${ARGS[@]}" "$BAG" --topics "${TOPICS[@]}"
