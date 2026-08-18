#!/bin/bash
# Play a trial bag with --clock, skipping /marker_* (markers come from
# marker_transformer bag-backed reader). Keeps playback from stalling.
set -euo pipefail
source /opt/ros/noetic/setup.bash

BAG=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bag) BAG="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
if [[ -z "$BAG" ]]; then
  echo "Usage: $0 --bag BAG.bag [rosbag play args...]" >&2
  exit 1
fi

# Build topic list: everything except /marker_N
mapfile -t TOPICS < <(python3 - "$BAG" <<'PY'
import sys
import rosbag
bag = rosbag.Bag(sys.argv[1], "r")
topics = sorted(t for t in bag.get_type_and_topic_info().topics if not t.startswith("/marker_"))
print("\n".join(topics))
PY
)

if [[ ${#TOPICS[@]} -eq 0 ]]; then
  echo "[WARN] No non-marker topics found; playing full bag" >&2
  exec rosbag play "${EXTRA[@]}" "$BAG"
fi

echo "[INFO] rosbag play excluding /marker_* (${#TOPICS[@]} topics)" >&2
exec rosbag play "${EXTRA[@]}" "$BAG" --topics "${TOPICS[@]}"
