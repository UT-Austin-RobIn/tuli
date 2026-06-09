#!/bin/bash

source /opt/ros/noetic/setup.bash
set -euo pipefail

INTERVAL_SEC="${INTERVAL_SEC:-3}"
TOPICS=(
  "/cam_L/color/image_raw"
  "/cam_L/aligned_depth_to_color/image_raw"
  "/cam_M/color/image_raw"
  "/cam_M/aligned_depth_to_color/image_raw"
  "/cam_R/color/image_raw"
  "/cam_R/aligned_depth_to_color/image_raw"
)

echo "[INFO] ROS_MASTER_URI=${ROS_MASTER_URI:-<unset>}"
if ! timeout 2s rostopic list >/dev/null 2>&1; then
  echo "[ERROR] Cannot reach ROS master."
  echo "[HINT] Start camera stack first (./start_all.sh or ./start_all_staggered.sh)."
  exit 2
fi

trap 'echo; echo "[INFO] Stopping camera check."; exit 0' INT TERM

while true; do
  echo
  echo "[INFO] Checking one message on each topic..."
  missing=()
  present=()
  for topic in "${TOPICS[@]}"; do
    if timeout 3s rostopic echo -n 1 "$topic" >/dev/null 2>&1; then
      echo "[OK] $topic"
      present+=("$topic")
    else
      echo "[MISS] $topic"
      missing+=("$topic")
    fi
  done

  if ((${#present[@]} > 0)); then
    echo
    echo "[INFO] Sampling rates for available topics (5s)..."
    timeout 5s rostopic hz "${present[@]}" || true
  fi

  if ((${#missing[@]} > 0)); then
    echo
    echo "[WARN] Missing ${#missing[@]} topic(s): ${missing[*]}"
    echo "[HINT] USB contention likely if RealSense logs show RS2_USB_STATUS_BUSY."
  else
    echo
    echo "[OK] All 6 camera topics are active."
  fi

  echo "[INFO] Rechecking in ${INTERVAL_SEC}s. Press Ctrl+C to stop."
  sleep "$INTERVAL_SEC"
done
