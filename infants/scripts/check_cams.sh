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
AUDIO_TOPIC="/audio/audio"

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

  if timeout 3s rostopic echo -n 1 "$AUDIO_TOPIC" >/dev/null 2>&1; then
    echo "[OK] $AUDIO_TOPIC"
    present+=("$AUDIO_TOPIC")
  else
    echo "[MISS] $AUDIO_TOPIC"
    missing+=("$AUDIO_TOPIC")
  fi

  if ((${#present[@]} > 0)); then
    echo
    echo "[INFO] Sampling rates for available topics (5s)..."
    timeout 5s rostopic hz "${present[@]}" || true
  fi

  if ((${#missing[@]} > 0)); then
    echo
    echo "[WARN] Missing ${#missing[@]} topic(s): ${missing[*]}"
    camera_missing=false
    audio_missing=false
    for topic in "${missing[@]}"; do
      if [[ "$topic" == "$AUDIO_TOPIC" ]]; then
        audio_missing=true
      else
        camera_missing=true
      fi
    done
    if $camera_missing; then
      echo "[HINT] USB contention likely if RealSense logs show RS2_USB_STATUS_BUSY."
    fi
    if $audio_missing; then
      echo "[HINT] Audio may have failed to start. Check: rosnode list | grep audio"
      echo "[HINT] Verify ALSA device with: arecord -l  (USB mic is often hw:2,0)"
    fi
  else
    echo
    echo "[OK] All 6 camera topics and audio are active."
  fi

  echo "[INFO] Rechecking in ${INTERVAL_SEC}s. Press Ctrl+C to stop."
  sleep "$INTERVAL_SEC"
done
