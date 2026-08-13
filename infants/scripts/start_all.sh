#!/bin/bash

# Ensure ROS tools are on PATH even if venv activation reordered PATH.
source /opt/ros/noetic/setup.bash

set -euo pipefail

TOPICS=(
  "/cam_L/color/image_raw"
  "/cam_L/aligned_depth_to_color/image_raw"
  "/cam_M/color/image_raw"
  "/cam_M/aligned_depth_to_color/image_raw"
  "/cam_R/color/image_raw"
  "/cam_R/aligned_depth_to_color/image_raw"
)

echo "[INFO] Launching cameras/audio..."
roslaunch launch/rs_cam.launch &
LAUNCH_PID=$!

cleanup() {
  if kill -0 "$LAUNCH_PID" 2>/dev/null; then
    kill "$LAUNCH_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

# Allow roslaunch to bring up roscore and nodes.
sleep 6

# Live cameras use wall-clock time; clear leftover from bag viz launches.
rosparam set /use_sim_time false || true

deadline=$((SECONDS + 120))
while (( SECONDS < deadline )); do
  missing=()
  for topic in "${TOPICS[@]}"; do
    if ! timeout 2s rostopic echo -n 1 "$topic" >/dev/null 2>&1; then
      missing+=("$topic")
    fi
  done

  if (( ${#missing[@]} == 0 )); then
    echo "[OK] All 6 camera streams are publishing."
    wait "$LAUNCH_PID"
    exit $?
  fi

  echo "[WAIT] Still waiting on ${#missing[@]} topic(s): ${missing[*]}"
  sleep 5
done

echo "[INFO] Launching audio..."
roslaunch --wait audio_capture capture.launch \
  ns:=audio/ device:=hw:1,0 sample_rate:=44100 channels:=1 &

echo "[ERROR] Timed out waiting for all camera topics."
echo "[ERROR] Missing topics: ${missing[*]}"
echo "[HINT] Re-run ./start_all.sh; USB contention can delay startup."
cleanup
exit 1
