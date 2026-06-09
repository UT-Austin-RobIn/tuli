#!/bin/bash

source /opt/ros/noetic/setup.bash

# Keep ROS logs within writable workspace
export ROS_HOME="/home/robotlearning2/infants/.ros"
export ROS_LOG_DIR="${ROS_HOME}/log"
mkdir -p "$ROS_LOG_DIR"

set -euo pipefail

TOPICS=(
  "/cam_L/color/image_raw"
  "/cam_L/aligned_depth_to_color/image_raw"
  "/cam_M/color/image_raw"
  "/cam_M/aligned_depth_to_color/image_raw"
  "/cam_R/color/image_raw"
  "/cam_R/aligned_depth_to_color/image_raw"
)

LAUNCH_PIDS=()
ROSCORE_PID=""
STARTED_ROSCORE=0

cleanup() {
  for pid in "${LAUNCH_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  if [[ "$STARTED_ROSCORE" -eq 1 ]] && [[ -n "${ROSCORE_PID:-}" ]] && kill -0 "$ROSCORE_PID" 2>/dev/null; then
    kill "$ROSCORE_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

wait_for_master() {
  for _ in {1..12}; do
    if timeout 1s rostopic list >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_for_topics() {
  local label="$1"
  local timeout_sec="$2"
  shift 2
  local topics=("$@")
  local end_ts=$((SECONDS + timeout_sec))
  local missing=()

  while ((SECONDS < end_ts)); do
    missing=()
    for topic in "${topics[@]}"; do
      if ! timeout 2s rostopic echo -n 1 "$topic" >/dev/null 2>&1; then
        missing+=("$topic")
      fi
    done

    if ((${#missing[@]} == 0)); then
      return 0
    fi

    sleep 1
  done

  echo "[WARN] ${label} missing topic(s): ${missing[*]}"
  return 1
}

launch_camera_with_retry() {
  local cam_name="$1"
  local serial="$2"
  local cam_topics=(
    "/${cam_name}/color/image_raw"
    "/${cam_name}/aligned_depth_to_color/image_raw"
  )

  local attempt
  for attempt in 1 2 3; do
    echo "[INFO] Launching ${cam_name} (attempt ${attempt}/3)..."
    roslaunch --wait realsense2_camera rs_camera.launch \
      camera:="${cam_name}" serial_no:="${serial}" usb_port_id:= \
      align_depth:=true depth_width:=848 depth_height:=480 depth_fps:=30 \
      color_width:=1280 color_height:=720 color_fps:=30 \
      enable_sync:=true &
    local pid=$!

    sleep 2
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[WARN] ${cam_name} launcher exited immediately."
      sleep 2
      continue
    fi

    if wait_for_topics "${cam_name}" 25 "${cam_topics[@]}"; then
      echo "[OK] ${cam_name} topics are publishing."
      LAUNCH_PIDS+=("$pid")
      return 0
    fi

    echo "[WARN] ${cam_name} did not stabilize. Restarting launcher..."
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sleep 4
  done

  echo "[ERROR] Failed to bring up ${cam_name} after 3 attempts."
  return 1
}

echo "[INFO] Cleaning stale camera/audio processes..."
pkill -f realsense2_camera || true
pkill -f audio_capture || true
sleep 2

if wait_for_master; then
  echo "[INFO] Reusing existing ROS master at ${ROS_MASTER_URI:-http://localhost:11311}"
else
  echo "[INFO] Launching roscore..."
  roscore >/tmp/roscore.out 2>&1 &
  ROSCORE_PID=$!
  STARTED_ROSCORE=1
  if ! wait_for_master; then
    echo "[ERROR] roscore did not become ready. See /tmp/roscore.out"
    exit 1
  fi
fi

# Ensure live camera checks use wall-clock time.
rosparam set /use_sim_time false || true

launch_camera_with_retry cam_L 332322072918
sleep 6
launch_camera_with_retry cam_M 332522077342
sleep 6
launch_camera_with_retry cam_R 327122075069
sleep 2

echo "[INFO] Launching audio..."
roslaunch --wait audio_capture capture.launch \
  ns:=audio/ device:=hw:1,0 sample_rate:=44100 channels:=1 &
LAUNCH_PIDS+=("$!")

STABLE_ROUNDS=4
stable_ok_rounds=0
deadline=$((SECONDS + 90))

while ((SECONDS < deadline)); do
  for pid in "${LAUNCH_PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR] A launch process exited unexpectedly (pid=$pid)."
      echo "[HINT] Check ROS logs under $ROS_LOG_DIR/latest/"
      cleanup
      exit 1
    fi
  done

  missing=()
  for topic in "${TOPICS[@]}"; do
    if ! timeout 2s rostopic echo -n 1 "$topic" >/dev/null 2>&1; then
      missing+=("$topic")
    fi
  done

  if ((${#missing[@]} == 0)); then
    stable_ok_rounds=$((stable_ok_rounds + 1))
    echo "[WAIT] All topics seen this round (${stable_ok_rounds}/${STABLE_ROUNDS})..."
    if ((stable_ok_rounds >= STABLE_ROUNDS)); then
      echo "[OK] All 6 camera streams are stably publishing."
      wait "${LAUNCH_PIDS[@]}"
      exit $?
    fi
  else
    stable_ok_rounds=0
    echo "[WAIT] Still waiting on ${#missing[@]} topic(s): ${missing[*]}"
  fi

  sleep 3
done

echo "[ERROR] Timed out waiting for stable topic publish."
cleanup
exit 1
