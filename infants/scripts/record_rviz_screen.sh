#!/bin/bash

set -euo pipefail

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[ERROR] ffmpeg not found. Install it first."
  exit 1
fi

if ! command -v xdpyinfo >/dev/null 2>&1; then
  echo "[ERROR] xdpyinfo not found. Install x11-utils first."
  exit 1
fi

OUTPUT_PATH="${1:-}"
if [[ -z "$OUTPUT_PATH" ]]; then
  echo "Usage: $0 OUTPUT_PATH.mp4 [DISPLAY_ID]"
  echo "Example: $0 /home/robotlearning2/infants/recordings/rviz_$(date +%Y%m%d_%H%M%S).mp4 :0"
  exit 1
fi

DISPLAY_ID="${2:-${DISPLAY:-:0}}"
FPS="${FPS:-30}"
CRF="${CRF:-23}"
PRESET="${PRESET:-veryfast}"
VIDEO_SIZE="${VIDEO_SIZE:-}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

set +e
DIMENSIONS="$(xdpyinfo -display "$DISPLAY_ID" 2>/dev/null | awk '/dimensions:/ {print $2; exit}')"
set -e

if [[ -z "$DIMENSIONS" ]]; then
  if [[ -n "$VIDEO_SIZE" ]]; then
    DIMENSIONS="$VIDEO_SIZE"
    echo "[WARN] Could not query display size from $DISPLAY_ID. Using VIDEO_SIZE=$DIMENSIONS."
  else
    DIMENSIONS="1920x1080"
    echo "[WARN] Could not query display size from $DISPLAY_ID. Falling back to $DIMENSIONS."
  fi
fi

echo "[INFO] Recording display $DISPLAY_ID at ${DIMENSIONS}, ${FPS} FPS"
echo "[INFO] Output: $OUTPUT_PATH"
echo "[INFO] Press Ctrl+C to stop recording."

ffmpeg \
  -y \
  -loglevel info \
  -f x11grab \
  -video_size "$DIMENSIONS" \
  -framerate "$FPS" \
  -i "${DISPLAY_ID}+0,0" \
  -c:v libx264 \
  -preset "$PRESET" \
  -crf "$CRF" \
  -pix_fmt yuv420p \
  "$OUTPUT_PATH"
