#!/bin/bash
# Rebuild combined bags (all C3D markers) + overlay with matching --num-markers.
set -euo pipefail
cd /home/robotlearning2/infants
source /home/robotlearning2/envs/infants/bin/activate

run_one() {
  local t="$1"
  local n="$2"
  local tt
  tt="$(printf '%03d' "$t")"
  local dir="data/2026-06-29_15-03-28/trial_${tt}"
  local log="/tmp/infant017_trial_${tt}.log"
  {
    echo "===== START trial_${tt} num_markers=${n} $(date) ====="
    python infants/scripts/process_marker_c3d.py \
      --file_path "${dir}/26_06_29_017_${t}.c3d" \
      --tsv "${dir}/26_06_29_017_${t}.tsv" \
      --camera-bag "${dir}/trial_ros.bag"
    python infants/scripts/overlay_markers_on_image.py \
      --bag "${dir}/trial_ros_combined.bag" \
      --calib-config data/calibration_data/26_06_29_infant_017/calibration_markers.yaml \
      --camera L \
      --save-mp4 \
      --no-display \
      --num-markers "${n}"
    echo "===== DONE trial_${tt} $(date) ====="
  } >"$log" 2>&1
  echo "finished trial_${tt} -> $log"
}

export -f run_one

# 3 at a time to avoid melting disk on huge bags
printf '%s\n' \
  '1 764' \
  '2 208' \
  '3 235' \
  '4 462' \
  '5 664' \
  '6 1544' \
  '7 877' \
  '8 635' \
| xargs -P 3 -n 2 bash -c 'run_one "$0" "$1"'

echo "ALL_TRIALS_FINISHED"
