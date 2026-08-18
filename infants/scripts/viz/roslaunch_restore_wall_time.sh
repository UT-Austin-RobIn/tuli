#!/usr/bin/env bash
# Run a roslaunch that enables /use_sim_time, and always restore wall time on exit.
# Usage: roslaunch_restore_wall_time.sh launch/visualize_data.launch bag_file:=...
set -euo pipefail
cleanup() {
  rosparam set /use_sim_time false >/dev/null 2>&1 || true
  echo "[INFO] Restored /use_sim_time=false (wall time)"
}
trap cleanup EXIT INT TERM
exec roslaunch "$@"
