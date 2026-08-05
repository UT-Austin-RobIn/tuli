#!/bin/bash
# Optional guided calibration CAPTURE helper.
# Does NOT replace the manual docs/calibration.md flow.
#
# Flow: session name → create Windows folder → record L/R/mid →
#       scp Qualisys dump to from_windows/ (never touches Linux bags) →
#       organize names/layout → print prepare / stereo-calib / export commands.
#
# Usage:
#   cd ~/infants
#   ./infants/scripts/guided_calibration_capture.sh
#   ./infants/scripts/guided_calibration_capture.sh --date 26_07_30 --infant 017

set -euo pipefail

INFANTS_ROOT="${INFANTS_ROOT:-/home/robotlearning2/infants}"
cd "$INFANTS_ROOT"

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
warn() { printf '\033[33m  [!] %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m  [OK] %s\033[0m\n' "$*"; }
cmd()  { printf '\033[36m    %s\033[0m\n' "$*"; }

divider() {
  echo
  echo "================================================================"
  bold "$*"
  echo "================================================================"
}

pause() {
  local msg="${1:-Press ENTER to continue (or type q then ENTER to quit)}"
  echo
  while true; do
    read -r -p "  >>> $msg: " ans || exit 1
    case "${ans,,}" in
      q|quit|exit) echo "Exiting. You can re-run this script anytime."; exit 0 ;;
      *) return 0 ;;
    esac
  done
}

ask() {
  # ask VAR "Prompt" "default"
  local __var="$1"
  local __prompt="$2"
  local __default="${3:-}"
  local __ans
  if [[ -n "$__default" ]]; then
    read -r -p "  $__prompt [$__default]: " __ans || exit 1
    __ans="${__ans:-$__default}"
  else
    read -r -p "  $__prompt: " __ans || exit 1
  fi
  printf -v "$__var" '%s' "$__ans"
}

confirm_yes() {
  local msg="$1"
  local ans
  read -r -p "  $msg [y/N]: " ans || exit 1
  case "${ans,,}" in
    y|yes) return 0 ;;
    *) return 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# Args / session name
# ---------------------------------------------------------------------------

DATE_ARG=""
INFANT_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --date) DATE_ARG="$2"; shift 2 ;;
    --infant) INFANT_ARG="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--date YY_MM_DD] [--infant NNN]"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

default_date="$(date +%y_%m_%d)"

divider "Infant calibration capture — guided helper"
info "This walks you through recording calibration videos (left, right, mid)."
info "It uses the SAME commands as docs/calibration.md — just fills in names."
info "You can still run everything manually from the docs if you prefer."
echo
info "Tip: type q then ENTER at any pause to quit safely."
pause "Press ENTER to begin"

divider "Step 0 — Session name"
info "We need a date code and the infant / subject number."
info "Example: date 26_07_30 and infant 017 → folder 26_07_30_infant_017"
echo
info "Today's date on this computer: $default_date"
info "Press ENTER to use that, or type a different yy_mm_dd."
echo

if [[ -z "$DATE_ARG" ]]; then
  ask DATE_ARG "Date code (yy_mm_dd)" "$default_date"
else
  info "Using --date $DATE_ARG"
fi

# Normalize 2026-07-30 → 26_07_30
if [[ "$DATE_ARG" =~ ^20([0-9]{2})-([0-9]{2})-([0-9]{2})$ ]]; then
  DATE_ARG="${BASH_REMATCH[1]}_${BASH_REMATCH[2]}_${BASH_REMATCH[3]}"
fi
if [[ ! "$DATE_ARG" =~ ^[0-9]{2}_[0-9]{2}_[0-9]{2}$ ]]; then
  warn "Date should look like 26_07_30 (got: $DATE_ARG)"
  if ! confirm_yes "Continue anyway?"; then exit 1; fi
fi

if [[ -z "$INFANT_ARG" ]]; then
  ask INFANT_ARG "Infant / subject number (e.g. 17 or 017)" ""
else
  info "Using --infant $INFANT_ARG"
fi
# Pad to 3 digits if numeric
if [[ "$INFANT_ARG" =~ ^[0-9]+$ ]]; then
  INFANT_ARG="$(printf '%03d' "$((10#$INFANT_ARG))")"
fi

SESSION="${DATE_ARG}_infant_${INFANT_ARG}"
DATA_DIR="$INFANTS_ROOT/data/calibration_data/$SESSION"

echo
bold "  Session folder name: $SESSION"
info "Data will go under: $DATA_DIR"
if ! confirm_yes "Does this look correct?"; then
  echo "OK — re-run the script with the right values."
  exit 0
fi

mkdir -p "$DATA_DIR"
ok "Created (or already had) Linux folder: $DATA_DIR"

# Where Qualisys / Windows files will live (short names inside this folder)
WIN_WORKSPACE='C:\Users\UT Austin\Documents\Qualisys0326\Data'
WIN_DIR="${WIN_WORKSPACE}\\${SESSION}"

# Activate venv if present
if [[ -f "$HOME/envs/infants/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/envs/infants/bin/activate"
  ok "Activated ~/envs/infants"
else
  warn "Could not find ~/envs/infants — continue if ROS/python already set up."
fi

# ---------------------------------------------------------------------------
# Windows folder first (Qualisys side)
# ---------------------------------------------------------------------------

divider "Step 1 — Create folder on the Windows (Qualisys) computer"
info "Do this on the WINDOWS machine (Qualisys PC), in PowerShell."
echo
bold "A) Go to the Qualisys data workspace"
info "Copy-paste:"
echo
cmd "cd \"$WIN_WORKSPACE\""
echo
info "(If that path is wrong on your PC, cd to wherever Qualisys saves Data, e.g. Documents\\Qualisys0326\\Data)"
pause "Press ENTER after you are in that folder in PowerShell"

bold "B) Make today's session folder"
info "Create a folder named exactly:"
echo
cmd "$SESSION"
echo
info "Copy-paste:"
echo
cmd "New-Item -ItemType Directory -Force -Path \".\\$SESSION\""
echo
info "All Qualisys videos / TSV / mocap files for this session go IN that folder."
info "Use SHORT recording names (no date/infant prefix):"
info "  • left_to_qualisys"
info "  • right_to_qualisys"
info "  • (and the mocap calibration file — put a copy in the same folder)"
echo
warn "Full Windows path to remember for later copy:"
cmd "$WIN_DIR"
pause "Press ENTER after the session folder exists"

# ---------------------------------------------------------------------------
# Pre-flight checklist
# ---------------------------------------------------------------------------

divider "Step 2 — Before recording (checklist)"
info "On the Qualisys (Windows) computer, confirm:"
info "  • Resolution 720p, aspect 16:9, FPS 30 Hz"
info "  • Mark has finished mocap system calibration (if required today)"
echo
info "On Linux we will record three pairs:"
info "  1) left RealSense  ↔  Qualisys"
info "  2) right RealSense ↔  Qualisys"
info "  3) left RealSense  ↔  mid RealSense  (no Qualisys)"
echo
info "Board tip: keep the Charuco board clearly visible to BOTH cameras"
info "of the pair for most of ~1 minute, several poses, not too small/blurry."
pause "Press ENTER when Qualisys settings look good"

# ---------------------------------------------------------------------------
# Helper: one stereo/Qualisys capture block
# ---------------------------------------------------------------------------

run_pair() {
  local pair_label="$1"          # human label
  local cam_flags="$2"           # e.g. --left
  local record_flag="$3"         # e.g. --left_to_qualisys
  local need_qualisys="$4"       # yes|no
  local qualisys_name="$5"       # recording name on Qualisys (or "")

  divider "Recording: $pair_label"
  info "Folder: data/calibration_data/$SESSION/${record_flag#--}/"
  # record_flag is --left_to_qualisys → pair dir left_to_qualisys
  local pair_dir="${record_flag#--}"

  echo
  bold "A) Start cameras"
  info "In a SEPARATE terminal window, copy-paste this exactly:"
  echo
  cmd "cd ~/infants"
  cmd "source ~/envs/infants/bin/activate"
  cmd "./infants/scripts/start_camera_for_calibration $cam_flags"
  echo
  info "Leave that terminal running. Come back here when cameras are up."
  pause "Press ENTER after cameras are started"

  echo
  bold "B) Check camera topics"
  info "Running one camera check (~30 seconds). Look for ~30 Hz rates."
  echo
  if ./infants/scripts/check_cams.sh --once; then
    ok "check_cams finished (look at rates above — ~30 Hz expected)."
  else
    warn "check_cams reported a problem. Fix cameras or ask Arpit/Daniel."
    if ! confirm_yes "Continue to recording anyway?"; then
      warn "Stop the camera terminal with Ctrl+C, then re-run this script."
      exit 1
    fi
  fi
  pause "Press ENTER to start the Linux recorder"

  echo
  bold "C) Record on Linux (+ Qualisys if needed)"
  if [[ "$need_qualisys" == "yes" ]]; then
    warn "On Qualisys, set the recording name EXACTLY to (short name, no date prefix):"
    echo
    cmd "$qualisys_name"
    echo
    info "After recording, put the AVI + TSV into the Windows folder:"
    cmd "$WIN_DIR"
    info "(Qualisys may add a camera id to the AVI filename — that is OK. Keep the TSV too.)"
    echo
    info "When the Linux recorder says to press SPACE to start:"
    info "  1) Press SPACE here (Linux) to start rosbag"
    info "  2) Immediately start Qualisys (red button / record)"
    info "Move the Charuco board for ~1 minute in BOTH cameras' view."
    info "Then: stop Qualisys (red button), then SPACE here to stop Linux."
  else
    info "No Qualisys for this step — RealSense only."
    info "Press SPACE to start, move the board ~1 minute, SPACE to stop."
  fi
  echo
  info "Starting recorder now (this uses SPACE to start/stop):"
  cmd "python infants/scripts/record_for_calibration.py $record_flag --folder_name $SESSION"
  echo
  pause "Press ENTER to launch the recorder"

  # record_for_calibration handles its own SPACE start/stop
  set +e
  python infants/scripts/record_for_calibration.py "$record_flag" --folder_name "$SESSION"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    warn "Recorder exited with code $rc"
    if ! confirm_yes "Continue with the guide anyway?"; then exit 1; fi
  else
    ok "Linux recording finished for $pair_dir"
  fi

  if [[ "$need_qualisys" == "yes" ]]; then
    echo
    bold "D) Export TSV on Qualisys + put files in the session folder"
    info "In Qualisys:  File → Export → to_tsv"
    info "Enable:"
    info "  • Include TSV header"
    info "  • Export time data for every frame"
    echo
    info "Move / copy the AVI and TSV into:"
    cmd "$WIN_DIR"
    info "Names should look like left_to_qualisys... / right_to_qualisys... (short names)."
    pause "Press ENTER after AVI + TSV are in the Windows session folder"
  fi

  echo
  bold "E) Stop cameras"
  info "Go to the camera terminal and press Ctrl+C to stop the cameras."
  pause "Press ENTER after cameras are stopped"
  ok "Done with: $pair_label"
}

# ---------------------------------------------------------------------------
# Three captures
# ---------------------------------------------------------------------------

run_pair \
  "LEFT RealSense ↔ Qualisys" \
  "--left" \
  "--left_to_qualisys" \
  "yes" \
  "left_to_qualisys"

run_pair \
  "RIGHT RealSense ↔ Qualisys" \
  "--right" \
  "--right_to_qualisys" \
  "yes" \
  "right_to_qualisys"

run_pair \
  "LEFT RealSense ↔ MID RealSense (no Qualisys)" \
  "--left --mid" \
  "--left_to_mid" \
  "no" \
  ""

# ---------------------------------------------------------------------------
# Transfer whole Windows folder → Linux (print only)
# ---------------------------------------------------------------------------

divider "Step 6 — Copy mocap calibration into the Windows folder"
info "Also put a copy of today's mocap calibration file (.qca / .qca.txt) into:"
cmd "$WIN_DIR"
info "Any name is fine — the organizer will find it."
pause "Press ENTER after the mocap calibration file is in the Windows folder"

divider "Step 7 — Transfer Windows files to a SEPARATE Linux folder"
info "Linux RealSense bags stay here (do not scp on top of this):"
cmd "$DATA_DIR/"
echo
warn "scp goes to from_windows/ — a different directory — so bags cannot be overwritten."
echo
info "In PowerShell, from the Qualisys Data parent folder, run:"
echo
cmd "cd \"$WIN_WORKSPACE\""
cmd "scp -r \".\\$SESSION\" robotlearning2@192.168.253.201:~/infants/data/calibration_data/from_windows/"
echo
info "That creates:"
cmd "~/infants/data/calibration_data/from_windows/$SESSION/"
info "Your Linux bags remain under:"
cmd "~/infants/data/calibration_data/$SESSION/"
pause "Press ENTER after the scp transfer finished"

divider "Step 8 — Organize files for prepare / calibration scripts"
info "Moves Qualisys files from from_windows/ into the Linux session layout."
info "Does not touch ros.bag."
echo
cmd "python infants/scripts/organize_calibration_session.py --folder $SESSION"
echo
pause "Press ENTER to run the organizer"

set +e
python infants/scripts/organize_calibration_session.py --folder "$SESSION"
org_rc=$?
set -e
if [[ $org_rc -ne 0 ]]; then
  warn "Organizer reported problems with Qualisys files."
  warn "Fix those, then re-run:"
  cmd "python infants/scripts/organize_calibration_session.py --folder $SESSION"
  exit "$org_rc"
fi

bold "Session name to remember: $SESSION"
ok "Guided capture + organize finished. Use the Next commands printed above."
echo
