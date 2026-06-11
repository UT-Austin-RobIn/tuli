# RealSense ↔ Qualisys Calibration

## Overview

This document describes the procedure for calibrating the following transforms:

- `RS_left` → `Qualisys_left`
- `RS_right` → `Qualisys_right`
- `RS_mid` → `RS_left`

---

# 1. Mocap Calibration

Calibrate the mocap system using Qualisys (performed by Mark).

This step outputs a `TODO` file.

---

# 2. Initial Setup

## Qualisys Settings

Ensure the following settings on the Qualisys side:

- Resolution: `720p`
- Aspect ratio: `16:9`
- FPS: `30 Hz`

## Linux Setup

Navigate to the project directory:

```bash
cd /home/robotlearning2/infants
source ~/envs/infants/bin/activate
```

## Naming Convention

Use the following naming convention for this calibration data collection session:
```bash
{yy_mm_dd_infant_XXX} e.g. 26_05_09_infant_010
```


# 3. Collect data for calibrating `RS_left` → `Qualisys_left`

## 3.1 Start Camera Stream

```bash
./infants/scripts/start_camera_for_calibration --left
```

## 3.2 Verify Cameras

In a new terminal:

```bash
./infants/scripts/check_cams.sh
```

This should output: 
```
                 topic                     rate   min_delta   max_delta   std_dev    window
===========================================================================================
/cam_L/color/image_raw                    29.97   0.01206     0.04849     0.002946   141   
/cam_L/aligned_depth_to_color/image_raw   29.95   0.01237     0.04816     0.002854   141   
```

If not, contact Arpit or Daniel.

## 3.3 Record Calibration Data

In a new terminal on linux:

```bash
python infants/scripts/record_for_calibration.py --left_to_qualisys --folder_name {yy_mm_dd_infant_XXX}
```

Immediately start recording on the Qualisys side. Set the recording name as `yy_mm_dd_left_to_qualisys`. Perform the calibration data collection process for approximately 1 minute.

Then stop:
- the Qualisys recording by pressing the red button
- the Linux recording script by pressing space bar

## 3.4 Export TSV File

In Qualisys:

```text
File → Export → to_tsv
```

Enable:
- "Include TSV header"
- "Export time data for every frame"

Expected output:

```text
yy_mm_dd_left_to_qualisys.tsv
```

## 3.5 Stop Camera Stream

Press `Ctrl+C` in the terminal that was running `./start_camera_for_calibration --left`

---

# 4. Collect data for calibrating `RS_right` → `Qualisys_right`

## 4.1 Start Camera Stream

```bash
./infants/scripts/start_camera_for_calibration --right
```

## 4.2 Verify Cameras

In a new terminal:

```bash
./infants/scripts/check_cams.sh
```

This should output `TODO`.

If not, contact Arpit or Daniel.

## 4.3 Record Calibration Data

In a new terminal:

```bash
python infants/scripts/record_for_calibration.py --right_to_qualisys --folder_name {yy_mm_dd_infant_XXX}
```

Immediately start recording on the Qualisys side. Set the recording name as `yy_mm_dd_right_to_qualisys`. Perform the calibration data collection process for approximately 1 minute.

Then stop:
- the Qualisys recording by pressing the red button
- the Linux recording script by pressing space bar

## 4.4 Export TSV File

In Qualisys:

```text
File → Export → to_tsv
```

Enable:
- "Include TSV header"
- "Export time data for every frame"

Expected output:

```text
yy_mm_dd_right_to_qualisys.tsv
```

## 4.5 Stop Camera Stream

Press `Ctrl+C` in the terminal that was running `./start_camera_for_calibration --right`

---

# 5. Collect data for calibrating `RS_mid` → `RS_left`

No Qualisys recording is required for this step.

## 5.1 Start Camera Streams

```bash
./infants/scripts/start_camera_for_calibration --left --mid
```

## 5.2 Verify Cameras

In a new terminal:

```bash
./infants/scripts/check_cams.sh
```

This should output `TODO`.

If not, contact Arpit or Daniel.

## 5.3 Record Calibration Data

In a new terminal:

```bash
python infants/scripts/record_for_calibration.py --left_to_mid --folder_name {yy_mm_dd_infant_XXX}
```

Perform the calibration data collection procedure.

---

# 6. Transfer Files to the Linux Machine

Open Windows PowerShell and run:

```bash
scp 'C:\Users\UT Austin\Documents\Qualisys0326\Data\{yy_mm_dd_XXX}_left_to_qualisys_Miqus_1_31039.avi' robotlearning2@192.168.253.201:~/infants/data/calibration_data/{yy_mm_dd_infant_XXX}/left_to_qualisys/

scp 'C:\Users\UT Austin\Documents\Qualisys0326\Data\{yy_mm_dd_XXX}_left_to_qualisys.tsv' robotlearning2@192.168.253.201:~/infants/data/calibration_data/{yy_mm_dd_infant_00X}/left_to_qualisys/

scp 'C:\Users\UT Austin\Documents\Qualisys0326\Data\{yy_mm_dd_XXX}_right_to_qualisys_Miqus_10_31041.avi' robotlearning2@192.168.253.201:~/infants/data/calibration_data/{yy_mm_dd_infant_XXX}/right_to_qualisys/

scp 'C:\Users\UT Austin\Documents\Qualisys0326\Data\{yy_mm_dd_XXX}_right_to_qualisys.tsv' robotlearning2@192.168.253.201:~/infants/data/calibration_data/{yy_mm_dd_infant_00X}/right_to_qualisys/

scp 'C:\Users\UT Austin\Downloads\{yy_mm_dd_XXX}_mocap_calibration.txt' robotlearning2@192.168.253.201:~/infants/data/calibration_data/{yy_mm_dd_infant_00X}/
```

So, finally we should have the following in the `~/infants/data/calibration_data/{yy_mm_dd_infant_00X}` folder:
1. `left_to_qualisys/ros.bag`
2. `right_to_qualisys/ros.bag`
3. `right_to_left/ros.bag`
4. `left_to_qualisys/{yy_mm_dd}_left_to_qualisys_Miqus_10_31039.avi`
5. `right_to_qualisys/{yy_mm_dd}_right_to_qualisys_Miqus_10_31041.avi`
6. `left_to_qualisys/{yy_mm_dd}_left_to_qualisys.tsv`
7. `right_to_qualisys/{yy_mm_dd}_right_to_qualisys.tsv`
8. `qualisys_calibration_file.qca`


# 7. Preparing images for calibration
```bash
python infants/scripts/prepare_calibration_image_rs.py --folder_name {folder_name}  #  26_06_09_infant_014
python infants/scripts/prepare_calibration_image.py --folder_name {folder_name}  #  26_06_09_infant_014
python infants/scripts/prepare_calibration_image_rs.py --folder_name {folder_name}  #  26_06_09_infant_014
```

# 8. Run calibration
```bash
```