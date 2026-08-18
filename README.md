# Infant Experiments

You can open this file in chrome for a better viewing experience.
Point the browser to 
`file:///home/robotlearning2/infants/notes.md` 

Windows machine log in info:
username: "ut austin" (you will need to use "" cause of the spacebar)
password: 1234
To ssh: ssh "ut austin"@192.168.253.101

Linux machine log in info:  
username: `robotlearning2`  
password: `robotlearning2`  

## Step-by-step instructions on experiment day
1. Ensure time-sync is good using [Verifying NTP time sync](docs/time_sync.md#verifying-ntp-time-sync)
2. Open a terminal on linux machine.
3. Check if NAS (synology) is mounted:
  run: `mountpoint -q ~/synology-tuli && echo "Mounted" || echo "Not mounted"`
  If output is "Not mounted":
         Run: `sudo mount -t nfs 192.168.253.1:/volume1/tuli ~/synology-tuli` (password: robotlearning2 )
4. Perform calibration by following [the calibration guide](docs/calibration.md)


## Running Trials

1. Note: change settings on Qualisys as follows:
- Resolution: 1080p
- Aspect Ratio: 16:9
- FPS: `25 Hz`

2. `cd ~/infants/`
3. [Optional] `arecord -l` Check what card # does "USB Audio" shows and ensure rs_cam.launch has that.
4. Launch the cameras: `./infants/scripts/start_all.sh`
5. Verify that they start with `./infants/scripts/check_cams.sh`
You should see all 6 camera topics (color image raw and aligned depth image raw for cameras L, M, R) and audio!
```bash
                 topic                     rate   min_delta   max_delta   std_dev    window
===========================================================================================
/cam_L/color/image_raw                    29.93   0.02236     0.04497     0.003658   141   
/cam_L/aligned_depth_to_color/image_raw   29.93   0.02354     0.04423     0.003293   141   
/cam_M/color/image_raw                    29.94   0.02401     0.04426     0.003387   142   
/cam_M/aligned_depth_to_color/image_raw   29.93   0.02458     0.04402     0.003249   142   
/cam_R/color/image_raw                    29.99   0.02847     0.03942     0.001995   141   
/cam_R/aligned_depth_to_color/image_raw   29.99   0.02894     0.039       0.001779   141   
/audio/audio                              38.28   0.01948     0.0305      0.003658   141   
``` 
6. Activate the virtualenv, `source ~/envs/infants/bin/activate` 
7. Run the experiment script: `python infants/experiment/experiment_driver.py`
8. It will prompt for `subject ID`, `task name`, and `condition ID`. 
Make sure the subject ID 3 digits. Example (1) write 001 for (2) write 002
Subject ID should be an integer. Task should be in `[bang, slide, hammer]`.  

9. Press SPACEBAR to start recording that trial.
10. Start recording on Qualisys side. Choose appropriate name
11. When done, stop Qualisys recording
12. Press SPACEBAR to stop recording on linux
13. Say `[y/n]` to keep trial or delete.


<!-- Once the entire session for an infant is finished, and Mark has transferred all the mocap data to "Roberto project", run the following script to transfer the data from windows to NAS  
```bash
python infants/scripts/transfer_windows_to_nas.py \
  'D:\Roberto_project\{014}' \
  {2026-06-09_14-02-01} \
  --host 192.168.253.101 \
  --user "ut austin" \
  --nas-root ~/synology-tuli
# Change the values in {}
``` -->

## Visualizing Data

1. Transfer Qualisys trial files (C3D, TSV, Miqus AVIs) into each `trial_*` folder:
```bash
python infants/scripts/organize_trial_session.py \
  --session 2026-08-18_16-10-10 \
  --infant 050
```

2. Put marker data into rosbag
```bash
python infants/scripts/process_marker_c3d.py \
  data/2026-08-18_16-10-10/trial_001
# optional: --num-markers 500
```

4. Visualize on Mocap RGB video (markers overlaid)
```bash
python infants/scripts/viz/overlay_miqus_markers.py \
  --trial-dir data/2026-08-18_16-10-10/trial_001 \
  --calibration-dir data/calibration_data/26_08_18_050
```

5. Visualize on RealSense RGB video (markers overlaid)

Basic (no audio, OpenCV window only):
```bash
python infants/scripts/viz/overlay_realsense_markers.py \
  --trial-dir data/2026-08-18_16-10-10/trial_001 \
  --calibration-dir data/calibration_data/26_08_18_050 \
  --camera L \
  --save-mp4
```

Optional flags:
- `--audio` — play bag `/audio/audio` while showing frames (paced to bag time)
- `--save-mp4` — write `visualizations/realsense/realsense_marker_overlay_<L|M|R>.mp4` (includes audio if present)
- `--output /path/to/out.mp4` — custom MP4 path (same as `--save-mp4` but choose the name)
- `--no-display` — skip the OpenCV window (useful with `--save-mp4` only)

Example with audio + MP4 export:
```bash
python infants/scripts/viz/overlay_realsense_markers.py \
  --trial-dir data/2026-06-29_15-03-28/trial_001 \
  --calibration-dir data/calibration_data/26_06_29_infant_017 \
  --camera L \
  --audio \
  --save-mp4
```

6. Visualize in RViz (point clouds ± markers)

Basic:
```bash
python infants/scripts/viz/run_trial_viz.py \
  --bag /home/robotlearning2/infants/data/2026-06-29_15-03-28/trial_001/trial_ros_combined.bag \
  --cameras L \
  --markers \
  --calib-config data/calibration_data/26_06_29_infant_017/calibration_markers.yaml
```

Add `--audio` to also start `audio_play` so bag audio plays with RViz (same bag must contain `/audio/audio`).
Add `--record` to capture the screen (RViz + your view moves) via `record_rviz_screen.sh`:
```bash
python infants/scripts/viz/run_trial_viz.py \
  --bag /home/robotlearning2/infants/data/2026-06-29_15-03-28/trial_001/trial_ros_combined.bag \
  --cameras L \
  --markers \
  --calib-config data/calibration_data/26_06_29_infant_017/calibration_markers.yaml \
  --audio \
  --record
```

Optional recording flags:
- `--record-output /path/to/out.mp4` — custom output path (default: `recordings/rviz_<bag>_<timestamp>.mp4`)
- `--record-delay 2` — seconds to wait for RViz before starting capture (default: 2)

Note: `--record` grabs the screen, then muxes bag `/audio/audio` into the MP4 (aligned using `--record-delay` vs `--bag-delay`). Live speaker playback still needs `--audio`.

Manual audio-only test (without the viz scripts):
```bash
# Terminal 1
roslaunch audio_play play.launch
# Terminal 2
rosbag play --clock /path/to/trial_ros.bag
```

---
Old 

1. RViz + rosbag playback (restores wall time on exit / Ctrl+C):
  `./infants/scripts/viz/roslaunch_restore_wall_time.sh launch/visualize_data.launch bag_file:=/absolute/path/to/trial_ros.bag`
2. Record RViz screen:
  `./infants/scripts/viz/record_rviz_screen.sh /home/robotlearning2/infants/recordings/rviz_$(date +%Y%m%d_%H%M%S).mp4 "$DISPLAY"`
3. Stop recording with `Ctrl+C`.

`rqt_bag <path to bag>` and open a bagfile with the gui. This will show images but audio will not play properly.  
rqt_image_view first and then rosbag play 
You can replay the audio with 

```
c
roslaunch audio_play play.launch
rosbag play <path to bag>
```


### Quicksheet for commands 
To rsync data:  
`sudo rsync -avh --progress --no-owner --no-group --ignore-existing 2026-07-17_10-57-17 /home/robotlearning2/synology-tuli/`

### Synology Troubleshooting
1. Make sure it is switched on (blue light should be visible on the big black box)
2. http://192.168.253.1:5000/#/signin
3. https://finds.synology.com/#
4. http://169.254.68.74:5000/


Troubleshooting:

If you see "no new messgaes"
# Stop leftover bag playback
rosnode kill /play_*   # or kill the trial_viz / visualize launch
# Restore wall-clock time
rosparam set /use_sim_time false
