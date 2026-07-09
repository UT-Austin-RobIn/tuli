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
5. `cd ~/infants/`
6. `arecord -l` Check what card # does "USB Audio" shows and ensure rs_cam.launch has that.
7. Launch the cameras: `./infants/scripts/start_all.sh`
8. Verify that they start with `./infants/scripts/check_cams.sh`
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

## Running Trials

1. Activate the virtualenv, `source ~/envs/infants/bin/activate` 
2. Run the experiment script: `python infants/experiment/experiment_driver.py`
3. It will prompt for `subject ID`, `task name`, and `condition ID`. 
Make sure the subject ID 3 digits. Example (1) write 001 for (2) write 002
Subject ID should be an integer. Task should be in `[bang, slide, hammer]`.  
`TODO change this`
    Condition numbers: 
    1. Soft Board - Headphones		(low haptics,  low audio )
    2. Soft Board - No Headphones 		(low haptics,  high audio)
    3. Hard Board - Headphones 		(high haptics, low audio )
    4. Hard Board - No Headphones 		(high haptics, high audio)
    5. Wash Board - Headphones 		(high haptics, low audio )
    6. Wash Board - No Headphones 		(high haptics, high audio)
    7. Soft Board and Button - Headphones 	(high haptics, low audio )
    8. Soft Board and Button - No Headphones 	(high haptics, high audio)

4. Press SPACEBAR to start recording that trial.
5. Start recording on Qualisys side. Choose appropriate name
6. When done, stop Qualisys recording
7. Press SPACEBAR to stop recording on linux
8. Say `[y/n]` to keep trial or delete.


Once the entire session for an infant is finished, and Mark has transferred all the mocap data to "Roberto project", run the following script to transfer the data from windows to NAS  
```bash
python infants/scripts/transfer_windows_to_nas.py \
  'D:\Roberto_project\{014}' \
  {2026-06-09_14-02-01} \
  --host 192.168.253.101 \
  --user "ut austin" \
  --nas-root ~/synology-tuli
# Change the values in {}
```

## Visualizing Data

`TODO: Daniel add exact instructions here`
1. Annotate the mocap markers (Optional)
2. Transfer the 3 RGB camera videos and the tsv file to the data folder which has ros bags on synology
```bash
scp windows:D:/Roberto_project/015/26_06_24_infant_015_2.tsv /home/robotlearning2/synology-tuli/2026-06-24_09-40-52/trial_002/

scp windows:D:/Roberto_project/015/26_06_29_infant_017_1.c3d /home/robotlearning2/infants/data/2026-06-29_15-03-28/trial_001/

```
3. Put marker data into rosbag
```bash
  python infants/scripts/process_marker_c3d.py   \
  --file_path /home/robotlearning2/infants/data/2026-06-29_15-03-28/trial_001/26_06_29_017_1.c3d \
  --tsv /home/robotlearning2/infants/data/2026-06-29_15-03-28/trial_001/26_06_29_017_1.tsv \
  --num-markers 1 \
  --camera-bag /home/robotlearning2/infants/data/2026-06-29_15-03-28/trial_001/trial_ros.bag
```

4. Visualize on RS image
```bash  
python infants/scripts/overlay_markers_on_image.py \
  --bag data/2026-06-29_15-03-28/trial_001/trial_ros_combined.bag \
  --calib-config data/calibration_data/26_06_29_infant_017/calibration_markers.yaml \
  --camera L \
  --num-markers 300
```

5. Visualize in rviz 
```bash
python infants/scripts/run_trial_viz.py \
  --bag  /home/robotlearning2/infants/data/2026-06-29_15-03-28/trial_001/trial_ros_combined.bag \
  --cameras L \
  --markers \
  --calib-config data/calibration_data/26_06_29_infant_017/calibration_markers.yaml
```

Test audio:
Terminal 1: roslaunch audio_play play.launch
Terminal 2: rosbag play --clock /path/to/trial_ros.bag

---
Old 

1. RViz + rosbag playback:
  `roslaunch launch/visualize_data.launch bag_file:=/absolute/path/to/trial_ros.bag`
2. Record RViz screen:
  `./record_rviz_screen.sh /home/robotlearning2/infants/recordings/rviz_$(date +%Y%m%d_%H%M%S).mp4 "$DISPLAY"`
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
To rsync data: `rsync -r --info=progress2 data /home/robotlearning2/synology-tuli/`  

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