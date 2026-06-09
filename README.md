# Infant Experiments

You can open this file in chrome for a better viewing experience.
Point the browser to 
`file:///home/robotlearning2/infants/notes.md` 

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
6. Launch the cameras: `./infants/scripts/start_all.sh`
7. Verify that they start with `./infants/scripts/check_cams.sh`
You should see all 6 camera topics (color image raw and aligned depth image raw for cameras L, M, R). 

## Running Trials

1. Activate the virtualenv, `source ~/envs/infants/bin/activate` 
2. Run the experiment script: `python experiment/experiment_driver.py`.
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


## Visualizing Data

`TODO: Daniel add exact instructions here`
1. Annotate the mocap markers (Usually Mark does this)
2. Transfer the 3 RGB camera videos and the tsv file to the data folder which has ros bags on synology
3. ...

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