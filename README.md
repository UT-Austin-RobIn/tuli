# Infant Experiments
Linux machine log in info:   
username: `robotlearning2`  
password: `robotlearning2`  

You can open this file in chrome for a better viewing experience.
Point the browser to 
`file:///home/robotlearning2/infants/notes.md` 

## Setting Up NTP time sync 
The Windows machine will serve time to the Linux machine.   
Connect the two via ethernet, and make sure Windows machine is reachable at `192.168.253.101`

On Windows PowerShell, make sure the Windows time service is running:

`Get-Service W32Time`

If it is not running, start it with:

`Start-Service W32Time`

On Linux, configure `chrony` to use the Windows machine as its NTP server.

In `/etc/chrony/chrony.conf`, make sure this line exists:

  `server 192.168.253.101 iburst maxpoll 4 minpoll 4`

Then restart `chrony`:

  `sudo systemctl restart chrony`

Verify synchronization using the steps in **Verifying NTP time sync**.

## Verifying NTP time sync 
In Linux terminal: 

   `chronyc sources -v`

That should report `^* 192.168.253.101` (where 192.168.253.101 is Windows machine), and `Reach` as non-zero.

If not, run:

   `sudo chronyc burst 4/4`

   `sudo chronyc makestep`

Then try again.

If this still fails, check if NTP server on windows is running. 

In Windows Powershell: 

   `Get-Service w32time` should show `Running`

   `w32tm /query /status` should show no errors, and have a valid source. Example: `Source: Local CMOS Clock`

If not running, start:

   `Start-Service w32time`

To monitor Linux time offset relative to Windows machine, run:

   `while true; do chronyc tracking | grep -E "System time|Last offset|RMS offset"; sleep 1; done`
   
Check the `System time` line.


## Running Trials
1. Open a terminal on linux machine. 
2. Check if NAS (synology) is mounted:
	Run: `mountpoint -q ~/synology-tuli && echo "Mounted" || echo "Not mounted"`
	If output is "Not mounted":
	        Run: `sudo mount -t nfs 192.168.253.1:/volume1/tuli ~/synology-tuli`
3. `cd ~/infants/`
4. Launch the cameras: `./start_all.sh`   
Verify that they start with the command:   
```
./check_cams.sh
```  
You should see all 6 camera topics (color image raw and aligned depth image raw for cameras L, M, R).   

4. Activate the virtualenv, `source ~/envs/infants/bin/activate` and run the experiment script: 
`python experiment/experiment_driver.py`.
5. It will prompt for `subject ID`, `task name`, and `condition ID`. 
Make sure the subject ID 3 digits. Example (1) write 001 for (2) write 002
Subject ID should be an integer. Task should be in `[bang, slide, hammer]`.    
    Condition numbers: 
    1. Soft Board - Headphones		(low haptics,  low audio )
    2. Soft Board - No Headphones 		(low haptics,  high audio)
    3. Hard Board - Headphones 		(high haptics, low audio )
    4. Hard Board - No Headphones 		(high haptics, high audio)
    5. Wash Board - Headphones 		(high haptics, low audio )
    6. Wash Board - No Headphones 		(high haptics, high audio)
    7. Soft Board and Button - Headphones 	(high haptics, low audio )
    8. Soft Board and Button - No Headphones 	(high haptics, high audio)

6. Press ENTER to stop recording that trial. 
7. Say `[y/n]` to keep trial or delete. 
8. Press `ctrl+c` to interrupt and kill the program. 

# Network Setup
Windows IP: `192.168.253.101`  
Linux IP: `192.168.253.201`   
Synology NAS: `192.168.253.1` 
Web interface for synology: `192.168.253.1:5000`.   
Synology: `robin`, `Robot123`.  
To mount: `sudo mount -t nfs 192.168.253.1:/volume1/tuli ~/synology-tuli/`, ps: robotlearning2    
To rsync data: `rsync -r --info=progress2 data /home/robotlearning2/synology-tuli/`   

## ping windows desktop 
set this IP on windows manually if needed. 
`ping 192.168.253.101`

## visualizing data
`rqt_bag <path to bag>` and open a bagfile with the gui. This will show images but audio will not play properly.   
rqt_image_view first and then rosbag play 
You can replay the audio with 
```
c
roslaunch audio_play play.launch
rosbag play <path to bag>
```

RViz + rosbag playback (single command):
`roslaunch launch/visualize_data.launch bag_file:=/absolute/path/to/trial_ros.bag`

Record RViz screen (reproducible):
`./record_rviz_screen.sh /home/robotlearning2/infants/recordings/rviz_$(date +%Y%m%d_%H%M%S).mp4 "$DISPLAY"`

Recommended workflow for a recorded demo:
1. Start visualization:
   `roslaunch launch/visualize_data.launch bag_file:=/absolute/path/to/trial_ros.bag`
2. Start screen capture:
   `./record_rviz_screen.sh /home/robotlearning2/infants/recordings/rviz_$(date +%Y%m%d_%H%M%S).mp4 "$DISPLAY"`
3. Stop recording with `Ctrl+C`.

additional commands:
1. rs-enumerate-devices


After recording video:
1. python process_marker_tsv.py
2. python visualize_data_on_image.py


## Debugging Synology
We want the following ip addresses:
Windows IP: 192.168.253.101
Linux IP: 192.168.253.201
Synology NAS: 192.168.253.1 
Web interface for synology: 192.168.253.1:5000.

So, when we run robotlearning2@robinlab:~$ ip -4 addr show
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
2: enp4s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UP group default qlen 1000
    inet 192.168.253.201/24 scope global enp4s0
       valid_lft forever preferred_lft forever
4: wlx503dd129b771: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    inet 100.64.217.160/16 brd 100.64.255.255 scope global dynamic noprefixroute wlx503dd129b771
       valid_lft 552sec preferred_lft 552sec


1. Manually add ip addr through terminal on linux
sudo ip addr flush dev enp4s0
sudo ip addr add 192.168.253.201/24 dev enp4s0
sudo ip link set enp4s0 up

Another thing is to login into Synology assistant. We found two ways
- From terminal: sudo arp-scan --interface=enp4s0 169.254.0.0/16 
Interface: enp4s0, type: EN10MB, MAC: 10:ff:e0:bb:34:1b, IPv4: 192.168.253.1
Starting arp-scan 1.9.7 with 65536 hosts (https://github.com/royhills/arp-scan)
169.254.68.74	90:09:d0:6f:1f:09	(Unknown)


From app: download synology assistant



## timesync test

Test 1: Change time on windows manually and see if linux catches up

On windows do:
- net stop w32time 
- Set-Date -Date "2026-04-29 00:00:00" 
- net start w32time 
On limux:
- sudo systemctl restart chrony
- Then, check on limux if the time is changed to the one on windows 
- If time doesn't chage on linux, try changing maxdistance to a very high value (like 1000000000000000000) in sudo vim /etc/chrony/chrony.conf

To revert on windows: 
- w32tm /config /manualpeerlist:"time.google.com,0x8 pool.ntp.org,0x8" /syncfromflags:manual /update 
- w32tm /config /manualpeerlist:"" /syncfromflags:manual /update 
- net stop w32time 
- net start w32time


Test 2: Stop timesync, Manually change time on ubuntu, start timesync again 

On linux do:
- sudo systemctl stop chronyd 
- sudo timedatectl set-time "2026-04-29 00:00:00"
- sudo systemctl start chronyd
After this, time should be reset to the original time on windows
- systemctl status chronyd
- chronyc sources -v
- while true; do chronyc tracking | grep -E "System time|Last offset|RMS offset"; sleep 1; done

