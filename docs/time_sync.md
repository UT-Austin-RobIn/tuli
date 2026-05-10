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

#### 1. Verify chronyc source is Windows
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

#### 2. Verify time offset between the two machines is very low

To monitor Linux time offset relative to Windows machine, run:

   `while true; do chronyc tracking | grep -E "System time|Last offset|RMS offset"; sleep 1; done`

Check the `System time` line. It should be less than 0.001 seconds


## Other Timesync Tests

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