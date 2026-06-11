#!/usr/bin/env python3
"""Launch full-chain calibration visualization.

Starts:
  1. Marker transformer (publishes TF + marker spheres)
  2. roslaunch (bag playback + RViz)
"""
import os
import sys
import signal
import subprocess


def main():
    print("=== Full-Chain Calibration Visualization ===\n")

    combined_bag = "/home/robotlearning2/infants/data/combined_latest.bag"
    config = os.path.expanduser("~/stereo-calib/examples/fullchain_config_example.yaml")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    launch_file = os.path.abspath(os.path.join(script_dir, "..", "launch", "visualize_fullchain.launch"))
    marker_script = os.path.abspath(os.path.join(script_dir, "visualize_fullchain.py"))
    config = os.path.abspath(config)

    # 1. Start marker transformer (system Python for ROS)
    marker_cmd = ["/usr/bin/python3", marker_script, "--config", config]
    print(f"Starting marker transformer: {' '.join(marker_cmd)}")
    marker_proc = subprocess.Popen(marker_cmd)

    # 2. roslaunch (bag + rviz)
    launch_cmd = [
        "roslaunch", launch_file,
        f"bag_file:={combined_bag}",
        "loop:=false",
    ]
    print(f"Starting roslaunch: {' '.join(launch_cmd)}\n")

    try:
        launch_proc = subprocess.Popen(launch_cmd)
        launch_proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        marker_proc.send_signal(signal.SIGINT)
        marker_proc.wait()
        print("\nAll processes stopped.")


if __name__ == "__main__":
    main()
