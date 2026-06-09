import csv
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "matched_timestamp_5_minutes.csv"

ros_frames = []
qualisys_frames = []
time_diffs = []

with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        ros_frames.append(int(row["ros_frame"]))
        qualisys_frames.append(int(row["qualisys_frame"]))
        time_diffs.append(float(row["time_difference_sec"]))

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(ros_frames, time_diffs, "o-", markersize=1, linewidth=0.5, label="time difference")

ax.set_ylabel("Time Difference (sec)")
ax.set_title("Time Difference Between Matched ROS and Qualisys Frames")

# Dual x-axis labels: ROS on top, Qualisys on bottom
num_ticks = 10
step = len(ros_frames) // num_ticks
tick_indices = list(range(0, len(ros_frames), step))

ros_tick_labels = [str(ros_frames[i]) for i in tick_indices]
qualisys_tick_labels = [str(qualisys_frames[i]) for i in tick_indices]
tick_positions = [ros_frames[i] for i in tick_indices]

ax.set_xticks(tick_positions)
ax.set_xticklabels([f"ros: {r}\nqualisys: {q}" for r, q in zip(ros_tick_labels, qualisys_tick_labels)],
                    fontsize=8)
ax.set_xlabel("Frame")

ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "time_diff_plot.png", dpi=150)
plt.show()
print(f"Saved to {SCRIPT_DIR / 'time_diff_plot.png'}")
