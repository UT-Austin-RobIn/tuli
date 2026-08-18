#!/usr/bin/env python3
"""Start/stop helpers for unattended RViz trial recording.

RViz, roslaunch, ffmpeg x11grab, and orbit-drag must die as a process group.
Otherwise Ctrl+C leaves zombies that keep /use_sim_time=true and grab DISPLAY
→ black recordings on the next trial.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from typing import Iterable, Optional

STALE_PATTERN = re.compile(
    r"trial_viz\.launch|"
    r"run_trial_viz\.py|"
    r"record_rviz_screen\.sh|"
    r"orbit_drag_yaw\.py|"
    r"orbit_cam_broadcaster|"
    r"x11grab|"
    r"depth_to_pointcloud|"
    r"calib_tf_broadcaster|"
    r"marker_transformer|"
    r"rosbag play"
)

KEEP_PATTERN = re.compile(
    r"make_trial_overlay|"
    r"overlay_realsense|"
    r"overlay_miqus|"
    r"make_trial_rviz_videos|"
    r"process_marker"
)


def set_use_sim_time(value: bool) -> bool:
    flag = "true" if value else "false"
    try:
        result = subprocess.run(
            ["rosparam", "set", "/use_sim_time", flag],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print(f"[INFO] /use_sim_time={flag}")
            return True
        err = (result.stderr or result.stdout or "").strip()
        print(f"[WARN] Could not set /use_sim_time={flag}: {err or 'rosparam failed'}")
        return False
    except Exception as exc:
        print(f"[WARN] Could not set /use_sim_time={flag}: {exc}")
        return False


def restore_wall_clock_time() -> None:
    set_use_sim_time(False)


def popen_session(cmd: list[str], env: Optional[dict] = None) -> subprocess.Popen:
    """Start a child in its own session so Ctrl+C hits only this script."""
    return subprocess.Popen(
        cmd,
        env=env,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
    )


def _killpg(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def stop_proc(proc: Optional[subprocess.Popen], name: str, timeout: float = 8.0) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    print(f"[stop] {name} (pid={proc.pid})")
    _killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        print(f"[stop] {name} SIGINT timed out; SIGTERM")
    _killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        print(f"[stop] {name} SIGTERM timed out; SIGKILL")
    _killpg(proc.pid, signal.SIGKILL)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        print(f"[stop] {name} still alive after SIGKILL")


def sleep_interruptible(
    seconds: float,
    procs: Iterable[Optional[subprocess.Popen]] = (),
    label: str = "",
) -> Optional[int]:
    """Sleep in small slices. Returns early process exit code if a watched proc dies."""
    seconds = max(0.0, float(seconds))
    if label:
        print(f"[wait] {label} ({seconds:.1f}s)", flush=True)
    deadline = time.time() + seconds
    procs = tuple(p for p in procs if p is not None)
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return None
        for proc in procs:
            code = proc.poll()
            if code is not None:
                return code
        time.sleep(min(0.25, remaining))


def list_stale_viz(exclude_pids: Iterable[int] = ()) -> list[tuple[int, str]]:
    exclude = {os.getpid(), *exclude_pids}
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return []
    found: list[tuple[int, str]] = []
    for line in (out.stdout or "").splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        try:
            pid_s, args = line.split(None, 1)
            pid = int(pid_s)
        except ValueError:
            continue
        if pid in exclude:
            continue
        if KEEP_PATTERN.search(args):
            continue
        if STALE_PATTERN.search(args):
            found.append((pid, args))
    return found


def kill_stale_viz(exclude_pids: Iterable[int] = (), quiet: bool = False) -> int:
    """Kill leftover RViz/ffmpeg/rosbag jobs from a previous run."""
    leftover = list_stale_viz(exclude_pids=exclude_pids)
    if not leftover:
        return 0
    if not quiet:
        print(f"[cleanup] Stopping {len(leftover)} leftover viz process(es)")
    for pid, args in leftover:
        if not quiet:
            print(f"  pid {pid}: {args[:140]}")
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
    time.sleep(0.8)
    leftover = list_stale_viz(exclude_pids=exclude_pids)
    for pid, args in leftover:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    time.sleep(0.3)
    leftover = list_stale_viz(exclude_pids=exclude_pids)
    if leftover and not quiet:
        print(f"[WARN] {len(leftover)} viz process(es) still running after cleanup")
    return len(leftover)


def install_stop_signals(handler) -> None:
    """Map SIGINT/SIGTERM/SIGHUP to the same handler (KeyboardInterrupt-friendly)."""
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, handler)
        except Exception:
            pass


def raise_keyboard(_signum, _frame):
    raise KeyboardInterrupt
