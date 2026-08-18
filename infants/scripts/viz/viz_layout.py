"""Per-trial visualization folder layout.

  trial_NNN/visualizations/{realsense,mocap,rviz}/
"""
from __future__ import annotations

from pathlib import Path

KINDS = ("realsense", "mocap", "rviz")


def vis_dir(trial_dir: Path, kind: str, mkdir: bool = True) -> Path:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    path = Path(trial_dir) / "visualizations" / kind
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def classify_mp4(name: str) -> str | None:
    n = name.lower()
    if n.startswith("rviz_"):
        return "rviz"
    if n.startswith("realsense_marker_overlay"):
        return "realsense"
    if n.startswith("mocap_marker_overlay"):
        return "mocap"
    return None
