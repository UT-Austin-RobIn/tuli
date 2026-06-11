"""Visualization utilities for charuco detection diagnostics.

Produces:
- Scatter plot of charuco interpolation counts per processed image (left vs right).
- Per-image annotated images with all detected charuco corners and common corners
  (corners shared between left and right views) labeled distinctly.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm


# BGR colors for OpenCV drawing
COLOR_ALL_CORNERS = (0, 0, 255)      # red — all interpolated charuco corners
COLOR_COMMON_CORNERS = (0, 255, 0)   # green — corners shared across left/right


def save_interpolation_plot(detection_log: List[Dict[str, Any]],
                            output_path: Union[str, Path]) -> None:
    """Scatter plot of charuco interpolation counts per image.

    x-axis = image number (counter from process_images)
    y-axis = number of interpolated charuco corners
    Two colors: left (blue) and right (orange). Dots only, no connecting lines.
    """
    if not detection_log:
        logger.warning("detection_log is empty; skipping interpolation plot")
        return

    counters = [e['counter'] for e in detection_log]
    retvals_l = [e['retval_l'] for e in detection_log]
    retvals_r = [e['retval_r'] for e in detection_log]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(counters, retvals_l, s=12, c='tab:blue', alpha=0.7, label='left')
    ax.scatter(counters, retvals_r, s=12, c='tab:orange', alpha=0.7, label='right')
    ax.set_xlabel('Image number')
    ax.set_ylabel('Number of interpolated charuco corners')
    ax.set_title(f'Charuco interpolation counts per image  (n={len(detection_log)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved interpolation plot to {output_path}")


def _draw_corners(img: np.ndarray,
                  corners: Optional[np.ndarray],
                  ids: Optional[np.ndarray],
                  color: tuple,
                  radius: int = 6,
                  thickness: int = 2,
                  label: bool = True) -> np.ndarray:
    """Draw circles + id labels on corners. Returns new image."""
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()
    if corners is None or ids is None or len(corners) == 0:
        return out
    corners_flat = np.asarray(corners).reshape(-1, 2)
    ids_flat = np.asarray(ids).flatten()
    for (pt, cid) in zip(corners_flat, ids_flat):
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(out, (x, y), radius, color, thickness)
        if label:
            cv2.putText(out, str(int(cid)), (x + radius + 2, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def save_detection_visualizations(detection_log: List[Dict[str, Any]],
                                  output_dir: Union[str, Path],
                                  max_images: Optional[int] = None,
                                  only_accepted: bool = False) -> None:
    """Save per-image charuco detection visualizations.

    For each entry in detection_log (optionally filtered), reads the source
    image and draws:
      - All interpolated charuco corners in red
      - Corners common to left and right views in green (over-drawn)
    Left and right are saved side-by-side as ``{counter:05d}_{left_name}.png``.

    Args:
        detection_log: list of per-image detection dicts from StereoCalibration.
        output_dir: directory to write annotated images into.
        max_images: cap the number of saved pairs (None = all).
        only_accepted: if True, only draw pairs that passed acceptance.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    entries = [e for e in detection_log if not only_accepted or e['accepted']]
    if max_images is not None:
        entries = entries[:max_images]

    if not entries:
        logger.warning("No entries to visualize")
        return

    for e in tqdm(entries, desc="Saving detection visualizations"):
        img_l = cv2.imread(str(e['left_path']), cv2.IMREAD_COLOR)
        img_r = cv2.imread(str(e['right_path']), cv2.IMREAD_COLOR)
        if img_l is None or img_r is None:
            continue

        # All detected charuco corners (red)
        vis_l = _draw_corners(img_l, e.get('left_charuco_corners'),
                              e.get('left_charuco_ids'), COLOR_ALL_CORNERS)
        vis_r = _draw_corners(img_r, e.get('right_charuco_corners'),
                              e.get('right_charuco_ids'), COLOR_ALL_CORNERS)

        # Common corners (green) — overlay on top
        vis_l = _draw_corners(vis_l, e.get('left_common_corners'),
                              e.get('left_common_ids'), COLOR_COMMON_CORNERS,
                              radius=9, thickness=2)
        vis_r = _draw_corners(vis_r, e.get('right_common_corners'),
                              e.get('right_common_ids'), COLOR_COMMON_CORNERS,
                              radius=9, thickness=2)

        # Header text
        status = "ACCEPTED" if e['accepted'] else f"REJECTED ({e['reject_reason']})"
        header = (f"#{e['counter']}  L:{e['left_name']}  R:{e['right_name']}  "
                  f"interp L={e['retval_l']} R={e['retval_r']}  common={e['num_common_pts']}  {status}")

        # Side-by-side
        h = max(vis_l.shape[0], vis_r.shape[0])
        w = vis_l.shape[1] + vis_r.shape[1]
        banner_h = 30
        canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
        canvas[banner_h:banner_h + vis_l.shape[0], :vis_l.shape[1]] = vis_l
        canvas[banner_h:banner_h + vis_r.shape[0], vis_l.shape[1]:] = vis_r
        cv2.putText(canvas, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        # Legend
        cv2.putText(canvas, "red=detected  green=common",
                    (w - 260, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        left_stem = Path(e['left_name']).stem
        right_stem = Path(e['right_name']).stem
        if left_stem == right_stem:
            out_name = f"{left_stem}.png"
        else:
            out_name = f"{left_stem}__{right_stem}.png"
        cv2.imwrite(str(output_dir / out_name), canvas)

    logger.info(f"Saved {len(entries)} detection visualizations to {output_dir}")
