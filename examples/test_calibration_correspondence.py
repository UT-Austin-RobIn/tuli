#!/usr/bin/env python3
"""test_calibration_correspondence.py

Sanity check stereo calibration without rectification.

For each selected image pair:
  1. Detect charuco corners in the LEFT image.
  2. solvePnP to get board pose in the left camera frame.
  3. Transform each 3D board corner into the right camera frame via (R, T).
  4. Project into the right image via (K_r, dist_r).
  5. Overlay matching-colored dots on left (detected) and right (predicted).
     If calibration is good, projected dots land ON the actual board corners
     in the right image.

Works fine for wildly convergent / wide-baseline rigs — no rectification.

Usage
-----
    python examples/test_calibration_correspondence.py \
        --calib results/run_<ts>/calibration_results.json \
        --left-dir dataset/left --right-dir dataset/right \
        --out-dir results/run_<ts>/correspondence \
        --n-frames 8 --n-points 8
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

from stereo_calib.charuco import CharucoBoard, CharucoBoardData
from stereo_calib.charuco import CharucoConfig as C


IMG_EXTS = ("*.png", "*.jpg", "*.jpeg")


def load_calib(json_path: Path):
    with open(json_path) as f:
        d = json.load(f)
    K_l = np.array(d["left_camera_calibration_data"]["camera_matrix"], dtype=np.float64)
    dc_l = np.array(d["left_camera_calibration_data"]["dist_coeffs"], dtype=np.float64).reshape(-1)
    K_r = np.array(d["right_camera_calibration_data"]["camera_matrix"], dtype=np.float64)
    dc_r = np.array(d["right_camera_calibration_data"]["dist_coeffs"], dtype=np.float64).reshape(-1)
    R = np.array(d["rot"], dtype=np.float64)
    T = np.array(d["trans"], dtype=np.float64).reshape(3, 1)
    return K_l, dc_l, K_r, dc_r, R, T


def glob_images(d: Path):
    files = []
    for ext in IMG_EXTS:
        files.extend(d.glob(ext))

    def _key(p: Path):
        s = p.stem
        if s.isdigit():
            return int(s)
        tail = s.split("_")[-1]
        return int(tail) if tail.isdigit() else 10**9
    return sorted(files, key=_key)


def detect_charuco(gray: np.ndarray, board):
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(gray)
    return corners, ids


def find_good_frames(left_dir: Path, right_dir: Path, board,
                     min_corners: int, max_frames: int):
    left_files = glob_images(left_dir)
    right_files = glob_images(right_dir)
    if not left_files or not right_files:
        raise FileNotFoundError(f"No images in {left_dir} or {right_dir}")

    right_by_stem = {p.stem: p for p in right_files}
    good = []
    for lp in left_files:
        rp = right_by_stem.get(lp.stem)
        if rp is None:
            continue
        img_l = cv2.imread(str(lp))
        img_r = cv2.imread(str(rp))
        if img_l is None or img_r is None:
            continue
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_charuco(gray_l, board)
        if ids is not None and len(ids) >= min_corners:
            good.append((lp.stem, img_l, img_r, corners, ids))
            if len(good) >= max_frames:
                break
    if not good:
        raise RuntimeError(
            f"No frames with >= {min_corners} charuco corners in left image")
    logger.info(f"Selected {len(good)} pairs for correspondence test")
    return good


def project_points_to_right(obj_pts, img_pts, K_l, dc_l, K_r, dc_r, R, T):
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K_l, dc_l)
    if not ok:
        return None, None, None
    R_board_to_l, _ = cv2.Rodrigues(rvec)
    pts_3d_l = (R_board_to_l @ obj_pts.reshape(-1, 3).T + tvec).T   # Nx3
    pts_3d_r = (R @ pts_3d_l.T + T).T                                # Nx3
    proj_r, _ = cv2.projectPoints(pts_3d_r, np.zeros(3), np.zeros(3),
                                  K_r, dc_r)
    return proj_r.reshape(-1, 2), rvec, tvec


def make_frame_vis(img_l, img_r, corners, ids, K_l, dc_l, K_r, dc_r, R, T,
                   board, n_points: int):
    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    if obj_pts is None or len(obj_pts) < 4:
        return None, None

    proj_r_px, rvec, tvec = project_points_to_right(
        obj_pts, img_pts, K_l, dc_l, K_r, dc_r, R, T)
    if proj_r_px is None:
        return None, None

    detected_l_px = img_pts.reshape(-1, 2)

    n = min(n_points, len(obj_pts))
    step = max(1, len(obj_pts) // n)
    indices = list(range(0, len(obj_pts), step))[:n]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(indices)))

    vis_l = img_l.copy()
    vis_r = img_r.copy()

    H, W = vis_r.shape[:2]
    oob_count = 0
    for ci, idx in enumerate(indices):
        color_bgr = tuple(int(c * 255) for c in colors[ci][:3][::-1])
        px_l = tuple(detected_l_px[idx].astype(int))
        px_r_xy = proj_r_px[idx]
        px_r = tuple(px_r_xy.astype(int))
        cv2.circle(vis_l, px_l, 7, color_bgr, -1)
        cv2.putText(vis_l, str(idx), (px_l[0] + 8, px_l[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)

        in_bounds = 0 <= px_r[0] < W and 0 <= px_r[1] < H
        if not in_bounds:
            oob_count += 1
        cv2.circle(vis_r, px_r, 7, color_bgr, -1)
        cv2.circle(vis_r, px_r, 11, (255, 255, 255), 2)
        cv2.putText(vis_r, str(idx), (px_r[0] + 8, px_r[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)

    # If the right-side points are detectable via charuco, measure residual.
    mean_err = None
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    corners_r, ids_r = detect_charuco(gray_r, board)
    if ids_r is not None and len(ids_r) > 0:
        obj_pts_r, img_pts_r = board.matchImagePoints(corners_r, ids_r)
        if obj_pts_r is not None and len(obj_pts_r) > 0:
            # Build lookup: 3D -> measured pixel in right
            key = lambda p: tuple(np.round(p.ravel(), 6))
            meas = {key(o): ip.ravel() for o, ip in zip(obj_pts_r, img_pts_r)}
            errs = []
            for o, proj_px in zip(obj_pts, proj_r_px):
                m = meas.get(key(o))
                if m is not None:
                    errs.append(np.linalg.norm(proj_px - m))
            if errs:
                mean_err = float(np.mean(errs))

    combined = np.hstack((vis_l, vis_r))
    banner_h = 36
    canvas = np.zeros((combined.shape[0] + banner_h, combined.shape[1], 3),
                      dtype=np.uint8)
    canvas[banner_h:, :] = combined
    header = f"n_corners={len(obj_pts)}  shown={len(indices)}  oob_right={oob_count}"
    if mean_err is not None:
        header += f"  reproj_err_right={mean_err:.2f}px"
    cv2.putText(canvas, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

    stats = dict(
        n_corners=int(len(obj_pts)),
        n_shown=int(len(indices)),
        oob_right=int(oob_count),
        reproj_err_right_px=mean_err,
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), stats


def visualize(calib_json, left_dir, right_dir, out_dir, n_frames, n_points,
              min_corners):
    K_l, dc_l, K_r, dc_r, R, T = load_calib(calib_json)
    logger.info(f"K_l=\n{K_l}")
    logger.info(f"K_r=\n{K_r}")
    logger.info(f"R=\n{R}")
    logger.info(f"T={T.ravel()}  |T|={np.linalg.norm(T):.4f}m")

    charuco_board = CharucoBoard(charuco_data=CharucoBoardData(
        aruco_dict=C.ARUCO_DICT,
        squares_vertically=C.SQUARES_VERTICALLY,
        squares_horizontally=C.SQUARES_HORIZONTALLY,
        square_length=C.SQUARE_LENGTH,
        marker_length=C.MARKER_LENGTH,
    ))
    board = charuco_board.board

    good = find_good_frames(left_dir, right_dir, board,
                            min_corners=min_corners, max_frames=n_frames)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = []
    all_stats = []
    for stem, img_l, img_r, corners, ids in good:
        vis, stats = make_frame_vis(img_l, img_r, corners, ids,
                                    K_l, dc_l, K_r, dc_r, R, T, board, n_points)
        if vis is None:
            logger.warning(f"  skip {stem}: matchImagePoints failed")
            continue
        out_path = out_dir / f"correspondence_{stem}.png"
        plt.imsave(str(out_path), vis)
        logger.info(f"  saved {out_path.name}  "
                    f"oob_right={stats['oob_right']}  "
                    f"err={stats['reproj_err_right_px']}")
        panels.append((stem, vis))
        all_stats.append({"frame": stem, **stats})

    if not panels:
        raise RuntimeError("No valid frames produced")

    # Multi-row summary image
    nrows = len(panels)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 6 * nrows))
    if nrows == 1:
        axes = [axes]
    for ax, (stem, vis) in zip(axes, panels):
        ax.imshow(vis)
        ax.set_title(f"frame {stem}  —  LEFT (detected) | RIGHT (projected)")
        ax.axis("off")
    plt.tight_layout()
    summary_img = out_dir / "summary.png"
    plt.savefig(str(summary_img), dpi=100)
    plt.close(fig)
    logger.info(f"Summary image: {summary_img}")

    # Aggregate metrics
    errs = [s["reproj_err_right_px"] for s in all_stats
            if s["reproj_err_right_px"] is not None]
    oob_total = sum(s["oob_right"] for s in all_stats)
    report = {
        "n_frames": len(all_stats),
        "total_oob_right": oob_total,
        "reproj_err_right_px": {
            "n": len(errs),
            "mean": float(np.mean(errs)) if errs else None,
            "median": float(np.median(errs)) if errs else None,
            "max": float(np.max(errs)) if errs else None,
        },
        "per_frame": all_stats,
    }
    report_path = out_dir / "correspondence_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report: {report_path}")

    # Interpret
    if errs:
        mean_err = np.mean(errs)
        if mean_err < 2.0:
            logger.success(f"Mean right-side reprojection error {mean_err:.2f}px → calibration looks GOOD")
        elif mean_err < 10.0:
            logger.warning(f"Mean right-side reprojection error {mean_err:.2f}px → marginal")
        else:
            logger.error(f"Mean right-side reprojection error {mean_err:.2f}px → BAD calibration")
    else:
        logger.warning("Could not compute right-side reprojection error "
                       "(charuco not detected in right images)")


def parse_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__)
    ap.add_argument("--calib", required=True,
                    help="Path to calibration_results.json")
    ap.add_argument("--left-dir", default="dataset/left")
    ap.add_argument("--right-dir", default="dataset/right")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir (default: <calib_dir>/correspondence)")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--n-points", type=int, default=8)
    ap.add_argument("--min-corners", type=int, default=6,
                    help="Min charuco corners in left image to accept the pair")
    return ap.parse_args()


def main():
    args = parse_args()
    calib = Path(args.calib).resolve()
    if not calib.exists():
        raise FileNotFoundError(calib)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else \
        calib.parent / "correspondence"
    visualize(
        calib_json=calib,
        left_dir=Path(args.left_dir).resolve(),
        right_dir=Path(args.right_dir).resolve(),
        out_dir=out_dir,
        n_frames=args.n_frames,
        n_points=args.n_points,
        min_corners=args.min_corners,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
