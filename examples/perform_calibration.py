import sys
import json
import argparse
import datetime
import numpy as np
from pathlib import Path
from loguru import logger

from stereo_calib.charuco import CharucoBoard, CharucoBoardData
from stereo_calib.charuco import CharucoConfig as C
from stereo_calib.calibration import StereoCalibration
from stereo_calib import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Stereo Calibration")
    parser.add_argument("--data-path", type=str, help="Path to input data folder")
    parser.add_argument("--viz-max-images", type=int, default=None,
                        help="Cap for per-image detection visualizations (default: all)")
    parser.add_argument("--viz-include-rejected", action="store_true",
                        help="Also save detection viz for rejected pairs (default: accepted only)")
    parser.add_argument("--skip-detection-viz", action="store_true",
                        help="Skip saving per-image annotated detection images")
    return parser.parse_args()


def prompt_path(prompt_label: str,
                default_path: Path,
                must_exist: bool = True) -> Path:
    entered = input(f"{prompt_label} [{default_path}]: ").strip()
    chosen = Path(entered).expanduser() if entered else default_path
    resolved = chosen.resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{prompt_label} does not exist: {resolved}")
    return resolved


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    default_data_path = Path(args.data_path).expanduser().resolve() if args.data_path else repo_root / "dataset"
    left_path = default_data_path / "left_images"
    right_path = default_data_path / "right_images"

    # Root for all calibration run outputs.
    default_output_root = repo_root / "results"
    output_root = prompt_path("Results root folder", default_output_root, must_exist=False)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_root / f"run_{timestamp}"
    plots_dir = run_dir / "plots"
    detections_dir = run_dir / "detections"
    run_dir.mkdir(exist_ok=True, parents=True)
    plots_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Calibration run directory: {run_dir}")

    charuco_board = CharucoBoard(charuco_data=CharucoBoardData(aruco_dict=C.ARUCO_DICT,
                                                               squares_vertically=C.SQUARES_VERTICALLY,
                                                               squares_horizontally=C.SQUARES_HORIZONTALLY,
                                                               square_length=C.SQUARE_LENGTH,
                                                               marker_length=C.MARKER_LENGTH))

    # Construct calibration object (runs process_images in __init__).
    # Save detection diagnostics immediately — even if calibrate() blows up later.
    calib: StereoCalibration = StereoCalibration(data_path=default_data_path,
                                                 charuco_board=charuco_board,
                                                 left_dir=left_path,
                                                 right_dir=right_path)
    calib.output_dir = run_dir  # tells log_calibration_summary where to write

    # --- Detection diagnostics (always save, before calibrate() can fail) ---
    utils.save_interpolation_plot(calib.detection_log,
                                  plots_dir / "interpolations_per_image.png")

    if not args.skip_detection_viz:
        utils.save_detection_visualizations(calib.detection_log,
                                            detections_dir,
                                            max_images=args.viz_max_images,
                                            only_accepted=not args.viz_include_rejected)

    # --- Run stereo calibration (no longer raises on validation failure) ---
    calib_results: utils.StereoCalibrationData = calib.calibrate()

    if not calib.validation_passed:
        logger.warning("Stereo calibration failed validation: "
                       + "; ".join(calib.validation_reasons)
                       + ". Outputs will still be saved for inspection.")

    # --- Save each output independently so one failure does not block others ---
    def _safe(step_name, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
            logger.info(f"[OK] {step_name}")
        except Exception as exc:
            logger.exception(f"[FAIL] {step_name}: {exc}")

    _safe("save calibration JSON",
          utils.save_calibration_data, calib_results, run_dir)

    # --- Consolidated run-level summary ---
    run_info = {
        "timestamp": timestamp,
        "data_path": str(default_data_path),
        "left_path": str(left_path),
        "right_path": str(right_path),
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "validation_passed": calib.validation_passed,
        "validation_reasons": calib.validation_reasons,
        "total_image_pairs_loaded": len(calib.left_images_path),
        "filtered_image_pairs_used": len(calib.stereo_obj_points),
        "left_rms_reprojection_error": calib.left_camera_calib_results.rms_reprojection_error,
        "right_rms_reprojection_error": calib.right_camera_calib_results.rms_reprojection_error,
        "stereo_rms_reprojection_error": float(calib_results.rms_stereo_reprojection_error),
        "recalibration_triggered": calib.recalibrate,
        "baseline_distance": float(np.linalg.norm(calib_results.trans)),
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    if calib.validation_passed:
        logger.success(f"All calibration outputs saved to {run_dir}")
    else:
        logger.error(f"Calibration saved to {run_dir} but validation FAILED. "
                     "Do not use these results without manual verification.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(0)
