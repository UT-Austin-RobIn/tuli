import logging
import json
import datetime

import cv2
import glob
import numpy as np
from tqdm import tqdm
from termcolor import colored
from pathlib import Path
from loguru import logger
from typing import Union, Optional, List, Tuple, Dict, Any

from stereo_calib.charuco import CharucoBoard
from stereo_calib.utils import CameraCalibrationData, StereoCalibrationData


class StereoCalibration:
    """
    Class for performing stereo calibration using Charuco boards.

    Args:
        data_path (Union[str, Path]): Path to the directory containing left and right camera images.
        charuco_board (CharucoBoard): Charuco board object used for calibration.
        initial_camera_matrix_left (Optional[np.ndarray]): Optional 3x3 intrinsics guess for the left camera.
        initial_camera_matrix_right (Optional[np.ndarray]): Optional 3x3 intrinsics guess for the right camera.

    Attributes:
        data_path (Path): Path to the directory containing calibration images.
        charuco_board (CharucoBoard): Charuco board object used for calibration.
        left_images_path (List[str]): List of paths to left camera images.
        right_images_path (List[str]): List of paths to right camera images.
        stereo_obj_points (List[np.ndarray]): List to store stereo object points.
        stereo_charuco_points_l (List[np.ndarray]): List to store stereo charuco points in left image plane.
        stereo_charuco_ids_l (List[np.ndarray]): List to store stereo charuco ids in left image.
        stereo_charuco_points_r (List[np.ndarray]): List to store stereo charuco points in right image plane.
        stereo_charuco_ids_r (List[np.ndarray]): List to store stereo charuco ids in right image.
        frame_size (Optional[Tuple[int, int]]): Size of the calibration images.
        criteria (Tuple[int, int, float]): Termination criteria for calibration.
        stereo_criteria (Tuple[int, int, float]): Termination criteria for stereo calibration.
        left_camera_calib_results (Optional[CameraCalibrationData]): Results of left camera calibration.
        right_camera_calib_results (Optional[CameraCalibrationData]): Results of right camera calibration.
        best_calib_images_indices (List[int]): Indices of the best calibration images.
        recalibrate (bool): Flag indicating whether recalibration is required.
        max_allowable_rms_error (float): Maximum allowable RMS error for calibration.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 charuco_board: CharucoBoard,
                 left_dir: Optional[Union[str, Path]] = None,
                 right_dir: Optional[Union[str, Path]] = None):
        """
         Initialize StereoCalibration object.

         Args:
             data_path (Union[str, Path]): Path to the directory containing left and right camera images.
             charuco_board (CharucoBoard): Charuco board object used for calibration.
             left_dir (Optional[Union[str, Path]]): Optional explicit path to left camera images.
             right_dir (Optional[Union[str, Path]]): Optional explicit path to right camera images.
         """
        self.data_path = Path(data_path) if isinstance(data_path, str) else data_path
        self.charuco_board = charuco_board
        self.left_dir = Path(left_dir) if isinstance(left_dir, str) else left_dir
        self.right_dir = Path(right_dir) if isinstance(right_dir, str) else right_dir

        left_images_glob = (self.left_dir if self.left_dir is not None else self.data_path / "left").joinpath("*.jpg")
        right_images_glob = (self.right_dir if self.right_dir is not None else self.data_path / "right").joinpath("*.jpg")
        self.left_images_path = self.load_images(left_images_glob)
        self.right_images_path = self.load_images(right_images_glob)
        assert len(self.left_images_path) == len(self.right_images_path)

        # List to store stereo object points, charuco points and charuco ids from all the images.
        self.stereo_obj_points: List[np.ndarray] = []  # 3d point in real world space
        self.stereo_charuco_points_l: List[np.ndarray] = []  # stereo charuco points in left image plane.
        self.stereo_charuco_ids_l: List[np.ndarray] = []  # stereo charuco ids in left image
        self.stereo_charuco_points_r: List[np.ndarray] = []  # stereo charuco points in rightimage plane.
        self.stereo_charuco_ids_r: List[np.ndarray] = []  # stereo charuco ids in right image

        self.frame_size: Optional[Tuple[int, int]] = None
        self._min_points: int = 6
        self.skip_factor: int = 5

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-5)
        self.stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 70, 1e-6)

        self.left_camera_calib_results: Optional[CameraCalibrationData] = None
        self.right_camera_calib_results: Optional[CameraCalibrationData] = None
        self.best_calib_images_indices: List[int] = []
        self.recalibrate: bool = False
        self.max_allowable_rms_error = 0.4
        self.metrics: Dict[str, Any] = {}
        self.detection_log: List[Dict[str, Any]] = []
        self.output_dir: Optional[Path] = None
        self.validation_passed: bool = True
        self.validation_reasons: List[str] = []
        self.accepted_left_names: List[str] = []
        self.accepted_right_names: List[str] = []
        self.process_images()

    @staticmethod
    def load_images(directory_path: Union[str, Path]) -> List[str]:
        """
        Load images from a directory.

        Args:
            directory_path (Union[str, Path]): Path to the directory containing images.

        Returns:
            List[str]: List of paths to the images.
        """
        def image_sort_key(image_path: str) -> int:
            stem = Path(image_path).stem
            if stem.isdigit():
                return int(stem)
            tail = stem.split("_")[-1]
            if tail.isdigit():
                return int(tail)
            raise ValueError(f"Unsupported image filename format: {image_path}")

        images = sorted(glob.glob(str(directory_path)), key=image_sort_key)
        return images

    def init_camera_matrix(self): #side = 'left'
        """
        Initialize camera matrix.

        Args:
            side: "left" or "right" camera.

        Returns:
            np.ndarray: Initial camera matrix.
        """
        # if side == "left":
        #     initial_camera_matrix = np.array([
        #         [1120.887, 0, 438.127],
        #         [0, 1120.821, 234.046],
        #         [0, 0, 1]
        #     ], dtype=np.float32)
        # else:
        #     initial_camera_matrix = np.array([
        #         [905.26, 0, 651.21],
        #         [0, 904.85, 377.03],
        #         [0, 0, 1]
        #     ], dtype=np.float32)

        initial_camera_matrix = np.array([
            [1121, 0, 651],
            [0, 1186, 383],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return initial_camera_matrix

    def process_images(self):
        """
         Process calibration images to extract charuco points and ids.
        """
        params = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(self.charuco_board.aruco_dict, params)

        counter = 0
        self.left_frame_size = (1280, 720)
        self.right_frame_size = (1280, 720)
        left_interp_counter, right_interp_counter = 0, 0
        accepted_image_numbers = []
        images_processed = 0
        images_both_detected = 0
        common_pts_counts = []
        self.imgs_l, self.imgs_r = [], []
        for img_left_path, img_right_path in tqdm(zip(self.left_images_path, self.right_images_path),
                                                  total=len(self.left_images_path),
                                                  desc="Processing calibration images"):
            counter += 1
            if counter % self.skip_factor != 0:
                continue
            images_processed += 1

            if len(self.stereo_obj_points) > 300:
                print("got 300 image pairs")
                break

            img_l = cv2.imread(img_left_path, cv2.IMREAD_COLOR)
            img_r = cv2.imread(img_right_path, cv2.IMREAD_COLOR)

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            if self.frame_size is None:
                self.frame_size = gray_l.shape
            else:
                # TODO: change this later
                #assert self.frame_size == gray_l.shape == gray_r.shape
                pass

            corners_l, ids_l, rejected_img_points_l = aruco_detector.detectMarkers(gray_l)
            corners_r, ids_r, rejected_img_points_r = aruco_detector.detectMarkers(gray_r)

            print("=== image number: ", counter)
            # print("Number of corners detected in left image: ", len(corners_l))
            # print("Number of corners detected in right image: ", len(corners_r))

            if ids_l is None:
                print(colored(f"No aruco markers found in left image {img_left_path.split('/')[-1]}", "red"))
            if ids_r is None:
                print(colored(f"No aruco markers found in right image {img_right_path.split('/')[-1]}", "red"))

            # img_markers = cv2.aruco.drawDetectedMarkers(gray_l.copy(), corners_l, ids_l)
            # cv2.imshow("Markers", img_markers)
            # print("Detected marker IDs:", ids_l.ravel())
            # print("Charuco board marker IDs:", self.charuco_board.board.getIds().ravel())

            det_entry: Dict[str, Any] = {
                'counter': counter,
                'left_path': img_left_path,
                'right_path': img_right_path,
                'left_name': Path(img_left_path).name,
                'right_name': Path(img_right_path).name,
                'retval_l': 0,
                'retval_r': 0,
                'left_charuco_corners': None,
                'left_charuco_ids': None,
                'right_charuco_corners': None,
                'right_charuco_ids': None,
                'left_common_corners': None,
                'left_common_ids': None,
                'right_common_corners': None,
                'right_common_ids': None,
                'num_common_pts': 0,
                'accepted': False,
                'reject_reason': 'no_markers_detected',
            }

            if ids_l is not None and ids_r is not None:
                images_both_detected += 1
                retval_l, charuco_corners_l, charuco_ids_l = cv2.aruco.interpolateCornersCharuco(corners_l, ids_l,
                                                                                                 gray_l,
                                                                                                 self.charuco_board.board)
                retval_r, charuco_corners_r, charuco_ids_r = cv2.aruco.interpolateCornersCharuco(corners_r, ids_r,
                                                                                                 gray_r,
                                                                                                 self.charuco_board.board)

                det_entry['retval_l'] = int(retval_l)
                det_entry['retval_r'] = int(retval_r)
                det_entry['left_charuco_corners'] = charuco_corners_l
                det_entry['left_charuco_ids'] = charuco_ids_l
                det_entry['right_charuco_corners'] = charuco_corners_r
                det_entry['right_charuco_ids'] = charuco_ids_r
                det_entry['reject_reason'] = 'low_interpolation'

                print("Left Image Number of interpolations: ", retval_l)
                if retval_l > 0:
                    left_interp_counter += 1

                print("Right Image Number of interpolations: ", retval_r)
                if retval_r > 0:
                    right_interp_counter += 1

                if (charuco_corners_l is not None and charuco_corners_r is not None and
                        retval_l > self._min_points and retval_r > self._min_points):
                    obj_pts_l, img_pts_l = cv2.aruco.getBoardObjectAndImagePoints(self.charuco_board.board,
                                                                                  charuco_corners_l,
                                                                                  charuco_ids_l)
                    obj_pts_r, img_pts_r = cv2.aruco.getBoardObjectAndImagePoints(self.charuco_board.board,
                                                                                  charuco_corners_r,
                                                                                  charuco_ids_r)

                    # make sure img_pts_l and img_pts_r are within range
                    width, height = 1280, 720
                    x = img_pts_l[:, 0, 0]
                    y = img_pts_l[:, 0, 1]
                    if np.any(x < 0) or np.any(x > width) or np.any(y < 0) or np.any(y > height):
                        print(f"Left Image {counter} has corners out of bounds!")
                    x = img_pts_r[:, 0, 0]
                    y = img_pts_r[:, 0, 1]
                    if np.any(x < 0) or np.any(x > width) or np.any(y < 0) or np.any(y > height):
                        print(f"Right Image {counter} has corners out of bounds!")

                    pts_l = {tuple(a): tuple(b) for a, b in zip(obj_pts_l[:, 0], img_pts_l[:, 0])}
                    pts_r = {tuple(a): tuple(b) for a, b in zip(obj_pts_r[:, 0], img_pts_r[:, 0])}
                    ids_l_map = {tuple(a): b for a, b in zip(obj_pts_l[:, 0], charuco_ids_l[:, 0])}
                    ids_r_map = {tuple(a): b for a, b in zip(obj_pts_r[:, 0], charuco_ids_r[:, 0])}
                    common_pts = set(pts_l.keys()) & set(pts_r.keys())
                    print("common_pts: ", common_pts)

                    print(len(pts_l), len(pts_r), len(common_pts))
                    common_pts_counts.append(len(common_pts))
                    det_entry['num_common_pts'] = len(common_pts)

                    if len(common_pts) >= self._min_points:
                        obj = np.zeros((len(common_pts), 1, 3), dtype=np.float32)
                        left_corners = np.zeros((len(common_pts), 1, 2), dtype=np.float32)
                        right_corners = np.zeros((len(common_pts), 1, 2), dtype=np.float32)
                        left_corner_ids = np.zeros((len(common_pts), 1), dtype=np.int32)
                        right_corner_ids = np.zeros((len(common_pts), 1), dtype=np.int32)
                        for i, pts in enumerate(common_pts):
                            obj[i] = pts
                            left_corners[i] = np.reshape(pts_l[pts], (1, 2))
                            right_corners[i] = np.reshape(pts_r[pts], (1, 2))
                            left_corner_ids[i] = ids_l_map[pts]
                            right_corner_ids[i] = ids_r_map[pts]

                        self.stereo_obj_points.append(obj)
                        self.stereo_charuco_points_l.append(left_corners)
                        self.stereo_charuco_points_r.append(right_corners)
                        self.stereo_charuco_ids_l.append(left_corner_ids)
                        self.stereo_charuco_ids_r.append(right_corner_ids)
                        self.imgs_l.append(img_l)
                        self.imgs_r.append(img_r)
                        self.accepted_left_names.append(Path(img_left_path).name)
                        self.accepted_right_names.append(Path(img_right_path).name)
                        accepted_image_numbers.append((counter, Path(img_left_path).name, Path(img_right_path).name, len(common_pts)))

                        det_entry['accepted'] = True
                        det_entry['reject_reason'] = None
                        det_entry['left_common_corners'] = left_corners
                        det_entry['left_common_ids'] = left_corner_ids
                        det_entry['right_common_corners'] = right_corners
                        det_entry['right_common_ids'] = right_corner_ids
                    else:
                        det_entry['reject_reason'] = 'insufficient_common_points'

            self.detection_log.append(det_entry)

        assert len(self.stereo_obj_points) == len(self.stereo_charuco_points_l) == len(
            self.stereo_charuco_points_r) == len(
            self.stereo_charuco_ids_l) == len(self.stereo_charuco_ids_r)

        print("Number of filtered images (will be used for stereo calibration): ", len(self.stereo_obj_points))
        for img_num, left_name, right_name, n_pts in accepted_image_numbers:
            print(f"  ACCEPTED: image #{img_num} | {left_name} / {right_name} | {n_pts} common points")

        # Store image processing metrics
        self.metrics["image_processing"] = {
            "total_image_pairs_in_dataset": len(self.left_images_path),
            "skip_factor": self.skip_factor,
            "image_pairs_processed": images_processed,
            "image_pairs_both_detected": images_both_detected,
            "image_pairs_used_for_calibration": len(self.stereo_obj_points),
            "common_pts_per_image": {
                "min": int(np.min(common_pts_counts)) if common_pts_counts else 0,
                "max": int(np.max(common_pts_counts)) if common_pts_counts else 0,
                "mean": float(np.mean(common_pts_counts)) if common_pts_counts else 0,
                "std": float(np.std(common_pts_counts)) if common_pts_counts else 0,
            },
        }

        if not self.stereo_obj_points:
            raise ValueError(
                "No usable stereo calibration pairs were found. "
                f"Processed {images_processed} image pairs out of {len(self.left_images_path)} "
                f"(skip factor {self.metrics['image_processing']['skip_factor']}). "
                f"{images_both_detected} pairs had detections in both views, but 0 survived the "
                "shared-ChArUco filtering. This usually means the board is missing from many frames, "
                "too small to detect reliably, or not sufficiently visible in both cameras at the same time."
            )

        init_camera_matrix = self.init_camera_matrix()

        # init_left = self.init_camera_matrix() #side="left"
        # init_right = self.init_camera_matrix() #side="right"

        # for j in range(len(self.stereo_charuco_points_l)):
        #     print(self.stereo_charuco_points_l[j].shape, self.stereo_charuco_points_r[j].shape)

        # left camera calibration
        print("starting left camera calibration")
        self.left_camera_calib_results = self.calibrate_camera(charuco_points=self.stereo_charuco_points_l,
                                                               charuco_ids=self.stereo_charuco_ids_l,
                                                               initial_camera_matrix=init_camera_matrix,
                                                               criteria=self.criteria,
                                                               frame_size=self.left_frame_size)
        print("left camera calibration completed")

        # right camera calibration
        print("starting right camera calibration")
        self.right_camera_calib_results = self.calibrate_camera(charuco_points=self.stereo_charuco_points_r,
                                                                charuco_ids=self.stereo_charuco_ids_r,
                                                                initial_camera_matrix=init_camera_matrix,
                                                                criteria=self.criteria,
                                                                frame_size=self.right_frame_size)
        print("right camera calibration completed")

        print("left_intr: ", self.left_camera_calib_results.camera_matrix, self.left_camera_calib_results.rms_reprojection_error)
        print("right_intr: ", self.right_camera_calib_results.camera_matrix, self.right_camera_calib_results.rms_reprojection_error)

        # Store initial intrinsic calibration metrics
        left_per_image_errors = self.calculate_reprojection_error(
            self.stereo_obj_points, self.stereo_charuco_points_l, self.left_camera_calib_results)
        right_per_image_errors = self.calculate_reprojection_error(
            self.stereo_obj_points, self.stereo_charuco_points_r, self.right_camera_calib_results)

        self.metrics["initial_intrinsic_calibration"] = {
            "num_images_used": len(self.stereo_obj_points),
            "left": {
                "initial_camera_matrix": init_camera_matrix.tolist(),
                "calibrated_camera_matrix": self.left_camera_calib_results.camera_matrix.tolist(),
                "drift_fx": float(self.left_camera_calib_results.camera_matrix[0, 0] - init_camera_matrix[0, 0]),
                "drift_fy": float(self.left_camera_calib_results.camera_matrix[1, 1] - init_camera_matrix[1, 1]),
                "drift_cx": float(self.left_camera_calib_results.camera_matrix[0, 2] - init_camera_matrix[0, 2]),
                "drift_cy": float(self.left_camera_calib_results.camera_matrix[1, 2] - init_camera_matrix[1, 2]),
                "rms_reprojection_error": self.left_camera_calib_results.rms_reprojection_error,
                "dist_coeffs": self.left_camera_calib_results.dist_coeffs.flatten().tolist(),
                "per_image_error_min": float(np.min(left_per_image_errors)),
                "per_image_error_max": float(np.max(left_per_image_errors)),
                "per_image_error_mean": float(np.mean(left_per_image_errors)),
                "per_image_error_std": float(np.std(left_per_image_errors)),
            },
            "right": {
                "initial_camera_matrix": init_camera_matrix.tolist(),
                "calibrated_camera_matrix": self.right_camera_calib_results.camera_matrix.tolist(),
                "drift_fx": float(self.right_camera_calib_results.camera_matrix[0, 0] - init_camera_matrix[0, 0]),
                "drift_fy": float(self.right_camera_calib_results.camera_matrix[1, 1] - init_camera_matrix[1, 1]),
                "drift_cx": float(self.right_camera_calib_results.camera_matrix[0, 2] - init_camera_matrix[0, 2]),
                "drift_cy": float(self.right_camera_calib_results.camera_matrix[1, 2] - init_camera_matrix[1, 2]),
                "rms_reprojection_error": self.right_camera_calib_results.rms_reprojection_error,
                "dist_coeffs": self.right_camera_calib_results.dist_coeffs.flatten().tolist(),
                "per_image_error_min": float(np.min(right_per_image_errors)),
                "per_image_error_max": float(np.max(right_per_image_errors)),
                "per_image_error_mean": float(np.mean(right_per_image_errors)),
                "per_image_error_std": float(np.std(right_per_image_errors)),
            },
        }

        def draw_reprojections(img, objpoints, imgpoints, camera_matrix, dist_coeffs, rvec, tvec):
            proj, _ = cv2.projectPoints(objpoints, rvec, tvec, camera_matrix, dist_coeffs)
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape)==2 else img.copy()
            for p_obs, p_proj in zip(np.array(imgpoints).reshape(-1,2), proj.reshape(-1,2)):
                cv2.circle(vis, tuple(p_obs.astype(int)), 3, (0,255,0), -1)   # observed (green)
                cv2.circle(vis, tuple(p_proj.astype(int)), 3, (0,0,255), -1)  # reprojected (red)
            return vis

        # # Example for the i-th calibration image:
        # vis = draw_reprojections(imgs_r[0],
        #                         self.stereo_obj_points[0],
        #                         self.stereo_charuco_points_r[0],
        #                         self.right_camera_calib_results.camera_matrix,
        #                         self.right_camera_calib_results.dist_coeffs,
        #                         self.right_camera_calib_results.rotation_vectors[0],
        #                         self.right_camera_calib_results.translation_vectors[0])
        # cv2.imshow("reproj", vis); cv2.waitKey(0); cv2.destroyAllWindows()


        if self.left_camera_calib_results.rms_reprojection_error > self.max_allowable_rms_error or \
                self.right_camera_calib_results.rms_reprojection_error > self.max_allowable_rms_error:
            print("Will recalibrate!")
            self.recalibrate = True
            self.select_best_calib_images()

    def calibrate_camera(self,
                         charuco_points: List[np.ndarray],
                         charuco_ids: List[np.ndarray],
                         initial_camera_matrix: np.ndarray,
                         criteria: Tuple[int, int, float],
                         frame_size) -> CameraCalibrationData:
        """
        Calibrate a camera.

        Args:
            charuco_points (List[np.ndarray]): Charuco points detected in the calibration images.
            charuco_ids (List[np.ndarray]): Charuco IDs detected in the calibration images.
            initial_camera_matrix (np.ndarray): Initial camera matrix.
            criteria (Tuple[int, int, float]): Termination criteria for calibration.

        Returns:
            CameraCalibrationData: Results of camera calibration.
        """
        flags = 0
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_RATIONAL_MODEL

        # flags_no_guess = flags & ~cv2.CALIB_USE_INTRINSIC_GUESS  # remove that bit

        print("start calibrateCameraCharuco")
        width, height = 1280, 720
        # breakpoint()

        for i, corners in enumerate(charuco_points):
            # print(corners.shape)
            # breakpoint()
            # corners has shape (N, 1, 2)
            x = corners[:, 0, 0]
            y = corners[:, 0, 1]
            
            if np.any(x < 0) or np.any(x > width) or np.any(y < 0) or np.any(y > height):
                print(f"Image {i} has corners out of bounds!")

        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=charuco_points,
            charucoIds=charuco_ids,
            board=self.charuco_board.board,
            imageSize=frame_size,
            cameraMatrix=initial_camera_matrix,
            distCoeffs=None,
            flags=flags,
            criteria=criteria)
        print("end calibrateCameraCharuco")

        image_width, image_height = self.frame_size[::-1]

        undistort_map_x, undistort_map_y = cv2.initUndistortRectifyMap(cameraMatrix=camera_matrix,
                                                                       distCoeffs=dist_coeffs,
                                                                       R=None,
                                                                       newCameraMatrix=None,
                                                                       size=self.frame_size[::-1],
                                                                       m1type=cv2.CV_16SC2)

        return CameraCalibrationData(rms_reprojection_error=retval,
                                     camera_matrix=camera_matrix,
                                     dist_coeffs=dist_coeffs,
                                     rotation_vectors=list(rvecs),
                                     translation_vectors=list(tvecs),
                                     undistort_map_x=undistort_map_x,
                                     undistort_map_y=undistort_map_y,
                                     image_width=image_width,
                                     image_height=image_height,
                                     image_dim=(image_width, image_height))

    def select_best_calib_images(self, keep_best_ratio: float = 0.75) -> None:
        """
        Select the best calibration images based on reprojection error.

        Args:
            keep_best_ratio (float): Ratio of best images to keep.
        """
        left_reprojection_errors = self.calculate_reprojection_error(obj_pts=self.stereo_obj_points,
                                                                     image_pts=self.stereo_charuco_points_l,
                                                                     camera_calib_results=self.left_camera_calib_results)
        right_reprojection_errors = self.calculate_reprojection_error(obj_pts=self.stereo_obj_points,
                                                                      image_pts=self.stereo_charuco_points_r,
                                                                      camera_calib_results=self.right_camera_calib_results)

        mean_errors = (np.array(left_reprojection_errors) + np.array(right_reprojection_errors)) / 2.0
        mean_errors = mean_errors.tolist()
        mean_errors_sorted_indices = sorted(range(len(mean_errors)), key=lambda i: mean_errors[i])
        threshold_index = int(len(mean_errors_sorted_indices) * keep_best_ratio)
        self.best_calib_images_indices = sorted(mean_errors_sorted_indices[:threshold_index])

    @staticmethod
    def calculate_reprojection_error(obj_pts: List[np.ndarray],
                                     image_pts: List[np.ndarray],
                                     camera_calib_results: CameraCalibrationData) -> List[float]:
        """
        Calculate reprojection error.

        Args:
            obj_pts (List[np.ndarray]): Object points.
            image_pts (List[np.ndarray]): Image points.
            camera_calib_results (CameraCalibrationData): Results of camera calibration.

        Returns:
            List[float]: List of reprojection errors.
        """
        reprojected_errors: List[float] = [0.0] * len(obj_pts)
        for id, obj in enumerate(obj_pts):
            projected_pts, _ = cv2.projectPoints(objectPoints=obj,
                                                 rvec=camera_calib_results.rotation_vectors[id],
                                                 tvec=camera_calib_results.translation_vectors[id],
                                                 cameraMatrix=camera_calib_results.camera_matrix,
                                                 distCoeffs=camera_calib_results.dist_coeffs)
            error_pts = image_pts[id] - projected_pts
            rms_error_per_image = np.sqrt(np.mean(np.sum(error_pts.squeeze(axis=1) ** 2, axis=1)))
            reprojected_errors[id] = rms_error_per_image
        return reprojected_errors

    def recalibrate_cameras(self):
        """
        Recalibrate cameras using the best calibration images.
        """
        self.left_images_path = [self.left_images_path[i] for i in self.best_calib_images_indices]
        self.right_images_path = [self.right_images_path[i] for i in self.best_calib_images_indices]
        self.stereo_obj_points = [self.stereo_obj_points[i] for i in self.best_calib_images_indices]
        self.stereo_charuco_points_l = [self.stereo_charuco_points_l[i] for i in self.best_calib_images_indices]
        self.stereo_charuco_ids_l = [self.stereo_charuco_ids_l[i] for i in self.best_calib_images_indices]
        self.stereo_charuco_points_r = [self.stereo_charuco_points_r[i] for i in self.best_calib_images_indices]
        self.stereo_charuco_ids_r = [self.stereo_charuco_ids_r[i] for i in self.best_calib_images_indices]
        self.imgs_l = [self.imgs_l[i] for i in self.best_calib_images_indices]
        self.imgs_r = [self.imgs_r[i] for i in self.best_calib_images_indices]
        if self.accepted_left_names:
            self.accepted_left_names = [self.accepted_left_names[i] for i in self.best_calib_images_indices]
            self.accepted_right_names = [self.accepted_right_names[i] for i in self.best_calib_images_indices]

        self.left_camera_calib_results = self.calibrate_camera(charuco_points=self.stereo_charuco_points_l,
                                                               charuco_ids=self.stereo_charuco_ids_l,
                                                               initial_camera_matrix=self.left_camera_calib_results.camera_matrix,
                                                               criteria=self.criteria,
                                                               frame_size=self.left_frame_size)

        self.right_camera_calib_results = self.calibrate_camera(charuco_points=self.stereo_charuco_points_r,
                                                                charuco_ids=self.stereo_charuco_ids_r,
                                                                initial_camera_matrix=self.right_camera_calib_results.camera_matrix,
                                                                criteria=self.criteria,
                                                                frame_size=self.right_frame_size)
        
        print("After recalibration:")
        print("Number of images for recalibration: ", len(self.best_calib_images_indices))
        print("left_intr: ", self.left_camera_calib_results.camera_matrix, self.left_camera_calib_results.rms_reprojection_error)
        print("right_intr: ", self.right_camera_calib_results.camera_matrix, self.right_camera_calib_results.rms_reprojection_error)

        self.metrics["recalibration"] = {
            "triggered": True,
            "images_before": self.metrics["image_processing"]["image_pairs_used_for_calibration"],
            "images_after": len(self.best_calib_images_indices),
            "images_dropped": self.metrics["image_processing"]["image_pairs_used_for_calibration"] - len(self.best_calib_images_indices),
            "left_rms_after": self.left_camera_calib_results.rms_reprojection_error,
            "right_rms_after": self.right_camera_calib_results.rms_reprojection_error,
            "left_camera_matrix_after": self.left_camera_calib_results.camera_matrix.tolist(),
            "right_camera_matrix_after": self.right_camera_calib_results.camera_matrix.tolist(),
        }


    def calibrate(self) -> StereoCalibrationData:
        """
        Perform stereo calibration.

        Returns:
            StereoCalibrationData: Results of stereo calibration.
        """
        logger.info("Starting Calibration")
        flags = 0
        flags |= cv2.CALIB_RATIONAL_MODEL

        if self.recalibrate:
            self.recalibrate_cameras()
            flags |= cv2.CALIB_FIX_INTRINSIC
        else:
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS

        print("Starting stereo calibration!")
        # breakpoint()
        ret_stereo, new_camera_matrix_l, new_dist_coeffs_l, new_camera_matrix_r, new_dist_coeffs_r, rot, trans, \
            essential_matrix, fundamental_matrix = cv2.stereoCalibrate(objectPoints=self.stereo_obj_points,
                                                                       imagePoints1=self.stereo_charuco_points_l,
                                                                       imagePoints2=self.stereo_charuco_points_r,
                                                                       cameraMatrix1=self.left_camera_calib_results.camera_matrix.copy(),
                                                                       distCoeffs1=self.left_camera_calib_results.dist_coeffs.copy(),
                                                                       cameraMatrix2=self.right_camera_calib_results.camera_matrix.copy(),
                                                                       distCoeffs2=self.right_camera_calib_results.dist_coeffs.copy(),
                                                                       imageSize=self.frame_size[::-1],
                                                                       criteria=self.stereo_criteria,
                                                                       flags=flags)

        # np.testing.assert_array_equal(new_camera_matrix_l, self.left_camera_calib_results.camera_matrix)
        # np.testing.assert_array_equal(new_dist_coeffs_l, self.left_camera_calib_results.dist_coeffs)

        # np.testing.assert_array_equal(new_camera_matrix_r, self.right_camera_calib_results.camera_matrix)
        # np.testing.assert_array_equal(new_dist_coeffs_r, self.right_camera_calib_results.dist_coeffs)

        # stereo rectification
        rect_l, rect_r, proj_matrix_l, proj_matrix_r, Q, roi_l, roi_r = cv2.stereoRectify(
            cameraMatrix1=new_camera_matrix_l,
            distCoeffs1=new_dist_coeffs_l,
            cameraMatrix2=new_camera_matrix_r,
            distCoeffs2=new_dist_coeffs_r,
            imageSize=self.frame_size[::-1],
            R=rot,
            T=trans,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=1)

        image_size = (1280, 720)
        newCamMatrix_left, roi_left = cv2.getOptimalNewCameraMatrix(
            new_camera_matrix_l, new_dist_coeffs_l, image_size, alpha=0)  # alpha=0 crops to valid pixels

        newCamMatrix_right, roi_right = cv2.getOptimalNewCameraMatrix(
            new_camera_matrix_r, new_dist_coeffs_r, image_size, alpha=0)

        stereo_rectify_map_l_x, stereo_rectify_map_l_y = cv2.initUndistortRectifyMap(cameraMatrix=new_camera_matrix_l,
                                                                                     distCoeffs=new_dist_coeffs_l,
                                                                                     R=rect_l,
                                                                                     newCameraMatrix=proj_matrix_l,                                                                                     
                                                                                    #  newCameraMatrix=newCamMatrix_left,
                                                                                     size=self.frame_size[::-1],
                                                                                     m1type=cv2.CV_16SC2)

        stereo_rectify_map_r_x, stereo_rectify_map_r_y = cv2.initUndistortRectifyMap(cameraMatrix=new_camera_matrix_r,
                                                                                     distCoeffs=new_dist_coeffs_r,
                                                                                     R=rect_r,
                                                                                     newCameraMatrix=proj_matrix_r,
                                                                                    #  newCameraMatrix=newCamMatrix_right,
                                                                                     size=self.frame_size[::-1],
                                                                                     m1type=cv2.CV_16SC2)

        self.left_camera_calib_results.stereo_rectify_map_x = stereo_rectify_map_l_x
        self.left_camera_calib_results.stereo_rectify_map_y = stereo_rectify_map_l_y
        self.right_camera_calib_results.stereo_rectify_map_x = stereo_rectify_map_r_x
        self.right_camera_calib_results.stereo_rectify_map_y = stereo_rectify_map_r_y

        # ===================================
        rect_pass_count = 0
        rect_fail_count = 0
        rect_common_ids_counts = []

        rect_viz_dir: Optional[Path] = None
        if self.output_dir is not None:
            rect_viz_dir = Path(self.output_dir) / "rectified_viz"
            rect_viz_dir.mkdir(exist_ok=True, parents=True)

        params = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(self.charuco_board.aruco_dict, params)

        def detect_charuco(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco_detector.detectMarkers(gray)
            charuco_corners = None
            charuco_ids = None
            out = img.copy()
            if ids is not None and len(ids) > 0:
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, self.charuco_board.board)
                if charuco_corners is not None and len(charuco_corners) > 0:
                    cv2.aruco.drawDetectedCornersCharuco(out, charuco_corners, charuco_ids, (0, 0, 255))
            return out, charuco_corners, charuco_ids

        for i in range(len(self.imgs_l)):
            left_img = self.imgs_l[i]
            right_img = self.imgs_r[i]

            left_rect = cv2.remap(left_img, stereo_rectify_map_l_x, stereo_rectify_map_l_y, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_img, stereo_rectify_map_r_x, stereo_rectify_map_r_y, cv2.INTER_LINEAR)

            left_anno, _, left_charuco_ids = detect_charuco(left_rect)
            right_anno, _, right_charuco_ids = detect_charuco(right_rect)

            n_common = 0
            passed = False
            if left_charuco_ids is not None and right_charuco_ids is not None:
                common_ids = np.intersect1d(left_charuco_ids, right_charuco_ids)
                n_common = int(len(common_ids))
                if n_common > 0:
                    rect_pass_count += 1
                    rect_common_ids_counts.append(n_common)
                    passed = True
                else:
                    rect_fail_count += 1
            else:
                rect_fail_count += 1

            status = "PASS" if passed else "FAIL"
            l_det = 0 if left_charuco_ids is None else int(len(left_charuco_ids))
            r_det = 0 if right_charuco_ids is None else int(len(right_charuco_ids))
            print(f"[rect_viz] i={i} {status} L_det={l_det} R_det={r_det} common={n_common}")

            if rect_viz_dir is None:
                continue

            # ensure 3-channel for side-by-side canvas
            if left_anno.ndim == 2:
                left_anno = cv2.cvtColor(left_anno, cv2.COLOR_GRAY2BGR)
            if right_anno.ndim == 2:
                right_anno = cv2.cvtColor(right_anno, cv2.COLOR_GRAY2BGR)

            h = max(left_anno.shape[0], right_anno.shape[0])
            w = left_anno.shape[1] + right_anno.shape[1]
            banner_h = 30
            canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
            canvas[banner_h:banner_h + left_anno.shape[0], :left_anno.shape[1]] = left_anno
            canvas[banner_h:banner_h + right_anno.shape[0], left_anno.shape[1]:] = right_anno

            # horizontal epi-lines across full canvas (rectified rows should align)
            for y in range(banner_h, h + banner_h, 40):
                cv2.line(canvas, (0, y), (w, y), (0, 255, 0), 1)

            left_name = self.accepted_left_names[i] if i < len(self.accepted_left_names) else f"idx{i}"
            right_name = self.accepted_right_names[i] if i < len(self.accepted_right_names) else f"idx{i}"
            header = (f"#{i} {status}  L:{left_name}  R:{right_name}  "
                      f"L_det={l_det} R_det={r_det} common={n_common}")
            cv2.putText(canvas, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

            left_stem = Path(left_name).stem
            right_stem = Path(right_name).stem
            out_name = f"{i:04d}_{status}_{left_stem}" + (
                f"__{right_stem}.png" if left_stem != right_stem else ".png")
            cv2.imwrite(str(rect_viz_dir / out_name), canvas)
        # ===================================

        # Store stereo calibration metrics
        stereo_flags_str = "CALIB_FIX_INTRINSIC" if self.recalibrate else "CALIB_USE_INTRINSIC_GUESS"
        r_diag = np.diag(rot).tolist()
        r_off_max = float(np.max(np.abs(rot - np.diag(np.diag(rot)))))
        self.metrics["stereo_calibration"] = {
            "num_image_pairs_used": len(self.stereo_obj_points),
            "stereo_rms_reprojection_error": float(ret_stereo),
            "rotation_matrix": rot.tolist(),
            "rotation_diag": r_diag,
            "rotation_offdiag_max_abs": r_off_max,
            "translation_vector": trans.ravel().tolist(),
            "baseline_distance": float(np.linalg.norm(trans)),
            "flags": stereo_flags_str,
            "roi_left": list(roi_l),
            "roi_right": list(roi_r),
            "stereo_rectify_alpha": 1,
            "rect_l": rect_l.tolist(),
            "rect_r": rect_r.tolist(),
            "proj_matrix_l": proj_matrix_l.tolist(),
            "proj_matrix_r": proj_matrix_r.tolist(),
        }

        if "recalibration" not in self.metrics:
            self.metrics["recalibration"] = {"triggered": False}

        self.metrics["rectification_validation"] = {
            "total_pairs_tested": rect_pass_count + rect_fail_count,
            "pairs_passed": rect_pass_count,
            "pairs_failed": rect_fail_count,
            "pass_rate": rect_pass_count / max(rect_pass_count + rect_fail_count, 1),
            "common_ids_in_passing_pairs": {
                "min": int(np.min(rect_common_ids_counts)) if rect_common_ids_counts else 0,
                "max": int(np.max(rect_common_ids_counts)) if rect_common_ids_counts else 0,
                "mean": float(np.mean(rect_common_ids_counts)) if rect_common_ids_counts else 0,
            },
        }

        unusable_reasons = []
        if rect_pass_count == 0:
            unusable_reasons.append("rectification validation found 0 passing stereo pairs")
        if roi_r[2] == 0 or roi_r[3] == 0:
            unusable_reasons.append(f"right rectified ROI collapsed to {list(roi_r)}")
        if not np.isfinite(ret_stereo) or ret_stereo > 1.0:
            unusable_reasons.append(f"stereo RMS error is too high ({ret_stereo:.4f})")

        self.metrics["config"] = {
            "frame_size": list(self.frame_size) if self.frame_size else None,
            "left_frame_size": list(self.left_frame_size),
            "right_frame_size": list(self.right_frame_size),
            "charuco_board": {
                "aruco_dict": str(self.charuco_board.charuco_data.aruco_dict),
                "squares_vertically": self.charuco_board.charuco_data.squares_vertically,
                "squares_horizontally": self.charuco_board.charuco_data.squares_horizontally,
                "square_length": self.charuco_board.charuco_data.square_length,
                "marker_length": self.charuco_board.charuco_data.marker_length,
            },
            "max_allowable_rms_error": self.max_allowable_rms_error,
        }

        self.metrics["stereo_validation"] = {
            "usable": len(unusable_reasons) == 0,
            "reasons": unusable_reasons,
        }

        self.validation_passed = len(unusable_reasons) == 0
        self.validation_reasons = unusable_reasons

        if unusable_reasons:
            logger.error("Stereo calibration failed validation: " + "; ".join(unusable_reasons))
            logger.warning("Returning calibration result anyway; inspect metrics to decide usability.")
        else:
            logger.success("Calibration complete!")
            logger.info(f"Stereo Calibration result: RMS error = {ret_stereo:.4f}")

        self.log_calib_info()
        self.log_calibration_summary(self.output_dir)

        return StereoCalibrationData(rms_stereo_reprojection_error=ret_stereo,
                                     left_camera_calibration_data=self.left_camera_calib_results,
                                     right_camera_calibration_data=self.right_camera_calib_results,
                                     left_camera_rectification_transform=rect_l,
                                     right_camera_rectification_transform=rect_r,
                                     rot=rot,
                                     trans=trans,
                                     essential_matrix=essential_matrix,
                                     fundamental_matrix=fundamental_matrix,
                                     projection_matrix_left=proj_matrix_l,
                                     projection_matrix_right=proj_matrix_r,
                                     perspective_transformation_matrix_Q=Q,
                                     transform_r_to_l=None,
                                     left_cam_to_base_transform=None,
                                     right_cam_to_base_transform=None,
                                     left_cam_serial_id=None,
                                     right_cam_serial_id=None,
                                     robot_joint_vals=None)

    def log_calib_info(self):
        logger.info(f"Left Camera Calibration: RMS error = {self.left_camera_calib_results.rms_reprojection_error:.4f}")
        logger.info(
            f"Right Camera Calibration: RMS error = {self.right_camera_calib_results.rms_reprojection_error:.4f}")

        logger.info(f"Left Camera Calibration Matrix = {self.left_camera_calib_results.camera_matrix}")
        logger.info(f"Right Camera Calibration Matrix = {self.right_camera_calib_results.camera_matrix}")

    def log_calibration_summary(self, output_dir: Optional[Union[str, Path]] = None):
        """Print and save a structured summary of all calibration metrics.

        Args:
            output_dir: directory to save summary.txt and metrics.json into.
                If None, falls back to ``data_path.parent/calibration_logs`` (legacy behavior).
        """
        if "timestamp" not in self.metrics:
            self.metrics["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        m = self.metrics

        sep = "=" * 70
        lines = [
            sep,
            f"  CALIBRATION SUMMARY  —  {m['timestamp']}",
            sep,
            "",
            "--- IMAGE PROCESSING ---",
            f"  Total image pairs in dataset:        {m['image_processing']['total_image_pairs_in_dataset']}",
            f"  Skip factor:                         every {m['image_processing']['skip_factor']}th image",
            f"  Image pairs processed:               {m['image_processing']['image_pairs_processed']}",
            f"  Image pairs with charuco in both:    {m['image_processing']['image_pairs_both_detected']}",
            f"  Image pairs used for calibration:    {m['image_processing']['image_pairs_used_for_calibration']}",
            f"  Common points per pair (min/max/mean/std): "
            f"{m['image_processing']['common_pts_per_image']['min']} / "
            f"{m['image_processing']['common_pts_per_image']['max']} / "
            f"{m['image_processing']['common_pts_per_image']['mean']:.1f} / "
            f"{m['image_processing']['common_pts_per_image']['std']:.1f}",
            "",
            "--- INITIAL INTRINSIC CALIBRATION ---",
            f"  Images used: {m['initial_intrinsic_calibration']['num_images_used']}",
            "",
            "  LEFT CAMERA:",
            f"    RMS reprojection error:  {m['initial_intrinsic_calibration']['left']['rms_reprojection_error']:.4f}",
            f"    Per-image error (min/max/mean/std): "
            f"{m['initial_intrinsic_calibration']['left']['per_image_error_min']:.4f} / "
            f"{m['initial_intrinsic_calibration']['left']['per_image_error_max']:.4f} / "
            f"{m['initial_intrinsic_calibration']['left']['per_image_error_mean']:.4f} / "
            f"{m['initial_intrinsic_calibration']['left']['per_image_error_std']:.4f}",
            f"    Drift from initial (fx/fy/cx/cy): "
            f"{m['initial_intrinsic_calibration']['left']['drift_fx']:.2f} / "
            f"{m['initial_intrinsic_calibration']['left']['drift_fy']:.2f} / "
            f"{m['initial_intrinsic_calibration']['left']['drift_cx']:.2f} / "
            f"{m['initial_intrinsic_calibration']['left']['drift_cy']:.2f}",
            "",
            "  RIGHT CAMERA:",
            f"    RMS reprojection error:  {m['initial_intrinsic_calibration']['right']['rms_reprojection_error']:.4f}",
            f"    Per-image error (min/max/mean/std): "
            f"{m['initial_intrinsic_calibration']['right']['per_image_error_min']:.4f} / "
            f"{m['initial_intrinsic_calibration']['right']['per_image_error_max']:.4f} / "
            f"{m['initial_intrinsic_calibration']['right']['per_image_error_mean']:.4f} / "
            f"{m['initial_intrinsic_calibration']['right']['per_image_error_std']:.4f}",
            f"    Drift from initial (fx/fy/cx/cy): "
            f"{m['initial_intrinsic_calibration']['right']['drift_fx']:.2f} / "
            f"{m['initial_intrinsic_calibration']['right']['drift_fy']:.2f} / "
            f"{m['initial_intrinsic_calibration']['right']['drift_cx']:.2f} / "
            f"{m['initial_intrinsic_calibration']['right']['drift_cy']:.2f}",
            "",
            "--- RECALIBRATION ---",
            f"  Triggered: {m['recalibration']['triggered']}",
        ]

        if m['recalibration']['triggered']:
            lines += [
                f"  Images before / after:   {m['recalibration']['images_before']} / {m['recalibration']['images_after']}  (dropped {m['recalibration']['images_dropped']})",
                f"  Left RMS after:          {m['recalibration']['left_rms_after']:.4f}",
                f"  Right RMS after:         {m['recalibration']['right_rms_after']:.4f}",
            ]

        lines += [
            "",
            "--- STEREO CALIBRATION ---",
            f"  Image pairs used:          {m['stereo_calibration']['num_image_pairs_used']}",
            f"  Stereo RMS error:          {m['stereo_calibration']['stereo_rms_reprojection_error']:.4f}",
            f"  Baseline distance:         {m['stereo_calibration']['baseline_distance']:.4f} m",
            f"  Translation T:             {['%.4f' % v for v in m['stereo_calibration']['translation_vector']]}",
            f"  Rotation diag(R):          {['%.4f' % v for v in m['stereo_calibration']['rotation_diag']]}  (want ~1,1,1)",
            f"  Rotation off-diag max|.|:  {m['stereo_calibration']['rotation_offdiag_max_abs']:.4f}  (want small)",
            f"  Rotation R:                {m['stereo_calibration']['rotation_matrix']}",
            f"  Rect R_l:                  {m['stereo_calibration']['rect_l']}",
            f"  Rect R_r:                  {m['stereo_calibration']['rect_r']}",
            f"  Proj P_l:                  {m['stereo_calibration']['proj_matrix_l']}",
            f"  Proj P_r:                  {m['stereo_calibration']['proj_matrix_r']}",
            f"  Flags:                     {m['stereo_calibration']['flags']}",
            f"  ROI left:                  {m['stereo_calibration']['roi_left']}",
            f"  ROI right:                 {m['stereo_calibration']['roi_right']}",
            f"  stereoRectify alpha:       {m['stereo_calibration']['stereo_rectify_alpha']}",
            "",
            "--- RECTIFICATION VALIDATION ---",
            f"  Pairs tested:     {m['rectification_validation']['total_pairs_tested']}",
            f"  Passed / Failed:  {m['rectification_validation']['pairs_passed']} / {m['rectification_validation']['pairs_failed']}",
            f"  Pass rate:        {m['rectification_validation']['pass_rate']:.1%}",
            f"  Common IDs in passing (min/max/mean): "
            f"{m['rectification_validation']['common_ids_in_passing_pairs']['min']} / "
            f"{m['rectification_validation']['common_ids_in_passing_pairs']['max']} / "
            f"{m['rectification_validation']['common_ids_in_passing_pairs']['mean']:.1f}",
            "",
            "--- CONFIG ---",
            f"  Frame size:          {m['config']['frame_size']}",
            f"  Charuco board:       {m['config']['charuco_board']['squares_horizontally']}x{m['config']['charuco_board']['squares_vertically']}, "
            f"square={m['config']['charuco_board']['square_length']}m, marker={m['config']['charuco_board']['marker_length']}m",
            f"  Max allowable RMS:   {m['config']['max_allowable_rms_error']}",
            sep,
        ]

        summary = "\n".join(lines)
        print(summary)

        # Save to file
        if output_dir is None:
            log_dir = self.data_path.parent / "calibration_logs"
            timestamp = m["timestamp"]
            summary_path = log_dir / f"calibration_summary_{timestamp}.txt"
            json_path = log_dir / f"calibration_metrics_{timestamp}.json"
        else:
            log_dir = Path(output_dir)
            summary_path = log_dir / "summary.txt"
            json_path = log_dir / "metrics.json"

        log_dir.mkdir(exist_ok=True, parents=True)

        with open(summary_path, "w") as f:
            f.write(summary)

        with open(json_path, "w") as f:
            json.dump(m, f, indent=2)

        logger.info(f"Calibration summary saved to {summary_path}")
        logger.info(f"Calibration metrics saved to {json_path}")
