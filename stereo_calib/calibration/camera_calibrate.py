import logging

import cv2
import glob
import numpy as np
from tqdm import tqdm
from termcolor import colored
from pathlib import Path
from loguru import logger
from typing import Union, Optional, List, Tuple

from stereo_calib.charuco import CharucoBoard
from stereo_calib.utils import CameraCalibrationData, StereoCalibrationData


class StereoCalibration:
    """
    Class for performing stereo calibration using Charuco boards.

    Args:
        data_path (Union[str, Path]): Path to the directory containing left and right camera images.
        charuco_board (CharucoBoard): Charuco board object used for calibration.

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

    def __init__(self, data_path: Union[str, Path], charuco_board: CharucoBoard):
        """
         Initialize StereoCalibration object.

         Args:
             data_path (Union[str, Path]): Path to the directory containing left and right camera images.
             charuco_board (CharucoBoard): Charuco board object used for calibration.
         """
        self.data_path = Path(data_path) if isinstance(data_path, str) else data_path
        self.charuco_board = charuco_board

        self.left_images_path = self.load_images(self.data_path.joinpath("left", "*.jpg"))
        self.right_images_path = self.load_images(self.data_path.joinpath("right", "*.jpg"))
        assert len(self.left_images_path) == len(self.right_images_path)

        # List to store stereo object points, charuco points and charuco ids from all the images.
        self.stereo_obj_points: List[np.ndarray] = []  # 3d point in real world space
        self.stereo_charuco_points_l: List[np.ndarray] = []  # stereo charuco points in left image plane.
        self.stereo_charuco_ids_l: List[np.ndarray] = []  # stereo charuco ids in left image
        self.stereo_charuco_points_r: List[np.ndarray] = []  # stereo charuco points in rightimage plane.
        self.stereo_charuco_ids_r: List[np.ndarray] = []  # stereo charuco ids in right image

        self.frame_size: Optional[Tuple[int, int]] = None
        self._min_points: int = 5

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-5)
        self.stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 70, 1e-6)

        self.left_camera_calib_results: Optional[CameraCalibrationData] = None
        self.right_camera_calib_results: Optional[CameraCalibrationData] = None
        self.best_calib_images_indices: List[int] = []
        self.recalibrate: bool = False
        self.max_allowable_rms_error = 0.4
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
        images = sorted(glob.glob(str(directory_path)), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return images

    def init_camera_matrix(self):
        """
        Initialize camera matrix.

        Returns:
            np.ndarray: Initial camera matrix.
        """
        fx, fy = 570.0, 570.0
        height, width = self.frame_size
        cx, cy = width / 2.0, height / 2.0

        # initial_camera_matrix = np.array([
        #     [fx, 0, cx],
        #     [0, fy, cy],
        #     [0, 0, 1]
        # ], dtype=np.float32)

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
        self.imgs_l, self.imgs_r = [], []
        for img_left_path, img_right_path in tqdm(zip(self.left_images_path, self.right_images_path),
                                                  total=len(self.left_images_path),
                                                  desc="Processing calibration images"):
            counter += 1
            if counter % 5 != 0:
                continue            

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
                assert self.frame_size == gray_l.shape == gray_r.shape
                # pass

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

            if ids_l is not None and ids_r is not None:
                retval_l, charuco_corners_l, charuco_ids_l = cv2.aruco.interpolateCornersCharuco(corners_l, ids_l,
                                                                                                 gray_l,
                                                                                                 self.charuco_board.board)
                retval_r, charuco_corners_r, charuco_ids_r = cv2.aruco.interpolateCornersCharuco(corners_r, ids_r,
                                                                                                 gray_r,
                                                                                                 self.charuco_board.board)
                
                # print("--", retval_l, retval_r)
                if retval_l > 0:
                    print("Number of interpolations: ", retval_l)
                    left_interp_counter += 1

                if retval_r > 0:
                    print("Number of interpolations: ", retval_r)
                    right_interp_counter += 1


                if charuco_corners_l is None or charuco_corners_r is None:
                    continue
                
                if retval_l > self._min_points and retval_r > self._min_points:
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
                        breakpoint()
                    x = img_pts_r[:, 0, 0]
                    y = img_pts_r[:, 0, 1]
                    if np.any(x < 0) or np.any(x > width) or np.any(y < 0) or np.any(y > height):
                        print(f"Right Image {counter} has corners out of bounds!")
                        breakpoint()


                    pts_l = {tuple(a): tuple(b) for a, b in zip(obj_pts_l[:, 0], img_pts_l[:, 0])}
                    pts_r = {tuple(a): tuple(b) for a, b in zip(obj_pts_r[:, 0], img_pts_r[:, 0])}
                    ids_l = {tuple(a): b for a, b in zip(obj_pts_l[:, 0], charuco_ids_l[:, 0])}
                    ids_r = {tuple(a): b for a, b in zip(obj_pts_r[:, 0], charuco_ids_r[:, 0])}
                    common_pts = set(pts_l.keys()) & set(pts_r.keys())
                    print("common_pts: ", common_pts)

                    # has_gt_one = any(val > 1.0 for point in common_pts for val in point)
                    # if has_gt_one:
                    #     print(has_gt_one)
                    #     breakpoint()
                    
                    # Added by Arpit
                    print(len(pts_l), len(pts_r), len(common_pts))
                    if len(common_pts) < self._min_points:
                        continue

                    obj = np.zeros((len(common_pts), 1, 3), dtype=np.float32)
                    left_corners = np.zeros((len(common_pts), 1, 2), dtype=np.float32)
                    right_corners = np.zeros((len(common_pts), 1, 2), dtype=np.float32)
                    left_corner_ids = np.zeros((len(common_pts), 1), dtype=np.int32)
                    right_corner_ids = np.zeros((len(common_pts), 1), dtype=np.int32)
                    for i, pts in enumerate(common_pts):
                        obj[i] = pts
                        left_corners[i] = np.reshape(pts_l[pts], (1, 2))
                        right_corners[i] = np.reshape(pts_r[pts], (1, 2))
                        # print("left_corenrs, right_coreners: ", left_corners)
                        left_corner_ids[i] = ids_l[pts]
                        right_corner_ids[i] = ids_r[pts]

                    self.stereo_obj_points.append(obj)
                    self.stereo_charuco_points_l.append(left_corners)
                    self.stereo_charuco_points_r.append(right_corners)
                    self.stereo_charuco_ids_l.append(left_corner_ids)
                    self.stereo_charuco_ids_r.append(right_corner_ids)
                    self.imgs_l.append(img_l)
                    self.imgs_r.append(img_r)
                    # breakpoint()

        assert len(self.stereo_obj_points) == len(self.stereo_charuco_points_l) == len(
            self.stereo_charuco_points_r) == len(
            self.stereo_charuco_ids_l) == len(self.stereo_charuco_ids_r)

        print("Number of filtered images (will be used for stereo calibration): ", len(self.stereo_obj_points))
        # breakpoint()

        init_camera_matrix = self.init_camera_matrix()

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
        # breakpoint()

        # visualize                 
        # objpoints_list and imgpoints_list should be the same lists used for calibrateCameraCharuco
        # camera_matrix, dist_coeffs are results from calibration
        
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
        # breakpoint()


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
                                                                       cameraMatrix1=self.left_camera_calib_results.camera_matrix,
                                                                       distCoeffs1=self.left_camera_calib_results.dist_coeffs,
                                                                       cameraMatrix2=self.right_camera_calib_results.camera_matrix,
                                                                       distCoeffs2=self.right_camera_calib_results.dist_coeffs,
                                                                       imageSize=self.frame_size,
                                                                       criteria=self.stereo_criteria,
                                                                       flags=flags)

        np.testing.assert_array_equal(new_camera_matrix_l, self.left_camera_calib_results.camera_matrix)
        np.testing.assert_array_equal(new_dist_coeffs_l, self.left_camera_calib_results.dist_coeffs)

        np.testing.assert_array_equal(new_camera_matrix_r, self.right_camera_calib_results.camera_matrix)
        np.testing.assert_array_equal(new_dist_coeffs_r, self.right_camera_calib_results.dist_coeffs)

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
            alpha=-0.7)

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
        for i in range(len(self.imgs_l)):
            left_img = self.imgs_l[i]
            right_img = self.imgs_r[i]
            img_size = left_img.shape

            params = cv2.aruco.DetectorParameters()
            aruco_detector = cv2.aruco.ArucoDetector(self.charuco_board.aruco_dict, params)
            
            def detect_charuco(img):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = aruco_detector.detectMarkers(gray)
                charuco_ids = None
                if ids is not None and len(ids) > 0:
                    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.charuco_board.board)
                    if charuco_corners is not None and len(charuco_corners) > 0:
                        cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids, (0, 0, 255))
                return img, charuco_ids

            left_rect = cv2.remap(left_img, stereo_rectify_map_l_x, stereo_rectify_map_l_y, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_img, stereo_rectify_map_r_x, stereo_rectify_map_r_y, cv2.INTER_LINEAR)

            # # Compute valid overlap region (intersection of ROI_l and ROI_r)
            # x1 = max(roi_l[0], roi_r[0])
            # y1 = max(roi_l[1], roi_r[1])
            # x2 = min(roi_l[0] + roi_l[2], roi_r[0] + roi_r[2])
            # y2 = min(roi_l[1] + roi_l[3], roi_r[1] + roi_r[3])

            # # Crop both images to valid region
            # left_rect  = left_rect[y1:y2, x1:x2]
            # right_rect = right_rect[y1:y2, x1:x2]

            left_anno, left_charuco_ids = detect_charuco(left_rect.copy())
            right_anno, right_charuco_ids = detect_charuco(right_rect.copy())
            if left_charuco_ids is not None and right_charuco_ids is not None:
                common_ids = np.intersect1d(left_charuco_ids, right_charuco_ids)
                print("i, len(common_ids): ", i, len(common_ids))
            else:
                print(f"{i}, None")

            # # Step 3: Detect checkerboard corners
            # swapped_l = self.stereo_charuco_points_l[0][:, :, ::-1] 
            # swapped_r = self.stereo_charuco_points_r[0][:, :, ::-1]
            # cv2.aruco.drawDetectedCornersCharuco(left_rect, swapped_l, swapped_r, (0, 0, 255))
            # cv2.aruco.drawDetectedCornersCharuco(right_rect, self.stereo_charuco_points_r[0], self.stereo_charuco_ids_r[0], (0, 0, 255))

            # import matplotlib.pyplot as plt
            # # breakpoint()

            # # Draw epipolar lines on combined image
            # combined = np.hstack((left_anno, right_anno))
            # for y in range(0, img_size[0], 50):
            #     cv2.line(combined, (0, y), (2 * img_size[1], y), (0, 255, 0), 1)

            # cv2.imshow("Rectified Stereo with Checkerboard + Epipolar Lines", combined)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        # ===================================

        logger.success("Calibration complete!")
        logger.info(f"Stereo Calibration result: RMS error = {ret_stereo:.4f}")
        self.log_calib_info()

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
