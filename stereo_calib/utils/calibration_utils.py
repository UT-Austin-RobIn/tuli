import json
import cv2
import numpy as np
from pathlib import Path
from copy import deepcopy
from json import JSONEncoder
from loguru import logger
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config
from typing import List, Tuple, Union, Optional
# import open3d as o3d

@dataclass_json
@dataclass
class CameraCalibrationData:
    """
    A data class to hold camera calibration information.

    Attributes:
        rms_reprojection_error (float): The root mean square (RMS) reprojection error.
        camera_matrix (np.ndarray): The camera matrix.
        dist_coeffs (np.ndarray): The distortion coefficients.
        rotation_vectors (List[np.ndarray]): List of rotation vectors for each calibration image.
        translation_vectors (List[np.ndarray]): List of translation vectors for each calibration image.
        image_width (int): The width of the calibration image.
        image_height (int): The height of the calibration image.
        image_dim (Tuple[int, int]): The dimensions of the calibration image [width, height].
        undistort_map_x (Optional[np.ndarray]): Optional map for undistortion in x-direction.
        undistort_map_y (Optional[np.ndarray]): Optional map for undistortion in y-direction.
        stereo_rectify_map_x (Optional[np.ndarray]): Optional map for stereo rectification in x-direction.
        stereo_rectify_map_y (Optional[np.ndarray]): Optional map for stereo rectification in y-direction.
    """
    rms_reprojection_error: float
    camera_matrix: np.ndarray = field(metadata=config(decoder=np.asarray))
    dist_coeffs: np.ndarray = field(metadata=config(decoder=np.asarray))
    rotation_vectors: List[np.ndarray] = field(metadata=config(decoder=lambda l: [np.asarray(item) for item in l]))
    translation_vectors: List[np.ndarray] = field(metadata=config(decoder=lambda l: [np.asarray(item) for item in l]))
    image_width: int
    image_height: int
    image_dim: Tuple[int, int]
    undistort_map_x: Optional[np.ndarray] = None
    undistort_map_y: Optional[np.ndarray] = None
    stereo_rectify_map_x: Optional[np.ndarray] = None
    stereo_rectify_map_y: Optional[np.ndarray] = None


# Define data class for stereo calibration results
@dataclass_json
@dataclass
class StereoCalibrationData:
    """
    A data class to hold stereo calibration information.

    Attributes:
        rms_stereo_reprojection_error (float): The root mean square (RMS) stereo reprojection error.
        left_camera_calibration_data (CameraCalibrationData): Calibration data for the left camera.
        right_camera_calibration_data (CameraCalibrationData): Calibration data for the right camera.
        projection_matrix_left (np.ndarray): Projection matrix for the left camera.
        projection_matrix_right (np.ndarray): Projection matrix for the right camera.
        left_camera_rectification_transform (np.ndarray): Rectification transform for the left camera.
        right_camera_rectification_transform (np.ndarray): Rectification transform for the right camera.
        rot (np.ndarray): The rotation matrix between the two cameras.
        trans (np.ndarray): The translation vector between the two cameras.
        essential_matrix (np.ndarray): The essential matrix.
        fundamental_matrix (np.ndarray): The fundamental matrix.
        perspective_transformation_matrix_Q (np.ndarray): Perspective transformation matrix.
    """
    rms_stereo_reprojection_error: float
    left_camera_calibration_data: CameraCalibrationData
    right_camera_calibration_data: CameraCalibrationData
    projection_matrix_left: np.ndarray = field(metadata=config(decoder=np.asarray))
    projection_matrix_right: np.ndarray = field(metadata=config(decoder=np.asarray))
    left_camera_rectification_transform: np.ndarray = field(metadata=config(decoder=np.asarray))
    right_camera_rectification_transform: np.ndarray = field(metadata=config(decoder=np.asarray))
    rot: np.ndarray = field(metadata=config(decoder=np.asarray))
    trans: np.ndarray = field(metadata=config(decoder=np.asarray))
    essential_matrix: np.ndarray = field(metadata=config(decoder=np.asarray))
    fundamental_matrix: np.ndarray = field(metadata=config(decoder=np.asarray))
    perspective_transformation_matrix_Q: np.ndarray = field(metadata=config(decoder=np.asarray))

    # add a transform from right to left camera
    transform_r_to_l: Optional[np.ndarray] = field(metadata=config(decoder=np.asarray))
    left_cam_to_base_transform: Optional[np.ndarray] = field(metadata=config(decoder=np.asarray))
    right_cam_to_base_transform: Optional[np.ndarray] = field(metadata=config(decoder=np.asarray))
    left_cam_serial_id: Optional[str] = field(metadata=config(decoder=str))
    right_cam_serial_id: Optional[str] = field(metadata=config(decoder=str))
    robot_joint_vals: Optional[np.ndarray] = field(metadata=config(decoder=np.asarray))

class CustomJSONEncoder(JSONEncoder):
    """
    A custom JSON encoder to handle special data types like numpy arrays, lists, and tuples.
    Overrides the default() method of JSONEncoder to handle custom serialization.
    Methods:
        default(obj): Override of the default() method to handle custom serialization.
    Attributes:
        None
    """

    def default(self, obj):
        """
        Override the default() method of JSONEncoder to handle custom serialization.
        Args:
            obj: The object to be serialized.
        Returns:
            Serialized JSON-compatible representation of the object.
        Raises:
            None
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [self.default(item) for item in obj]
        if isinstance(obj, tuple):
            return list(obj)
        # If not a special type, use default serialization
        return JSONEncoder.default(self, obj)


def save_calibration_data(data: StereoCalibrationData, file_path: Union[str, Path]) -> None:
    """
    Save stereo calibration data to a JSON file.
    Args:
        data (StereoCalibrationData): Stereo calibration data to be saved.
        file_path (Union[str, Path]): File path where the calibration data will be saved.

    Returns:
        None
    """
    logger.info("Saving calibration results ...")

    data_to_save = deepcopy(data)

    # setting undistort and rectify map to None and only save useful calib data
    for camera_data in [data_to_save.left_camera_calibration_data, data_to_save.right_camera_calibration_data]:
        camera_data: CameraCalibrationData
        camera_data.undistort_map_x = None
        camera_data.undistort_map_y = None
        camera_data.stereo_rectify_map_x = None
        camera_data.stereo_rectify_map_y = None

    data_dict = data_to_save.to_dict()

    if isinstance(file_path, str):
        file_path = Path(file_path)

    file_path.mkdir(exist_ok=True, parents=True)
    with open(str(file_path.joinpath("calibration_results.json")), 'w') as file:
        json.dump(data_dict, file, cls=CustomJSONEncoder, indent=4)
    logger.success("Calibration results successfully saved!")


def load_stereo_rectify_maps(data: StereoCalibrationData) -> StereoCalibrationData:
    """
    Load stereo rectify maps and update the StereoCalibrationData object.

    This function loads stereo rectify maps for each camera and updates the given StereoCalibrationData
    object with the computed rectify maps.

    Args:
        data (StereoCalibrationData): Stereo calibration data containing camera calibration information.

    Returns:
        StereoCalibrationData: Updated stereo calibration data object with stereo rectify maps.
    """
    for camera_data, rect, projection_mat in zip(
            [data.left_camera_calibration_data, data.right_camera_calibration_data],
            [data.left_camera_rectification_transform, data.right_camera_rectification_transform],
            [data.projection_matrix_left, data.projection_matrix_right]):
        camera_data: CameraCalibrationData
        rect: np.ndarray
        projection_mat: np.ndarray

        stereo_rectify_map_x, stereo_rectify_map_y = cv2.initUndistortRectifyMap(cameraMatrix=camera_data.camera_matrix,
                                                                                 distCoeffs=camera_data.dist_coeffs,
                                                                                 R=rect,
                                                                                 newCameraMatrix=projection_mat,
                                                                                 size=camera_data.image_dim,
                                                                                 m1type=cv2.CV_16SC2)

        camera_data.stereo_rectify_map_x = stereo_rectify_map_x
        camera_data.stereo_rectify_map_y = stereo_rectify_map_y

    return data


def load_calibration_data(file_path: Union[str, Path]) -> StereoCalibrationData:
    """
    Load stereo calibration data from a JSON file.

    This function loads stereo calibration data from a JSON file at the specified file path,
    and returns the corresponding StereoCalibrationData object.

    Args:
        file_path (Union[str, Path]): File path to the JSON file containing stereo calibration data.

    Returns:
        StereoCalibrationData: Stereo calibration data loaded from the JSON file.

    Raises:
        FileNotFoundError: If the specified file path does not exist or is not a file.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if file_path.is_file() and file_path.exists():
        with open(file_path, 'r') as file:
            data = json.load(file)
        stereo_calib_data = StereoCalibrationData.from_dict(data)
        stereo_calib_data = load_stereo_rectify_maps(stereo_calib_data)
        return stereo_calib_data
    else:
        raise FileNotFoundError(f"File '{str(file_path)}' not found or is not a file.")


class SimpleStereoCalibFromJSON:
    """
    A simplified wrapper class for stereo calibration data loaded from JSON.
    This class provides easy access to the most commonly used calibration parameters.
    """

    def __init__(self, json_file_path: Union[str, Path]):
        """
        Initialize the class by loading calibration data from a JSON file.

        Args:
            json_file_path (Union[str, Path]): Path to the calibration results JSON file
        """
        if isinstance(json_file_path, str):
            json_file_path = Path(json_file_path)

        if not json_file_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {json_file_path}")

        # Load the full calibration data
        self._full_calib_data = load_calibration_data(json_file_path)

        # Extract commonly used parameters
        self._extract_parameters()

    def _extract_parameters(self):
        """Extract commonly used parameters from the full calibration data."""
        # Camera matrices
        self.left_camera_matrix = self._full_calib_data.left_camera_calibration_data.camera_matrix
        self.right_camera_matrix = self._full_calib_data.right_camera_calibration_data.camera_matrix

        # Distortion coefficients
        self.left_dist_coeffs = self._full_calib_data.left_camera_calibration_data.dist_coeffs
        self.right_dist_coeffs = self._full_calib_data.right_camera_calibration_data.dist_coeffs

        # Stereo extrinsics
        self.rot = self._full_calib_data.rot
        self.trans = self._full_calib_data.trans

        # Rectification transforms
        self.left_rect_transform = self._full_calib_data.left_camera_rectification_transform
        self.right_rect_transform = self._full_calib_data.right_camera_rectification_transform

        # Projection matrices
        self.projection_matrix_left = self._full_calib_data.projection_matrix_left
        self.projection_matrix_right = self._full_calib_data.projection_matrix_right

        # Essential and fundamental matrices
        self.essential_matrix = self._full_calib_data.essential_matrix
        self.fundamental_matrix = self._full_calib_data.fundamental_matrix

        # Perspective transformation matrix
        self.perspective_transform_Q = self._full_calib_data.perspective_transformation_matrix_Q

        # Additional robot-specific transforms (if available)
        self.transform_r_to_l = getattr(self._full_calib_data, 'transform_r_to_l', None)
        self.left_cam_to_base_transform = getattr(self._full_calib_data, 'left_cam_to_base_transform', None)
        self.right_cam_to_base_transform = getattr(self._full_calib_data, 'right_cam_to_base_transform', None)
        self.left_cam_serial_id = getattr(self._full_calib_data, 'left_cam_serial_id', None)
        self.right_cam_serial_id = getattr(self._full_calib_data, 'right_cam_serial_id', None)
        self.robot_joint_vals = getattr(self._full_calib_data, 'robot_joint_vals', None)

    @property
    def left_camera_to_base(self):
        """Alias for left_cam_to_base_transform for backward compatibility."""
        return self.left_cam_to_base_transform

    def get_camera_matrix(self, camera_side: str = 'left') -> np.ndarray:
        """
        Get camera matrix for the specified camera side.

        Args:
            camera_side (str): Either 'left' or 'right'

        Returns:
            np.ndarray: Camera matrix
        """
        if camera_side.lower() == 'left':
            return self.left_camera_matrix
        elif camera_side.lower() == 'right':
            return self.right_camera_matrix
        else:
            raise ValueError("camera_side must be 'left' or 'right'")

    def get_dist_coeffs(self, camera_side: str = 'left') -> np.ndarray:
        """
        Get distortion coefficients for the specified camera side.

        Args:
            camera_side (str): Either 'left' or 'right'

        Returns:
            np.ndarray: Distortion coefficients
        """
        if camera_side.lower() == 'left':
            return self.left_dist_coeffs
        elif camera_side.lower() == 'right':
            return self.right_dist_coeffs
        else:
            raise ValueError("camera_side must be 'left' or 'right'")

    def get_image_dim(self, camera_side: str = 'left') -> Tuple[int, int]:
        """
        Get image dimensions for the specified camera side.

        Args:
            camera_side (str): Either 'left' or 'right'

        Returns:
            Tuple[int, int]: Image dimensions (width, height)
        """
        if camera_side.lower() == 'left':
            return self.left_image_dim
        elif camera_side.lower() == 'right':
            return self.right_image_dim
        else:
            raise ValueError("camera_side must be 'left' or 'right'")

    def __repr__(self):
        return f"SimpleStereoCalibFromJSON(left_cam={self.left_cam_serial_id}, right_cam={self.right_cam_serial_id})"


def fuse_stereo_pointclouds_live(
    stereo_calib: 'SimpleStereoCalibFromJSON',
    left_rgb_path: Union[str, Path],
    left_depth_path: Union[str, Path],
    right_rgb_path: Union[str, Path],
    right_depth_path: Union[str, Path],
    extra_transform: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse stereo point clouds from RGB and depth images using calibration data.
    
    This function loads RGB and depth images from both cameras, creates point clouds,
    transforms the right camera points to the left camera coordinate system, and fuses them.
    
    Args:
        stereo_calib (SimpleStereoCalibFromJSON): Stereo calibration data
        left_rgb_path (Union[str, Path]): Path to left RGB image
        left_depth_path (Union[str, Path]): Path to left depth image (.npy file)
        right_rgb_path (Union[str, Path]): Path to right RGB image
        right_depth_path (Union[str, Path]): Path to right depth image (.npy file)
        extra_transform (Optional[np.ndarray]): Optional 4x4 transformation matrix to apply
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (points, colors) where points is (N, 3) and colors is (N, 3)
    
    Raises:
        FileNotFoundError: If any of the image files don't exist
        ValueError: If depth images have invalid dimensions or values
    """
    import cv2
    import open3d as o3d
    
    # Convert paths to Path objects
    left_rgb_path = Path(left_rgb_path)
    left_depth_path = Path(left_depth_path) if left_depth_path else None
    right_rgb_path = Path(right_rgb_path)
    right_depth_path = Path(right_depth_path) if right_depth_path else None
    
    # Check if files exist
    if not left_rgb_path.exists():
        raise FileNotFoundError(f"Left RGB image not found: {left_rgb_path}")
    if not right_rgb_path.exists():
        raise FileNotFoundError(f"Right RGB image not found: {right_rgb_path}")
    
    # Load RGB images
    left_rgb = cv2.imread(str(left_rgb_path))
    right_rgb = cv2.imread(str(right_rgb_path))
    
    if left_rgb is None:
        raise ValueError(f"Failed to load left RGB image: {left_rgb_path}")
    if right_rgb is None:
        raise ValueError(f"Failed to load right RGB image: {right_rgb_path}")
    
    # Convert BGR to RGB
    left_rgb = cv2.cvtColor(left_rgb, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_rgb, cv2.COLOR_BGR2RGB)
    
    # Load depth images if available
    left_depth = None
    right_depth = None
    
    if left_depth_path and left_depth_path.exists():
        left_depth = np.load(str(left_depth_path))
        if left_depth.ndim == 3 and left_depth.shape[2] == 1:
            left_depth = left_depth.squeeze()
    
    if right_depth_path and right_depth_path.exists():
        right_depth = np.load(str(right_depth_path))
        if right_depth.ndim == 3 and right_depth.shape[2] == 1:
            right_depth = right_depth.squeeze()
    
    # Create point clouds from depth images
    left_points = None
    left_colors = None
    right_points = None
    right_colors = None
    
    # Process left camera
    if left_depth is not None:
        left_pcd = _pcd_from_depth(left_depth, stereo_calib.left_camera_matrix)
        if left_pcd is not None:
            left_points = np.asarray(left_pcd.points)
            left_colors = left_rgb.reshape(-1, 3)
    
    # Process right camera
    if right_depth is not None:
        right_pcd = _pcd_from_depth(right_depth, stereo_calib.right_camera_matrix)
        if right_pcd is not None:
            right_points = np.asarray(right_pcd.points)
            right_colors = right_rgb.reshape(-1, 3)
            
            # Transform right camera points to left camera coordinate system
            if len(right_points) > 0:
                # Apply the transform from right to left camera
                if stereo_calib.transform_r_to_l is not None:
                    # Add homogeneous coordinate
                    right_points_homo = np.concatenate((right_points, np.ones((right_points.shape[0], 1))), axis=-1)
                    # Apply transformation
                    right_points_transformed = (stereo_calib.transform_r_to_l @ right_points_homo.T).T
                    right_points = right_points_transformed[:, :3]
    
    # Combine point clouds
    all_points = []
    all_colors = []
    
    if left_points is not None and len(left_points) > 0:
        all_points.append(left_points)
        all_colors.append(left_colors)
    
    if right_points is not None and len(right_points) > 0:
        all_points.append(right_points)
        all_colors.append(right_colors)
    
    if not all_points:
        logger.warning("No valid points found in either camera")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    # Concatenate all points and colors
    fused_points = np.concatenate(all_points, axis=0)
    fused_colors = np.concatenate(all_colors, axis=0)
    
    if extra_transform is not None:
        fused_points = (extra_transform @ np.concatenate((fused_points, np.ones((fused_points.shape[0], 1))), axis=-1).T).T
        fused_points = fused_points[:, :3]
    
    return fused_points, fused_colors


def _pcd_from_depth(
    depth,
    intrinsic_matrix,
    depth_trunc=5,
    depth_scale=1000,
):
    """
    Copy of the pcd_from_depth function from grx.utils.misc_utils
    """
    width, height = depth.shape[:2]

    pinholecameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height,
        intrinsic_matrix=intrinsic_matrix
    )

    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth.astype(np.uint16)),
        pinholecameraIntrinsic,
        depth_trunc=depth_trunc,
        depth_scale=depth_scale,
        project_valid_depth_only=False,
    )
    return pcd