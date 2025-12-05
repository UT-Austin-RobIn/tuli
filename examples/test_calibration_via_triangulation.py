import cv2
import numpy as np
import matplotlib.pyplot as plt

from stereo_calib.charuco import CharucoBoard, CharucoBoardData
from stereo_calib.charuco import CharucoConfig as C


# -------------------------
# Load stereo calibration parameters from JSON file
# -------------------------
import json

def load_calibration_from_json(json_file_path):
    """Load K, R, T matrices from JSON calibration file"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract left camera matrix (K1)
    K1 = np.array(data['left_camera_calibration_data']['camera_matrix'])
    
    # Extract right camera matrix (K2) - assuming similar structure
    K2 = np.array(data['right_camera_calibration_data']['camera_matrix'])
    
    # Extract distortion coefficients
    dist1 = np.array(data['left_camera_calibration_data']['dist_coeffs']).flatten()
    dist2 = np.array(data['right_camera_calibration_data']['dist_coeffs']).flatten()
    
    # Extract stereo calibration parameters (R and T)
    R = np.array(data['rot'])
    T = np.array(data['trans'])
    
    return K1, K2, dist1, dist2, R, T

# Load calibration parameters
K1, K2, dist1, dist2, R, T = load_calibration_from_json("results/calibration_results.json")

# Using K1 from realsense camera (905.2623291015625, 0.0, 651.2127075195312, 0.0, 904.84765625, 377.03375244140625, 0.0, 0.0, 1.0)
# K1 = np.array([[905.2623291015625, 0.0, 651.2127075195312], [0.0, 904.84765625, 377.03375244140625], [0.0, 0.0, 1.0]])

# # Adversarial testing
# # add some noise along z axis of T
# T[1] += 0.01
# T[0] = -1*T[0]
# breakpoint()

# # Add noise to rotation matrix using axis-angle format
# # Choose which axis to add noise to (x, y, or z)
# axis_to_noise = 'x'  # Change to 'y' or 'z' to add noise to different axes
# noise_magnitude = 0.1  # noise in radians

# # Create axis-angle vector for the chosen axis
# if axis_to_noise == 'x':
#     axis_angle = np.array([noise_magnitude, 0, 0])
# elif axis_to_noise == 'y':
#     axis_angle = np.array([0, noise_magnitude, 0])
# elif axis_to_noise == 'z':
#     axis_angle = np.array([0, 0, noise_magnitude])

# # Convert axis-angle to rotation matrix using Rodrigues formula
# R_noise, _ = cv2.Rodrigues(axis_angle)

# # Apply noise to the rotation matrix
# R = R @ R_noise

# print(f"Added rotation noise along {axis_to_noise}-axis: {noise_magnitude:.4f} radians")


# Projection matrices
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))
P2 = K2 @ np.hstack((R, T))

# -------------------------
# Read two images
# -------------------------
img1 = cv2.imread("/home/robotlearning2/stereo-calib/dataset/left/img_0850.jpg")
img2 = cv2.imread("/home/robotlearning2/stereo-calib/dataset/right/img_0850.jpg")

# -------------------------
# Detect ChArUco corners
# -------------------------
charuco_board = CharucoBoard(charuco_data=CharucoBoardData(aruco_dict=C.ARUCO_DICT,
                                                               squares_vertically=C.SQUARES_VERTICALLY,
                                                               squares_horizontally=C.SQUARES_HORIZONTALLY,
                                                               square_length=C.SQUARE_LENGTH,
                                                               marker_length=C.MARKER_LENGTH))

aruco_params = cv2.aruco.DetectorParameters()

def detect_charuco(img, K, dist):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, charuco_board.aruco_dict, parameters=aruco_params)
    if len(corners) > 0:
        # cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=[])
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            # corners, ids, gray, charuco_board.board, cameraMatrix=K, distCoeffs=dist
            corners, ids, gray, charuco_board.board
        )
        return charuco_corners, charuco_ids
    return None, None

charuco_corners1, ids1 = detect_charuco(img1, K1, dist1)
charuco_corners2, ids2 = detect_charuco(img2, K2, dist2)

# # visualize the charuco corners
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# cv2.aruco.drawDetectedCornersCharuco(img1, charuco_corners1, ids1, (0, 0, 255))
# cv2.aruco.drawDetectedCornersCharuco(img2, charuco_corners2, ids2, (0, 0, 255))
# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(img1)
# axs[1].imshow(img2)
# plt.show()


# -------------------------
# Match corners between both images (by their IDs)
# -------------------------
matched_pts1, matched_pts2 = [], []
if ids1 is not None and ids2 is not None:
    for i, id1 in enumerate(ids1.flatten()):
        matches = np.where(ids2.flatten() == id1)[0]
        if len(matches) > 0:
            matched_pts1.append(charuco_corners1[i][0])  # (x, y)
            matched_pts2.append(charuco_corners2[matches[0]][0])

matched_pts1 = np.array(matched_pts1).T  # shape (2, N)
matched_pts2 = np.array(matched_pts2).T  # shape (2, N)

# -------------------------
# Triangulation
# -------------------------
pts4D = cv2.triangulatePoints(P1, P2, matched_pts1, matched_pts2)
pts3D = (pts4D[:3] / pts4D[3]).T  # Nx3 points in cam1 frame

# -------------------------
# Reprojection check
# -------------------------
def project_points(P, pts3D):
    pts_h = np.hstack([pts3D, np.ones((pts3D.shape[0],1))]).T
    proj = P @ pts_h
    proj = proj[:2] / proj[2]
    return proj.T

proj1 = project_points(P1, pts3D)
proj2 = project_points(P2, pts3D)

# -------------------------
# Draw results
# -------------------------
for (u,v), (u_proj, v_proj) in zip(matched_pts1.T, proj1):
    cv2.circle(img1, (int(u), int(v)), 4, (0,255,0), -1)     # original (green)
    cv2.circle(img1, (int(u_proj), int(v_proj)), 4, (0,0,255), -1) # reprojected (red)

for (u,v), (u_proj, v_proj) in zip(matched_pts2.T, proj2):
    cv2.circle(img2, (int(u), int(v)), 4, (0,255,0), -1)
    cv2.circle(img2, (int(u_proj), int(v_proj)), 4, (0,0,255), -1)

cv2.imshow("Cam1 reprojection", img1)
cv2.imshow("Cam2 reprojection", img2)
cv2.waitKey(0)
