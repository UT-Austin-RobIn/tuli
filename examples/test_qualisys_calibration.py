import numpy as np
import cv2

def overlay_point_on_image(image, pixel, color=(0, 0, 255), radius=12, thickness=-1):
    """
    Draw a point on the image at given pixel coordinates.

    Args:
        image (np.ndarray): BGR image
        pixel (tuple or np.ndarray): (u, v) pixel coordinates
        color (tuple): BGR color
        radius (int): circle radius
        thickness (int): -1 fills the circle
    """
    img_out = image.copy()
    u, v = int(pixel[0]), int(pixel[1])
    cv2.circle(img_out, (u, v), radius, color, thickness)
    return img_out

def project_point_to_image(point_3d, K, dist_coeffs=None):
    """
    Project a 3D point (in camera frame) to 2D pixel coordinates.

    Args:
        point_3d (np.ndarray): shape (3,) or (N,3) -> 3D points in camera frame (X, Y, Z)
        K (np.ndarray): 3x3 intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        dist_coeffs (np.ndarray): optional, distortion coefficients (k1, k2, p1, p2, k3)

    Returns:
        np.ndarray: 2D pixel coordinates (u, v)
    """
    # Ensure correct shape
    points_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 1, 3)
    
    # No rotation/translation since point is already in camera frame
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    
    # breakpoint()
    # Use OpenCV’s projection function
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist_coeffs)
    return points_2d.reshape(-1, 2)


# Compute the intrinsic matrix in pixel units
scale_u = 1919 / 122879
scale_v = 1087 / 69631

fx_px = 107605.140625 * scale_u
fy_px = 107598.804688 * scale_v
cx_px = 62540.17578 * scale_u
cy_px = 34756.410156 * scale_v

# # scale the K to another resolution
# sx = 1280 / 1920
# sy = 768 / 1088
# fx_px = fx_px * sx
# fy_px = fy_px * sy
# cx_px = cx_px * sx
# cy_px = cy_px * sy

K = np.array([
    [fx_px, 0, cx_px],
    [0, fy_px, cy_px],
    [0, 0, 1]
])
print("K in pixel units:\n", K)

dist_coeffs = np.array([-0.046081, 0.134643, 0.000513, 0.000115, 0.0])

# Load your mocap RGB image
img = cv2.imread("/home/robotlearning2/stereo-calib/test_calib_2_qualisys/0000.jpg")

# Obtain the position of mocap marker wrt mocap reference frame
# point_wrt_mcR = np.array([-112.0, 220.0, 12.0, 1000.0]) / 1000.0   # X, Y, Z in meters wrt camera
point_wrt_mcR = np.array([210.845, 159.02, 6.159, 1000.0]) / 1000.0   # X, Y, Z in meters wrt camera

# # Pose of mocap RGB camera wrt mocap reference frame
# T_mc_wrt_mcR = np.array([
#         [-0.952129, -0.281458, -0.119293, -207.692825 / 1000],
#         [0.032076, -0.480063, 0.876647, 1005.463013 / 1000],
#         [-0.304008, 0.830855, 0.466110, 562.679993 / 1000],
#         [0.0, 0.0, 0.0, 1.0]
#     ]) 

T_mc_wrt_mcR = np.array([
        [-0.964936, -0.170986, 0.199153, 613.657959 / 1000],
        [0.251068, -0.379914, 0.890298, 1280.002319 / 1000],
        [-0.076567, 0.909082, 0.409521, 783.635193 / 1000],
        [0.0, 0.0, 0.0, 1.0]
    ]) 

# Pose of mocap marker wrt mocap RGB camera
point_wrt_mc = np.linalg.inv(T_mc_wrt_mcR) @ point_wrt_mcR
# point_wrt_mc = T_mc_wrt_mcR @ point_wrt_mcR
print("point_wrt_mc: ", point_wrt_mc)

# Maybe account for image convention
R = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])
# R = np.array([
#     [0, -1, 0],
#     [-1, 0, 0],
#     [0, 0, -1]
# ])
point_wrt_mc = R @ point_wrt_mc[:3]
point_wrt_mc = np.append(point_wrt_mc, 1.0)

# Project the mocap marker position to the image
pixel = project_point_to_image(point_wrt_mc[:3], K)
pixel_w_distortion = project_point_to_image(point_wrt_mc[:3], K, dist_coeffs)
print("Projected pixel:", pixel)
print("Projected pixel raw:", pixel_w_distortion)

# Overlay point
img_overlay = overlay_point_on_image(img, pixel[0], color=(0, 0, 255))
# img_overlay = overlay_point_on_image(img_overlay, pixel_w_distortion[0], color=(0, 255, 0))

# Show result
cv2.imshow("Reprojected Point", img_overlay)
cv2.waitKey(0)
# cv2.destroyAllWindows()