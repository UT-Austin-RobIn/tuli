import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import matplotlib.pyplot as plt

import numpy as np
np.set_printoptions(suppress=True, precision=4)
import open3d as o3d
from sensor_msgs.msg import CameraInfo

def show_pointcloud_with_point(pcd, point_xyz, point_color=[1, 0, 0], sphere_radius=0.03):
    """
    Display a point cloud with a single highlighted 3D point (e.g., in red).

    Args:
        pcd (o3d.geometry.PointCloud): The original point cloud.
        point_xyz (list or np.ndarray): 3D coordinates of the point to highlight [x, y, z].
        point_color (list): RGB color of the point (default red).
        sphere_radius (float): Radius of the sphere used to show the point.
    """
    # Create a small sphere at the point location for visibility
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere.translate(point_xyz)
    sphere.paint_uniform_color(point_color)

    # Optional: add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Visualize together
    o3d.visualization.draw_geometries([pcd, sphere, coord_frame],
                                      window_name="PointCloud with Highlighted Point")

def show_colored_pointcloud(color_img, depth_img, cam_info: CameraInfo, depth_scale=1000.0):
    """
    Display a colored point cloud using Open3D.
    
    Args:
        color_img (np.ndarray): Color image (BGR from OpenCV).
        depth_img (np.ndarray): Depth image (uint16 or float32).
        cam_info (CameraInfo): CameraInfo message with intrinsic parameters.
        depth_scale (float): Depth scale (e.g., 1000.0 if depth is in millimeters).
    """
    # Convert BGR (OpenCV) to RGB (Open3D)
    color_img_rgb = color_img[:, :, ::-1].copy()

    # breakpoint()

    # Normalize depth if it's in uint16 (e.g., millimeters)
    if depth_img.dtype == np.uint16:
        depth_img = depth_img.astype(np.float32) / depth_scale

    # Create Open3D images
    color_o3d = o3d.geometry.Image(color_img_rgb)
    depth_o3d = o3d.geometry.Image(depth_img)

    # Camera intrinsics from CameraInfo
    width = cam_info.width
    height = cam_info.height
    fx = cam_info.K[0]
    fy = cam_info.K[4]
    cx = cam_info.K[2]
    cy = cam_info.K[5]

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,  # Already normalized
        convert_rgb_to_intensity=False
    )

    # Create and visualize point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic
    )

    # # Flip it for correct orientation
    # pcd.transform([[1, 0, 0, 0],
    #                [0, -1, 0, 0],
    #                [0, 0, -1, 0],
    #                [0, 0, 0, 1]])

    # Visualize
    # o3d.visualization.draw_geometries([pcd], window_name="Colored Point Cloud")
    # o3d.visualization.draw_geometries_with_editing([pcd])

    # INSERT_YOUR_CODE

    # # Add Open3D visualizer with picking support
    # def pick_points_and_print(pcd):
    #     print("Instructions: Please pick a point by clicking on it in the visualizer window, then press 'Q' to finish.")
    #     vis = o3d.visualization.VisualizerWithEditing()
    #     vis.create_window(window_name="Pick a Point (Press Q to finish)")
    #     vis.add_geometry(pcd)
    #     vis.run()  # user picks points
    #     vis.destroy_window()
    #     picked_indices = vis.get_picked_points()
    #     if not picked_indices:
    #         print("No point was picked.")
    #     else:
    #         for idx in picked_indices:
    #             pt = np.asarray(pcd.points)[idx]
    #             print(f"Picked point index: {idx}, coordinates: {pt}")

    # pick_points_and_print(pcd)
    
    # Add camera coordinate frame at origin
    # geometries = [pcd]
    # show_camera_frame = True
    # if show_camera_frame:
    #     cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    #     cam_frame.transform(flip)  # Align frame with flipped point cloud
    #     geometries.append(cam_frame)

    # o3d.visualization.draw_geometries(geometries, window_name="Colored Point Cloud with Camera Frame")

    return pcd


def get_color_image_and_info():
    rospy.init_node('simple_image_listener', anonymous=True)
    bridge = CvBridge()
    
    image_msg = rospy.wait_for_message("/cam_L/color/image_raw", Image)
    depth_msg = rospy.wait_for_message('/cam_L/aligned_depth_to_color/image_raw', Image)
    info_msg = rospy.wait_for_message("/cam_L/color/camera_info", CameraInfo)
    
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
    cv_depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
    # cv_depth = None
    
    return cv_image, cv_depth, info_msg

# Example usage:
if __name__ == '__main__':
    img, depth, cam_info = get_color_image_and_info()
    print("Camera K matrix:", cam_info.K)
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(img)
    # ax[1].imshow(depth)
    # plt.show()

    T_mc_wrt_mcR = np.array([
        [-0.952129, 0.032076, -0.304008, -207.692825 / 1000],
        [-0.281458, -0.480063, 0.830855, 1005.463013 / 1000],
        [-0.119293, 0.876647, 0.466110, 562.679993 / 1000],
        [0.0, 0.0, 0.0, 1.0]
    ]) 

    # point_wrt_mcR = np.array([-174.0, 419.0, 25.0, 1000.0]) / 1000.0    
    # 78.71	184.111	14.02
    point_wrt_mcR = np.array([19.0, 220.25, 77.40, 1000.0]) / 1000.0   # X, Y, Z in meters wrt camera

    point_wrt_mc = np.linalg.inv(T_mc_wrt_mcR) @ point_wrt_mcR
    print("point_wrt_mc: ", point_wrt_mc)

    # Account for image convention change
    R = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    point_wrt_mc = R @ point_wrt_mc[:3]
    point_wrt_mc = np.append(point_wrt_mc, 1.0)

    R =  np.array([
        [ 0.7780564627749914, 0.17704388149961886, -0.6027301259748158],
        [ -0.2978532687426021, 0.9487285426025455, -0.10581863139635345],
        [ 0.5530927327613478, 0.26185800823117666, 0.7908974728065485]])
    T = np.array([
        [
            0.8142749429152566
        ],
        [
            0.0373086855203048
        ],
        [
            0.20257702653342857
        ]])

    trans_matrix = np.eye(4)
    trans_matrix[:3, :3] = R
    trans_matrix[:3, 3] = T.squeeze()
    temp1 = trans_matrix @ point_wrt_mc
    temp2 = np.linalg.inv(trans_matrix) @ point_wrt_mc

    print("temp1: ", temp1)
    print("temp2: ", temp2)
    breakpoint()

    pcd = show_colored_pointcloud(img, depth, cam_info)
    show_pointcloud_with_point(pcd, temp2[:3])
    # ====================================================

    # # From calibration
    # # T_mc_wrt_rs = np.array([
    # #     [0.7728119043488829, 0.16489096932055147, -0.6128398883339611,  0.9329320913352112],
    # #     [-0.27310698869462935, 0.9580762539578125, -0.08661676701607281, 0.0185173121258862],
    # #     [0.5728650218182331, 0.23430932512107577, 0.7854456104266554, 0.00822586669189619],
    # #     [0.0, 0.0, 0.0, 1.0]
    # # ])
    # T_mc_wrt_rs = np.array([
    #     [0.7702271844351884, 0.1731200513184427, -0.613823697969166, 0.8438201983047298],
    #     [-0.30053877967391807, 0.9474153310704713, -0.10991193003828764, 0.040289812708229594],
    #     [0.5625180230616209, 0.26913498153181253, 0.7817543319014718, 0.14936088329822786],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])

    # # Axis alignment from mocap frame to RealSense frame
    # F = np.diag([-1, -1, 1, 1])
    # T_mc_wrt_rs = F @ T_mc_wrt_rs
    # print("T_mc_wrt_rs: ", T_mc_wrt_rs)
    # breakpoint()

    # # rvec, _ = cv2.Rodrigues(T_mc_wrt_rs[:3, :3])
    # # # Now tweak rvec (in radians), e.g.
    # # rvec[1] += 0.2  # small rotation around X
    # # # Convert back to rotation matrix
    # # R_new, _ = cv2.Rodrigues(rvec)
    # # T_mc_wrt_rs[:3, :3] = R_new

    # T_mcR_wrt_rs = T_mc_wrt_rs @ np.linalg.inv(T_mc_wrt_mcR)

    # T_rs_wrt_mcR = np.linalg.inv(T_mcR_wrt_rs)
    # print("T_rs_wrt_mcR: ", T_rs_wrt_mcR)

    # # breakpoint()
    # point_wrt_rs = T_mcR_wrt_rs @ point_wrt_mcR

    # print("point: ", point_wrt_rs)
    # # # breakpoint()
    # # flip = np.array([[1, 0, 0, 0],
    # #              [0, -1, 0, 0],
    # #              [0, 0, -1, 0],
    # #              [0, 0, 0, 1]])
    # # point_wrt_rs = flip @ point_wrt_rs

    # pcd = show_colored_pointcloud(img, depth, cam_info)

    # # breakpoint()
    # show_pointcloud_with_point(pcd, point_wrt_rs[:3])

    