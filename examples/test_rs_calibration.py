import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import matplotlib.pyplot as plt

import numpy as np
np.set_printoptions(suppress=True, precision=4)
import open3d as o3d
from sensor_msgs.msg import CameraInfo

def show_pointcloud_with_point(pcd, point_xyz, point_color=[1, 0, 0], sphere_radius=0.05):
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


def get_color_image_and_info(cam_name):
    rospy.init_node('simple_image_listener', anonymous=True)
    bridge = CvBridge()
    
    image_msg = rospy.wait_for_message(f"/{cam_name}/color/image_raw", Image)
    depth_msg = rospy.wait_for_message(f'/{cam_name}/aligned_depth_to_color/image_raw', Image)
    info_msg = rospy.wait_for_message(f"/{cam_name}/color/camera_info", CameraInfo)
    
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
    cv_depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
    # cv_depth = None
    
    return cv_image, cv_depth, info_msg

# Example usage:
if __name__ == '__main__':
    img_L, depth_L, cam_info_L = get_color_image_and_info("cam_L")
    img_M, depth_M, cam_info_M = get_color_image_and_info("cam_M")
    print("Camera L K matrix:", cam_info_L.K)
    print("Camera M K matrix:", cam_info_M.K)
    

    # R = np.array([
    #     [
    #         0.5112488327460185,
    #         0.14645775453176482,
    #         -0.8468617107611768
    #     ],
    #     [
    #         -0.17605127644886667,
    #         0.9823240823650885,
    #         0.06360302875108664
    #     ],
    #     [
    #         0.8412078096859054,
    #         0.11657411094707157,
    #         0.5279961151186072
    #     ]
    # ])

    # T =  np.array([
    #     [
    #         0.8852560627705376
    #     ],
    #     [
    #         -0.08736871615668282
    #     ],
    #     [
    #         0.08579026326347071
    #     ]
    # ])

    R = np.array([
        [
            0.4617615120266097,
            0.1636960951356345,
            -0.8717682573070922
        ],
        [
            -0.168284758616144,
            0.9811406803178834,
            0.09509576984738795
        ],
        [
            0.8708941072417504,
            0.10279374427807621,
            0.4806005619107079
        ]
    ])
    T = np.array([
        [
            0.8847490265848044
        ],
        [
            -0.1131774738506267
        ],
        [
            0.11741134311767307
        ]
    ])

    trans_matrix = np.eye(4)
    trans_matrix[:3, :3] = R
    trans_matrix[:3, 3] = T.squeeze()

    pcd_L = show_colored_pointcloud(img_L, depth_L, cam_info_L)
    pcd_M = show_colored_pointcloud(img_M, depth_M, cam_info_M)

    # breakpoint()
    pcd_M_points = np.asarray(pcd_M.points)
    # INSERT_YOUR_CODE
    # Make pcd_M_points homogeneous (Nx4)
    ones = np.ones((pcd_M_points.shape[0], 1))
    pcd_M_points_hom = np.hstack([pcd_M_points, ones]).T  # shape (4, N)
    pcd_M_points_wrt_L = np.linalg.inv(trans_matrix) @ pcd_M_points_hom
    pcd_M_points_wrt_L = pcd_M_points_wrt_L.T
    pcd_M_points_wrt_L = pcd_M_points_wrt_L[:, :3]
    pcd_M_wrt_L = o3d.geometry.PointCloud()
    pcd_M_wrt_L.points = o3d.utility.Vector3dVector(pcd_M_points_wrt_L)
    pcd_M_wrt_L.colors = pcd_M.colors
    # pcd_M_wrt_L = pcd_M.transform(trans_matrix)

    print(np.min(depth_L), np.max(depth_L))
    print(np.min(depth_M), np.max(depth_M))
    print(len(pcd_L.points), len(pcd_M.points))

    o3d.visualization.draw_geometries([pcd_L, pcd_M_wrt_L])


    