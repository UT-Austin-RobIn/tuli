import argparse
import json
from pathlib import Path

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import matplotlib.pyplot as plt

import numpy as np
np.set_printoptions(suppress=True, precision=4)
import open3d as o3d
from sensor_msgs.msg import CameraInfo


def load_RT_from_calib(json_path: Path):
    with open(json_path) as f:
        d = json.load(f)
    R = np.array(d["rot"], dtype=np.float64)
    T = np.array(d["trans"], dtype=np.float64).reshape(3)
    return R, T

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
    ap = argparse.ArgumentParser(
        description="Overlay cam_L and cam_M point clouds using calibration R,T")
    ap.add_argument("--calib", required=True,
                    help="Path to calibration_results.json")
    ap.add_argument("--left-cam", default="cam_L",
                    help="ROS namespace of LEFT camera (matches dataset/left)")
    ap.add_argument("--right-cam", default="cam_M",
                    help="ROS namespace of RIGHT camera (matches dataset/right)")
    args = ap.parse_args()

    R, T_vec = load_RT_from_calib(Path(args.calib).resolve())
    print("Loaded R=\n", R)
    print("Loaded T=", T_vec, " |T|=", np.linalg.norm(T_vec))

    img_L, depth_L, cam_info_L = get_color_image_and_info(args.left_cam)
    img_M, depth_M, cam_info_M = get_color_image_and_info(args.right_cam)
    print("Camera L K matrix:", cam_info_L.K)
    print("Camera M K matrix:", cam_info_M.K)

    T = T_vec.reshape(3, 1)

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

    print("depth_L min/max:", np.min(depth_L), np.max(depth_L))
    print("depth_M min/max:", np.min(depth_M), np.max(depth_M))
    print("n_points L / M:", len(pcd_L.points), len(pcd_M.points))

    # Bounding boxes so you can tell if clouds exist and roughly where
    bbL = pcd_L.get_axis_aligned_bounding_box()
    bbM = pcd_M_wrt_L.get_axis_aligned_bounding_box()
    print(f"pcd_L bounds: min={bbL.min_bound}  max={bbL.max_bound}")
    print(f"pcd_M_wrt_L bounds: min={bbM.min_bound}  max={bbM.max_bound}")

    # Tint each cloud so they're distinguishable when overlapping.
    # cam_L cloud: keep RGB but bias red channel up so it reads warm
    # cam_M cloud (transformed): paint uniformly cyan
    pcd_L_vis = o3d.geometry.PointCloud(pcd_L)
    L_colors = np.asarray(pcd_L_vis.colors)
    if len(L_colors) > 0:
        L_colors[:, 0] = np.clip(L_colors[:, 0] * 0.6 + 0.4, 0, 1)  # redden
        pcd_L_vis.colors = o3d.utility.Vector3dVector(L_colors)

    pcd_M_wrt_L_vis = o3d.geometry.PointCloud(pcd_M_wrt_L)
    M_colors = np.asarray(pcd_M_wrt_L_vis.colors)
    if len(M_colors) > 0:
        # keep RGB but bias green+blue up so it reads cyan while still showing texture
        M_colors[:, 1] = np.clip(M_colors[:, 1] * 0.6 + 0.4, 0, 1)
        M_colors[:, 2] = np.clip(M_colors[:, 2] * 0.6 + 0.4, 0, 1)
        pcd_M_wrt_L_vis.colors = o3d.utility.Vector3dVector(M_colors)

    # Coord frames at origin (cam_L frame) and at cam_M position
    frame_L = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    # cam_M origin in cam_L frame = -R^T @ T
    cam_M_origin_in_L = (-R.T @ T_vec).ravel()
    frame_M = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    M_pose = np.eye(4)
    M_pose[:3, :3] = R.T
    M_pose[:3, 3] = cam_M_origin_in_L
    frame_M.transform(M_pose)

    print(f"cam_M origin in cam_L frame: {cam_M_origin_in_L}")

    print("\nOpen3D controls:  mouse-drag=rotate  shift+drag=pan  scroll=zoom  "
          "R=reset view  Q=quit")
    print("Keys: 1=toggle cam_L (red)  2=toggle cam_M (cyan)  3=show both")
    print("Legend:  RED-TINTED = cam_L.  CYAN = cam_M transformed into cam_L frame.\n")

    # Interactive viewer with keyboard toggles
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Stereo calib check — cam_L (red) + cam_M->L (cyan)")
    for g in (pcd_L_vis, pcd_M_wrt_L_vis, frame_L, frame_M):
        vis.add_geometry(g)

    state = {"L": True, "M": True}
    def _refresh():
        vis.clear_geometries()
        if state["L"]:
            vis.add_geometry(pcd_L_vis, reset_bounding_box=False)
        if state["M"]:
            vis.add_geometry(pcd_M_wrt_L_vis, reset_bounding_box=False)
        vis.add_geometry(frame_L, reset_bounding_box=False)
        vis.add_geometry(frame_M, reset_bounding_box=False)

    def toggle_L(v):
        state["L"] = not state["L"]; _refresh(); return False
    def toggle_M(v):
        state["M"] = not state["M"]; _refresh(); return False
    def show_both(v):
        state["L"] = True; state["M"] = True; _refresh(); return False

    vis.register_key_callback(ord("1"), toggle_L)
    vis.register_key_callback(ord("2"), toggle_M)
    vis.register_key_callback(ord("3"), show_both)
    vis.run()
    vis.destroy_window()


    