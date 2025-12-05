#!/usr/bin/env python3
"""
Script to save calibration intrinsics as a separate file
"""

import numpy as np
np.set_printoptions(precision=3, suppress=True)
import xml.etree.ElementTree as ET
import cv2
import logging
from lxml import etree

def read_qca(qca_path, binning_factor):
    '''
    Reads a Qualisys .qca.txt calibration file
    Returns 6 lists of size N (N=number of cameras)
    
    INPUTS: 
    - qca_path: path to .qca.txt calibration file: string
    - binning_factor: usually 1: integer

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of 3x3 arrays of floats
    - T: extrinsic translation: list of arrays of floats
    '''

    root = etree.parse(qca_path).getroot()
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []
    res = []
    vid_id = []
    
    # Camera name
    for i, tag in enumerate(root.findall('cameras/camera')):
        ret += [float(tag.attrib.get('avg-residual'))]
        C += [tag.attrib.get('serial')]
        res += [int(tag.attrib.get('video_resolution')[:-1]) if tag.attrib.get('video_resolution') not in (None, "N/A") else 1080]
        if tag.attrib.get('model') in ('Miqus Video', 'Miqus Video UnderWater', 'none'):
            vid_id += [i]
    
    # Image size
    for i, tag in enumerate(root.findall('cameras/camera/fov_video')):
        w = (float(tag.attrib.get('right')) - float(tag.attrib.get('left')) +1) /binning_factor \
            / (1080/res[i]) 
        h = (float(tag.attrib.get('bottom')) - float(tag.attrib.get('top')) +1) /binning_factor \
            / (1080/res[i])
        S += [[w, h]]
    
    # Intrinsic parameters: distorsion and intrinsic matrix
    for i, tag in enumerate(root.findall('cameras/camera/intrinsic')):
        k1 = float(tag.get('radialDistortion1'))/64/binning_factor
        k2 = float(tag.get('radialDistortion2'))/64/binning_factor
        p1 = float(tag.get('tangentalDistortion1'))/64/binning_factor
        p2 = float(tag.get('tangentalDistortion2'))/64/binning_factor
        D+= [np.array([k1, k2, p1, p2])]
        
        fu = float(tag.get('focalLengthU'))/64/binning_factor \
            / (1080/res[i])
        fv = float(tag.get('focalLengthV'))/64/binning_factor \
            / (1080/res[i])
        cu = (float(tag.get('centerPointU'))/64/binning_factor \
            - float(root.findall('cameras/camera/fov_video')[i].attrib.get('left'))) \
            / (1080/res[i])
        cv = (float(tag.get('centerPointV'))/64/binning_factor \
            - float(root.findall('cameras/camera/fov_video')[i].attrib.get('top'))) \
            / (1080/res[i])
        K += [np.array([fu, 0., cu, 0., fv, cv, 0., 0., 1.]).reshape(3,3)]

    # Extrinsic parameters: rotation matrix and translation vector
    for tag in root.findall('cameras/camera/transform'):
        tx = float(tag.get('x'))/1000
        ty = float(tag.get('y'))/1000
        tz = float(tag.get('z'))/1000
        r11 = float(tag.get('r11'))
        r12 = float(tag.get('r12'))
        r13 = float(tag.get('r13'))
        r21 = float(tag.get('r21'))
        r22 = float(tag.get('r22'))
        r23 = float(tag.get('r23'))
        r31 = float(tag.get('r31'))
        r32 = float(tag.get('r32'))
        r33 = float(tag.get('r33'))

        # Rotation (by-column to by-line)
        R += [np.array([r11, r12, r13, r21, r22, r23, r31, r32, r33]).reshape(3,3).T]
        T += [np.array([tx, ty, tz])]
   
    breakpoint()
    return ret, C, S, D, K, R, T


def calib_qca_fun(file_to_convert_path, binning_factor=1):
    '''
    Convert a Qualisys .qca.txt calibration file
    Converts from camera view to object view, Pi rotates cameras, 
    and converts rotation with Rodrigues formula

    INPUTS:
    - file_to_convert_path: path of the .qca.text file to convert
    - binning_factor: when filming in 540p, one out of 2 pixels is binned so that the full frame is used

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats
    - T: extrinsic translation: list of arrays of floats
    '''
    
    logging.info(f'Converting {file_to_convert_path} to .toml calibration file...')
    ret, C, S, D, K, R, T = read_qca(file_to_convert_path, binning_factor)
    
    RT = [world_to_camera_persp(r,t) for r, t in zip(R, T)]
    R = [rt[0] for rt in RT]
    T = [rt[1] for rt in RT]

    RT = [rotate_cam(r, t, ang_x=np.pi, ang_y=0, ang_z=0) for r, t in zip(R, T)]
    R = [rt[0] for rt in RT]
    T = [rt[1] for rt in RT]

    #R = [np.array(cv2.Rodrigues(r)[0]).flatten() for r in R]
    #T = np.array(T)

    return ret, C, S, D, K, R, T

def calculate_intrinsics_from_calibration(camera_data):
    """Calculate intrinsic matrices from calibration data"""
    intrinsics = []
    
    for camera in camera_data:
        # Create intrinsic matrix
        K = np.eye(3)
        
        # Use scaling factor 1.565 to match reference format
        scale_factor = 64
        
        K[0, 0] = camera['focalLengthU'] / scale_factor  # focal length x
        K[1, 1] = camera['focalLengthV'] / scale_factor  # focal length y
        K[0, 2] = camera['centerPointU'] / scale_factor  # principal point x
        K[1, 2] = camera['centerPointV'] / scale_factor  # principal point y
        K[0, 1] = camera['skew']
        
        intrinsics.append(K)
        
        print(f"Camera {camera['index']+1} intrinsics:")
        print(K)
        print()
    
    return intrinsics

def world_to_camera_persp(r, t):
    '''
    Converts rotation R and translation T 
    from Qualisys world centered perspective
    to OpenCV camera centered perspective,
    and inversely.

    Qc = RQ+T --> Q = R-1.Qc - R-1.T

    INPUTS:
    - r: rotation matrix (3x3)
    - t: translation vector (3x1)

    OUTPUTS:
    - r: rotation matrix (3x3)
    - t: translation vector (3x1)
    '''

    r = r.T
    t = - r @ t 

    return r, t


def rotate_cam(r, t, ang_x=0, ang_y=0, ang_z=0):
    '''
    Apply rotations around x, y, z in cameras coordinates
    Angle in radians
    '''

    r,t = np.array(r), np.array(t)
    if r.shape == (3,3):
        rt_h = np.block([[r,t.reshape(3,1)], [np.zeros(3), 1 ]]) 
    elif r.shape == (3,):
        rt_h = np.block([[cv2.Rodrigues(r)[0],t.reshape(3,1)], [np.zeros(3), 1 ]])
    
    r_ax_x = np.array([1,0,0, 0,np.cos(ang_x),-np.sin(ang_x), 0,np.sin(ang_x),np.cos(ang_x)]).reshape(3,3) 
    r_ax_y = np.array([np.cos(ang_y),0,np.sin(ang_y), 0,1,0, -np.sin(ang_y),0,np.cos(ang_y)]).reshape(3,3)
    r_ax_z = np.array([np.cos(ang_z),-np.sin(ang_z),0, np.sin(ang_z),np.cos(ang_z),0, 0,0,1]).reshape(3,3) 
    r_ax = r_ax_z @ r_ax_y @ r_ax_x

    r_ax_h = np.block([[r_ax,np.zeros(3).reshape(3,1)], [np.zeros(3), 1]])
    r_ax_h__rt_h = r_ax_h @ rt_h
    
    r = r_ax_h__rt_h[:3,:3]
    t = r_ax_h__rt_h[:3,3]

    return r, t



def save_calibration_intrinsics(calibration_path, output_path, num_frames=5000):
    """Save calibration intrinsics to npz file similar to cameras.npz"""
    
    # Use pose2sim functions to parse and transform calibration data
    ret, C, S, D, K, R, T = calib_qca_fun(calibration_path, binning_factor=1)
    
    print(f"Found {len(C)} cameras:")
    for i, serial in enumerate(C):
        print(f"  Camera {i}: {serial}")
    
    # Number of cameras: real cameras
    num_cameras = len(C)
    
    # Initialize arrays for final output with specified number of frames
    intrins_final = np.zeros((num_cameras, num_frames, 3, 3))
    w2c_final = np.zeros((num_cameras, num_frames, 4, 4))
    dist_final = np.zeros((num_cameras, num_frames, 5))
    
    
    # Set real cameras from pose2sim data
    for i in range(len(C)):        
        # Create 4x4 transformation matrix
        w2c = np.eye(4)
        w2c[:3, :3] = R[i]
        w2c[:3, 3] = T[i]
        
        # Create distortion coefficients array [k1, k2, p1, p2, k3]
        # D[i] has 4 coefficients, we need 5, so add 0 for k3
        dist = np.array([D[i][0], D[i][1], D[i][2], D[i][3], 0.0])
        
        for frame in range(num_frames):
            intrins_final[i, frame] = K[i]
            w2c_final[i, frame] = w2c
            dist_final[i, frame] = dist
        
        print(f"Camera {i}: {C[i]}")
    
    # Save to npz file
    print(f"Saving calibration intrinsics to: {output_path}")
    np.savez(output_path,
             intrins=intrins_final,
             w2c=w2c_final,
             dist=dist_final)
    
    print(f"Saved {num_cameras} cameras with {num_frames} frames")
    print(f"Array shapes: intrins {intrins_final.shape}, w2c {w2c_final.shape}, dist {dist_final.shape}")
    
    return intrins_final, w2c_final, dist_final


if __name__ == "__main__":
    read_qca("/home/robotlearning2/infants/calibration_output.txt", binning_factor=1)
