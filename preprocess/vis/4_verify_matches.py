import os
import json
import numpy as np
import json
import pyproj
import random
from scipy.spatial.transform import Rotation as R
from wrapper import Pose, Camera
import cv2
import torch
from tqdm import tqdm
from torch import nn
from get_depth import sample_points_with_valid_depth, get_3D_samples, transform_ecef_origin, get_points2D_ECEF_projection
from transform import ECEF_to_WGS84, WGS84_to_ECEF, get_rotation_enu_in_ecef, visualize_matches
def load_poses(pose_file):
    """Load poses from the pose file."""
    pose_dict = {}
    with open(pose_file, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                pose_dict[parts[0]] = parts[1: ]  # Add the first element to the name list
    return pose_dict
if __name__ == "__main__":
    query_files = []
    reference_rgb_path = "/home/ubuntu/Documents/code/FPVLoc/src_open/preprocess/datasets/Netherland_seq1/Switzerland/193_0.png"
    reference_depth_path = "/home/ubuntu/Documents/code/FPVLoc/src_open/preprocess/datasets/Netherland_seq1/Switzerland/193_1.png"
    query_rgb_path = "/home/ubuntu/Documents/code/FPVLoc/src_open/preprocess/datasets/Netherland_seq1/Switzerland/269_0.png"
    query_depth_path = "/home/ubuntu/Documents/code/FPVLoc/src_open/preprocess/datasets/Netherland_seq1/Switzerland/269_1.png"
    pose_txt = "/mnt/sda/MapScape/pose/switzerland_seq1@500@30_50.txt"

    pose_dict = load_poses(pose_txt)
    # Generate pairings
    origin = None
    min_valid_threshold = 4000    # 有效点数必须不少于此值
    max_attempts = 10
    attempt = 0
    ref_pose_name = reference_rgb_path.split('/')[-1].split('_')[0] +'.jpg'
    query_pose_name = query_rgb_path.split('/')[-1].split('_')[0] +'.jpg'
    ref_pose = pose_dict[ref_pose_name]
    query_pose = pose_dict[query_pose_name]
    # get query pose
    lon, lat, alt, roll, pitch, yaw = map(float, query_pose)

    euler_angles = [pitch, roll, yaw]
    translation = [lon, lat, alt]
    rot_pose_in_ned = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
    rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
    t_c2w = WGS84_to_ECEF(translation)
    query_T = np.eye(4)
    query_T[:3, :3] = R_c2w
    query_T[:3, 3] = t_c2w
    query_T = query_T.tolist()

    query_pose = euler_angles + translation
        # query_T_candidates = add_noise_to_pose(euler_angles, translation, query_T)

    # get query intrinscis
    K_w2c = [1600, 1200, 1900.0, 1900.0, 800.0, 600.0]
         
    K = K_w2c
    width, height = K[0], K[1]
    cam_query = {
    'model': 'PINHOLE',
    'width': width,
    'height': height,
    'params': [K[2], K[3], K[4], K[5]] #np.array(K[2:]
    }   
    cam_ref = {
    'model': 'PINHOLE',
    'width': width,
    'height': height,
    'params': [K[2], K[3], K[4], K[5]] #np.array(K[2:]
    }   
    origin = [0, 0, 0]     
    qcamera = Camera.from_colmap(cam_query)
    rcamera = Camera.from_colmap(cam_ref)


       
    lon, lat, alt, roll, pitch, yaw = map(float, ref_pose)
    euler_angles_ref = [pitch, roll, yaw]
    translation_ref = [lon, lat, alt]
    lon, lat, height = translation_ref
    rot_pose_in_ned = R.from_euler('xyz', euler_angles_ref, degrees=True).as_matrix()  # ZXY 东北天  
    rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
    t_c2w = WGS84_to_ECEF(translation_ref)
    ref_T = np.eye(4)
    ref_T[:3, :3] = R_c2w
    ref_T[:3, 3] = t_c2w
    ref_T = ref_T.tolist()
            

    # get query & ref rgb path/depth
    rgb_image = cv2.imread(query_rgb_path)
    ref_image = cv2.imread(reference_rgb_path)
    ref_depth_image = cv2.imread(reference_depth_path, cv2.IMREAD_UNCHANGED)
    ref_depth_image = cv2.flip(ref_depth_image, 0)
           
            
            
    points2d_ref = sample_points_with_valid_depth(ref_depth_image, num_points=100000, max_depth=2000)
    points2d_ref_valid, point3D_from_ref, _ = get_3D_samples(points2d_ref, ref_depth_image, ref_T, rcamera)
    pose_query_origin = transform_ecef_origin(np.array(query_T), origin=origin)
    points2d_query, _, Points_3D_ECEF_origin, valid = get_points2D_ECEF_projection(pose_query_origin, qcamera, point3D_from_ref, points2d_ref_valid, use_valid = False, num_samples=20000)

    depth_image = cv2.imread(query_depth_path, cv2.IMREAD_UNCHANGED)
    depth_image = cv2.flip(depth_image, 0)

    render_height_px, render_width_px = depth_image.shape[:2]
    valid_x = (points2d_query[:, 0] >= 0) & (points2d_query[:, 0] < render_width_px)
    valid_y = (points2d_query[:, 1] >= 0) & (points2d_query[:, 1] < render_height_px)
    valid = valid_x & valid_y
    true_indices = np.where(valid)[0]
    false_indices = np.where(~valid)[0]

    points2d_query_with_valid_depth, point3D_from_query, _= get_3D_samples(points2d_query[true_indices], depth_image, query_T, qcamera)
    pose_ref_origin = transform_ecef_origin(np.array(ref_T), origin=origin)
    points2d_ref_rej, _, _, _ = get_points2D_ECEF_projection(pose_ref_origin, rcamera, point3D_from_query, points2d_query_with_valid_depth, use_valid = False, num_samples=10000)
    points2d_ref_sample = points2d_ref_valid[true_indices]

    # 检查是否query像素坐标与反投影回query图像素坐标是否对不上
    max_distance = 2
    if points2d_ref_sample.shape != points2d_ref_rej.shape:
        raise ValueError("Both point sets must have the same shape.")
    # 计算每对点之间的欧几里得距离
    distances = np.linalg.norm(points2d_ref_sample.cpu().numpy() - points2d_ref_rej, axis=1)
    # 找到距离小于等于最大距离的索引
    num_samples = 5000
    valid_indices = np.where(distances <= max_distance)[0]
    # print(f"当前参考帧 {ref_indices} 有效点数: {len(valid_indices)}")
    # 如果有效点数满足要求，则跳出循环
    true_indices_valid = true_indices[valid_indices]
    final_indices = np.array(true_indices_valid.tolist())

    step = len(final_indices) / num_samples
    # 根据间隔采样索引
    valid_indices_final = [final_indices[int(i * step)] for i in range(num_samples)]
    # num_points = 5000
    # valid_indices_final = np.random.choice(len(valid_indices), size=num_points, replace=False)
    points2d_query = points2d_query[valid_indices_final]
    points2d_ref = points2d_ref_valid[valid_indices_final]
    Points_3D_ECEF_origin = Points_3D_ECEF_origin[valid_indices_final]

        
    t = np.random.choice(len(points2d_query), size=50, replace=False)
    vis_query_points = points2d_query[t]
    vis_ref_points = points2d_ref[t]
    visualize_matches(rgb_image, ref_image, 
                    vis_query_points, 
                    vis_ref_points)
    # t = np.random.choice(len(points2d_query_with_valid_depth[valid_indices]), size=100, replace=False)
    # visualize_points_on_images(rgb_image, ref_image, 
    #                 points2d_query_with_valid_depth[valid_indices][t], 
    #                 points2d_ref_rej[valid_indices][t])
    # visualize_points_on_images(rgb_image, ref_image, 
    #                 points2d_query_with_valid_depth[valid_indices][t], 
    #                 points2d_ref_sample[valid_indices][t])