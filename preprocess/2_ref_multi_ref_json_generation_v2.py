import os
import json
import numpy as np
import json
import pyproj
import random
from scipy.spatial.transform import Rotation as R
from wrapper import Pose, Camera
from transform import visualize_points_on_images
import cv2
import torch
from tqdm import tqdm
from torch import nn
from get_depth import sample_points_with_valid_depth, get_3D_samples, transform_ecef_origin, get_points2D_ECEF_projection
def add_noise_to_pose(euler_angles, translation, query_T, noise_std_angle=5.0, noise_std_translation=0.5, num_candidates=7):
    """
    Generate candidate poses by adding noise to Euler angles and translations.

    :param euler_angles: List or array of 3 Euler angles (roll, pitch, yaw) in degrees
    :param t_c2w: List or array of 3 translations (x, y, z)
    :param noise_std_angle: Standard deviation for angle noise in degrees
    :param noise_std_translation: Standard deviation for translation noise
    :param num_candidates: Number of candidate poses to generate
    :return: List of candidate poses, each pose is a dictionary with 'euler_angles' and 't_c2w'
    """
    candidates = []
    lon, lat, _ = translation
    rot_pose_in_enu = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
    rot_enu_to_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_enu_to_ecef, rot_pose_in_enu)
    t_c2w = WGS84_to_ECEF(translation)
    # Initialize a 4x4 identity matrix
    render_T = np.eye(4)
    render_T[:3, :3] = R_c2w
    render_T[:3, 3] = t_c2w
    candidates.append(render_T.tolist())



    are_close = np.allclose(render_T, query_T)
    if not are_close:
        print("!")
    
    for _ in range(num_candidates):
        noisy_euler_angles = euler_angles + np.random.normal(0, noise_std_angle, size=3)
        noisy_t_c2w = t_c2w + np.random.normal(0, noise_std_translation, size=3)

        noise_trans = ECEF_to_WGS84(noisy_t_c2w)
        lon, lat, _ =noise_trans
        rot_pose_in_enu = R.from_euler('xyz', noisy_euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
        rot_enu_to_ecef = get_rotation_enu_in_ecef(lon, lat)
        noisy_R_c2w = np.matmul(rot_enu_to_ecef, rot_pose_in_enu)
        
        # Initialize a 4x4 identity matrix
        noisy_render_T = np.eye(4)
        noisy_render_T[:3, :3] = noisy_R_c2w
        noisy_render_T[:3, 3] = noisy_t_c2w

        candidates.append(noisy_render_T.tolist())


    min_offset_angle = -5.0
    max_offset_angle = 5.0
    min_offset_translation = -5.0
    max_offset_translation = 5.0
    render_T_w2c = np.eye(4)
    render_T_w2c[:3, :3] = render_T[:3, :3].T
    render_T_w2c[:3, 3] = -render_T_w2c[:3, :3] @ render_T[:3, 3]
    render_T_w2c = render_T_w2c.astype(np.float32)
    render_T = render_T.astype(np.float32)
    pose = Pose.from_Rt(render_T_w2c[:3, :3], render_T_w2c[:3, 3])

    pose_c2w = Pose.from_Rt(render_T[:3, :3], render_T[:3, 3])
    random_aa, random_t = generate_random_aa_and_t(min_offset_angle, max_offset_angle,
                                                  min_offset_translation, max_offset_translation)
    random_pose = Pose.from_aa(random_aa, random_t)
    body2view_pose = pose_c2w @ random_pose[0]
    
    # R_c2w = (body2view_pose.R).T
    # up_t_c2w = -R_c2w @ body2view_pose.t
    R_c2w = body2view_pose.R
    up_t_c2w = body2view_pose.t

    uodate_translation = ECEF_to_WGS84(up_t_c2w)
    lon, lat, _ = uodate_translation

    rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    # 将ECEF姿态矩阵转换为ENU姿态矩阵
    rot_ecef_in_enu = rot_ned_in_ecef.T  #! ECEF TO WGS84 transformation
    # 将ECEF姿态矩阵转换为ENU姿态矩阵
    rot_pose_in_enu = np.matmul(rot_ecef_in_enu, R_c2w)
    # 从ENU姿态矩阵中提取欧拉角
    rot_pose_in_enu_obj = R.from_matrix(rot_pose_in_enu)
    uodate_euler_angles = rot_pose_in_enu_obj.as_euler('xyz', degrees=True)
    print("euler: ", euler_angles, uodate_euler_angles)
    print("t: ", t_c2w, up_t_c2w)


    return candidates
def ECEF_to_WGS84(pos):
    xpjr, ypjr, zpjr = pos
    transprojr = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        "EPSG:4326",
        always_xy=True,
    )
    lon, lat, height = transprojr.transform(xpjr, ypjr, zpjr, radians=False)
    return [lon, lat, height]  
def WGS84_to_ECEF(pos):
    lon, lat, height = pos
    transprojr = pyproj.Transformer.from_crs(
        "EPSG:4326",
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        always_xy=True,
    )
    xpjr, ypjr, zpjr = transprojr.transform(lon, lat, height, radians=False)
    return [xpjr, ypjr, zpjr]

def get_rotation_enu_in_ecef(lon, lat):
    """
    @param: lon, lat Longitude and latitude in degree
    @return: 3x3 rotation matrix of heading-pith-roll ENU in ECEF coordinate system
    Reference: https://apps.dtic.mil/dtic/tr/fulltext/u2/a484864.pdf, Section 4.3, 4.1
    Reference: https://www.fossen.biz/wiley/ed2/Ch2.pdf, p29
    """
    # 将角度转换为弧度
    latitude_rad = np.radians(lat)
    longitude_rad = np.radians(lon)
    
    # 计算向上的向量（Up Vector）
    up = np.array([
        np.cos(longitude_rad) * np.cos(latitude_rad),
        np.sin(longitude_rad) * np.cos(latitude_rad),
        np.sin(latitude_rad)
    ])
    
    # 计算向东的向量（East Vector）
    east = np.array([
        -np.sin(longitude_rad),
        np.cos(longitude_rad),
        0
    ])
    
    # 计算向北的向量（North Vector），即up向量和east向量的外积（叉积）
    north = np.cross(up, east)
    
    # 构建局部到世界坐标系的转换矩阵
    local_to_world = np.zeros((3, 3))
    local_to_world[:, 0] = east  # 东向分量
    local_to_world[:, 1] = north  # 北向分量
    local_to_world[:, 2] = up  # 上向分量
    return local_to_world
def generate_random_aa_and_t(min_offset_angle, max_offset_angle, min_offset_translation, max_offset_translation):
    if isinstance(min_offset_angle, float):
        min_offset_angle = torch.tensor([min_offset_angle], dtype=torch.float32)
    if isinstance(max_offset_angle, float):
        max_offset_angle = torch.tensor([max_offset_angle], dtype=torch.float32)
    if isinstance(min_offset_translation, float):
        min_offset_translation = torch.tensor([min_offset_translation], dtype=torch.float32)
    if isinstance(max_offset_translation, float):
        max_offset_translation = torch.tensor([max_offset_translation], dtype=torch.float32)

    n = min_offset_angle.shape[0]
    axis = nn.functional.normalize(torch.rand(n, 3) * 2 - 1, dim=-1)
    angle = (torch.rand(n) * (max_offset_angle - min_offset_angle) + min_offset_angle).unsqueeze(-1) / 180 * 3.1415926
    aa = axis * angle

    direction = nn.functional.normalize(torch.rand(n, 3) * 2 - 1, dim=-1)
    t = (torch.rand(n) * (max_offset_translation - min_offset_translation) + min_offset_translation).unsqueeze(-1)
    trans = direction * t

    return aa, trans

def generate_refer_info(sequence_folder, output_json, seq_name, ref_pose_dict , root):
    """
    Generate a JSON file containing query and reference pairings.

    :param sequence_folder: Folder containing the six sequences of query images.
    :param output_json: Path to save the generated JSON file.
    """
    query_files = []
    reference_rgb_path = "Reference/"+seq_name+"/RGB"
    reference_depth_path = "Reference/"+seq_name+"/Depth"

    query_pose_dict = load_poses(os.path.join(sequence_folder, 'pose.txt'))
    # Generate pairings
    refer_info = {}
    points_dir = os.path.join(sequence_folder, 'Points3D')
    if not os.path.exists(points_dir):
        os.mkdir(points_dir)
    origin = None
    min_valid_threshold = 4000    # 有效点数必须不少于此值
    max_attempts = 10
    attempt = 0
    for pose_name, pose in tqdm(query_pose_dict.items()):
        # pose_name = '1_28410.png'
        # pose = query_pose_dict[pose_name]
        # get query pose
        lon, lat, alt, roll, pitch, yaw = map(float, pose)

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
        K_w2c = [640, 480, 1200.0, 1200.0, 320.0, 240.0]
         # get query rgb path
        query_rgb_path = os.path.join(sequence_folder, 'RGB', pose_name)
        # get reference info
        query_index = int(pose_name.split('.')[0].split('_')[-1])

        # Post-process indices
        lower_bound = max(0, query_index - 100)
        upper_bound = min(len(ref_pose_dict)-1, query_index + 100)


        while attempt < max_attempts:
            attempt += 1
            # print(f"尝试第 {attempt} 次选择参考帧...")
            ref_indices = random.randint(lower_bound, upper_bound)
                
            # ref_indices = 28128
            # ref pose
            ref_pose = ref_pose_dict[str(ref_indices)+'.png']
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
            # ref name
            ref_files = f"{ref_indices}.png"

            refer_info[pose_name] = {
                "img_pose": query_T,
                "img_pose_6": query_pose,
                "img_path": query_rgb_path,
                "img_intrisic": K_w2c,
                "img_depth": os.path.join(reference_depth_path, str(query_index)+'_1.png'),
                "ref_info":{"ref_name": ref_files,
                            "ref_rgb": os.path.join(reference_rgb_path, str(ref_indices)+'_0.png'),
                            "ref_depth": os.path.join(reference_depth_path, str(ref_indices)+'_1.png'),
                            "ref_poses" : ref_T,
                            "ref_intrinsics": K_w2c}
            }



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
            # depth_relative_path = refer_info[pose_name]['img_depth']
            # query_depth_path = os.path.join(root, depth_relative_path)
            # depth_image = cv2.imread(query_depth_path, cv2.IMREAD_UNCHANGED)
            # depth_image = cv2.flip(depth_image, 0)
            rgb_image = cv2.imread(query_rgb_path)
            ref_image = cv2.imread(os.path.join(root, reference_rgb_path, f"{os.path.splitext(ref_files)[0]}_0{os.path.splitext(ref_files)[-1]}"))
            # points2d_total_sample = sample_points_with_valid_depth(depth_image, num_points=100000, max_depth=2000)
            # points2d_total, point3D, _= get_3D_samples(points2d_total_sample, depth_image, query_T, qcamera)
            
            # pose_ref_origin = transform_ecef_origin(np.array(ref_T), origin=origin)
            # points2d_ref, points2d_query, Points_3D_ECEF_origin, _ = get_points2D_ECEF_projection(pose_ref_origin, qcamera, point3D, points2d_total, num_samples=20000)
            
            
            depth_relative_path = refer_info[pose_name]["ref_info"]['ref_depth']
            ref_depth_path = os.path.join(root, depth_relative_path)
            ref_depth_image = cv2.imread(ref_depth_path, cv2.IMREAD_UNCHANGED)
            ref_depth_image = cv2.flip(ref_depth_image, 0)
            points2d_ref = sample_points_with_valid_depth(ref_depth_image, num_points=100000, max_depth=2000)
            points2d_ref_valid, point3D_from_ref, _ = get_3D_samples(points2d_ref, ref_depth_image, ref_T, rcamera)
            pose_query_origin = transform_ecef_origin(np.array(query_T), origin=origin)
            points2d_query, _, Points_3D_ECEF_origin, valid = get_points2D_ECEF_projection(pose_query_origin, qcamera, point3D_from_ref, points2d_ref_valid, use_valid = False, num_samples=20000)

            query_depth_path = os.path.join(root, depth_relative_path)
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
            if len(valid_indices) >= min_valid_threshold:
                attempt = 0
                break
        else:
            raise ValueError("经过多次尝试后，未能找到有效点数足够的参考帧！")
        true_indices_valid = true_indices[valid_indices]
        final_indices = np.array(false_indices.tolist() + true_indices_valid.tolist())

        step = len(final_indices) / num_samples
        # 根据间隔采样索引
        valid_indices_final = [final_indices[int(i * step)] for i in range(num_samples)]
        # num_points = 5000
        # valid_indices_final = np.random.choice(len(valid_indices), size=num_points, replace=False)
        points2d_query = points2d_query[valid_indices_final]
        points2d_ref = points2d_ref_valid[valid_indices_final]
        Points_3D_ECEF_origin = Points_3D_ECEF_origin[valid_indices_final]

        
        # t = np.random.choice(len(points2d_query), size=100, replace=False)
        # visualize_points_on_images(rgb_image, ref_image, 
        #                 points2d_query[t], 
        #                 points2d_ref[t])
        # t = np.random.choice(len(points2d_query_with_valid_depth[valid_indices]), size=100, replace=False)
        # visualize_points_on_images(rgb_image, ref_image, 
        #                 points2d_query_with_valid_depth[valid_indices][t], 
        #                 points2d_ref_rej[valid_indices][t])
        # visualize_points_on_images(rgb_image, ref_image, 
        #                 points2d_query_with_valid_depth[valid_indices][t], 
        #                 points2d_ref_sample[valid_indices][t])



        points3D_save_path = os.path.join(sequence_folder, 'Points3D', pose_name.split('.')[0]+'.npy')
        np.save(points3D_save_path, Points_3D_ECEF_origin)

    # Save JSON
    save_path = os.path.join(sequence_folder, output_json)
    with open(save_path, 'w') as json_file:
        json.dump(refer_info, json_file, indent=4)
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
    root = "/home/ubuntu/Documents/code/FPVLoc/datasets/Aero/"
    folder_path = "/home/ubuntu/Documents/code/FPVLoc/datasets/Aero/Query"
    ref_path = "/home/ubuntu/Documents/code/FPVLoc/datasets/Aero/Reference"
    sequence_folder = ["Jan_seq2"]  # Replace with your actual query folder path

    for seq in sequence_folder:
        seq_dirs = os.listdir(os.path.join(folder_path, seq))
        # get reference number
        ref_pose_dict = load_poses(os.path.join(ref_path, seq, 'pose.txt'))
        for folder_index, folder in enumerate(seq_dirs):
            seq_dir = os.path.join(folder_path, seq, folder)
            output_json = "refer_info.json"
            generate_refer_info(seq_dir, output_json, seq, ref_pose_dict, root)
            print(f"Refer info JSON saved to {output_json}")