import torch
import torch.nn.functional as F
import math
import pyproj
import os
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
import cv2
# from render2loc_tan import Render2Loc

### SETTING EXP PLACE (TODO:不同实验地点影响get_CRS和json文件选择)
def visualize_points_on_images(img1, img2, points1, points2, point_color=(0, 255, 0), radius=5, thickness=-1, save_path = None, extra = None):
    """
    Visualize points on two images side by side.

    :param img1: First image (numpy array, BGR format).
    :param img2: Second image (numpy array, BGR format).
    :param points1: List of points (x, y) for the first image.
    :param points2: List of points (x, y) for the second image.
    :param point_color: Color of the points (default green).
    :param radius: Radius of the points to draw.
    :param thickness: Thickness of the point circles (-1 means filled circle).
    :return: Combined image with visualized points.
    """
    # Make copies to avoid modifying original images
    # if torch.is_tensor(img1):
    #     # 检查 Tensor 是否在 GPU 上
    #     if img1.is_cuda:
    #         img1 = img1.cpu()  # 先将 GPU Tensor 移动到 CPU
    #     img1 = img1.numpy()
    #     if img2.is_cuda:
    #         img2 = img2.cpu() 
    #     img2 = img2.numpy()
    img1 = np.transpose(img1, (1, 2, 0))
    img2 = np.transpose(img2, (1, 2, 0))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img1_vis = img1.copy()
    img2_vis = img2.copy()
    if img1_vis.min() < 0:  # If image is in [-1, 1], normalize to [0, 255]
        img1_vis = ((img1_vis + 1) * 127.5).clip(0, 255).astype(np.uint8)
    elif img1_vis.max() <= 1:  # If image is in [0, 1], scale to [0, 255]
        img1_vis = (img1_vis * 255).clip(0, 255).astype(np.uint8)
    elif img1_vis.dtype != np.uint8:  # If not already uint8, convert
        img1_vis = img1_vis.clip(0, 255).astype(np.uint8)

    if img2_vis.min() < 0:  # If image is in [-1, 1], normalize to [0, 255]
        img2_vis = ((img2_vis + 1) * 127.5).clip(0, 255).astype(np.uint8)
    elif img2_vis.max() <= 1:  # If image is in [0, 1], scale to [0, 255]
        img2_vis = (img2_vis * 255).clip(0, 255).astype(np.uint8)
    elif img2_vis.dtype != np.uint8:  # If not already uint8, convert
        img2_vis = img2_vis.clip(0, 255).astype(np.uint8)
    if not np.isfinite(img2_vis).all():
        raise ValueError("Image contains NaN or Infinity values.")
    # Draw points on the first image
    if save_path is None:
        save_path = "/mnt/sda/MapScape/verify"
        save_name = os.path.join(save_path, str(len(os.listdir(save_path))+1)+'.png')
    else:
        if extra is not None:
            save_name = os.path.join(save_path, str(len(os.listdir(save_path))+1)+'_'+extra+'.png')
        else:
            save_name = os.path.join(save_path, str(len(os.listdir(save_path))+1)+'.png')
    
    for (x, y) in points1:
        # import ipdb; ipdb.set_trace()
        cv2.circle(img1_vis, (int(x), int(y)), radius, point_color, thickness)

    # Draw points on the second image
    for (x, y) in points2:
        cv2.circle(img2_vis, (int(x), int(y)), radius, point_color, thickness)

    # Combine the two images side by side
    combined_img = cv2.hconcat([img1_vis, img2_vis])
    print(save_name)
    cv2.imwrite(save_name, combined_img)


    return combined_img

def features_to_RGB(*Fs, skip=1):
    """Project a list of d-dimensional feature maps to RGB colors using PCA."""
    from sklearn.decomposition import PCA

    def normalize(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
    flatten = []
    shapes = []
    for F in Fs:
        c, h, w = F.shape
        F = np.rollaxis(F, 0, 3)
        F = F.reshape(-1, c)
        flatten.append(F)
        shapes.append((h, w))
    flatten = np.concatenate(flatten, axis=0)

    pca = PCA(n_components=3)
    if skip > 1:
        pca.fit(normalize(flatten[::skip]))
        flatten = normalize(pca.transform(normalize(flatten)))
    else:
        flatten = normalize(pca.fit_transform(normalize(flatten)))
    flatten = (flatten + 1) / 2

    Fs = []
    for h, w in shapes:
        F, flatten = np.split(flatten, [h*w], axis=0)
        F = F.reshape((h, w, 3))
        Fs.append(F)
    assert flatten.shape[0] == 0
    return Fs

def pixloc_to_osg(T_refined_c2w):
    R_c2w, t_c2w = T_refined_c2w[:3, :3], T_refined_c2w[:3, 3]
    t_c2w_wgs84 = ECEF_to_WGS84(t_c2w)
    lon, lat, _ = t_c2w_wgs84
    # 计算从ENU到ECEF的旋转矩阵
    rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    # 将ECEF姿态矩阵转换为ENU姿态矩阵
    rot_ecef_in_enu = rot_ned_in_ecef.T  #! ECEF TO WGS84 transformation
    # 将ECEF姿态矩阵转换为ENU姿态矩阵
    rot_pose_in_enu = np.matmul(rot_ecef_in_enu, R_c2w)
    # 从ENU姿态矩阵中提取欧拉角
    rot_pose_in_enu_obj = R.from_matrix(rot_pose_in_enu)
    euler_angles_in_enu = rot_pose_in_enu_obj.as_euler('xyz', degrees=True)
    
    R_c2w = convert_euler_to_matrix(euler_angles_in_enu)
    R_w2c = R_c2w.T
    t_w2c = np.array(-R_w2c.dot(t_c2w))  
    # R_w2c = R_c2w.T
    # t_w2c = np.array(-R_w2c.dot(t_c2w))  

    T_ECEF = np.concatenate((R_w2c, np.array([t_w2c]).transpose()), axis=1)
    
    R_c2w = R_w2c.T
    t_c2w = np.array(-R_c2w.dot(t_w2c)) 
    return euler_angles_in_enu, t_c2w_wgs84, T_ECEF
    
def colmap_to_osg(T_colmap_in_ned):
    '''
    Args:
    - T_colmap_in_ned: Transform from World(ECEF) to Camera
    Return:
    - T_ECEF: Transform from Camera to World(ECEF)
    - T_ECEF_W2C: Transform from World to Camera, 2D-3D,   s * p2d(左上角为原点) = K(Normal)@(R_ECEF_W2C @ P3D + t_ECEF_W2C)
    
    '''
    R_w2c, t_w2c = np.array(T_colmap_in_ned.R), np.array(T_colmap_in_ned.t)
    R_c2w = R_w2c.T
    t_c2w = np.array(-R_c2w.dot(t_w2c))  
    t_c2w_wgs84 = ECEF_to_WGS84(t_c2w)
    lon, lat, _ = t_c2w_wgs84
    # 计算从ENU到ECEF的旋转矩阵
    rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    # 将ECEF姿态矩阵转换为ENU姿态矩阵
    rot_ecef_in_enu = rot_ned_in_ecef.T  #! ECEF TO WGS84 transformation
    # 将ECEF姿态矩阵转换为ENU姿态矩阵
    rot_pose_in_enu = np.matmul(rot_ecef_in_enu, R_c2w)
    # 从ENU姿态矩阵中提取欧拉角
    rot_pose_in_enu_obj = R.from_matrix(rot_pose_in_enu)
    euler_angles_in_enu = rot_pose_in_enu_obj.as_euler('xyz', degrees=True)
    

    
    T_ECEF = np.concatenate((R_c2w, np.array([t_c2w]).transpose()), axis=1)
    T_ECEF = np.concatenate((T_ECEF, np.array([[0, 0, 0, 1]])), axis=0)   
    
    # render_T_pnp = copy.deepcopy(R_c2w)
    # render_T_pnp[:3, 1] = -render_T_pnp[:3, 1]  # screen coordinate problem
    # render_T_pnp[:3, 2] = -render_T_pnp[:3, 2]  # depth problem
    # R_w2c = render_T_pnp.T
    # t_w2c = -R_w2c @ t_c2w
    
    
    # T_ECEF_W2C = np.concatenate((R_w2c, np.array([t_w2c]).transpose()), axis=1)
    # T_ECEF_W2C = np.concatenate((T_ECEF_W2C, np.array([[0, 0, 0, 1]])), axis=0)   
    T_ECEF_W2C = None
    return euler_angles_in_enu, t_c2w_wgs84, T_ECEF, T_ECEF_W2C
def colmap_to_osg_bak2(T_colmap_in_ned):
    # colmap_T_init = (np.array([0.8820744480611885, -0.27987676553527197, -0.1105136880498206 ,-0.36249191569563494 ]),
    #              np.array([-2482099.8221656354, -1586322.2628436063, 1135492.5201693343]))
    # R_w2c_colmap, t_w2c_colmap = qvec2rotmat(colmap_T_init[0]), colmap_T_init[1]
    # R_w2c_colmap = R_w2c_colmap.T
    # q_c2w_colmap = rotmat2qvec(R_w2c_colmap)
    # euler_angles_in_ned_colmap = convert_quaternion_to_euler(q_c2w_colmap)
    # euler_angles_colmap = [euler_angles_in_ned_colmap[0], -euler_angles_in_ned_colmap[1], -euler_angles_in_ned_colmap[2]]
    
    R_w2c, t_w2c = np.array(T_colmap_in_ned.R), np.array(T_colmap_in_ned.t)
    T_CGCS2000 = np.concatenate((R_w2c, np.array([t_w2c]).transpose()), axis=1)
    T_CGCS2000 = np.concatenate((T_CGCS2000, np.array([[0, 0, 0, 1]])), axis=0)    
    
    R_c2w = R_w2c.T
    q_c2w = rotmat2qvec(R_c2w)
    euler_angles_in_ned = convert_quaternion_to_euler(q_c2w)
    euler_angles = [euler_angles_in_ned[0] + 180, euler_angles_in_ned[1], euler_angles_in_ned[2]]
    R_c2w = np.asmatrix(R_w2c).transpose()  
    t_c2w = np.array(-R_c2w.dot(t_w2c))  
    # Convert coordinates from CGCS2000 to WGS84.get_query_intrinsic
    t_c2w = cgcs2000towgs84(t_c2w, 0)
    
    lon, lat, _ = t_c2w
    rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    rot_pose_in_enu = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天
    
    # 计算最终的旋转矩阵
    r = np.matmul(rot_enu_in_ecef, rot_pose_in_enu)
    
    # 将经纬度转换为ECEF坐标系下的点
    xyz = WGS84_to_ECEF(t_c2w)
    
    # 创建变换矩阵T，将旋转矩阵和平移向量合并
    T_ECEF = np.concatenate((r, np.array([xyz]).transpose()), axis=1)
    T_ECEF = np.concatenate((T_ECEF, np.array([[0, 0, 0, 1]])), axis=0)    
    
    r_w2c = r.T
    xyz_w2c = -r_w2c @ xyz
    T_ECEF_W2C = np.concatenate((r_w2c, np.array([xyz_w2c]).transpose()), axis=1)
    T_ECEF_W2C = np.concatenate((T_ECEF_W2C, np.array([[0, 0, 0, 1]])), axis=0)   
    
    return euler_angles, t_c2w, T_ECEF, T_ECEF_W2C
def colmap_to_osg_bak(T_colmap_in_ned):
    # input NED colmap format
    R_w2c, t_w2c = np.array(T_colmap_in_ned.R), np.array(T_colmap_in_ned.t)
    T_CGCS2000 = np.concatenate((R_w2c, np.array([t_w2c]).transpose()), axis=1)
    T_CGCS2000 = np.concatenate((T_CGCS2000, np.array([[0, 0, 0, 1]])), axis=0)    
    
    R_c2w = R_w2c.T
    q_c2w = rotmat2qvec(R_c2w)
    euler_angles_in_ned = convert_quaternion_to_euler(q_c2w)
    euler_angles = [euler_angles_in_ned[0], -euler_angles_in_ned[1], -euler_angles_in_ned[2]]
    R_c2w = np.asmatrix(R_w2c).transpose()  
    t_c2w = np.array(-R_c2w.dot(t_w2c))  
    # Convert coordinates from CGCS2000 to WGS84.get_query_intrinsic
    t_c2w = cgcs2000towgs84(t_c2w, 0)
    
    lon, lat, _ = t_c2w
    rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    rot_pose_in_enu = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天
    
    # 计算最终的旋转矩阵
    r = np.matmul(rot_enu_in_ecef, rot_pose_in_enu)
    
    # 将经纬度转换为ECEF坐标系下的点
    xyz = WGS84_to_ECEF(t_c2w)
    
    # 创建变换矩阵T，将旋转矩阵和平移向量合并
    T_ECEF = np.concatenate((r, np.array([xyz]).transpose()), axis=1)
    T_ECEF = np.concatenate((T_ECEF, np.array([[0, 0, 0, 1]])), axis=0)    
    
    r_w2c = r.T
    xyz_w2c = -r_w2c @ xyz
    T_ECEF_W2C = np.concatenate((r_w2c, np.array([xyz_w2c]).transpose()), axis=1)
    T_ECEF_W2C = np.concatenate((T_ECEF_W2C, np.array([[0, 0, 0, 1]])), axis=0)   
    
    return euler_angles, t_c2w, T_ECEF, T_ECEF_W2C
def euler_to_rotation(euler_angles, translation, origin = None):
    t_c2w = WGS84_to_ECEF(translation)
    lon, lat, alt = translation
    rot_pose_in_ned = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
    rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_enu_in_ecef, rot_pose_in_ned)
    # Initialize a 4x4 identity matrix
    T_c2w = np.eye(4)
    T_c2w[:3, :3] = R_c2w
    T_c2w[:3, 3] = t_c2w
    if origin is not None:
        T_c2w[:3, 1] = -T_c2w[:3, 1]
        T_c2w[:3, 2] = -T_c2w[:3, 2]
        T_c2w[:3, 3] -= origin  # 加上 origin 偏移量
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = T_c2w[:3, :3].T
    T_w2c[:3, 3] = - T_w2c[:3, :3] @ T_c2w[:3, 3]
    import ipdb; ipdb.set_trace()
    return T_w2c    
def get_c2w(T_candidtas, origin):
    B = T_candidtas.shape[0]
    T_opt_c2w = torch.eye(4, dtype=torch.float64).repeat(B, 1, 1).to(T_candidtas.device)
    # 提取旋转矩阵和位移向量
    R_w2c = T_candidtas.R.cpu().numpy().astype(np.float64)  # [B, 3, 3]
    t_w2c = T_candidtas.t.cpu().numpy().astype(np.float64)  # [B, 3]
    # 对旋转矩阵进行正交化处理
    for i in range(B):
        R_w2c[i] = orthogonalize_rotation_matrix(R_w2c[i])
    # 更新旋转矩阵和位移
    T_opt_c2w[:, :3, :3] = torch.tensor(R_w2c).transpose(-1, -2)  # 转置旋转矩阵
    T_opt_c2w[:, :3, 3] = -torch.bmm(torch.tensor(R_w2c).transpose(-1, -2), torch.tensor(t_w2c).unsqueeze(-1)).squeeze(-1)

    # 对 T_opt_c2w 进行调整
    T_opt_c2w[:, :3, 1] = -T_opt_c2w[:, :3, 1]
    T_opt_c2w[:, :3, 2] = -T_opt_c2w[:, :3, 2]
    T_opt_c2w[:, :3, 3] += origin  # 加上 origin 偏移量

    R_c2w, t_c2w = T_opt_c2w[0, :3, :3], T_opt_c2w[0, :3, 3]

    t_c2w_wgs84 = ECEF_to_WGS84(t_c2w.cpu().numpy())
    lon, lat, _ = t_c2w_wgs84
    # 计算从ENU到ECEF的旋转矩阵
    rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    # 将ECEF姿态矩阵转换为ENU姿态矩阵
    rot_ecef_in_enu = rot_enu_in_ecef.T  #! ECEF TO WGS84 transformation
    # 将ECEF姿态矩阵转换为ENU姿态矩阵
    rot_pose_in_enu = np.matmul(rot_ecef_in_enu, R_c2w.cpu().numpy())

    # 从ENU姿态矩阵中提取欧拉角
    rot_pose_in_enu_obj = R.from_matrix(rot_pose_in_enu)
    euler_angles_in_enu = rot_pose_in_enu_obj.as_euler('xyz', degrees=True)

    print("e, t:", euler_angles_in_enu, t_c2w_wgs84, t_c2w)
    
    return T_opt_c2w
def decimal_to_dms(decimal):
    """
    Convert decimal degrees to degrees, minutes and seconds.
    
    Args:
        decimal (float): The decimal degrees value to convert.

    Returns:
        (int, int, float): A tuple containing degrees, minutes, and seconds.
    """
    # Convert decimal to degrees and the remaining fraction
    degrees = int(decimal)
    fraction = decimal - degrees
    
    # Convert the fraction to minutes
    minutes_full = fraction * 60
    minutes = int(minutes_full)
    # The remaining fraction becomes seconds
    seconds = (minutes_full - minutes) * 60
    
    return degrees, minutes, seconds

def dms_to_string(degrees, minutes, seconds, direction):
    """
    Format the degrees, minutes, and seconds into a DMS string.
    
    Args:
        degrees (int): The degrees part of the DMS.
        minutes (int): The minutes part of the DMS.
        seconds (float): The seconds part of the DMS.
        direction (str): The direction (N/S/E/W).

    Returns:
        str: A string representing the DMS in format "D°M'S".
    """
    # Format seconds to ensure it has three decimal places
    seconds = round(seconds, 3)
    # Create the DMS string
    dms_string = f"{degrees}°{minutes}'{seconds}\" {direction}"
    return dms_string


def convert_quaternion_to_euler(q_c2w):
    """
    Convert a quaternion to Euler angles in 'xyz' order, with angles in degrees.
    
    Parameters:
    qvec (numpy.ndarray): The quaternion vector as [x, y, z, w].
    R (numpy.matrix, optional): If provided, this rotation matrix will be used
                                 instead of calculating it from the quaternion.
    
    Returns:
    numpy.ndarray: The Euler angles in 'xyz' order in degrees.
    """
    # Convert the quaternion in COLMAP format [QW, QX, QY, QZ] 
    # to the correct order for scipy's Rotation object: [x, y, z, w]
    q_xyzw = [np.float64(q_c2w[1]), np.float64(q_c2w[2]), np.float64(q_c2w[3]), np.float64(q_c2w[0])] 
    
    # Create a Rotation object from the quaternion
    ret = R.from_quat(q_xyzw)
    
    # Convert the quaternion to Euler angles in 'xyz' order, with angles in degrees
    euler_xyz = ret.as_euler('xyz', degrees=True)    
    euler_xyz[0] = euler_xyz[0] 
    # TODO
    return list(euler_xyz)

def convert_euler_to_matrix(euler_xyz):
    """
    Convert Euler angles in 'xyz' order to a rotation matrix.
    
    Parameters:
    euler_xyz (list or numpy.ndarray): The Euler angles in 'xyz' order in degrees.
    
    Returns:
    numpy.ndarray: The rotation matrix as a 3x3 numpy array.
    """
    # Convert the Euler angles from degrees to radians
    euler_xyz_rad = np.radians(euler_xyz)
    
    # Create a Rotation object from the Euler angles
    rotation = R.from_euler('xyz', euler_xyz_rad)
    
    # Convert the rotation to a matrix
    rotation_matrix = rotation.as_matrix()
    
    return rotation_matrix

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

def get_CRS(exp_place):
    wgs84 = pyproj.CRS('EPSG:4326')
    if exp_place == PLACE_CHANGSHA:
        cgcs2000 = pyproj.CRS('EPSG:4547')  # 翡翠湾
    else:
        cgcs2000 = pyproj.CRS('EPSG:4529')  # 宁波
    return wgs84, cgcs2000

def cgcs2000towgs84(c2w_t, exp_place):
    """Convert coordinates from CGCS2000 to WGS84.
    
    Args:
        c2w_t (list): [x, y, z] in CGCS2000 format
    """
    x, y = c2w_t[0][0], c2w_t[0][1]
    wgs84, cgcs2000 = get_CRS(exp_place)
    
    
    transformer = pyproj.Transformer.from_crs(cgcs2000, wgs84, always_xy=True)
    lon, lat = transformer.transform(x, y)
    height = c2w_t[0][2]
    return [lon, lat, height]

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
def orthogonalize_rotation_matrix(R):
    """确保旋转矩阵是正交的，使用SVD修正误差"""
    U, _, Vt = np.linalg.svd(R)
    R_orth = np.dot(U, Vt)
    return R_orth
  
def wgs84tocgcs2000(trans, exp_place):
    """Convert coordinates from WGS84 to CGCS2000.
    
    Args:
        trans (list): [lon, lat, height] in WGS84 format
    """
    lon, lat, height = trans  # Unpack the WGS84 coordinates
    
    wgs84, cgcs2000 = get_CRS(exp_place)
    
    # Create a transformer from WGS84 to CGCS2000
    transformer = pyproj.Transformer.from_crs(wgs84, cgcs2000, always_xy=True)
    
    # Perform the transformation
    x, y = transformer.transform(lon, lat)
    
    # Return the transformed coordinates as a list
    return [x, y, height]  # Keep the original height from WGS84        
        
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec
def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])
def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))
def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")
def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])
def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)
#!wxyz
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
#!wxyz
def compute_pixel_focal(sensorWidth, sensorHeight, focallength, imgWidth, imgHeight):
    
    # factor = np.sqrt(36*36+24*24)
    # sensorSize = np.sqrt(np.square(sensorWidth) + np.square(sensorHeight))
    # eq_length = factor * (focallength/sensorSize)
    pixelSizeW = sensorWidth/imgWidth
    pixelSizeH = sensorHeight/imgHeight
    fx = focallength / pixelSizeW
    fy = focallength / pixelSizeH
    
    return fx, fy
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles
def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

def torch_image_to_numpy(tensor):
    """
    Normalize the image tensor and reorder the dimensions for NumPy.
    
    Parameters:
    - tensor: PyTorch Tensor of shape (C, H, W) or (H, W).
    
    Returns:
    - NumPy array of shape (H, W, C) or (H, W) normalized to [0, 1].
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input must be a PyTorch Tensor.")
    
    # Ensure the tensor is on CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Convert to NumPy array
    array = tensor.numpy()
    
    if array.ndim == 3:  # Convert CxHxW to HxWxC
        array = array.transpose((1, 2, 0))  # CxHxW -> HxWxC
    elif array.ndim == 2:
        pass  # HxW stays the same
    else:
        raise ValueError(f"Not an image tensor: {tensor.shape}")
    
    return (array * 255).astype(np.uint8)  # Denormalize to 0-255 and cast to uint8


def get_rotation_enu_in_ecef_batch(lon, lat):
    lon = np.atleast_1d(lon)
    lat = np.atleast_1d(lat)
    lon_rad, lat_rad = np.radians(lon), np.radians(lat)
    cos_lon, sin_lon = np.cos(lon_rad), np.sin(lon_rad)
    cos_lat, sin_lat = np.cos(lat_rad), np.sin(lat_rad)
    up = np.stack([cos_lon * cos_lat, sin_lon * cos_lat, sin_lat], axis=1)
    east = np.stack([-sin_lon, cos_lon, np.zeros_like(lon)], axis=1)
    north = np.cross(up, east)
    Rm = np.zeros((lon.shape[0], 3, 3))
    Rm[:, :, 0], Rm[:, :, 1], Rm[:, :, 2] = east, north, up
    return Rm


def get_matrix_list_batch(translation_array, euler_angles_array, origin=(0, 0, 0)):
    rot_pose = R.from_euler('xyz', euler_angles_array, degrees=True).as_matrix()
    rot_enu = get_rotation_enu_in_ecef_batch(translation_array[:, 0], translation_array[:, 1])
    R_c2w = np.matmul(rot_enu, rot_pose)
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'}, always_xy=True)
    xs, ys, zs = transformer.transform(
        translation_array[:, 0], translation_array[:, 1], translation_array[:, 2])
    t_c2w = np.stack([xs, ys, zs], axis=1)
    N = translation_array.shape[0]
    T = np.zeros((N, 4, 4))
    T[:, 3, 3] = 1.0
    T[:, :3, :3] = R_c2w
    T[:, :3, 3] = t_c2w - np.array(origin)
    T[:, :3, 1] *= -1
    T[:, :3, 2] *= -1
    return T