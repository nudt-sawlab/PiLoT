import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import torch
import pyproj
import copy
from get_depth  import read_valid_depth
from transform import visualize_matches
from scipy.spatial.transform import Rotation as R
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
def ECEF_to_WGS84(pos):
    xpjr, ypjr, zpjr = pos
    transprojr = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        "EPSG:4326",
        always_xy=True,
    )
    lon, lat, height = transprojr.transform(xpjr, ypjr, zpjr, radians=False)
    return [lon, lat, height]  

def get_points2D_ECEF_projection(render_T, render_camera, points_3D, point2d_total = None, num_samples = 500, use_valid = True):  # points_3D[n,3]
    """
    根据相机的内参矩阵、姿态（旋转矩阵和平移向量）以及三维世界坐标，
    计算对应的二维图像坐标。

    参数:
    - R: 旋转矩阵，尺寸为 [3, 3]，表示从相机坐标系到世界坐标系的旋转。
    - t: 平移向量，尺寸为 [3, 1]，表示从相机坐标系到世界坐标系的平移。
    - K: 相机内参矩阵，尺寸为 [3, 3]，包含焦距和主点坐标。
    - points_3D: 三维世界坐标数组，尺寸为 [n, 3]，其中 n 是点的数量。
    返回:
    - point_2d: 二维图像坐标数组，尺寸为 [n, 2]。
    """
    # 将输入数据转换为高精度浮点数类型
    w, h, fx, fy, cx, cy = render_camera
    render_K = np.array([[fx, 0, cx],[0, fy, cy], [0, 0, 1]])
    points_3D = np.float64(points_3D)

    K = np.float64(render_K)
    R = render_T[:3, :3]
    t = render_T[:3, 3]
    R = np.float64(R)
    t = np.float64(t)
    t = np.tile(t, (points_3D.shape[0], 1))
    # 修改内参矩阵的最后一项，以适应透视投影
    K_c2w = np.linalg.inv(K)
    
    K_inverse = np.linalg.inv(K_c2w)
    R_inverse = np.linalg.inv(R)
    
    # 计算相机坐标系下的点
    # point_3d_camera = np.expand_dims(points_3D - t, 1)
    point_3d_camera = points_3D - t
    # 将世界坐标系下的点转换为相机坐标系下的点

    point_3d_camera_r = np.dot(R_inverse, point_3d_camera.T).T  # 使用转置计算旋转

    point_2d_homo = np.dot(K_inverse, point_3d_camera_r.T).T  # 投影到图像平面
    point_2d = point_2d_homo[:, :2] / point_2d_homo[:, 2, np.newaxis]
    # 将同质坐标转换为二维图像坐标
    
    # valid = np.logical_and(point_2d >= 0, point_2d <= (size - 1))
    if use_valid:
        valid_x = (point_2d[:, 0] >= 0) & (point_2d[:, 0] < render_width_px.numpy())
        valid_y = (point_2d[:, 1] >= 0) & (point_2d[:, 1] < render_height_px.numpy())

        # 结果是一个布尔数组，表示每个点是否有效
        valid = valid_x & valid_y

        true_indices = np.where(valid)[0]
        # sampled_indices = np.random.choice(true_indices, size=num_samples, replace=False)

        point2d_ref = point_2d[true_indices]
        points2d_query = point2d_total[true_indices]
        points_3D_ = points_3D[true_indices]
        return  point2d_ref, points2d_query, points_3D_, true_indices, point_2d_homo[:, 2, np.newaxis]
    # else:
    #     sampled_indices = np.random.choice(len(points_3D), size=num_samples, replace=False)
    #     point2d_ref = point_2d[sampled_indices]
    #     points2d_query = point2d_total[sampled_indices]
    #     points_3D_ = points_3D[sampled_indices]
    return  point_2d, point2d_total, points_3D, np.squeeze(point_2d_homo[:, 2, np.newaxis], axis=-1)
def get_3D_samples(mkpts_r, depth_mat, render_T, render_camera, origin = None, device = 'cuda'):
    # render T is in CGCS2000 format
    # in ECEF
    w, h,  fx, fy, cx, cy= render_camera
    render_K = np.array([[fx, 0, cx],[0, fy, cy], [0, 0, 1]])
    render_K = torch.tensor(render_K, device=device)
    K_c2w = render_K.inverse()
    
    render_T = torch.tensor(render_T, device=device)
    # if origin is not None:
    #     origin_tensor = torch.tensor(origin, device=device)
    #     render_T[:3, 3] -= origin_tensor  # t_c2w - origin
    # render_T[:3, 1] = -render_T[:3, 1]  # Y轴取反，投影后二维原点在左上角
    # render_T[:3, 2] = -render_T[:3, 2]  # Z轴取反
    mkpts_r = torch.tensor(mkpts_r, device=device)
    
    depth, valid = read_valid_depth(mkpts_r, depth = depth_mat, device=device)
    #!转换到OSG屏幕坐标系下反投影求3D点
    mkpts_r_in_osg = copy.deepcopy(mkpts_r[valid])

    R, t = render_T[:3, :3], render_T[:3, 3]
    
    K = K_c2w
    points = mkpts_r_in_osg
    if points.shape[-1] != 3:
        points_2D = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
        points_2D = points_2D.T
    else:
        points_2D = points.T

    # 扩展平移向量以匹配点的数量
    t = t.unsqueeze(1)  # 这相当于np.expand_dims(t, -1)
    t = t.repeat(1, points_2D.size(-1))  # 这相当于np.tile(t, points_2D.shape[-1])

    # 将所有输入转换为高精度浮点数类型
    points_2D = points_2D.double()
    K = K.double()
    R = R.double()
    depth = depth.double()
    t = t.double()

    # 计算三维世界坐标
    Points_3D = R @ (K @ (depth * points_2D)) + t
    Points_3D_ECEF = Points_3D.cpu().numpy().T

    if origin is not None:
        # 
        Points_3D_ECEF_origin = Points_3D_ECEF - np.tile(origin, (len(Points_3D_ECEF), 1))   
        return mkpts_r[valid], Points_3D_ECEF, origin, Points_3D_ECEF_origin
    
    return mkpts_r[valid], Points_3D_ECEF, origin, valid.cpu().numpy()
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
def WGS84_to_ECEF(pos):
    lon, lat, height = pos
    transprojr = pyproj.Transformer.from_crs(
        "EPSG:4326",
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        always_xy=True,
    )
    xpjr, ypjr, zpjr = transprojr.transform(lon, lat, height, radians=False)
    return [xpjr, ypjr, zpjr]

if __name__ == "__main__":
    query_files = []
    reference_rgb_path = "/mnt/sda/MapScape/query/depth/USA_seq5@8@cloudy@300-100@200/1_0.png"
    reference_depth_path = "/mnt/sda/MapScape/query/depth/USA_seq5@8@cloudy@300-100@200/1_1.png"
    query_rgb_path = "/mnt/sda/MapScape/query/depth/USA_seq5@8@cloudy@300-100@200/10_0.png"
    query_depth_path = "/mnt/sda/MapScape/query/depth/USA_seq5@8@cloudy@300-100@200/10_1.png"
    pose_txt = "/media/ubuntu/PS2000/poses/USA_seq5@8@cloudy@300-100@200.txt"
    vis_save_path = "/mnt/sda/ycb/"

    pose_dict = load_poses(pose_txt)
    # Generate pairings
    origin = None
    # ref_pose_name = reference_rgb_path.split('/')[-1].split('_')[0] +'.jpg'
    # query_pose_name = query_rgb_path.split('/')[-1].split('_')[0] +'.jpg'
    ref_pose_name = reference_rgb_path.split('/')[-1]
    query_pose_name = query_rgb_path.split('/')[-1]
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
    query_T[:3, 1] = -query_T[:3, 1]  # Y轴取反，投影后二维原点在左上角
    query_T[:3, 2] = -query_T[:3, 2]  # Z轴取反
    query_T = query_T.tolist()

    query_pose = euler_angles + translation
        # query_T_candidates = add_noise_to_pose(euler_angles, translation, query_T)

    # get query intrinscis
    # qcamera = [1600, 1200, 1931.7, 1931.7, 800.0, 600.0]
    
    # rcamera = [1600, 1200, 1931.7, 1931.7, 800.0, 600.0]
    qcamera = [960, 540, 1158.8, 1158.8, 480.0, 270.0]
    rcamera = [960, 540, 1158.8, 1158.8, 480.0, 270.0] 
    lon, lat, alt, roll, pitch, yaw = map(float, ref_pose)
    euler_angles_ref = [pitch, roll, yaw]
    translation_ref = [lon, lat, alt]
    lon, lat, _ = translation_ref
    rot_pose_in_ned = R.from_euler('xyz', euler_angles_ref, degrees=True).as_matrix()  # ZXY 东北天  
    rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
    t_c2w = WGS84_to_ECEF(translation_ref)
    ref_T = np.eye(4)
    ref_T[:3, :3] = R_c2w
    ref_T[:3, 3] = t_c2w
    ref_T[:3, 1] = -ref_T[:3, 1]  # Y轴取反，投影后二维原点在左上角
    ref_T[:3, 2] = -ref_T[:3, 2]  # Z轴取反
    ref_T = ref_T.tolist()
            

    # get query & ref rgb path/depth
    rgb_image = cv2.imread(query_rgb_path)
    ref_image = cv2.imread(reference_rgb_path)
    ref_depth_image = cv2.imread(reference_depth_path, cv2.IMREAD_UNCHANGED)
    ref_depth_image = cv2.flip(ref_depth_image, 0)
           
    num_samples = 1       
    width, height = qcamera[:2]
    ey = np.random.randint(0, height, size= num_samples)
    ex = np.random.randint(0, width, size= num_samples)
    points2d_ref = np.column_stack((ex, ey)) 
    points2d_ref = np.array([[24, 54]])   

    points2d_ref_valid, point3D_from_ref, _, _ = get_3D_samples(points2d_ref, ref_depth_image, ref_T, rcamera)
    points2d_query, _, Points_3D_ECEF_origin, valid = get_points2D_ECEF_projection(np.array(query_T), qcamera, point3D_from_ref, points2d_ref_valid, use_valid = False, num_samples=20000)
    print(point3D_from_ref[0])
    print(ECEF_to_WGS84(point3D_from_ref[0]))
    visualize_matches(rgb_image, ref_image, 
                    points2d_query, 
                    points2d_ref)
