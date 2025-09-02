import torch
import torch.nn.functional as F
import math
import pyproj
import os
import numpy as np
from pykalman import KalmanFilter
from scipy.spatial.transform import Rotation as R
from ..pixlib.geometry import Pose, Camera
import copy
import matplotlib.pyplot as plt
import cv2
# from render2loc_tan import Render2Loc

### SETTING EXP PLACE (TODO:不同实验地点影响get_CRS和json文件选择)
PLACE_CHANGSHA = 0 
def visualize_points_on_images(img1, img2, points1, points2, point_color=(0, 255, 0), radius=5, thickness=-1):
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
    img1_vis = img1.copy()
    img2_vis = img2.copy()

    # Draw points on the first image
    for (x, y) in points1:
        cv2.circle(img1_vis, (int(x), int(y)), radius, point_color, thickness)

    # Draw points on the second image
    for (x, y) in points2:
        cv2.circle(img2_vis, (int(x), int(y)), radius, point_color, thickness)

    # Combine the two images side by side
    combined_img = cv2.hconcat([img1_vis, img2_vis])
    save_path = "/mnt/sda/MapScape/query/estimation/matches/FPVLoc"
    save_name = os.path.join(save_path, str(len(os.listdir(save_path))+1)+'.png')
    cv2.imwrite(save_name, combined_img)

    return combined_img
def move_inputs_to_cuda(*args, device=None):
        """
        递归地将输入参数中的所有 Tensor、Pose、Camera（如果定义了 .to() 方法）转到 CUDA。
        返回一个 tuple，顺序与 args 保持一致。
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def move(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            elif hasattr(x, 'to') and callable(getattr(x, 'to')):  # 支持 Pose、Camera 等
                return x.to(device)
            elif isinstance(x, (list, tuple)):
                return type(x)(move(i) for i in x)
            elif isinstance(x, dict):
                return {k: move(v) for k, v in x.items()}
            else:
                return x

        return tuple(move(arg) for arg in args)
def euler_angles_to_matrix_ECEF(euler_angles, trans):
    lon, lat, _ = trans
    rot_pose_in_enu = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
    rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_enu_in_ecef, rot_pose_in_enu)
    # q_w2c = rotmat2qvec(R_c2w.transpose())  # return wxyz (colmap pnp return xyzw)
    t_c2w = WGS84_to_ECEF(trans)
    # R_w2c_in_ecef = R_c2w.transpose() # 和enu的差异是第二行和第三行取负号
    # t_w2c = -R_w2c_in_ecef.dot(t_c2w)

    T_render_in_ECEF_c2w = np.eye(4)
    T_render_in_ECEF_c2w[:3, :3] = R_c2w
    T_render_in_ECEF_c2w[:3, 3] = t_c2w
    return T_render_in_ECEF_c2w
def euler_angles_to_matrix_ECEF_batch_speical(euler_angles_list, translation):
    lon, lat, _ = translation
    rot_pose_in_enu = R.from_euler('xyz', euler_angles_list, degrees=True).as_matrix()  # ZXY 东北天  
    rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_enu_in_ecef, rot_pose_in_enu)
    t_c2w = WGS84_to_ECEF(translation)
    t_c2w = np.array(t_c2w)
    N = len(euler_angles_list)
    t_c2w_batch = np.tile(t_c2w, (N, 1))  # shape (N, 3)
    T_render_in_ECEF_c2w = np.tile(np.eye(4), (N, 1, 1))  # shape (N, 4, 4)
    T_render_in_ECEF_c2w[:, :3, :3] = R_c2w
    T_render_in_ECEF_c2w[:, :3, 3] = t_c2w_batch
    return T_render_in_ECEF_c2w

def orthogonalize_rotation_matrix(R):
    """确保旋转矩阵是正交的，使用SVD修正误差"""
    U, _, Vt = np.linalg.svd(R)
    R_orth = np.dot(U, Vt)
    return R_orth
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
    
    R_w2c = R_c2w.T
    t_w2c = np.array(-R_w2c.dot(t_c2w))  

    T_ECEF = np.concatenate((R_w2c, np.array([t_w2c]).transpose()), axis=1)

    # R_w2c = T_ECEF[:3, :3]
    # t_w2c = T_ECEF[:3, 3]
    # R_c2ww = R_w2c.T
    # t_c2ww = np.array(-R_c2ww.dot(t_w2c))  
    # t_c2w_wgs84 = ECEF_to_WGS84(t_c2ww)
    # lon, lat, _ = t_c2w_wgs84
    # # 计算从ENU到ECEF的旋转矩阵
    # rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    # # 将ECEF姿态矩阵转换为ENU姿态矩阵
    # rot_ecef_in_enu = rot_ned_in_ecef.T  #! ECEF TO WGS84 transformation
    # # 将ECEF姿态矩阵转换为ENU姿态矩阵
    # rot_pose_in_enu = np.matmul(rot_ecef_in_enu, R_c2ww)
    # # 从ENU姿态矩阵中提取欧拉角
    # rot_pose_in_enu_obj = R.from_matrix(rot_pose_in_enu)
    # euler_angles_in_enu = rot_pose_in_enu_obj.as_euler('xyz', degrees=True)

    # R_c2w = R_w2c.T
    # t_c2w = np.array(-R_c2w.dot(t_w2c)) 
    # kf_current_frame_es_pose = t_c2ww+euler_angles_in_enu
    kf_current_frame_es_pose = np.concatenate((t_c2w, euler_angles_in_enu))
    return euler_angles_in_enu, t_c2w_wgs84, T_ECEF, kf_current_frame_es_pose
def plot_kf_trajectory_save(filtered_states, predicted_pose, save_path="kf_prediction_plot.png"):
    """
    绘制并保存6个状态量的卡尔曼滤波历史轨迹和预测值（横坐标为帧编号，预测为最后一个点）

    Args:
        filtered_states: numpy array of shape (N, 12)
        predicted_pose: numpy array of shape (6,)
        save_path: str, 图片保存路径
    """
    labels = ['x', 'y', 'z', 'pitch', 'roll', 'yaw']
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    axes = axes.flatten()

    N = filtered_states.shape[0]
    x_hist = np.arange(N)
    x_pred = N  # 预测值在最后一个位置

    for i in range(6):
        y_hist = filtered_states[:, i]
        y_pred = predicted_pose[i]

        # 历史轨迹：蓝色线 + 圆点
        axes[i].plot(x_hist, y_hist, 'b-o', label='Filtered (history)')
        
        # 预测值：红色星号
        axes[i].plot(x_pred, y_pred, 'r*', markersize=10, label='Predicted')

        # 辅助线：灰色虚线
        axes[i].axvline(N - 1, color='gray', linestyle='--', label='Last Obs')

        axes[i].set_title(labels[i])
        axes[i].grid(True)
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 已保存预测轨迹图到: {save_path}")
def plot_kf_trajectory_save_dev(filtered_states, predicted_pose, save_path="kf_prediction_plot.png"):
    """
    绘制并保存6个状态量的卡尔曼滤波历史轨迹 + 预测值曲线图

    Args:
        filtered_states: numpy array of shape (N, 12)
        predicted_pose: numpy array of shape (6,)
        save_path: str, 图片保存路径
    """
    labels = ['x', 'y', 'z', 'pitch', 'roll', 'yaw']
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(6):
        axes[i].plot(filtered_states[:, i], label='Filtered')
        axes[i].axhline(predicted_pose[i], color='r', linestyle='--', label='Predicted')
        axes[i].set_title(labels[i])
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()

    # 确保目录存在
    plt.savefig(save_path)
    plt.close()
    print(f"已保存预测轨迹图到: {save_path}")
def visualize_prediction(observations, most_likely_pose, candidates,save_path="kf_prediction_plot.png"):
    observations = np.array(observations)
    most_likely_pose = np.array(most_likely_pose)
    candidates = np.array(candidates)

    t = np.arange(len(observations))  # 时间步
    labels = ['x', 'y', 'z', 'pitch', 'roll', 'yaw']

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ---- 第一部分：位置 x, y, z ----
    for i in range(3):
        axes[0].plot(t, observations[:, i], label=f'{labels[i]}', linewidth=2)

    # most_likely_pose 位置
    for i in range(3):
        axes[0].scatter(t[-1], most_likely_pose[i], 
                    color='red', marker='*', s=150, label='Most Likely Pose' if i == 0 else "")

    # candidates 位置点云
    # candidates 位置点云：分别画 x, y, z
    for i in range(3):
        axes[0].scatter([t[-1]] * len(candidates), candidates[:, i], 
                        color='gray', alpha=0.2, s=10, label='Candidates' if i == 0 else "")

    axes[0].set_ylabel("Position (m)")
    axes[0].legend()
    axes[0].set_title("Position Prediction")

    # ---- 第二部分：角度 pitch, roll, yaw ----
    for i in range(3, 6):
        axes[1].plot(t, observations[:, i], label=f'{labels[i]}', linewidth=2)

    for i in range(3, 6):
        axes[1].scatter(t[-1], most_likely_pose[i], 
                        color='red', marker='*', s=150, label='Most Likely Pose' if i == 3 else "")

    # candidates 旋转角：分别画 pitch, roll, yaw
    for i in range(3, 6):
        axes[1].scatter([t[-1]] * len(candidates), candidates[:, i], 
                        color='gray', alpha=0.2, s=10, label='Candidates' if i == 3 else "")

    axes[1].set_ylabel("Angle (deg)")
    axes[1].set_xlabel("Time step")
    axes[1].legend()
    axes[1].set_title("Rotation Prediction")

    plt.tight_layout()

    # 确保目录存在
    plt.savefig(save_path)
    print(f"已保存预测轨迹图到: {save_path}")
    plt.close()
def kf_predictor(observations, num_candidates = 128):
    # 初始化卡尔曼滤波器
    # 计算变化量 Δp 和 Δr
    observations = np.array(observations)
    assert observations.shape[1] == 6, "Expected input shape: (N, 6)"


    dt = 1.0  # 默认时间步长

    # 状态：位置+欧拉角+线速度+角速度 共12维
    # [x, y, z, pitch, roll, yaw, vx, vy, vz, v_pitch, v_roll, v_yaw]
    A = np.eye(12)
    for i in range(6):
        A[i, i + 6] = dt  # 添加速度分量影响

    H = np.hstack([np.eye(6), np.zeros((6, 6))])  # 观测只包含位置和姿态

    kf = KalmanFilter(
        transition_matrices=A,
        observation_matrices=H,
        initial_state_mean=np.zeros(12),
        initial_state_covariance=np.eye(12) * 0.1,
        transition_covariance=np.eye(12) * 0.05,
        observation_covariance=np.eye(6) * 0.1
    )

    # 初始化
    state_mean = np.zeros(12)
    state_mean[:6] = observations[0]
    state_cov = np.eye(12) * 0.1

    filtered_state_means = []
    filtered_state_covs = []
    for obs in observations:
        state_mean, state_cov = kf.filter_update(state_mean, state_cov, observation=obs)
        filtered_state_means.append(state_mean)
        filtered_state_covs.append(state_cov)
    # 取最后一步进行预测
    last_mean = filtered_state_means[-1]
    # last_cov = filtered_state_covs[-1]
    # 从预测分布中生成 100 个候选位姿
    
    # 预测下一时刻的状态
    next_state_mean = A @ last_mean
    # next_state_cov = A @ last_cov @ A.T + kf.transition_covariance

    # 从预测分布采样候选位姿
    # candidates = np.random.multivariate_normal(next_state_mean, next_state_cov, num_candidates)[:, :6]
    most_likely_pose = next_state_mean[:6]  # [x, y, z, pitch, roll, yaw]


    euler = most_likely_pose[3:]
    t_c2w_ecef = most_likely_pose[:3]
    # rot_pose_in_enu = R.from_euler('xyz', euler, degrees=True).as_matrix()
    t_c2w_wgs84 = ECEF_to_WGS84(t_c2w_ecef)
    # lon, lat, _ = t_c2w_wgs84

    # new_origin = t_c2w_ecef

    # rot_enu_to_ecef = get_rotation_enu_in_ecef(lon, lat)  # 假设函数返回 (3, 3) 矩阵
    # R_c2w = np.matmul(rot_enu_to_ecef, rot_pose_in_enu)
    
    # most_likely_pose_T_w2c = np.eye(4)
    # most_likely_pose_T_w2c[:3, :3] = R_c2w.T
    # most_likely_pose_T_w2c[:3, 3] = -R_c2w.T @ t_c2w_ecef
    
    # T_next_frame = Pose.from_Rt(most_likely_pose_T_w2c[:3, :3], most_likely_pose_T_w2c[:3, 3])  # w2c
    
    # 候选位姿构建
    # candidate_delta_poses = candidates - observations[-1, :]
    # translations = candidate_delta_poses[:, :3]
    # euler_angles = candidate_delta_poses[:, 3:]

    # 欧拉角转旋转向量
    # rotations = R.from_euler('xyz', euler_angles, degrees=False)
    # rotation_vectors = rotations.as_rotvec()

    # 构建 torch Tensor
    # translations_tensor = torch.tensor(translations, dtype=torch.float32)
    # directions = F.normalize(translations_tensor, dim=-1)
    # translation_lengths = torch.norm(translations_tensor, dim=-1, keepdim=True)
    # random_t = directions * translation_lengths

    # random_aa = torch.tensor(rotation_vectors, dtype=torch.float32)
    # random_pose = Pose.from_aa(random_aa, random_t)

    # 更新观测队列
    observations = np.vstack([observations, most_likely_pose])
    observations = observations[1:, :]
    filtered_state_means = np.array(filtered_state_means)  # (N, 12)
    # plot_kf_trajectory_save(filtered_state_means, most_likely_pose)
    # visualize_prediction(filtered_state_means, most_likely_pose, candidates)

    return observations, euler, t_c2w_wgs84

def kf_predictor_dev(observations, origin, num_candidates = 128):
    # 初始化卡尔曼滤波器
    kf = KalmanFilter(
        initial_state_mean=np.zeros(12),  # [x, y, z, phi, theta, psi, vx, vy, vz, omega_phi, omega_theta, omega_psi]
        initial_state_covariance=np.eye(12),
        transition_matrices=np.eye(12) + np.diag([1, 1, 1, 1, 1, 1], k=6),  # 状态转移矩阵
        observation_matrices=np.hstack([np.eye(6), np.zeros((6, 6))])  # 观测矩阵
    )

    kf = kf.em(observations, n_iter=10)  # 估计模型参数

    # 滤波和平滑
    filtered_state_means, filtered_state_covariances = kf.filter(observations)

    # 获取最后状态的预测均值和协方差
    last_mean = filtered_state_means[-1]
    last_covariance = filtered_state_covariances[-1]

    # 从预测分布中生成 100 个候选位姿
    
    candidates = np.random.multivariate_normal(last_mean, last_covariance, num_candidates)

    # 提取下一帧参考图位姿 ,most_likely_pose w2c Pose.from_Rt
    most_likely_pose = last_mean[:6]  # 提取位姿部分（例如 [x, y, z, φ, θ, ψ]）
    euler = most_likely_pose[3:]
    t_c2w_ecef = most_likely_pose[:3]
    rot_pose_in_enu = R.from_euler('xyz', euler, degrees=True).as_matrix()
    t_c2w_wgs84 = ECEF_to_WGS84(t_c2w_ecef)
    lon, lat, _ = t_c2w_wgs84

    rot_enu_to_ecef = get_rotation_enu_in_ecef(lon, lat)  # 假设函数返回 (3, 3) 矩阵
    R_c2w = np.matmul(rot_enu_to_ecef, rot_pose_in_enu)
    
    most_likely_pose_T_w2c = np.eye(4)
    most_likely_pose_T_w2c[:3, :3] = R_c2w.T
    most_likely_pose_T_w2c[:3, :3] = -R_c2w.T@ t_c2w_ecef
    
    T_next_frame = Pose.from_Rt(most_likely_pose_T_w2c[:3, :3], most_likely_pose_T_w2c[:3, 3])  # w2c
    
    # 提取位姿候选
    # candidate_poses = candidates[:, :6]  # 取位置和欧拉角部分 [x, y, z, phi, theta, psi]
    # candidate_poses_euler = candidate_poses[:, 3:]
    # rot_pose_in_enu = R.from_euler('xyz', candidate_poses_euler, degrees=True).as_matrix() 
    # noisy_R_c2w = np.einsum('ij,njk->nik', rot_enu_to_ecef, rot_pose_in_enu) #! rot_enu_to_ecef不严格准确
    # noisy_t_c2w = candidate_poses[:, :3] - origin

    # noise_T_c2w = np.eye(4).repeat(num_candidates, 1, 1)
    # noise_T_c2w[:, :3, :3] = noisy_R_c2w
    # noise_T_c2w[:, :3, 3] = noisy_t_c2w
    # noise_T_c2w[:, :3, 3] -= origin  # t_c2w - origin
    # noise_T_c2w[:, :3, 1] = -noise_T_c2w[:, :3, 1]  # Y轴取反，投影后二维原点在左上角
    # noise_T_c2w[:, :3, 2] = -noise_T_c2w[:, :3, 2]  # Z轴取反



    # noise_T_w2c = np.eye(4).repeat(num_candidates, 1, 1)
    # noise_T_w2c[:, :3, :3] = noise_T_c2w[:, :3, :3].T
    # noise_T_w2c[:, :3, :3] = -noise_T_c2w[:, :3, :3].T@ noise_T_c2w[:, :3, 3]
    # T_candidates = Pose.from_Rt(noise_T_w2c[:, :3, :3], noise_T_w2c[:, :3, 3])  # w2c

    # 更新先验位姿信息
    observations = np.vstack([observations, most_likely_pose])  # 添加新观测
    observations = observations[1:,:]

    return observations, T_candidates, T_next_frame
def resize(image, size, fn=None, interp='linear'):
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h*scale)), int(round(w*scale))
        # TODO: we should probably recompute the scale like in the second case
        scale = (scale, scale)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f'Incorrect new size: {size}')
    mode = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST}[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale
def zero_pad(size, image):
    
    h, w = image.shape[:2]
    # import ipdb; ipdb.set_trace();
    padded = np.zeros((size, size)+image.shape[2:], dtype=image.dtype)
    padded[:h, :w] = image
        
    return padded
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
def WGS84_to_ECEF_tensor(pos, device='cuda', dtype=torch.float32):
    """
    Convert WGS84 (lon, lat, height) → ECEF (x, y, z) using PyTorch on GPU.
    
    Args:
        pos: Iterable of 3 elements or Tensor of shape (..., 3),
             where last dim = [lon_deg, lat_deg, height_m].
        device: 'cuda' or 'cpu'
        dtype: torch.float32 or torch.float64
        
    Returns:
        Tensor of shape (..., 3) = [X, Y, Z] in meters
    """
    # WGS84 constants
    a = torch.tensor(6378137.0, device=device, dtype=dtype)          # semi-major axis
    f = torch.tensor(1 / 298.257223563, device=device, dtype=dtype)  # flattening
    e2 = f * (2 - f)                                                  # first eccentricity squared

    # Pack input into tensor
    p = torch.as_tensor(pos, device=device, dtype=dtype)
    lon_deg = p[..., 0]
    lat_deg = p[..., 1]
    h       = p[..., 2]

    # to radians
    lon = lon_deg * (torch.pi / 180.0)
    lat = lat_deg * (torch.pi / 180.0)

    sin_lat = torch.sin(lat)
    cos_lat = torch.cos(lat)
    cos_lon = torch.cos(lon)
    sin_lon = torch.sin(lon)

    # Radius of curvature in the prime vertical
    N = a / torch.sqrt(1 - e2 * sin_lat * sin_lat)  # shape (...)

    # Compute ECEF
    X = (N + h) * cos_lat * cos_lon
    Y = (N + h) * cos_lat * sin_lon
    Z = (N * (1 - e2) + h) * sin_lat

    return torch.stack([X, Y, Z], dim=-1)
def WGS84_to_ECEF(pos):
    lon, lat, height = pos
    transprojr = pyproj.Transformer.from_crs(
        "EPSG:4326",
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        always_xy=True,
    )
    xpjr, ypjr, zpjr = transprojr.transform(lon, lat, height, radians=False)
    return [xpjr, ypjr, zpjr]

  
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

if __name__ == '__main__':
    euler_angles = [3.47238421e+01, -8.33447684e-02,  1.23082890e+02]
    trans = [112.98941023000003, 28.291636667466683, 172.533362288028]
    lon, lat, _ = trans
    rot_pose_in_enu = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
    rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_enu_in_ecef, rot_pose_in_enu)
    q_w2c = rotmat2qvec(R_c2w.transpose())  # return wxyz (colmap pnp return xyzw)
    t_c2w = WGS84_to_ECEF(trans)
    R_w2c_in_ecef = R_c2w.transpose() # 和enu的差异是第二行和第三行取负号
    t_w2c = -R_w2c_in_ecef.dot(t_c2w)
    print(q_w2c, t_w2c)