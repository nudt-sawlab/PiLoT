from itertools import cycle
import torch
import numpy as np
import pyproj
import copy
from torch import nn
import math
import cv2
import time
from pixloc.utils.transform import ECEF_to_WGS84, get_rotation_enu_in_ecef, WGS84_to_ECEF,WGS84_to_ECEF_tensor
from scipy.spatial.transform import Rotation as R
from ..pixlib.geometry import Pose, Camera
import itertools
import torch.nn.functional as F
from ..utils.transform import  pixloc_to_osg, euler_angles_to_matrix_ECEF_batch_speical
import torch
import torch.nn.functional as F

def interpolate_depth_grid(pos, depth):
    """
    pos: Tensor[N,2], 每行是 (i, j) 像素坐标
    depth: Tensor[H,W]
    返回: depth_interp[valid], pos_valid.T, ids_valid
    """
    # 1) 准备
    device = depth.device
    H, W = depth.shape
    # 把 depth 变为 [1,1,H,W]
    depth4 = depth.unsqueeze(0).unsqueeze(0)

    # 2) 归一化到 [-1,1]，注意 grid_sample 的 coord 是 (x=j, y=i)
    j = pos[:,1].to(device)
    i = pos[:,0].to(device)
    # x 栏：2*j/(W-1)-1;  y 栏：2*i/(H-1)-1
    x = 2.0*j/(W-1) - 1.0
    y = 2.0*i/(H-1) - 1.0
    grid = torch.stack([x, y], dim=1)        # [N,2]
    grid = grid.view(1, -1, 1, 2)            # [1, N,1,2]

    # 3) 调用 grid_sample
    sampled = F.grid_sample(
        depth4, grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )  # [1,1,N,1]
    sampled = sampled.view(-1)               # [N]

    # 4) 掩码：坐标越界 or depth==0
    valid = (
        (i >= 0) & (i <= H-1) &
        (j >= 0) & (j <= W-1) &
        (sampled > 0)
    )
    ids   = valid.nonzero(as_tuple=False).view(-1)
    pos_v = pos[ids].t()                     # [2, M]
    depth_v = sampled[ids]                   # [M]

    return depth_v, pos_v, ids
def interpolate_depth(pos, depth):
    ids = torch.arange(0, pos.shape[0])
    if depth.ndim != 2:
        if depth.ndim == 3:
            depth = depth[:,:,0]
        else:
            raise Exception("Invalid depth image!")
    h, w = depth.size()
    
    i = pos[:, 0]
    j = pos[:, 1]

    # Valid corners, check whether it is out of range
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    # j_top_right = torch.ceil(j).long()
    j_top_right = torch.floor(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    # i_bottom_left = torch.ceil(i).long()
    i_bottom_left = torch.floor(i).long()
    
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    # i_bottom_right = torch.ceil(i).long()
    # j_bottom_right = torch.ceil(j).long()
    i_bottom_right = torch.floor(i).long()
    j_bottom_right = torch.floor(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    # Valid depth
    valid_depth = torch.min(
        torch.min(
            depth[i_top_left, j_top_left] > 0,
            depth[i_top_right, j_top_right] > 0
        ),
        torch.min(
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0
        )
    )

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]
    # vaild index
    ids = ids.to(valid_depth.device)

    ids = ids[valid_depth]
    
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.double()
    dist_j_top_left = j - j_top_left.double()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    #depth is got from interpolation
    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left] +
        w_top_right * depth[i_top_right, j_top_right] +
        w_bottom_left * depth[i_bottom_left, j_bottom_left] +
        w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    return [interpolated_depth, pos, ids]


def read_valid_depth(mkpts1r, depth=None, device = 'cuda'):
    depth = torch.tensor(depth).to(device)
    mkpts1r = mkpts1r.float().to(device)
    mkpts1r_inter = mkpts1r[:, [1, 0]].to(device)
    # depth_interpolated, _, valid = interpolate_depth(mkpts1r_inter, depth)
    
    # mkpts1r = mkpts1r.float().to(device)
    # mkpts1r_inter = mkpts1r[:, [1, 0]].to(device)
    depth_interpolated, _, valid = interpolate_depth_grid(mkpts1r_inter, depth)

    return depth_interpolated, valid
def get_Points3D(depth, R, t, K, points):
    """
    根据相机的内参矩阵、姿态（旋转矩阵和平移向量）以及图像上的二维点坐标和深度信息，
    计算对应的三维世界坐标。

    参数:
    - depth: 深度值数组，尺寸为 [n,]，其中 n 是点的数量。
    - R: 旋转矩阵，尺寸为 [3, 3]，表示从世界坐标系到相机坐标系的旋转。
    - t: 平移向量，尺寸为 [3, 1]，表示从世界坐标系到相机坐标系的平移。
    - K: 相机内参矩阵，尺寸为 [3, 3]，包含焦距和主点坐标。
    - points: 二维图像坐标数组，尺寸为 [n, 2]，其中 n 是点的数量。

    返回:
    - Points_3D: 三维世界坐标数组，尺寸为 [n, 3]。
    """
    # 检查points是否为同质坐标，如果不是则扩展为同质坐标
    if points.shape[-1] != 3:
        points_2D = np.concatenate([points, np.ones_like(points[ :, [0]])], axis=-1)
        points_2D = points_2D.Trender_camera
    else:
        points_2D = points.T  # 确保points的形状为 [2, n]

    # 扩展平移向量以匹配点的数量
    
    t = np.expand_dims(t,-1)
    t = np.tile(t, points_2D.shape[-1])

    # 将所有输入转换为高精度浮点数类型
    points_2D = np.float64(points_2D)
    K = np.float64(K)
    R = np.float64(R)
    depth = np.float64(depth)
    t = np.float64(t)

    # 修改内参矩阵的最后一项，以适应透视投影
    K[-1, -1] = -1
    
    # 计算三维世界坐标
    Points_3D = R @ K @ (depth * points_2D) + t
    
    # 返回三维点坐标，形状为 [3, n]
    return Points_3D.T
def get_points2D_ECEF(R, t, K, points_3D):  # points_3D[n,3]
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
        points_3D = np.float64(points_3D)
        K = np.float64(K)
        R = np.float64(R)
        t = np.float64(t)
        # 修改内参矩阵的最后一项，以适应透视投影
        K[-1, -1] = -1
        
        K_inverse = np.linalg.inv(K)
        R_inverse = np.linalg.inv(R)
        # 计算相机坐标系下的点
        point_3d_camera = np.expand_dims(points_3D - t, 1)
        # 将世界坐标系下的点转换为相机坐标系下的点
        point_3d_camera_r = R_inverse @ point_3d_camera
        # 将相机坐标系下的点投影到图像平面，得到同质坐标
        point_2d_homo = K_inverse @ point_3d_camera_r
        # 将同质坐标转换为二维图像坐标
        point_2d = point_2d_homo / point_2d_homo[2]
        return point_2d.T
def get_points2D_CGCS2000(R, t, K, points_3D):  # points_3D[n,3]
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
    points_3D = np.float64(points_3D)
    K = np.float64(K)
    R = np.float64(R)
    t = np.float64(t)
    # 修改内参矩阵的最后一项，以适应透视投影
    
    K_inverse = np.linalg.inv(K)
    R_inverse = np.linalg.inv(R)
    # 计算相机坐标系下的点
    point_3d_camera = np.expand_dims(points_3D - t, 1)
    # 将世界坐标系下的点转换为相机坐标系下的点
    point_3d_camera_r = R_inverse @ point_3d_camera
    # 将相机坐标系下的点投影到图像平面，得到同质坐标
    point_2d_homo = K_inverse @ point_3d_camera_r
    # 将同质坐标转换为二维图像坐标
    point_2d = point_2d_homo / point_2d_homo[2]
    return point_2d.T
def get_Points3D_torch(depth, R, t, K, points):
    """
    根据相机的内参矩阵、姿态（旋转矩阵和平移向量）以及图像上的二维点坐标和深度信息，
    计算对应的三维世界坐标。

    参数:
    - depth: 深度值数组，尺寸为 [n,]，其中 n 是点的数量。
    - R: 旋转矩阵，尺寸为 [3, 3]，表示从世界坐标系到相机坐标系的旋转。
    - t: 平移向量，尺寸为 [3, 1]，表示从世界坐标系到相机坐标系的平移。
    - K: 相机内参矩阵，尺寸为 [3, 3]，包含焦距和主点坐标。
    - points: 二维图像坐标数组，尺寸为 [n, 2]，其中 n 是点的数量。

    返回:
    - Points_3D: 三维世界坐标数组，尺寸为 [n, 3]。
    """
    # 检查points是否为同质坐标，如果不是则扩展为同质坐标
    if points.shape[-1] != 3:
        points_2D = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
        points_2D = points_2D.T
    else:
        points_2D = points.T

    # 扩展平移向量以匹配点的数量
    t = t.unsqueeze(1)  # 这相当于np.expand_dims(t, -1)
    t = t.repeat(1, points_2D.size(-1))  # 这相当于np.tile(t, points_2D.shape[-1])

    # 将所有输入转换为高精度浮点数类型
    points_2D = points_2D.float()
    K = K.float()
    R = R.float()
    depth = depth.float()
    t = t.float()

    # 修改内参矩阵的最后一项，以适应透视投影
    K[-1, -1] = -1

    # 计算三维世界坐标
    Points_3D = R @ (K @ (depth * points_2D)) + t

    # 返回三维点坐标，形状为 [n, 3]
    return Points_3D.cpu().numpy().T
def ecef_to_gausskruger_pyproj(ecef_points, central_meridian=117):
    """
    使用 pyproj 批量将 ECEF 坐标转换为高斯-克吕格投影平面坐标 (CGCS2000).
    
    Args:
        ecef_points: (n, 3) 的 numpy 数组，每行是一个 (x, y, z) 点.
        central_meridian: 中央经线（默认为 117°，适合长沙地区）.
    
    Returns:
        平面坐标数组 (n, 2)，每行是 (X, Y).
    """
    # ECEF 转 地理坐标 (经纬度 + 高程)
    transformer_to_geodetic = pyproj.Transformer.from_crs(
        crs_from="EPSG:4978",  # ECEF 坐标系
        crs_to="EPSG:4326",    # 地理坐标系 (WGS84 / CGCS2000)
        always_xy=True         # 确保输入顺序是 (x, y, z)
    )
    
    # 地理坐标转高斯-克吕格投影坐标
    zone = int((central_meridian - 1) / 3 + 1)  # 计算高斯-克吕格带号
    # epsg_proj = f"EPSG:454{zone}"  # CGCS2000 高斯-克吕格投影 (3° 带)
    transformer_to_projected = pyproj.Transformer.from_crs(
        crs_from="EPSG:4326",  # 地理坐标系
        crs_to='EPSG:4547',      # CGCS2000 高斯-克吕格投影
        always_xy=True
    )   
    # 分解输入 ECEF 坐标
    x, y, z = ecef_points[:, 0], ecef_points[:, 1], ecef_points[:, 2]

    # 第一步: ECEF -> 地理坐标
    lon, lat, h = transformer_to_geodetic.transform(x, y, z)

    # 第二步: 地理坐标 -> 高斯-克吕格投影平面坐标
    proj_x, proj_y = transformer_to_projected.transform(lon, lat)

    # 返回结果
    return np.column_stack((proj_x, proj_y, h))
def generate_random_aa_and_t(min_offset_angle, max_offset_angle, min_offset_translation, max_offset_translation, n = None):
    if isinstance(min_offset_angle, float):
        min_offset_angle = torch.tensor([min_offset_angle], dtype=torch.float32)
    if isinstance(max_offset_angle, float):
        max_offset_angle = torch.tensor([max_offset_angle], dtype=torch.float32)
    if isinstance(min_offset_translation, float):
        min_offset_translation = torch.tensor([min_offset_translation], dtype=torch.float32)
    if isinstance(max_offset_translation, float):
        max_offset_translation = torch.tensor([max_offset_translation], dtype=torch.float32)
    if n == None:
        n = min_offset_angle.shape[0]
    axis = nn.functional.normalize(torch.rand(n, 3) * 2 - 1, dim=-1)
    angle = (torch.rand(n) * (max_offset_angle - min_offset_angle) + min_offset_angle).unsqueeze(-1) / 180 * 3.1415926

    aa = axis * angle

    direction = nn.functional.normalize(torch.rand(n, 3) * 2 - 1, dim=-1)
    t = (torch.rand(n) * (max_offset_translation - min_offset_translation) + min_offset_translation).unsqueeze(-1)
    trans = direction * t

    return aa, trans
def generate_random_aa_and_t_cuda(
    n: int,
    min_offset_angle: float, max_offset_angle: float,
    min_offset_translation: float, max_offset_translation: float,
    device=torch.device("cuda"),
    dtype=torch.float32,
):
    # 1) 生成随机方向（单位向量）
    axis = torch.randn(n, 3, device=device, dtype=dtype)
    axis = axis / axis.norm(dim=1, keepdim=True)  # 归一化为单位向量

    direction = torch.randn(n, 3, device=device, dtype=dtype)
    direction = direction / direction.norm(dim=1, keepdim=True)

    # 2) 生成角度（单位为度，再转弧度）
    delta_angle = max_offset_angle - min_offset_angle
    angles_deg = min_offset_angle + delta_angle * torch.rand(n, device=device, dtype=dtype)
    angles_rad = angles_deg * (math.pi / 180.0)
    aa = axis * angles_rad.unsqueeze(1)  # (n, 3)

    # 3) 生成平移（单位：米）
    delta_trans = max_offset_translation - min_offset_translation
    trans_mag = min_offset_translation + delta_trans * torch.rand(n, device=device, dtype=dtype)
    trans = direction * trans_mag.unsqueeze(1)  # (n, 3)
    aa[0] = 0.0
    trans[0] = 0.0
    return aa, trans
import torch
import torch.nn.functional as F
import itertools
from scipy.spatial.transform import Rotation as R

def generate_pitch_yaw_aa_and_random_t(pitch_angles_deg=[-10, -7, -4, -1, 1, 4, 7, 10],
                                       yaw_angles_deg=[-10, -7, -4, -1, 1, 4, 7, 10],
                                       min_offset_translation=-0.1, 
                                       max_offset_translation=0.1):
    # 组合 pitch 和 yaw
    combos = list(itertools.product(pitch_angles_deg, yaw_angles_deg))
    n = len(combos)

    # 只对 pitch(x) 和 yaw(z) 添加扰动，roll(y) 为 0
    rot_list = []
    for pitch_deg, yaw_deg in combos:
        # 单独绕 x（pitch）
        Rx = R.from_rotvec(np.deg2rad(pitch_deg) * np.array([1, 0, 0]))
        # 单独绕 z（yaw）
        Rz = R.from_rotvec(np.deg2rad(yaw_deg) * np.array([0, 0, 1]))
        # 组合旋转矩阵（注意旋转顺序：先 Rx 后 Rz）
        R_combined = Rz * Rx  # 相乘等于先绕 x 再绕 z
        aa = R_combined.as_rotvec()
        rot_list.append(torch.tensor(aa, dtype=torch.float32))

    aa = torch.stack(rot_list, dim=0)  # [n, 3]

    # 随机平移扰动
    direction = F.normalize(torch.rand(n, 3) * 2 - 1, dim=-1)
    t = (torch.rand(n) * (max_offset_translation - min_offset_translation) + min_offset_translation).unsqueeze(-1)
    trans = direction * t

    return aa, trans
def generate_rotvecs_cuda_sym3d(
    base_pitch, base_roll, base_yaw, 
    max_pitch: float, pitch_step: float,
    max_yaw:   float,   yaw_step:   float,
    max_roll:  float,  roll_step:  float,
    device:   str = 'cuda'):
    """
    在 CUDA 上生成 (pitch, roll, yaw) 三维对称网格采样：
      - pitch 从 -max_pitch 到 +max_pitch，步长为 pitch_step
      - yaw   从 -max_yaw   到 +max_yaw，   步长为 yaw_step
      - roll  从 -max_roll  到 +max_roll，  步长为 roll_step

    如果某个 max = 0，则只在该维度采样 0；否则会在 [-max, +max] 区间内按照 step 生成序列，
    包含正负两侧（如果 (2*max) 不是 step 的整数倍，则右边最后一个采样点会略小于 +max）。

    返回张量形状为 [N, 3]，其中
      N = num_pitch * num_yaw * num_roll，
    每行即为一个 (pitch, roll, yaw) 三元组，dtype=float32 且在 CUDA 上。

    示例：
        # 在 [-10, +10] 以 5 度步长采样 pitch、yaw、roll
        rotvecs = generate_rotvecs_cuda_sym3d(
            max_pitch=10.0,  pitch_step=5.0,
            max_yaw=10.0,    yaw_step=5.0,
            max_roll=10.0,  roll_step=5.0,
            device='cuda'
        )
        # 此时 pitch_vals = [-10, -5, 0, 5, 10]，yaw_vals、roll_vals 同理
        # N = 5 * 5 * 5 = 125，所以 rotvecs.shape = [125, 3]
    """

    # ------- 1. 针对每个维度，生成 [-max, +max] 等步长值 -------
    def _symmetric_range(max_val: float, step: float) -> torch.Tensor:
        """
        生成一个对称采样向量：
          如果 max_val == 0: 返回 tensor([0.0])
          否则：采样 [-max_val, -max_val+step, ..., 0, ..., max_val-step, max_val]（包含正负两端）
        """
        if step <= 0 or max_val < 0:
            raise ValueError("step 必须 > 0 且 max_val 必须 >= 0")
        if max_val == 0:
            return torch.tensor([0.0], device=device, dtype=torch.float32)

        # torch.arange(start, end, step) 会生成 [start, start+step, ..., < end]
        # 我们希望包含 -max_val 和 +max_val 两侧，先从 0 到 +max_val 再映射
        pos = torch.arange(0.0, max_val + 1e-6, step, device=device, dtype=torch.float32)
        # 如果 pos 的最后一个元素略大于 max_val（due to float 误差），可以 clamp
        if pos[-1] > max_val:
            pos[-1] = max_val

        # 剔除第一个 0，生成正半区采样；0 在最中心单独出现一次
        if pos.shape[0] > 1:
            pos_nonzero = pos[1:]  # 跳过 0
            # 负半区 = 正半区 取反并倒序，以保持顺序从小到大
            neg = (-pos_nonzero).flip(0)
            # 合并负、中、正
            return torch.cat((neg, pos[:1], pos_nonzero), dim=0)
        else:
            # 只有一个元素 0
            return pos

    # pitch_vals = _symmetric_range(max_pitch, pitch_step)  # 形如 [ -max, ..., 0, ..., +max ]
    # yaw_vals   = _symmetric_range(max_yaw,   yaw_step)
    # roll_vals  = _symmetric_range(max_roll,  roll_step)
    # pitch_vals = torch.tensor([15, 13, 11, 9, 7, 5, 3, 1, -1, -3, -5, -7, -9, -11, -13, -15]).to(device)
    # yaw_vals = torch.tensor([15, 13, 11, 9, 7, 5, 3, 1, -1, -3, -5, -7, -9, -11, -13, -15]).to(device)
    # roll_vals = torch.tensor([-1, 0, 1]).to(device)
    # pitch_vals = torch.tensor([5,4,3,2, 1, 0, -1, -2, -3, -4, -5]).to(device)
    # yaw_vals = torch.tensor([5,4,3,2, 1, 0, -1, -2, -3, -4, -5]).to(device)
    
    pitch_vals = torch.tensor([11, 9, 7, 5, 3, 1, -1, -3, -5, -7, -9, -11]).to(device)
    yaw_vals = torch.tensor([11, 9, 7, 5, 3, 1, -1, -3, -5, -7, -9, -11]).to(device)
    # pitch_vals = torch.tensor([7, 5, 3, 1, -1, -3, -5, -7]).to(device)
    # yaw_vals = torch.tensor([7, 5, 3, 1, -1, -3, -5, -7]).to(device)
    # pitch_vals = torch.tensor([0]).to(device)
    roll_vals = torch.tensor([0]).to(device)
    # yaw_vals = torch.tensor([0]).to(device)

    # ------- 2. 使用 meshgrid 生成三维网格 -------
    P, Y, R = torch.meshgrid(
        pitch_vals, yaw_vals, roll_vals, indexing='ij'
    )  # P.shape == (num_pitch, num_yaw, num_roll)
    
    # ------- 3. 展平至 (N,) 并拼成 (N,3) -------
    N = P.numel()
    P_flat = P.reshape(N)
    Y_flat = Y.reshape(N)
    R_flat = R.reshape(N)
    
    # P_flat[0] = 0
    # R_flat[0] = 0
    # Y_flat[0] = 0
    
    pitch = P_flat + base_pitch
    roll  = R_flat + base_roll
    yaw   = Y_flat + base_yaw
    # 拼成 (N,3)
    return torch.stack((pitch, roll, yaw), dim=1)       # float tensor on GPU
def generate_rotvecs_cuda_sym3dbak(
    max_pitch: float, pitch_step: float,
    max_yaw:   float,   yaw_step:   float,
    max_roll:  float,  roll_step:  float,
    device:   str = 'cuda'):
    """
    在 CUDA 上生成 (pitch, roll, yaw) 三维对称网格采样：
      - pitch 从 -max_pitch 到 +max_pitch，步长为 pitch_step
      - yaw   从 -max_yaw   到 +max_yaw，   步长为 yaw_step
      - roll  从 -max_roll  到 +max_roll，  步长为 roll_step

    如果某个 max = 0，则只在该维度采样 0；否则会在 [-max, +max] 区间内按照 step 生成序列，
    包含正负两侧（如果 (2*max) 不是 step 的整数倍，则右边最后一个采样点会略小于 +max）。

    返回张量形状为 [N, 3]，其中
      N = num_pitch * num_yaw * num_roll，
    每行即为一个 (pitch, roll, yaw) 三元组，dtype=float32 且在 CUDA 上。

    示例：
        # 在 [-10, +10] 以 5 度步长采样 pitch、yaw、roll
        rotvecs = generate_rotvecs_cuda_sym3d(
            max_pitch=10.0,  pitch_step=5.0,
            max_yaw=10.0,    yaw_step=5.0,
            max_roll=10.0,  roll_step=5.0,
            device='cuda'
        )
        # 此时 pitch_vals = [-10, -5, 0, 5, 10]，yaw_vals、roll_vals 同理
        # N = 5 * 5 * 5 = 125，所以 rotvecs.shape = [125, 3]
    """

    # ------- 1. 针对每个维度，生成 [-max, +max] 等步长值 -------
    def _symmetric_range(max_val: float, step: float) -> torch.Tensor:
        """
        生成一个对称采样向量：
          如果 max_val == 0: 返回 tensor([0.0])
          否则：采样 [-max_val, -max_val+step, ..., 0, ..., max_val-step, max_val]（包含正负两端）
        """
        if step <= 0 or max_val < 0:
            raise ValueError("step 必须 > 0 且 max_val 必须 >= 0")
        if max_val == 0:
            return torch.tensor([0.0], device=device, dtype=torch.float32)

        # torch.arange(start, end, step) 会生成 [start, start+step, ..., < end]
        # 我们希望包含 -max_val 和 +max_val 两侧，先从 0 到 +max_val 再映射
        pos = torch.arange(0.0, max_val + 1e-6, step, device=device, dtype=torch.float32)
        # 如果 pos 的最后一个元素略大于 max_val（due to float 误差），可以 clamp
        if pos[-1] > max_val:
            pos[-1] = max_val

        # 剔除第一个 0，生成正半区采样；0 在最中心单独出现一次
        if pos.shape[0] > 1:
            pos_nonzero = pos[1:]  # 跳过 0
            # 负半区 = 正半区 取反并倒序，以保持顺序从小到大
            neg = (-pos_nonzero).flip(0)
            # 合并负、中、正
            return torch.cat((neg, pos[:1], pos_nonzero), dim=0)
        else:
            # 只有一个元素 0
            return pos

    # pitch_vals = _symmetric_range(max_pitch, pitch_step)  # 形如 [ -max, ..., 0, ..., +max ]
    # yaw_vals   = _symmetric_range(max_yaw,   yaw_step)
    # roll_vals  = _symmetric_range(max_roll,  roll_step)
    # pitch_vals = torch.tensor([15, 13, 11, 9, 7, 5, 3, 1, -1, -3, -5, -7, -9, -11, -13, -15]).to(device)
    # yaw_vals = torch.tensor([15, 13, 11, 9, 7, 5, 3, 1, -1, -3, -5, -7, -9, -11, -13, -15]).to(device)
    # roll_vals = torch.tensor([-1.0, 0, 1]).to(device)
    pitch_vals = torch.tensor([9.0,7.0, 5.0, 3.0, 1.0, -1.0, -3.0, -5.0, -7.0,-9.0]).to(device)
    yaw_vals = torch.tensor([9.0,7.0, 5.0, 3.0, 1.0, -1.0, -3.0, -5.0, -7.0,-9.0]).to(device)
    roll_vals = torch.tensor([0.0]).to(device)

    # ------- 2. 使用 meshgrid 生成三维网格 -------
    P, Y, R = torch.meshgrid(
        pitch_vals, yaw_vals, roll_vals, indexing='ij'
    )  # P.shape == (num_pitch, num_yaw, num_roll)

    # ------- 3. 展平至 (N,) 并拼成 (N,3) -------
    N = P.numel()
    P_flat = P.reshape(N)
    Y_flat = Y.reshape(N)
    R_flat = R.reshape(N)

    # 我们返回 [pitch, roll, yaw] 顺序
    rotvecs = torch.stack((P_flat, R_flat, Y_flat), dim=1)  # [N, 3]
    return rotvecs
def generate_rotvecs_cuda_3d(
    pitch_start: float, pitch_step: float, pitch_count: int,
    yaw_start:   float, yaw_step:   float, yaw_count:   int,
    roll_start:  float, roll_step:  float, roll_count:  int,
    device:      str = 'cuda'):
    """
    在 CUDA 上生成 (pitch, roll, yaw) 三维网格采样：
      - pitch 从 pitch_start 开始，以 pitch_step 为步长，共采样 pitch_count 个值
      - yaw   从 yaw_start   开始，以 yaw_step   为步长，共采样 yaw_count   个值
      - roll  从 roll_start  开始，以 roll_step  为步长，共采样 roll_count  个值

    最终返回的张量形状为 [N, 3]，其中 N = pitch_count * yaw_count * roll_count，
    每行对应一个 (pitch, roll, yaw) 三元组，类型 float32 并位于 CUDA 上。

    示例调用：
        rotvecs = generate_rotvecs_cuda_3d(
            pitch_start= -10.0, pitch_step= 3.0, pitch_count= 8,
            yaw_start=   -10.0, yaw_step=   3.0, yaw_count=    8,
            roll_start=   0.0,  roll_step=  5.0, roll_count=  4,
            device='cuda'
        )
        # rotvecs.shape == [8 * 8 * 4, 3] == [256, 3]
    """

    # 1) 分别生成 pitch、yaw、roll 的 1D 采样列表
    # torch.linspace 也可用：torch.linspace(start, start + step*(count-1), count)
    pitch_vals = torch.arange(pitch_start,
                              pitch_start + pitch_step * pitch_count,
                              pitch_step,
                              device=device,
                              dtype=torch.float32)[:pitch_count]  # 确保不超长
    yaw_vals   = torch.arange(yaw_start,
                              yaw_start   + yaw_step   * yaw_count,
                              yaw_step,
                              device=device,
                              dtype=torch.float32)[:yaw_count]
    roll_vals  = torch.arange(roll_start,
                              roll_start  + roll_step  * roll_count,
                              roll_step,
                              device=device,
                              dtype=torch.float32)[:roll_count]

    # 2) meshgrid 生成三维网格：三个维度的笛卡尔积
    #    注意：使用 indexing='ij' 保持第一个维度对应 pitch，第二对应 yaw，第三对应 roll
    P, Y, R = torch.meshgrid(
        pitch_vals, yaw_vals, roll_vals, indexing='ij'
    )  # P.shape==(pitch_count, yaw_count, roll_count), 同理 Y, R

    # 3) 展平成 N = pitch_count * yaw_count * roll_count 行
    N = P.numel()
    P_flat = P.reshape(N)
    Y_flat = Y.reshape(N)
    R_flat = R.reshape(N)

    # 4) 按 (pitch, roll, yaw) 顺序拼成最终张量
    #    注意，此处顺序可根据实际需求调整：示例中为 [pitch, roll, yaw]
    rotvecs = torch.stack((P_flat, R_flat, Y_flat), dim=1)  # [N,3]

    return rotvecs  # float32, device='cuda'
def generate_rotvecs_cuda(pitch_angles = [-10, -7, -4, -1, 1, 4, 7, 10],yaw_angles   = [-10, -7, -4, -1, 1, 4, 7, 10],
                     base_pitch=0., base_roll=0., base_yaw=0.,
                     device='cuda'):
    # pitch_angles, yaw_angles: list 或 1D tensor
    pa = torch.tensor(pitch_angles, device=device, dtype=torch.float32)
    ya = torch.tensor(yaw_angles,   device=device, dtype=torch.float32)
    # 生成所有组合 (P, Y)
    P, Y = torch.meshgrid(pa, ya, indexing='ij')        # shape (len(pa), len(ya))
    N = P.numel()
    P = P.reshape(N)
    Y = Y.reshape(N)
    R = torch.full((N,), base_roll, device=device)
    # 加上偏移
    pitch = P + base_pitch
    roll  = R + base_roll
    yaw   = Y + base_yaw
    # 拼成 (N,3)
    return torch.stack((pitch, roll, yaw), dim=1)       # float tensor on GPU
def get_rotation_enu_in_ecef_tensor(lon, lat, device='cuda', dtype=torch.float32):
    """
    Compute the 3×3 ENU→ECEF rotation matrix on GPU.

    Args:
        lon (float or Tensor): Longitude in degrees.
        lat (float or Tensor): Latitude in degrees.
        device (str or torch.device): e.g. 'cuda' or 'cpu'.
        dtype (torch.dtype): e.g. torch.float32.

    Returns:
        Tensor of shape (3,3) on specified device.
    """
    # 1) 转成 tensor
    lon_t = torch.as_tensor(lon, device=device, dtype=dtype)
    lat_t = torch.as_tensor(lat, device=device, dtype=dtype)

    # 2) 角度转弧度
    lon_rad = lon_t * (torch.pi / 180.0)
    lat_rad = lat_t * (torch.pi / 180.0)

    # 3) 计算 up, east, north
    up = torch.stack([
        torch.cos(lon_rad) * torch.cos(lat_rad),
        torch.sin(lon_rad) * torch.cos(lat_rad),
        torch.sin(lat_rad)
    ])

    east = torch.stack([
        -torch.sin(lon_rad),
         torch.cos(lon_rad),
         torch.zeros_like(lon_rad)
    ])

    north = torch.cross(up, east, dim=0)

    # 4) 拼矩阵，列分别是 east/north/up
    rot = torch.stack([east, north, up], dim=1)  # shape (3,3)
    return rot
def euler_to_rotm_batch(angles, translation, degrees=True, device='cuda'):
    # angles: (N,3) tensor representing [pitch, roll, yaw]
    if degrees:
        angles = angles * (torch.pi/180.0)
    p, r, y = angles.unbind(1)   # each (N,)

    # 生成各自的旋转矩阵 Rx(p), Ry(r), Rz(y)
    # Rx (around x axis = pitch)
    zero = torch.zeros_like(p); one = torch.ones_like(p)
    Rx = torch.stack([
        torch.stack([ one, zero,    zero], dim=1),
        torch.stack([ zero, torch.cos(p), -torch.sin(p)], dim=1),
        torch.stack([ zero, torch.sin(p),  torch.cos(p)], dim=1),
    ], dim=1)  # (N,3,3)

    # Ry (around y axis = roll)
    Ry = torch.stack([
        torch.stack([ torch.cos(r), zero, torch.sin(r)], dim=1),
        torch.stack([ zero,        one,  zero      ], dim=1),
        torch.stack([-torch.sin(r), zero, torch.cos(r)], dim=1),
    ], dim=1)

    # Rz (around z axis = yaw)
    Rz = torch.stack([
        torch.stack([torch.cos(y), -torch.sin(y), zero], dim=1),
        torch.stack([torch.sin(y),  torch.cos(y), zero], dim=1),
        torch.stack([zero,          zero,         one ], dim=1),
    ], dim=1)

    # 按顺序乘：R = ENU2ECEF @ Rz @ Rx @ Ry  （根据你的定义调整）
    # 假设 get_rotation_enu_in_ecef 返回 (3,3) CPU or GPU tensor:
    lon, lat, _ = translation
    rot_enu_in_ecef = get_rotation_enu_in_ecef_tensor(lon, lat).to(device)
    R_local = Rz @ Rx @ Ry                         # (N,3,3)
    R_batch = rot_enu_in_ecef.unsqueeze(0) @ R_local  # (N,3,3)
    return R_batch

def euler_angles_to_matrix_ECEF_batch_special_cuda(euler_angles, translation,
                                              device='cuda'):
    # euler_angles: (N,3) GPU tensor; translation: (3,) CPU or GPU
    euler_angles = torch.tensor(euler_angles).to(device)
    translation = torch.tensor(translation).to(device)
    R_batch = euler_to_rotm_batch(euler_angles, translation, degrees=True, device=device)
    t_c2w = WGS84_to_ECEF_tensor(translation)
    t = torch.tensor(t_c2w, device=device, dtype=torch.float32)  # (3,)
    N = R_batch.shape[0]
    # 拼成 (N,4,4)
    T = torch.eye(4, device=device).unsqueeze(0).repeat(N,1,1)         # (N,4,4)
    T[:, :3, :3] = R_batch
    T[:, :3,  3] = t
    return T  # (N,4,4) GPU tensor
def generate_rotvecs(pitch_angles_deg = [-10, -7, -4, -1, 1, 4, 7, 10], yaw_angles_deg  = [-10, -7, -4, -1, 1, 4, 7, 10],
                        base_pitch=0.0, base_roll=0.0, base_yaw=0.0):
    rotvecs = []
    combos = list(itertools.product(pitch_angles_deg, yaw_angles_deg))
    for pitch_delta, yaw_delta in combos:
        # 计算新的欧拉角
        pitch = base_pitch + pitch_delta
        roll = base_roll  # 固定为 0
        yaw = base_yaw + yaw_delta

        rotvecs.append([pitch, roll, yaw])
    return np.array(rotvecs)  # [64, 3]
def add_noise_to_pose(euler_angles, t_c2w, noise_std_angle=5.0, noise_std_translation=0.5, num_candidates=127):
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

        candidates.append(noisy_render_T)

    return candidates
def pad_to_multiple(image, padd = 16):
    """
    Pads the input color image to make its dimensions multiples of 16.
    Args:
        image (np.ndarray): Input image of shape (H, W, C) or (H, W).
    Returns:
        padded_image (np.ndarray): Padded image.
    """
    h, w = image.shape[:2]
    target_h = (h + (padd-1)) // padd * padd  # Compute nearest multiple of 16
    target_w = (w + (padd-1)) // padd * padd
    
    # Create a blank canvas with padding
    padded_image = np.zeros((target_h, target_w, *image.shape[2:]), dtype=image.dtype)
    
    # Copy the original image to the top-left corner
    padded_image[:h, :w] = image
    
    return padded_image
def preprocess_param(camera, pose):
    pose[:3, 1] = -pose[:3, 1]  # Y轴取反，投影后二维原点在左上角
    pose[:3, 2] = -pose[:3, 2]  # Z轴取反

    _, h = camera.size
    camera.c[1] = h - camera.c[1]
    return camera, pose
def preprocess_param_cuda(camera, pose, device='cuda'):
    # pose: torch tensor (4,4) or (N,4,4)
    pose = pose.to(device)
    pose[..., 0:3, 1] *= -1
    pose[..., 0:3, 2] *= -1

    # camera.c 也是 torch tensor
    # h = camera.size[1]
    # camera.c[1] = h - camera.c[1]

    return camera, pose
def get_3D_samples(mkpts_r, depth_mat, T_c2w, camera, last_frame_info = {}, origin = None, device = 'cuda', num_init_pose = 32,  mul = None):
    min_offset_angle = 0.0
    max_offset_angle = 5
    min_offset_translation = 0.0
    max_offset_translation = 2 # 0.015, 0.025 
    # preprocess
    render_camera, render_T = preprocess_param(copy.deepcopy(camera), copy.deepcopy(T_c2w))

    # print('-------Tc2w', T_c2w)
    # euler_angles_refined, translation_refined, _, _ = pixloc_to_osg(T_c2w)
    # print("refined euler angles: ",  euler_angles_refined, translation_refined)
    cx, cy = render_camera.c
    fx, fy = render_camera.f
    _, render_height_px = render_camera.size
    render_K = np.array([[fx, 0, cx],[0, fy, cy], [0, 0, 1]])
    render_K = torch.tensor(render_K, device=device)
    K_c2w = render_K.inverse()
    
    render_T = torch.tensor(render_T, device=device)
    mkpts_r = torch.tensor(mkpts_r, device=device)
    
    depth, valid = read_valid_depth(mkpts_r, depth = depth_mat, device=device)
    # Compute 3D points
    Points_3D_ECEF = get_Points3D_torch_normal(
        depth,
        render_T[:3, :3],
        render_T[:3, 3],
        K_c2w,
        mkpts_r[valid])
    #----mul
    if mul is not None:
        Points_3D_ECEF = Points_3D_ECEF * mul
        render_T[:3, 3] = render_T[:3, 3] * mul
        max_offset_translation = 5 *mul
        origin = np.array(origin)*mul

    # origin 
    if origin is None:
        origin = Points_3D_ECEF[0]
    Points_3D_ECEF_origin = Points_3D_ECEF - np.tile(origin, (len(Points_3D_ECEF), 1))   
    
    render_T = render_T.cpu().numpy()
    render_T[:3, 3] -= origin  # t_c2w - origin
    # render_T_w2c = np.eye(4)
    # render_T_w2c[:3, :3] = render_T[:3, :3].T
    # render_T_w2c[:3, 3] = -render_T[:3, :3].T @ render_T[:3, 3]
    # ref w2c
    render_T_c2w = Pose.from_Rt(render_T[:3, :3], render_T[:3, 3])  # w2c
    T_render = render_T_c2w.inv()

    center = True
    if center:
        # center
        points3D_total = torch.from_numpy(Points_3D_ECEF_origin).float()
        points_max = points3D_total.max(dim=0)[0]
        points_min = points3D_total.min(dim=0)[0]
        points_size = points_max - points_min
        dd = points_min + points_size / 2
        Points_3D_ECEF_origin_center = points3D_total - dd

        tt = T_render.t + T_render.R @ dd.double()
        T_render = Pose.from_Rt(T_render.R, tt)

        # tt = initial_poses.t + initial_poses.R @ dd.double()
        # initial_poses = Pose.from_Rt(initial_poses.R, tt)
        # return Points_3D_ECEF_origin_center, T_render, initial_poses, dd
    # query initial w2c
    else:
        points3D_total = torch.from_numpy(Points_3D_ECEF_origin).float()
        Points_3D_ECEF_origin_center = points3D_total
        dd = torch.tensor([0,0,0])
    if 'candidate_poses' not in last_frame_info.keys():
        T_query_c2w = np.eye(4)
        T_query_c2w[:3, :3] = render_T_c2w.R
        T_query_c2w[:3, 3] = render_T_c2w.t
        pose_query_repeat = np.tile(T_query_c2w, (num_init_pose, 1, 1))
        initial_pose = Pose.from_Rt(pose_query_repeat[:, :3, :3], pose_query_repeat[:, :3, 3])  # w2c

        # random_aa, random_t = generate_random_aa_and_t(min_offset_angle, max_offset_angle, 
        #                                         min_offset_translation, max_offset_translation,
        #                                         n = num_init_pose)
        random_aa, random_t = generate_pitch_yaw_aa_and_random_t(min_offset_translation = min_offset_translation, max_offset_translation = max_offset_translation)
        
        random_pose = Pose.from_aa(random_aa, random_t)
        initial_poses = initial_pose @ random_pose.double()  #c2w
        initial_poses_w2c = initial_poses.inv()

        # T_query_w2c = np.eye(4)
        # T_query_w2c[:3, :3] = (T_render.R)
        # T_query_w2c[:3, 3] = (T_render.t)
        # pose_query_repeat = np.tile(T_query_w2c, (num_init_pose, 1, 1))
        # initial_pose = Pose.from_Rt(pose_query_repeat[:, :3, :3], pose_query_repeat[:, :3, 3])  # w2c

        # random_aa, random_t = generate_random_aa_and_t(min_offset_angle, max_offset_angle, 
        #                                         min_offset_translation, max_offset_translation,
        #                                         n = num_init_pose)
        # random_pose = Pose.from_aa(random_aa, random_t)
        # initial_poses = initial_pose @ random_pose.double()  #c2w

        # R_w2c = initial_poses.R.cpu().numpy().astype(np.float64)  # [B, 3, 3]
        # t_w2c = initial_poses.t.cpu().numpy().astype(np.float64)  # [B, 3]
        # B = R_w2c.shape[0]
        # for i in range(B):
        #     # R_w2c[i] = orthogonalize_rotation_matrix(R_w2c[i])
        #     if dd is not None:
        #         t_w2c[i] = t_w2c[i] - R_w2c[i] @ dd.cpu().numpy()
        # T_opt_c2w = torch.eye(4, dtype=torch.float64).repeat(B, 1, 1)
        # T_opt_c2w[:, :3, :3] = torch.tensor(R_w2c).transpose(-1, -2)  # 转置旋转矩阵
        # T_opt_c2w[:, :3, 3] = -torch.bmm(torch.tensor(R_w2c).transpose(-1, -2), torch.tensor(t_w2c).unsqueeze(-1)).squeeze(-1)

        # T_opt_c2w[:, :3, 3] = T_opt_c2w[:, :3, 3] / mul
        
        # # 对 T_opt_c2w 进行调整
        # T_opt_c2w[:, :3, 1] = -T_opt_c2w[:, :3, 1]
        # T_opt_c2w[:, :3, 2] = -T_opt_c2w[:, :3, 2]
        # T_opt_c2w[:, :3, 3] += (origin / mul)  # 加上 origin 偏移量

        # T_c2w = np.tile(T_c2w, (B, 1, 1))
        # T_c2w = Pose.from_Rt(T_c2w[:, :3, :3], T_c2w[:, :3, 3])
        # T_opt_c2w_Pose = Pose.from_Rt(T_opt_c2w[:, :3, :3], T_opt_c2w[:, :3, 3])
        # dR, dt = (T_c2w.inv() @ T_opt_c2w_Pose).magnitude()







        # pose_query_repeat = Pose.from_Rt(pose_query_repeat[:, :3, :3], pose_query_repeat[:, :3, 3])
        # dR, dt = (pose_query_repeat @ initial_poses.inv()).magnitude()


        # T_c2w_R=T_render.inv().R
        # T_c2w_t = T_render.inv().t
        # Points_3D_ECEF_dev = get_Points3D_torch_normal(
        # depth,
        # (T_render.inv().R).cuda(),
        # (T_render.inv().t).cuda(),
        # K_c2w,
        # mkpts_r[valid])
        # tt = T_render.t - T_render.R @ dd.double()
        # T_render = Pose.from_Rt(T_render.R, tt)
        # T_render_in_ECEF_c2w = np.eye(4)
        # T_render_in_ECEF_c2w[:3, :3] = T_render.R.T.cpu().numpy()
        # T_render_in_ECEF_c2w[:3, 1] = -T_render_in_ECEF_c2w[:3, 1]
        # T_render_in_ECEF_c2w[:3, 2] = -T_render_in_ECEF_c2w[:3, 2]
        
        # T_render_in_ECEF_c2w[:3, 3] = (-T_render.R.T @ T_render.t).cpu().numpy() 

        # T_render_in_ECEF_c2w[:3, 3] = T_render_in_ECEF_c2w[:3, 3] / mul
        # T_render_in_ECEF_c2w[:3, 3] += (origin / mul)
        # euler_angles_refined, translation_refined, _, _ = pixloc_to_osg(T_render_in_ECEF_c2w)
        # print("refined euler angles: ",  euler_angles_refined, translation_refined)
    return Points_3D_ECEF_origin_center, T_render, initial_poses_w2c, dd

def get_3D_samples_v2(mkpts_r, depth_mat, T_c2w, camera, euler_angles, translation, last_frame_info = {}, origin = None, device = 'cuda', num_init_pose = 32, T_query_in_ECEF_c2w = None, mul = None):
    '''
    1. 欧拉角转R， 经纬高转xyz
    2. preprocess_param， [:3, 1] = -[:3, 1], [:3, 2] = - [:3, 2]
    3. * mul
    4. -origin
    5. -dd
    6. 加噪*
    '''
    # euler_angles_gt, translation_gt, _, _ = pixloc_to_osg(T_c2w)
    # print('-----gt', euler_angles_gt, translation_gt)
    # print(T_c2w)
    # start_time = time.time()
    query_euler_angles = generate_rotvecs(base_pitch = euler_angles[0], base_roll=euler_angles[1], base_yaw=euler_angles[2])
    query_T_c2w = euler_angles_to_matrix_ECEF_batch_speical(query_euler_angles, translation)
    
    # end_time1 = time.time()
    # print('seed: ', end_time1 - start_time)
    # start_time = time.time()
    # query_T_c2w[0] = T_c2w
    query_T_c2w[:, :3, 1] = -query_T_c2w[:, :3, 1]  # Y轴取反，投影后二维原点在左上角
    query_T_c2w[:, :3, 2] = -query_T_c2w[:, :3, 2]  # Z轴取反
    min_offset_angle = 0.0
    max_offset_angle = 0.0
    min_offset_translation = 0.0
    max_offset_translation = 5.0 # 0.015, 0.025 
    # preprocess
    render_camera, render_T = preprocess_param(copy.deepcopy(camera), copy.deepcopy(T_c2w))
    cx, cy = render_camera.c
    fx, fy = render_camera.f
    _, render_height_px = render_camera.size
    render_K = np.array([[fx, 0, cx],[0, fy, cy], [0, 0, 1]])
    render_K = torch.tensor(render_K, device=device)
    K_c2w = render_K.inverse()
    
    render_T = torch.tensor(render_T, device=device)
    mkpts_r = torch.tensor(mkpts_r, device=device)
    # end_time2 = time.time()
    # print('seed 2: ', end_time2 - start_time)
    # start_time = time.time()
    depth, valid = read_valid_depth(mkpts_r, depth = depth_mat, device=device)
    
    # end_time3 = time.time()
    # print('seed 3: ', end_time3 - start_time)
    # start_time = time.time()
    # Compute 3D points
    Points_3D_ECEF = get_Points3D_torch_normal(
        depth,
        render_T[:3, :3],
        render_T[:3, 3],
        K_c2w,
        mkpts_r[valid])
    #----mul
    # end_time4 = time.time()
    # print('seed 4: ', end_time4 - start_time)
    # start_time = time.time()
    if mul is not None:
        Points_3D_ECEF = Points_3D_ECEF * mul
        render_T[:3, 3] = render_T[:3, 3] * mul
        max_offset_translation = max_offset_translation *mul
        origin = np.array(origin)*mul
        query_T_c2w[:, :3, 3] = query_T_c2w[:, :3, 3] * mul

    # origin 
    if origin is None:
        origin = Points_3D_ECEF[0]
    Points_3D_ECEF_origin = Points_3D_ECEF - np.tile(origin, (len(Points_3D_ECEF), 1))
    if render_T.is_cuda:
        render_T = render_T.cpu()
    render_T = render_T.numpy()   
    # render_T = render_T.cpu().numpy()
    render_T[:3, 3] -= origin  # t_c2w - origin
    render_T_c2w = Pose.from_Rt(render_T[:3, :3], render_T[:3, 3])  # w2c
    T_render = render_T_c2w.inv()

    query_T_c2w[:, :3, 3] -= origin
    T_query = Pose.from_Rt(query_T_c2w[:, :3, :3], query_T_c2w[:, :3, 3]).inv()  # w2c
    center = True
    if center:
        # center
        points3D_total = torch.from_numpy(Points_3D_ECEF_origin).float()
        points_max = points3D_total.max(dim=0)[0]
        points_min = points3D_total.min(dim=0)[0]
        points_size = points_max - points_min
        dd = points_min + points_size / 2
        Points_3D_ECEF_origin_center = points3D_total - dd

        tt = T_render.t + T_render.R @ dd.double()
        T_render = Pose.from_Rt(T_render.R, tt)

        tt = T_query.t + T_query.R @ dd.double()
        T_query = Pose.from_Rt(T_query.R, tt)

    else:
        points3D_total = torch.from_numpy(Points_3D_ECEF_origin).float()
        Points_3D_ECEF_origin_center = points3D_total
        dd = torch.tensor([0,0,0])
    if 'candidate_poses' not in last_frame_info.keys():
        T_query_c2w = T_query.inv()
        random_aa, random_t = generate_random_aa_and_t(min_offset_angle, max_offset_angle, min_offset_translation = min_offset_translation, max_offset_translation = max_offset_translation)
        
        random_pose = Pose.from_aa(random_aa, random_t)
        initial_poses = T_query_c2w @ random_pose.double()  #c2w
        initial_poses_w2c = initial_poses.inv()
    # end_time5 = time.time()
    # print('seed 5: ', end_time5 - start_time)
    return Points_3D_ECEF_origin_center, T_render.float(), initial_poses_w2c.float(), dd

def get_3D_samples_v3(mkpts_r, depth_mat, T_c2w, camera, euler_angles, translation,  query_euler_angles, query_translation, last_frame_info = {}, origin = None, device = 'cuda', num_init_pose = 32, T_query_in_ECEF_c2w = None, mul = None):
    '''
    1. 欧拉角转R， 经纬高转xyz
    2. preprocess_param， [:3, 1] = -[:3, 1], [:3, 2] = - [:3, 2]
    3. * mul
    4. -origin
    5. -dd
    6. 加噪*
    '''
    # 设置撒种子步长，步长为1米，种子数量为16
    # print(' euler: ', euler_angles, query_euler_angles)
    query_euler_angles = generate_rotvecs_cuda_sym3d( base_pitch = query_euler_angles[0], base_roll=query_euler_angles[1], base_yaw=query_euler_angles[2], max_pitch=15.0, pitch_step=2.0, max_yaw=15.0,yaw_step=2.0, max_roll=1.0, roll_step=1.0)
    
    # query_euler_angles = generate_rotvecs_cuda_sym3d( base_pitch = euler_angles[0], base_roll=euler_angles[1], base_yaw=euler_angles[2], max_pitch=15.0, pitch_step=2.0, max_yaw=15.0,yaw_step=2.0, max_roll=1.0, roll_step=1.0)
    # for ii in range(len(query_euler_angles)):
    #     print(ii, query_euler_angles[ii])
    # query_euler_angles = generate_rotvecs_cuda(pitch_angles = [-7, -5, -3, -1, 1, 3, 5, 7], yaw_angles  = [-7, -5, -3, -1, 1, 3, 5, 7], base_pitch = euler_angles[0], base_roll=euler_angles[1], base_yaw=euler_angles[2])
    # query_euler_angles = generate_rotvecs_cuda(pitch_angles = [-10, -7, -4, -1, 1, 4, 7, 10], yaw_angles  = [-10, -7, -4, -1, 1, 4, 7, 10], base_pitch = euler_angles[0], base_roll=euler_angles[1], base_yaw=euler_angles[2])
    
    query_T_c2w = euler_angles_to_matrix_ECEF_batch_special_cuda(query_euler_angles, translation)
    # end_time1 = time.time()
    # print('seed: ', end_time1 - start_time)
    # start_time = time.time()
    # query_T_c2w[0] = T_c2w
    query_T_c2w[:, :3, 1] = -query_T_c2w[:, :3, 1]  # Y轴取反，投影后二维原点在左上角
    query_T_c2w[:, :3, 2] = -query_T_c2w[:, :3, 2]  # Z轴取反
    min_offset_angle = 0.0
    max_offset_angle = 0.0
    min_offset_translation = 0.0
    max_offset_translation = 1.0 # 0.015, 0.025 
    # preprocess
    render_camera, render_T = preprocess_param_cuda(copy.deepcopy(camera), copy.deepcopy(T_c2w))
    cx, cy = render_camera.c
    fx, fy = render_camera.f
    _, render_height_px = render_camera.size
    render_K = torch.tensor([[fx, 0, cx],[0, fy, cy], [0, 0, 1]],device=device)
    K_c2w = render_K.inverse()
    
    render_T = torch.tensor(render_T, device=device)
    mkpts_r = torch.tensor(mkpts_r, device=device)
    # end_time2 = time.time()
    # print('seed 2: ', end_time2 - start_time)
    # start_time = time.time()
    depth, valid = read_valid_depth(mkpts_r, depth = depth_mat, device=device)
    # end_time3 = time.time()
    # print('seed 3: ', end_time3 - start_time)
    # start_time = time.time()
    # Compute 3D points
    Points_3D_ECEF = get_Points3D_torch_cuda(
        depth,
        render_T[:3, :3],
        render_T[:3, 3],
        K_c2w,
        mkpts_r[valid])
    # Points_3D_ECEF = torch.tensor(Points_3D_ECEF).to(device)
    #----mul
    # end_time4 = time.time()
    # print('seed 4: ', end_time4 - start_time)
    # start_time = time.time()
    if mul is not None:
        Points_3D_ECEF = Points_3D_ECEF * mul
        render_T[:3, 3] = render_T[:3, 3] * mul
        max_offset_translation = max_offset_translation *mul
        origin = origin*mul
        query_T_c2w[:, :3, 3] = query_T_c2w[:, :3, 3] * mul

    # origin 
    if origin is None:
        origin = Points_3D_ECEF[0]
    # Points_3D_ECEF_origin = Points_3D_ECEF - np.tile(origin, (len(Points_3D_ECEF), 1))
    Points_3D_ECEF_origin = Points_3D_ECEF - origin
    # if render_T.is_cuda:
    #     render_T = render_T.cpu()
    # render_T = render_T.numpy()   
    # render_T = render_T.cpu().numpy()
    render_T[:3, 3] -= origin  # t_c2w - origin
    render_T_c2w = Pose.from_Rt(render_T[:3, :3], render_T[:3, 3])  # w2c
    T_render = render_T_c2w.inv().float()

    query_T_c2w[:, :3, 3] -= origin
    T_query = Pose.from_Rt(query_T_c2w[:, :3, :3], query_T_c2w[:, :3, 3]).inv()  # w2c
    center = True
    if center:
        points3D_total = Points_3D_ECEF_origin
        points_max = points3D_total.max(dim=0)[0]
        points_min = points3D_total.min(dim=0)[0]
        points_size = points_max - points_min
        dd = points_min + points_size / 2
        Points_3D_ECEF_origin_center = points3D_total - dd

        tt = T_render.t + T_render.R @ dd
        T_render = Pose.from_Rt(T_render.R, tt)

        tt = T_query.t + T_query.R @ dd
        T_query = Pose.from_Rt(T_query.R, tt)

    else:
        points3D_total = Points_3D_ECEF_origin
        Points_3D_ECEF_origin_center = points3D_total
        dd = torch.tensor([0,0,0])
    if 'candidate_poses' not in last_frame_info.keys():
        T_query_c2w = T_query.inv()
        random_aa, random_t = generate_random_aa_and_t_cuda(
            n = len(query_euler_angles),
            min_offset_angle = min_offset_angle,
            max_offset_angle = max_offset_angle,
            min_offset_translation = min_offset_translation,
            max_offset_translation = max_offset_translation
        )
        random_pose = Pose.from_aa(random_aa, random_t)
        initial_poses = T_query_c2w @ random_pose  #c2w
        initial_poses_w2c = initial_poses.inv()
    # end_time5 = time.time()
    # print('seed 5: ', end_time5 - start_time)
    return Points_3D_ECEF_origin_center, T_render, initial_poses_w2c, dd
def generate_render_camera(camera):
    if len(camera) == 5:
        image_width_px, image_height_px, sensor_width_mm, sensor_height_mm, f_mm = camera
        # 计算内参矩阵中的焦距
        focal_ratio_x = f_mm / sensor_width_mm
        focal_ratio_y = f_mm / sensor_height_mm
        
        fx = image_width_px * focal_ratio_x
        fy = image_height_px * focal_ratio_y

        # 计算主点坐标
        cx = image_width_px / 2
        cy = image_height_px / 2
    elif len(camera) == 7:
        image_width_px, image_height_px, cx, cy, sensor_width_mm, sensor_height_mm, f_mm = camera
        # 计算内参矩阵中的焦距
        focal_ratio_x = f_mm / sensor_width_mm
        focal_ratio_y = f_mm / sensor_height_mm
        
        fx = image_width_px * focal_ratio_x
        fy = image_height_px * focal_ratio_y
    elif len(camera) == 8:
        image_width_px, image_height_px, cx, cy, sensor_width_mm, sensor_height_mm, fx_mm, fy_mm = camera
        # 计算内参矩阵中的焦距
        focal_ratio_x = fx_mm / sensor_width_mm
        focal_ratio_y = fy_mm / sensor_height_mm
        
        fx = image_width_px * focal_ratio_x
        fy = image_height_px * focal_ratio_y    
    elif len(camera) == 6:
        image_width_px, image_height_px, cx, cy, fx, fy= camera
        # 计算内参矩阵中的焦距
        focal_ratio_x = fx / image_width_px
        focal_ratio_y = fy / image_height_px
        
        fx = image_width_px * focal_ratio_x
        fy = image_height_px * focal_ratio_y   
    camera_param = {
    'model': 'PINHOLE',
    'width': image_width_px,
    'height': image_height_px,
    'params': np.array([fx, fy, cx, cy])}
    cams = Camera.from_colmap(camera_param)

    return cams
def get_Points3D_torch_cuda(depth, R, t, K, points):
    """
    根据相机的内参矩阵、姿态（旋转矩阵和平移向量）以及图像上的二维点坐标和深度信息，
    计算对应的三维世界坐标。

    参数:
    - depth: 深度值数组，尺寸为 [n,]，其中 n 是点的数量。
    - R: 旋转矩阵，尺寸为 [3, 3]，表示从世界坐标系到相机坐标系的旋转。
    - t: 平移向量，尺寸为 [3, 1]，表示从世界坐标系到相机坐标系的平移。
    - K: 相机内参矩阵，尺寸为 [3, 3]，包含焦距和主点坐标。
    - points: 二维图像坐标数组，尺寸为 [n, 2]，其中 n 是点的数量。

    返回:
    - Points_3D: 三维世界坐标数组，尺寸为 [n, 3]。
    """
    # 检查points是否为同质坐标，如果不是则扩展为同质坐标
    if points.shape[-1] != 3:
        points_2D = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
        points_2D = points_2D.T
    else:
        points_2D = points.T

    # 扩展平移向量以匹配点的数量
    t = t.unsqueeze(1)  # 这相当于np.expand_dims(t, -1)
    t = t.repeat(1, points_2D.size(-1))  # 这相当于np.tile(t, points_2D.shape[-1])

    # 将所有输入转换为高精度浮点数类型
    points_2D = points_2D.float()
    K = K.float()
    R = R.float()
    depth = depth.float()
    t = t.float()

    # 修改内参矩阵的最后一项，以适应透视投影
    # K[-1, -1] = -1

    # 计算三维世界坐标
    Points_3D = R @ (K @ (depth * points_2D)) + t

    # 返回三维点坐标，形状为 [n, 3]
    return Points_3D.T
def get_Points3D_torch_normal(depth, R, t, K, points):
    """
    根据相机的内参矩阵、姿态（旋转矩阵和平移向量）以及图像上的二维点坐标和深度信息，
    计算对应的三维世界坐标。

    参数:
    - depth: 深度值数组，尺寸为 [n,]，其中 n 是点的数量。
    - R: 旋转矩阵，尺寸为 [3, 3]，表示从世界坐标系到相机坐标系的旋转。
    - t: 平移向量，尺寸为 [3, 1]，表示从世界坐标系到相机坐标系的平移。
    - K: 相机内参矩阵，尺寸为 [3, 3]，包含焦距和主点坐标。
    - points: 二维图像坐标数组，尺寸为 [n, 2]，其中 n 是点的数量。

    返回:
    - Points_3D: 三维世界坐标数组，尺寸为 [n, 3]。
    """
    # 检查points是否为同质坐标，如果不是则扩展为同质坐标
    if points.shape[-1] != 3:
        points_2D = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
        points_2D = points_2D.T
    else:
        points_2D = points.T

    # 扩展平移向量以匹配点的数量
    t = t.unsqueeze(1)  # 这相当于np.expand_dims(t, -1)
    t = t.repeat(1, points_2D.size(-1))  # 这相当于np.tile(t, points_2D.shape[-1])

    # 将所有输入转换为高精度浮点数类型
    points_2D = points_2D.float()
    K = K.float()
    R = R.float()
    depth = depth.float()
    t = t.float()

    # 修改内参矩阵的最后一项，以适应透视投影
    # K[-1, -1] = -1

    # 计算三维世界坐标
    Points_3D = R @ (K @ (depth * points_2D)) + t

    # 返回三维点坐标，形状为 [n, 3]
    return Points_3D.cpu().numpy().T
def get_3D_samples_dev(mkpts_r, depth_mat, render_T, render_camera, device = 'cuda'):
    # render T is in CGCS2000 format
    
    # in ECEF
    cx, cy = render_camera.c
    fx, fy = render_camera.f
    render_width_px, render_height_px = render_camera.size
    render_K = np.array([[fx, 0, cx],[0, fy, cy], [0, 0, 1]])
    render_K = torch.tensor(render_K, device=device)
    K_c2w = render_K.inverse()
    
    render_T = torch.tensor(render_T, device=device)
    mkpts_r = torch.tensor(mkpts_r, device=device)
    
    depth, valid = read_valid_depth(mkpts_r, depth = depth_mat, device=device)
    # Compute 3D points
    #!转换到OSG屏幕坐标系下反投影求3D点
    mkpts_r_in_osg = copy.deepcopy(mkpts_r[valid])
    mkpts_r_in_osg[:, 1] = render_height_px - mkpts_r_in_osg[:, 1]
    Points_3D_ECEF = get_Points3D_torch(
        depth,
        render_T[:3, :3],
        render_T[:3, 3],
        K_c2w,
        mkpts_r_in_osg
    ) #ECEF format
    origin = Points_3D_ECEF[0]
    Points_3D_ECEF_origin = Points_3D_ECEF - np.tile(origin, (len(Points_3D_ECEF), 1))   
    
    render_T = render_T.cpu().numpy()

# --------seed
    R_c2w, t_c2w = render_T[:3, :3], render_T[:3, 3]

    translation = ECEF_to_WGS84(t_c2w)
    lon, lat, _ = translation
    # 计算从ENU到ECEF的旋转矩阵
    rot_enu_to_ecef = get_rotation_enu_in_ecef(lon, lat)

    rot_ecef_to_enu = rot_enu_to_ecef.T  #! ECEF TO WGS84 transformation
    # 将ECEF姿态矩阵转换为ENU姿态矩阵
    rot_pose_in_enu = np.matmul(rot_ecef_to_enu, R_c2w)
    # 从ENU姿态矩阵中提取欧拉角
    rot_pose_in_enu_obj = R.from_matrix(rot_pose_in_enu)
    euler_angles = rot_pose_in_enu_obj.as_euler('xyz', degrees=True)

    min_offset_angle = 1.0
    max_offset_angle = 3.0 # 15.0, 25.0
    min_offset_translation = 2.5
    max_offset_translation = 5.0 # 0.015, 0.025

    pose_candidates = add_noise_to_pose(euler_angles, t_c2w,noise_std_angle=1, noise_std_translation=1)
    pose_candidates.append(render_T) # 8 seeds

# -------------2D-3D
    initial_poses = []
    # origin = np.array([-2195256. ,  5173871. ,  3005338.5]).astype(float)
    for render_T in pose_candidates:
        render_T = torch.tensor(render_T)
        render_T[:3, 3] -= origin  # t_c2w - origin
        render_T[:3, 1] = -render_T[:3, 1]  # Y轴取反，投影后二维原点在左上角
        render_T[:3, 2] = -render_T[:3, 2]  # Z轴取反
        
        # T_render_in_ECEF_w2c = torch.eye(4)
        # T_render_in_ECEF_w2c[:3, :3] = render_T[:3, :3].T
        # T_render_in_ECEF_w2c[:3, 3] = -render_T[:3, :3].T @ render_T[:3, 3]

        T_render_in_ECEF_w2c = Pose.from_Rt(*(render_T[:3, :3].T, -render_T[:3, :3].T @ render_T[:3, 3]))
        initial_poses.append(T_render_in_ECEF_w2c)
    T_render = copy.deepcopy(initial_poses[-1])
    # render_width_px, render_height_px = render_camera.size
    # render_K = torch.tensor([[fx, 0, cx],[0, fy, render_height_px - cy], [0, 0, 1]]).cuda()
    # K_c2w = render_K.inverse()
    # mkpts_r_in_osg = copy.deepcopy(mkpts_r[valid])
    # for i in range(len(Points_3D_ECEF_origin)):
    #     point2d = get_points2D_CGCS2000(
    #         render_T[:3, :3],
    #         render_T[:3, 3],
    #         K_c2w.cpu(),
    #         torch.tensor(Points_3D_ECEF_origin[i])
    #     ) #ECEF format
    #     print(mkpts_r_in_osg[i], point2d)
    # print(render_T)
    # print(K_c2w)
    
# -------------verify 
    # points_3D = np.float64(Points_3D_ECEF_origin[0])
    # K = [[fx, 0, cx], [0, fy, 1080 - cy], [0, 0, 1]]
    # K = np.float64(K)
    # R = T_render_in_ECEF_w2c[:3, :3]
    # t = T_render_in_ECEF_w2c[:3, 3]
    # R = np.float64(R)
    # t = np.float64(t)
    # t = -R.T @ t
    # # 计算相机坐标系下的点
    # point_3d_camera = np.expand_dims(points_3D - t, 1)
    # # 将世界坐标系下的点转换为相机坐标系下的点
    # point_3d_camera_r = R @ point_3d_camera
    # # 将相机坐标系下的点投影到图像平面，得到同质坐标
    # point_2d_homo = K @ point_3d_camera_r
    # # 将同质坐标转换为二维图像坐标
    # point_2d = point_2d_homo / point_2d_homo[2]
    # print("verify updated pose and intrinsic: ", point_2d, mkpts_r[valid][0])
    
    
    return Points_3D_ECEF_origin, T_render, initial_poses, origin
    
def zero_pad(size, image):
    
    h, w = image.shape[:2]
    # import ipdb; ipdb.set_trace();
    padded = np.zeros((size, size)+image.shape[2:], dtype=image.dtype)
    padded[:h, :w] = image
        
    return padded