from itertools import cycle
import torch
import numpy as np
import pyproj
import copy
from torch import nn
import cv2

from pixloc.utils.transform import ECEF_to_WGS84, get_rotation_enu_in_ecef
from scipy.spatial.transform import Rotation as R
from ..pixlib.geometry import Pose, Camera
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
def read_valid_depth(mkpts1r, depth=None, device = 'cuda'):
    depth = torch.tensor(depth).to(device)
    mkpts1r = mkpts1r.double().to(device)

    mkpts1r_inter = mkpts1r[:, [1, 0]].to(device)

    depth, _, valid = interpolate_depth(mkpts1r_inter, depth)

    return depth, valid
def preprocess_param(camera, pose):
    pose[:3, 1] = -pose[:3, 1]  # Y轴取反，投影后二维原点在左上角
    pose[:3, 2] = -pose[:3, 2]  # Z轴取反

    _, h = camera.size
    camera.c[1] = h - camera.c[1]
    return camera, pose

def get_3D_samples(mkpts_r, depth_mat, render_T, render_camera, device = 'cuda'):
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

    return Points_3D_ECEF

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

