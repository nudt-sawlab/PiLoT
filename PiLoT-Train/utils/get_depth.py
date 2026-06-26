import os
import torch
import numpy as np
import pyproj
import copy
import time
import cv2
# from pixloc.utils.transform import ECEF_to_WGS84
def sample_points_with_valid_depth( depth_map, num_points=500, max_depth=65534):
    """
    Randomly sample points on an image, ensuring the corresponding depth value is valid.

    :param image: Input image (used for shape reference).
    :param depth_map: Depth map (same size as image).
    :param num_points: Number of points to sample.
    :param max_depth: Maximum valid depth value.
    :return: List of sampled points [(x1, y1), (x2, y2), ...].
    """

    # Find all valid depth positions
    valid_positions = np.argwhere(depth_map < max_depth)

    if len(valid_positions) == 0:
        raise ValueError("No valid points found with depth less than max_depth.")

    # Randomly select `num_points` indices from valid positions
    num_points = min(num_points, len(valid_positions))  # Ensure not exceeding available points
    selected_indices = np.random.choice(len(valid_positions), size=num_points, replace=False)

    # Extract the corresponding coordinates
    sampled_points = valid_positions[selected_indices]

    return np.array([[int(x[1]), int(x[0])] for x in sampled_points])  # Return as (x, y)
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

def optimize_depth_reading(pos, depth):
    # 1. Ensure depth is 2D
    if depth.ndimension() != 2:
        if depth.ndimension() == 3:
            depth = depth[:, :, 0]
        else:
            raise Exception("Invalid depth image!")

    h, w = depth.shape

    # 2. Extract i and j from pos
    i = pos[:, 0]
    j = pos[:, 1]

    # 3. Calculate floor and ceil once for efficiency
    i_floor = torch.floor(i).long()
    j_floor = torch.floor(j).long()

    i_ceil = torch.ceil(i).long()
    j_ceil = torch.ceil(j).long()

    # 4. Validity checks for corners (using a single line to compute the validity for all 4 corners)
    valid_top_left = (i_floor >= 0) & (j_floor >= 0)
    # valid_top_right = (i_floor >= 0) & (j_ceil < w)
    # valid_bottom_left = (i_ceil < h) & (j_floor >= 0)
    # valid_bottom_right = (i_ceil < h) & (j_ceil < w)

    valid_corners = valid_top_left #& valid_top_right & valid_bottom_left & valid_bottom_right

    # 5. Apply validity mask to get valid corners' indices
    i_floor = i_floor[valid_corners]
    j_floor = j_floor[valid_corners]

    # i_ceil = i_ceil[valid_corners]
    # j_ceil = j_ceil[valid_corners]
    # 6. Check depth validity for the valid corners
    valid_depth = (
        (depth[i_floor, j_floor] > 0)
        #  &
        # (depth[i_ceil, j_ceil] > 0)
    )[valid_corners]

    # 7. Apply depth validity to update i, j and corner coordinates
    i_floor = i_floor[valid_depth]
    j_floor = j_floor[valid_depth]

    # i_ceil = i_ceil[valid_depth]
    # j_ceil = j_ceil[valid_depth]

    # 8. Update ids based on valid depth
    ids = torch.arange(0, pos.shape[0], device=pos.device)
    ids = ids[valid_depth]

    i = i[ids]
    j = j[ids]

    # 9. Directly index depth values for valid corners
    depth_top_left = depth[i_floor, j_floor]
    # depth_top_right = depth[i_ceil, j_floor]
    # depth_bottom_left = depth[i_floor, j_ceil]
    # depth_bottom_right = depth[i_ceil, j_ceil]

    # 10. Combine valid depth values for the valid points
    # interpolated_depth = torch.stack([depth_top_left, depth_top_right, depth_bottom_left, depth_bottom_right], dim=1)

    # 11. Return depth values, position, and valid indices
    pos = torch.stack([i.view(1, -1), j.view(1, -1)], dim=0)

    return [depth_top_left, pos, ids]

# def read_valid_depth(mkpts1r, depth=None, device = 'cuda'):
#     depth = torch.tensor(depth).to(device)
#     mkpts1r = mkpts1r.double().to(device)

#     mkpts1r_a = torch.unsqueeze(mkpts1r[:,0],0)
#     mkpts1r_b =  torch.unsqueeze(mkpts1r[:,1],0)
#     mkpts1r_inter = torch.cat((mkpts1r_b ,mkpts1r_a),0).transpose(1,0).to(device)

#     depth, _, valid = interpolate_depth(mkpts1r_inter, depth)

#     return depth, valid

def read_valid_depth(mkpts1r, depth=None, device = 'cuda'):
    depth = torch.tensor(depth).to(device)
    mkpts1r = mkpts1r.double().to(device)

    mkpts1r_inter = mkpts1r[:, [1, 0]].to(device)
    depth, _, valid = optimize_depth_reading(mkpts1r_inter, depth)

    return depth, valid
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
def transform_ecef_origin(render_T_ecef, origin):
    render_T = copy.deepcopy(render_T_ecef)
    if render_T.ndim == 3:  # [B, 4, 4]
        render_T[:, :3, 3] -= origin  # 对所有批次的平移部分减去 origin
        # render_T[:, :3, 1] = -render_T[:, :3, 1]  # 对所有批次的 Y 轴取反
        # render_T[:, :3, 2] = -render_T[:, :3, 2]  # 对所有批次的 Z 轴取反

        render_T_c2w = copy.deepcopy(render_T)
        render_T_c2w[:, :3, :3] = np.transpose(render_T_c2w[:, :3, :3], (0, 2, 1))
        render_T_c2w[:, :3, 3] = -np.matmul(render_T_c2w[:, :3, :3], render_T_c2w[:, :3, 3][:, :, np.newaxis])[:, :, 0]
        
    elif render_T.ndim == 2:  # [4, 4]
        render_T[:3, 3] -= origin  # 对单个矩阵的平移部分减去 origin
        # render_T[:3, 1] = -render_T[:3, 1]  # Y 轴取反
        # render_T[:3, 2] = -render_T[:3, 2]  # Z 轴取反

        render_T_c2w = copy.deepcopy(render_T)
        render_T_c2w[:3, :3] = render_T_c2w[:3, :3].T
        render_T_c2w[:3, 3] =  - render_T_c2w[:3, :3] @ render_T_c2w[:3, 3]

        
    # import ipdb; ipdb.set_trace()

    return render_T, render_T_c2w.astype(np.float32)

def get_3D_samples(mkpts_r, depth_mat, render_T, render_camera, origin=None, device='cuda'):
    cx, cy = render_camera.c
    fx, fy = render_camera.f
    render_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    render_K = torch.tensor(render_K, device=device)
    K_c2w = render_K.inverse()
    render_T = torch.tensor(render_T, device=device)
    mkpts_r = torch.tensor(mkpts_r, device=device)

    if isinstance(depth_mat, (str, os.PathLike)):
        depth_mat = cv2.imread(str(depth_mat), cv2.IMREAD_UNCHANGED)
        depth_mat = cv2.flip(depth_mat, 0)

    depth, valid = read_valid_depth(mkpts_r, depth=depth_mat, device=device)
    mkpts_r_in_osg = copy.deepcopy(mkpts_r[valid])
    R, t = render_T[:3, :3], render_T[:3, 3]
    points = mkpts_r_in_osg
    points_2D = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1).T
    t = t.unsqueeze(1).repeat(1, points_2D.size(-1))
    points_2D = points_2D.double()
    K = K_c2w.double()
    R = R.double()
    depth = depth.double()
    t = t.double()
    Points_3D = R @ (K @ (depth * points_2D)) + t
    Points_3D_ECEF = Points_3D.cpu().numpy().T
    if origin is not None:
        Points_3D_ECEF_origin = Points_3D_ECEF - np.tile(origin, (len(Points_3D_ECEF), 1))
        return mkpts_r[valid], Points_3D_ECEF, origin, Points_3D_ECEF_origin
    return mkpts_r[valid], Points_3D_ECEF, origin, valid.cpu().numpy()

def get_points2D_ECEF(render_T, render_camera, points_3D):  # points_3D[n,3]
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

        cx, cy = render_camera.c
        fx, fy = render_camera.f
        render_width_px, render_height_px = render_camera.size
        render_K = np.array([[fx, 0, cx],[0, fy, cy], [0, 0, 1]])
        points_3D = np.float64(points_3D)

        K = np.float64(render_K)
        R = render_T[:3, :3]
        t = render_T[:3, 3]
        R = np.float64(R)
        t = np.float64(t)
        # 修改内参矩阵的最后一项，以适应透视投影
        K_c2w = np.linalg.inv(K)
        K_c2w[-1, -1] = -1
        
        K_inverse = np.linalg.inv(K_c2w)
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
def get_points2D_ECEF_projection(render_T, render_camera, points_3D, point2d_total, num_samples = 500, use_valid = True):  # points_3D[n,3]
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

        cx, cy = render_camera.c
        fx, fy = render_camera.f
        render_width_px, render_height_px = render_camera.size
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
        # point_3d_camera_r = R_inverse @ point_3d_camera

        # # 将相机坐标系下的点投影到图像平面，得到同质坐标
        # point_2d_homo = K_inverse @ point_3d_camera_r
        # # 将同质坐标转换为二维图像坐标
        # point_2d = point_2d_homo / point_2d_homo[2]
        # point_2d = point_2d.T   
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
            if len(true_indices) < num_samples:
                return None, None, None
            sampled_indices = np.random.choice(true_indices, size=num_samples, replace=False)

            point2d_ref = point_2d[sampled_indices]
            points2d_query = point2d_total[sampled_indices]
            points_3D_ = points_3D[sampled_indices]
            return point2d_ref, points2d_query, points_3D_
        return point_2d, point2d_total, points_3D, np.squeeze(point_2d_homo[:, 2, np.newaxis], axis=-1)