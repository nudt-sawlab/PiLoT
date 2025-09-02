import torch
import numpy as np
import pyproj
import copy
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
    mkpts1r = mkpts1r.double().to(device)

    mkpts1r_inter = mkpts1r[:, [1, 0]].to(device)

    depth, _, valid = interpolate_depth(mkpts1r_inter, depth)

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
def transform_ecef_origin(render_T, origin):
    if render_T.ndim == 3:  # [B, 4, 4]
        render_T[:, :3, 3] -= origin  # 对所有批次的平移部分减去 origin
        render_T[:, :3, 1] = -render_T[:, :3, 1]  # 对所有批次的 Y 轴取反
        render_T[:, :3, 2] = -render_T[:, :3, 2]  # 对所有批次的 Z 轴取反
    elif render_T.ndim == 2:  # [4, 4]
        render_T[:3, 3] -= origin  # 对单个矩阵的平移部分减去 origin
        render_T[:3, 1] = -render_T[:3, 1]  # Y 轴取反
        render_T[:3, 2] = -render_T[:3, 2]  # Z 轴取反
    return render_T
def uniform_sample_with_interval(width, height, interval):
    """
    在指定大小的图像上，按照给定的间隔均匀采样点。
    
    Parameters:
        width (int): 图像宽度
        height (int): 图像高度
        interval (int): 采样间隔（像素）

    Returns:
        points (numpy.ndarray): 均匀采样的点坐标列表，形状为 (N, 2)
    """
    # 使用 np.arange 生成均匀间隔的坐标
    x = np.arange(0, width, interval)
    y = np.arange(0, height, interval)
    
    # 生成网格点
    xv, yv = np.meshgrid(x, y)
    
    # 合并成点的坐标
    points = np.stack((xv.flatten(), yv.flatten()), axis=-1)
    return points
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
    valid_positions = valid_positions[:, [1, 0]]
    height, width = depth_map.shape

    if len(valid_positions) == 0:
        raise ValueError("No valid points found with depth less than max_depth.")

    # Randomly select `num_points` indices from valid positions
    # num_points = min(num_points, len(valid_positions))  # Ensure not exceeding available points
    # selected_indices = np.random.choice(len(valid_positions), size=num_points, replace=False)
    # sampled_points = valid_positions[selected_indices]
    selected_indices = uniform_sample_with_interval(width, height, 5)
    # 转换为集合以加速查找
    valid_positions_set = set(map(tuple, valid_positions))
    
    # 筛选出有效的索引
    filtered_indices = [index for index in selected_indices if tuple(index) in valid_positions_set]
    
    return np.array(filtered_indices)
    # Extract the corresponding coordinates
    

    # return selected_indices#np.array([[int(x[1]), int(x[0])] for x in sampled_points])  # Return as (x, y)
def get_3D_samples(mkpts_r, depth_mat, render_T, render_camera, origin = None, device = 'cuda'):
    # render T is in CGCS2000 format
    
    # in ECEF
    cx, cy = render_camera.c
    fx, fy = render_camera.f
    render_width_px, render_height_px = render_camera.size
    render_K = np.array([[fx, 0, cx],[0, fy, cy], [0, 0, 1]])
    render_K = torch.tensor(render_K, device=device)
    K_c2w = render_K.inverse()
    
    render_T = torch.tensor(render_T, device=device)
    if origin is not None:
        render_T[:3, 3] -= origin  # t_c2w - origin
    render_T[:3, 1] = -render_T[:3, 1]  # Y轴取反，投影后二维原点在左上角
    render_T[:3, 2] = -render_T[:3, 2]  # Z轴取反
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
        Points_3D_ECEF_origin = Points_3D_ECEF - np.tile(origin, (len(Points_3D_ECEF), 1))   
        return mkpts_r[valid], Points_3D_ECEF, origin, Points_3D_ECEF_origin
    
    return mkpts_r[valid], Points_3D_ECEF, origin
def get_3D_samples_v2(mkpts_r, depth_mat, render_T, render_camera, origin = None, device = 'cuda'):
    # render T is in CGCS2000 format
    
    # in ECEF
    w, h, fx, fy, cx, cy = render_camera
    render_width_px, render_height_px = render_camera.size
    render_K = np.array([[fx, 0, cx],[0, fy, cy], [0, 0, 1]])
    render_K = torch.tensor(render_K, device=device)
    K_c2w = render_K.inverse()
    
    render_T = torch.tensor(render_T, device=device)
    if origin is not None:
        render_T[:3, 3] -= origin  # t_c2w - origin
    render_T[:3, 1] = -render_T[:3, 1]  # Y轴取反，投影后二维原点在左上角
    render_T[:3, 2] = -render_T[:3, 2]  # Z轴取反
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
        Points_3D_ECEF_origin = Points_3D_ECEF - np.tile(origin, (len(Points_3D_ECEF), 1))   
        return mkpts_r[valid], Points_3D_ECEF, origin, Points_3D_ECEF_origin
    
    return mkpts_r[valid], Points_3D_ECEF, origin
def get_3D_samples_dev(mkpts_r, depth_mat, render_T, render_camera, origin = None, device = 'cuda'):
    # render T is in CGCS2000 format
    
    # in ECEF
    cx, cy = render_camera.c
    fx, fy = render_camera.f
    render_width_px, render_height_px = render_camera.size
    render_K = np.array([[fx, 0, cx],[0, fy, cy], [0, 0, 1]])
    render_K = torch.tensor(render_K, device=device)
    K_c2w = render_K.inverse()
    
    render_T = torch.tensor(render_T, device=device)
    
    # render_T[:3, 3] -= origin  # t_c2w - origin
    render_T[:3, 1] = -render_T[:3, 1]  # Y轴取反，投影后二维原点在左上角
    render_T[:3, 2] = -render_T[:3, 2]  # Z轴取反
    mkpts_r = torch.tensor(mkpts_r, device=device)
    
    depth, valid = read_valid_depth(mkpts_r, depth = depth_mat, device=device)
    # Compute 3D points
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
    points_2D = points_2D.float()
    K = K.float()
    R = R.float()
    depth = depth.float()
    t = t.float()

    # 计算三维世界坐标
    Points_3D = R @ (K @ (depth * points_2D)) + t
    Points_3D_ECEF = Points_3D.cpu().numpy().T

    # 返回三维点坐标，形状为 [n, 3]
    # return Points_3D.cpu().numpy().T
    # mkpts_r_in_osg[:, 1] = render_height_px - mkpts_r_in_osg[:, 1]
    # Points_3D_ECEF = get_Points3D_torch(
    #     depth,
    #     render_T[:3, :3],
    #     render_T[:3, 3],
    #     K_c2w,
    #     mkpts_r_in_osg
    # ) #ECEF format
    if origin is None:
        origin = Points_3D_ECEF[0]
    # origin = torch.tensor(origin, device=device)
    Points_3D_ECEF_origin = Points_3D_ECEF - np.tile(origin, (len(Points_3D_ECEF), 1))   
    
    
    # transform Points_3D to CGCS2000 format
    # Points_3D_CGCS2000 = ecef_to_gausskruger_pyproj(Points_3D_ECEF)  # renturn CGCS2000, set EPSG!!
    
    # point2d = get_points2D_ECEF(
    #     render_T[:3, :3].cpu(),
    #     render_T[:3, 3].cpu(),
    #     K_c2w.cpu(),
    #     torch.tensor(Points_3D_ECEF[0])
    #     )
    # print("points: ", render_T, point2d) 
# -------------2D-3D
    # mkpts_r_in_osg = copy.deepcopy(mkpts_r[valid])

    # render_T = render_T.cpu().numpy()
    # render_T[:3, 3] -= origin  # t_c2w - origin
    # render_T[:3, 1] = -render_T[:3, 1]  # Y轴取反，投影后二维原点在左上角
    # render_T[:3, 2] = -render_T[:3, 2]  # Z轴取反
    
    # T_render_in_ECEF_w2c = np.identity(4)
    # T_render_in_ECEF_w2c[:3, :3] = render_T[:3, :3].T
    # T_render_in_ECEF_w2c[:3, 3] = -render_T[:3, :3].T @ render_T[:3, 3]
    # render_width_px, render_height_px = render_camera.size
    # render_K = torch.tensor([[fx, 0, cx],[0, fy, render_height_px - cy], [0, 0, 1]]).cuda()
    # K_c2w = render_K.inverse()

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
# -------------2D-3D  
    
    
    
    # points_3D = np.float64(Points_3D_CGCS2000[0])
    # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    # K = np.float64(K)
    # R = render_T_cgcs2000.R
    # t = render_T_cgcs2000.t
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
    
    
    return mkpts_r[valid], Points_3D_ECEF, origin

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
            return  point2d_ref, points2d_query, points_3D_, true_indices
        # else:
        #     sampled_indices = np.random.choice(len(points_3D), size=num_samples, replace=False)
        #     point2d_ref = point_2d[sampled_indices]
        #     points2d_query = point2d_total[sampled_indices]
        #     points_3D_ = points_3D[sampled_indices]
        return  point_2d, point2d_total, points_3D, None
        # return point2d_ref, points2d_query, points_3D_, true_indices