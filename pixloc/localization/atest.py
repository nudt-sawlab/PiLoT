from omegaconf import DictConfig, OmegaConf as oc
from scipy.spatial.transform import Rotation as R
import os 
import numpy as np
import torch
import pyproj
import math
import time
def ECEF_to_WGS84(pos):
    xpjr, ypjr, zpjr = pos
    transprojr = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        "EPSG:4326",
        always_xy=True,
    )
    lon, lat, height = transprojr.transform(xpjr, ypjr, zpjr, radians=False)
    return [lon, lat, height]  
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

    kf_current_frame_es_pose = np.concatenate((t_c2w, euler_angles_in_enu))
    return euler_angles_in_enu, t_c2w_wgs84, T_ECEF, kf_current_frame_es_pose
def build_c2w_batch(T_batch, dd, mul, origin):
    """
    把一批 Pose（从世界到相机的 w2c 旋转和平移）转换成相机到世界的 c2w 坐标系下的 [B,4,4] 张量。
    - T_batch.R: Tensor[B,3,3]
    - T_batch.t: Tensor[B,3]
    - dd:       Tensor[3] 或 None
    - mul:      float 或 Tensor 标量
    - origin:   Tensor[3] 偏移量
    返回: Tensor[B,4,4]
    """
    # 1) Device & dtype
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.float64

    # 2) 准备输入
    R_in = T_batch.R.to(device=device, dtype=dtype)  # [B,3,3]
    t_in = T_batch.t.to(device=device, dtype=dtype)  # [B,3]
    if dd is not None:
        dd = dd.to(device=device, dtype=dtype)       # [3]

    mul    = torch.as_tensor(mul,    device=device, dtype=dtype)
    origin    = torch.as_tensor(origin,    device=device, dtype=dtype)
    # origin = origin.to(device=device, dtype=dtype)  # [3]

    B = R_in.shape[0]

    # 3) 批量正交化：SVD → U @ Vh
    U, S, Vh = torch.linalg.svd(R_in)
    R = U @ Vh                                        # [B,3,3] 仍是 w2c

    # 4) 调整平移（扣掉 dd）
    t = t_in
    if dd is not None:
        # dd 视作 [3]，自动广播到 [B,3]
        t = t - (R @ dd)

    # 5) 转置成 c2w
    R_c2w = R.transpose(-1, -2)                       # [B,3,3]

    # 6) 构建 [B,4,4] 单位矩阵
    T = torch.eye(4, device=device, dtype=dtype) \
             .unsqueeze(0).repeat(B,1,1)              # [B,4,4]
    T[:, :3, :3] = R_c2w

    # 7) 填平移、缩放、翻轴、加 origin
    #   t_unsq: [B,3,1]  => R_c2w @ t_unsq => [B,3,1]
    tr = (-R_c2w @ t.unsqueeze(-1)).squeeze(-1) / mul  # [B,3]
    T[:, :3, 3] = tr

    # 翻转 Y/Z 轴
    T[:, :3, 1:3] *= -1

    # 加上 origin 偏移
    T[:, :3, 3] += origin

    return T


def build_prior_batch(T_render, dd, mul, origin):
    """
    把单帧渲染 Pose 转为 ECEF c2w，然后复制 B 份：
    - T_render.R: Tensor[3,3]
    - T_render.t: Tensor[3]
    - dd, mul, origin 同上
    - B: 需要复制的 batch 大小
    返回: Tensor[B,4,4]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.float64

    # 上面函数里同样的输入强转
    R = T_render.R.to(device=device, dtype=dtype)
    t = T_render.t.to(device=device, dtype=dtype)
    if dd is not None:
        dd = dd.to(device=device, dtype=dtype)
        t = t - R @ dd

    mul    = torch.as_tensor(mul,    device=device, dtype=dtype)
    origin    = torch.as_tensor(origin,    device=device, dtype=dtype)

    # 转置成 c2w
    R_c2w = R.transpose(-1, -2)  # [3,3]

    # 计算平移
    tr = (-R_c2w @ t.unsqueeze(-1)).squeeze(-1) / mul  # [3]
    tr = tr + origin

    # 组装单个 4×4
    T_single = torch.eye(4, device=device, dtype=dtype)
    T_single[:3, :3] = R_c2w
    T_single[:3, 1:3] *= -1
    T_single[:3,  3] = tr

    # 复制 B 份
    return T_single        # [B,4,4]
def magnitude(T):
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    trace = torch.diagonal(R, dim1=-1, dim2=-2).sum(-1)
    cos = torch.clamp((trace - 1) / 2, -1, 1)
    dr = torch.acos(cos).abs() / math.pi * 180
    dt = torch.norm(t, dim=-1)
    return dr, dt
def inverse(T):
    R = T[..., :3, :3].transpose(-1, -2)
    t = -(R @ T[..., :3, 3].unsqueeze(-1)).squeeze(-1)
    return torch.cat([R, t.unsqueeze(-1)], dim=-1)  # [B,4,4]


ret = np.load('/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/ret.npy', allow_pickle=True).item()
for _ in range(100):
    t1 = time.time()
    overall_loss = ret['overall_loss']
    fail_list = ret['fail_list']
    T_candidtas = ret['T_opt']
    T_query_opt_poses = ret['T_opt']
    T_render = T_query_opt_poses[0]
    device = 'cuda'
    dd = torch.tensor([0,0,0]).to(device)
    origin = torch.tensor([100000,10000,10000]).to(device)
    mul = 0.001
    # ------
    T_opt_c2w = build_c2w_batch(T_candidtas, dd, mul, origin)
    B = T_candidtas.shape[0]
    T_render_in_ECEF_c2w = build_prior_batch(T_render, dd, mul, origin)

    T_prior_ECEF = T_render_in_ECEF_c2w.unsqueeze(0).repeat(B, 1, 1)  # [B,4,4]
    # T_prior_ECEF_Pose = Pose.from_Rt(T_prior_ECEF[:, :3, :3], T_prior_ECEF[:, :3, 3])
    # T_opt_c2w_Pose = Pose.from_Rt(T_opt_c2w[:, :3, :3], T_opt_c2w[:, :3, 3])
    T_prior_ECEF_Pose = T_prior_ECEF
    T_opt_c2w_Pose = T_opt_c2w
    dR, dt = magnitude(inverse(T_prior_ECEF_Pose) @ T_opt_c2w_Pose)

    t_indices = dt <= 8 #!
    R_indices = dR <= 8

    # 剔除旋转变化量和平移变化量过大的候选
    valid = (~fail_list) & t_indices & R_indices
    valid_loss = overall_loss[valid]

    if not any(valid):
        import ipdb; ipdb.set_trace()
    min_index_in_valid = torch.argmin(valid_loss)

    T_refined = ret['T_opt'][valid][min_index_in_valid]
    # ------
    pose_index = torch.nonzero(valid)[min_index_in_valid].item()
    dR, dt = (T_query_opt_poses[pose_index].inv() @ T_refined).magnitude()
    ret = {
        **ret,
        'T_refined': T_refined,
        'diff_R': dR.item(),
        'diff_t': dt.item(),
    }
    # choose the best estimate
    if T_opt_c2w.is_cuda:
        T_opt_c2w = T_opt_c2w.cpu()
    T_opt_c2w = T_opt_c2w.numpy()
    T_opt_c2w = T_opt_c2w[pose_index]
    euler_angles_refined, translation_refined, T_ECEF_estimated, kf_current_frame_es_pose = pixloc_to_osg(T_opt_c2w)

    ret['euler_angles'] = euler_angles_refined
    ret['translation'] = translation_refined
    t2 = time.time()
    print(t2-t1)