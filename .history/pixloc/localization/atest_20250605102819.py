import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
from omegaconf import DictConfig, OmegaConf as oc
from scipy.spatial.transform import Rotation as R
import os 
import numpy as np
import torch
import time
import copy
import cv2
from .feature_extractor import FeatureExtractor
from .tracker import BaseTracker
from ..pixlib.geometry import Pose, Camera
from ..pixlib.datasets.view import read_image
from ..utils.data import Paths
from ..utils.osg import osg_render
from ..utils.transform import kf_predictor, pixloc_to_osg, orthogonalize_rotation_matrix, move_inputs_to_cuda
from ..utils.get_depth import pad_to_multiple, zero_pad
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
overall_loss = ret['overall_loss']
fail_list = ret['fail_list']
T_candidtas = ret['T_opt']
# ------
T_opt_c2w = build_c2w_batch(T_candidtas, dd, mul, self.origin)
B = T_candidtas.shape[0]
T_render_in_ECEF_c2w = build_prior_batch(T_render, dd, mul, self.origin)

T_prior_ECEF = T_render_in_ECEF_c2w.unsqueeze(0).repeat(B, 1, 1)  # [B,4,4]
T_prior_ECEF_Pose = Pose.from_Rt(T_prior_ECEF[:, :3, :3], T_prior_ECEF[:, :3, 3])
T_opt_c2w_Pose = Pose.from_Rt(T_opt_c2w[:, :3, :3], T_opt_c2w[:, :3, 3])
dR, dt = (T_prior_ECEF_Pose.inv() @ T_opt_c2w_Pose).magnitude()

t_indices = dt <= dis_thes #!
R_indices = dR <= R_thes

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