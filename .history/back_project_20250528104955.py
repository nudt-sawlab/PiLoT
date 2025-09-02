import time, torch, copy, numpy as np
from contextlib import contextmanager
from pixloc.pixlib.geometry import Pose
from this import d
import threading
from pixloc.utils.data import Paths
import os
import queue
import glob
import argparse
import numpy as np
import logging
from pixloc.utils.osg import osg_render
from pixloc.utils.transform import colmap_to_osg
from tqdm import tqdm
from pprint import pformat
from pixloc.settings import DATA_PATH, LOC_PATH
from pixloc.localization import RenderLocalizer, SimpleTracker
from pixloc.pixlib.geometry import Camera, Pose
from pixloc.pixlib.datasets.view import read_image_list, read_render_image_list
from pixloc.utils.colmap import qvec2rotmat
from pixloc.utils.data import Paths
from pixloc.utils.eval import evaluate
from pixloc.utils.get_depth import get_3D_samples, pad_to_multiple, generate_render_camera, get_3D_samples_v2
from pixloc.utils.transform import euler_angles_to_matrix_ECEF, pixloc_to_osg, WGS84_to_ECEF
from pixloc.utils import video_generation
import time
import yaml
import copy
# ---------- 小工具：高分辨率计时 ----------
@contextmanager
def tic(msg):
    t0 = time.perf_counter()          # 若在 GPU 上，想让 CUDA kernel 计入 → torch.cuda.synchronize()
    yield
    t1 = time.perf_counter()
    print(f"{msg:<35s}: {(t1 - t0)*1e3:8.3f} ms")

def my_pipeline(euler_angles, translation, camera, T_c2w,
                mkpts_r, depth_mat, mul=None, origin=None,
                last_frame_info={}, device="cpu"):
    with tic("1. generate_rotvecs"):
        query_euler_angles = generate_rotvecs(
            base_pitch=euler_angles[0],
            base_roll=euler_angles[1],
            base_yaw=euler_angles[2]
        )

    with tic("2. euler_angles_to_matrix (batch)"):
        query_T_c2w = euler_angles_to_matrix_ECEF_batch_speical(
            query_euler_angles, translation
        )
    query_T_c2w[:, :3, 1] *= -1
    query_T_c2w[:, :3, 2] *= -1

    # -- 预处理相机参数 -------------------------------------------------
    with tic("3. preprocess_param"):
        render_camera, render_T_np = preprocess_param(
            copy.deepcopy(camera), copy.deepcopy(T_c2w)
        )
        cx, cy = render_camera.c
        fx, fy = render_camera.f
        _, render_h = render_camera.size
        render_K = torch.tensor([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]], device=device).float()
        K_c2w = torch.inverse(render_K)

    with tic("4. depth + valid mask"):
        depth, valid = read_valid_depth(
            torch.as_tensor(mkpts_r, device=device),
            depth=depth_mat, device=device
        )

    # -- 计算 3D 点 -----------------------------------------------------
    with tic("5. back-project Points3D"):
        Points_3D_ECEF = get_Points3D_torch_normal(
            depth, torch.as_tensor(render_T_np[:3, :3], device=device),
            torch.as_tensor(render_T_np[:3, 3], device=device),
            K_c2w, torch.as_tensor(mkpts_r[valid], device=device)
        )

    # -- 可选缩放 -------------------------------------------------------
    if mul is not None:
        with tic("6. scaling mul"):
            Points_3D_ECEF *= mul
            render_T_np[:3, 3] *= mul
            max_offset_translation = 5.0 * mul
            origin = np.array(origin) * mul if origin is not None else None
            query_T_c2w[:, :3, 3] *= mul

    # -- 原点变换 -------------------------------------------------------
    if origin is None:
        origin = Points_3D_ECEF[0].cpu().numpy()

    with tic("7. origin shift"):
        pts_origin = Points_3D_ECEF - torch.as_tensor(origin, device=device)

    # -- 转 Pose -------------------------------------------------------
    with tic("8. render_T to Pose"):
        render_T_np[:3, 3] -= origin
        T_render = Pose.from_Rt(render_T_np[:3, :3], render_T_np[:3, 3]).inv()

    with tic("9. query_T to Pose"):
        query_T_c2w[:, :3, 3] -= origin
        T_query = Pose.from_Rt(
            query_T_c2w[:, :3, :3], query_T_c2w[:, :3, 3]).inv()

    # -- 居中 -----------------------------------------------------------
    with tic("10. center"):
        pts_total = pts_origin.float()
        if True:
            points_max = pts_total.max(dim=0)[0]
            points_min = pts_total.min(dim=0)[0]
            dd = points_min + (points_max - points_min) / 2
            pts_center = pts_total - dd
            T_render = Pose.from_Rt(T_render.R, T_render.t + T_render.R @ dd.double())
            T_query  = Pose.from_Rt(T_query.R,  T_query.t  + T_query.R  @ dd.double())
        else:
            pts_center = pts_total
            dd = torch.zeros(3)

    # -- 生成初值 -------------------------------------------------------
    with tic("11. random pose gen"):
        if 'candidate_poses' not in last_frame_info:
            T_query_c2w = T_query.inv()
            random_aa, random_t = generate_random_aa_and_t(
                0., 0., 0., 5.0
            )
            random_pose = Pose.from_aa(random_aa, random_t)
            initial_poses = (T_query_c2w @ random_pose.double()).inv()

    return pts_center, T_render.float(), initial_poses.float(), dd
