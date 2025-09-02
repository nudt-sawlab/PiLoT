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
    def back_project(self, depth_frame, euler_angles, translation, num_samples = 200):
        # 
        T_render_in_ECEF_c2w = euler_angles_to_matrix_ECEF(euler_angles, translation)
        # 反投影得到3D点
        # print('T render_______', euler_angles, translation)
        width, height = self.render_camera_osg[:2]
        ey = np.random.randint(0, height, size= num_samples)
        ex = np.random.randint(0, width, size= num_samples)
        points2d = np.column_stack((ex, ey))
        Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates, dd = get_3D_samples_v2(points2d, depth_frame, T_render_in_ECEF_c2w, self.render_camera,  euler_angles, translation, origin = self.origin, num_init_pose=self.num_init_pose,mul = self.mul)
        if dd is not None:
            self.dd = dd
        return Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates
