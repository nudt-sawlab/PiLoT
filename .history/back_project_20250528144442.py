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
def back_project(depth_frame, euler_angles, translation, render_camera, num_samples = 200):
    # 
    start_time = time.time()
    T_render_in_ECEF_c2w = euler_angles_to_matrix_ECEF(euler_angles, translation)
    end_time = time.time()
    print('T_render_in_ECEF_c2w: ', end_time - start_time)
    # 反投影得到3D点
    # print('T render_______', euler_angles, translation)
    width, height = 512, 288
    ey = np.random.randint(0, height, size= num_samples)
    ex = np.random.randint(0, width, size= num_samples)
    points2d = np.column_stack((ex, ey))
    Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates, dd = get_3D_samples_v2(points2d, depth_frame, T_render_in_ECEF_c2w, render_camera,  euler_angles, translation, origin = [0,0,0], num_init_pose=200,mul = 0.01)

    return Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates
def generate_render_camera(camera):
    w, h, cx, cy, sensor_width, sensor_height, fmm = camera
    fx = w / sensor_width * fmm
    fy = h / sensor_height * fmm
    camera_param = {
    'model': 'PINHOLE',
    'width': w,
    'height': h,
    'params': np.array([fx, fy, cx, cy])}
    cams = Camera.from_colmap(camera_param)

    return cams
depth_frame = np.load('/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/outputs/Mapsape/switzerland_seq4@8@sunny@500@512/0_0.npy')
camera = [512, 288, 256, 144, 5.12, 2.88, 3]
render_camera = generate_render_camera(camera)
euler = [24.971615671291282, -0.0014066373185035689, -45.015328436935214]
trans = [7.621656338208605, 46.74082876209364, 1100.1373342024162 ]
for i in range(10):
    back_project(depth_frame, euler, trans, render_camera)
    depth_frame[0,0] = i

