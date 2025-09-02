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
def load_poses(pose_file, origin):
    """Load poses from the pose file."""
    pose_dict = []
    translation_list = []
    euler_angles_list = []
    name_list = []
    with open(pose_file, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                # pose_dict[parts[0]] = parts[1: ]  # Add the first element to the name list
                lon, lat, alt, roll, pitch, yaw = map(float, parts[1: ])
                translation_list.append([lon, lat, alt])
                euler_angles_list.append([pitch, roll, yaw])
                name_list.append(parts[0])
                euler_angles = [pitch, roll, yaw]
                translation = [lon, lat, alt]
                T_in_ECEF_c2w = euler_angles_to_matrix_ECEF(euler_angles, translation)
                T_in_ECEF_c2w[:3, 1] = -T_in_ECEF_c2w[:3, 1]  # Y轴取反，投影后二维原点在左上角
                T_in_ECEF_c2w[:3, 2] = -T_in_ECEF_c2w[:3, 2]  # Z轴取反
                T_in_ECEF_c2w[:3, 3] -= origin  # t_c2w - origin
                render_T_w2c = np.eye(4)
                render_T_w2c[:3, :3] = T_in_ECEF_c2w[:3, :3].T
                render_T_w2c[:3, 3] = -T_in_ECEF_c2w[:3, :3].T @ T_in_ECEF_c2w[:3, 3]
                
                
                pose_dict.append([pitch, roll, yaw,lon, lat, alt])
                
                
    return pose_dict
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
pose_dict = load_poses('/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/outputs/FPVLoc@switzerland_seq4@8@sunny@500@512.txt',[0,0,0])
depth_path = '/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/outputs/Mapsape/switzerland_seq4@8@sunny@500@512/'
renderer = osg_render.RenderImageProcessor(render_config)
def render_scene(self, euler_angles, translation):
    for _ in range(20):
        self.renderer.update_pose(translation, euler_angles, ref_camera = self.render_camera_osg)
        color_image = self.renderer.get_color_image()
        depth_image = self.renderer.get_depth_image()
    return depth_image
for i in range(len(pose_dict)):
    euler, trans = pose_dict[i][:3],  pose_dict[i][3:]
    start_time = time.time()
    depth_frame = render_scene(euler, trans)
    end_time1 = time.time()
    print('--depth: ', end_time1 - start_time)
    start_time = time.time()
    back_project(depth_frame, euler, trans, render_camera)
    end_time2 = time.time()
    print('--depth: ', end_time2 - start_time)
    # depth_frame[0,0] = i

