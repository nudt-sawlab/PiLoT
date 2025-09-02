import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pyproj import CRS, Transformer
import yaml
import os
from vis_fov_view import batch_fov_visualization
# ----------------------- 配置部分 -----------------------
# 起始点（WGS84，经度, 纬度, 高程）
yaml_path = '/home/ubuntu/Documents/code/github/src_open/preprocess/datasets/query_8/feicuiwan_seq1@8.yaml'
TITLE = yaml_path.split('/')[-1].split('.')[0]

with open(yaml_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

params = cfg['params']

# 从配置中读取各参数
ABS_INIT     = params['ABS_INIT']
HEIGHT_INIT  = params['HEIGHT_INIT']
ROLL         = params['ROLL']
NUM_POINTS   = params['NUM_POINTS']  # 轨迹点数量
TOTAL_LENGTH = params['TOTAL_LENGTH']  # # 飞行总距离（单位：米），用于 x 方向
N            = params['N']  # N cycles
PITCH_RANGE  = params['PITCH_RANGE'] #pitch角范围
# 输出路径
POSE_OUTPUT_DIR = params['POSE_OUTPUT_DIR']
TRAJECTORY_OUTPUT_DIR        = params['TRAJECTORY_OUTPUT_DIR']
# 正弦曲线生成轨迹时的参数（用于 y 和 z 的偏移）
A            = params['A']
B            = params['B']


lon1, lat1 = ABS_INIT
x_max = TOTAL_LENGTH    # 此处表示无穷符号横向全长为 100 米
# -------------------- 配置部分结束 ----------------------

def get_utm_epsg_from_lonlat(lon, lat):
    """
    根据经纬度 (lon, lat) 计算其对应的 UTM 分带 EPSG 号。
    - 北半球：EPSG:326XX
    - 南半球：EPSG:327XX

    :param lon: 经度 (float)，通常范围 [-180, 180]
    :param lat: 纬度 (float)，通常范围 [-90, 90]
    :return: 整型 EPSG，若无法计算则返回 None
    """
    if lon < -180 or lon > 180 or lat < -90 or lat > 90:
        return None
    zone = int(math.floor((lon + 180) / 6)) + 1
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone

def wgs84_to_utm(lon, lat, alt, epsg):
    """
    将 WGS84 坐标 (lon, lat, alt) 转换到对应 UTM 坐标系 (x, y, z)。
    这里 alt 保持不变（不做大地高与正高转换）。
    """
    crs_wgs84 = CRS.from_epsg(4326)
    crs_utm   = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return [x, y, alt]

def utm_to_wgs84(x, y, alt, epsg):
    """
    将 UTM 坐标 (x, y, z) 转换回 WGS84 坐标 (lon, lat, alt)。
    """
    crs_utm   = CRS.from_epsg(epsg)
    crs_wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_utm, crs_wgs84, always_xy=True)
    lon, lat = transformer.transform(x, y)
    return [lon, lat, alt]
# --- 计算两个经纬坐标之间的标准航向（以正北为0度，顺时针增加） ---
def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    计算从点1到点2的标准航向，单位：度。
    返回值范围 [0, 360)，其中北=0, 东=90, 南=180, 西=270。
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    diffLong = math.radians(lon2 - lon1)
    
    x = math.sin(diffLong) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - (math.sin(lat1_rad) *
         math.cos(lat2_rad) * math.cos(diffLong))
    
    initial_bearing = math.degrees(math.atan2(x, y))
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing
if __name__ == "__main__":
    # 0. 构造输出文件名
    trajectory_file = POSE_OUTPUT_DIR + TITLE + '.txt'
    # if os.path.exists(trajectory_file):
    #     raise RuntimeError('File has already existed!')
    # 1. 根据起始点计算对应 UTM EPSG
    epsg = get_utm_epsg_from_lonlat(ABS_INIT[0], ABS_INIT[1])
    print(f"自动判定的 UTM EPSG: {epsg}")
    if epsg is None:
        raise ValueError("经纬度超出可处理范围，无法确定 UTM 带。")
    
    # 2. 将起始点从 WGS84 转换到 UTM 坐标系
    abs_utm_init = wgs84_to_utm(ABS_INIT[0], ABS_INIT[1], HEIGHT_INIT, epsg)

    # 设置无穷符号的水平方向最长距离（单位：米）
    
    a = x_max / 2  # 由于无穷符号在 x 方向范围为 [-a, a]

    # 生成参数 t 及相对轨迹点
    t = np.linspace(0, 2 * np.pi, NUM_POINTS)
    # 使用 Lemniscate of Gerono 参数方程：x = a*sin(t), y = a*sin(t)*cos(t)
    traj_rel_x = a * np.sin(t)
    traj_rel_y = a * np.sin(t) * np.cos(t)

    # 将相对轨迹点转换为全局 UTM 坐标（在 abs_utm_init 基础上偏移）
    dx = np.gradient(traj_rel_x)
    dy = np.gradient(traj_rel_y)
    dz = np.gradient(abs_utm_init[2])
    
    tangent_angles = np.arctan2(dy, dx)
    yaw_degrees = (np.degrees(tangent_angles) - 90) % 360
    
    # 6. 计算所有轨迹点在 UTM 坐标系下的坐标，再转换回 WGS84
    utm_xyz_all = [
        [abs_utm_init[0] + traj_rel_x[i],
         abs_utm_init[1] + traj_rel_y[i],
         abs_utm_init[2]]
        for i in range(NUM_POINTS)
    ]
    wgs84_coords = [utm_to_wgs84(p[0], p[1], p[2], epsg) for p in utm_xyz_all]
    
    # 7. 写入轨迹文件，每行格式：图片名称 pitch roll yaw lon lat alt
    lon_list, lat_list, alt_list = [], [], []
    with open(trajectory_file, 'w', encoding='utf-8') as f:
        for i in tqdm(range(NUM_POINTS)):
            euler_enu = [0, 45, yaw_degrees[i]]
            wgs84_coord = wgs84_coords[i]
            line_data = wgs84_coord + euler_enu   # 拼接姿态和坐标  'Viewpoint_1.jpg' 135.400234 34.643935 150 0.0 0.0 0.0
            name_str = str(i) + '.jpg'
            out_str = ' '.join(map(str, line_data))
            f.write(f'{name_str} {out_str}\n')
            
            lon_list.append(wgs84_coord[0])
            lat_list.append(wgs84_coord[1])
            alt_list.append(wgs84_coord[2])
    print(f"轨迹文件已写入: {trajectory_file}")

    kml_output = TRAJECTORY_OUTPUT_DIR + TITLE + '.kml'
    batch_fov_visualization(trajectory_file, kml_output)
    print("Done. Open '.kml' in Google Earth to see multiple FOV pyramids.")
