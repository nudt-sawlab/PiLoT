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
yaml_path = '/home/ubuntu/Documents/code/FPVLoc/src_open/preprocess/datasets/taiwan_seq1@200@30_50.yaml'
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

K_NEW = 2 * np.pi * N / TOTAL_LENGTH  # y 正弦波角频率（周期约 500 米）
M_NEW = 2 * np.pi * N/ TOTAL_LENGTH  # z 正弦波角频率（周期约 500 米）
PHI   = 0         # y 正弦波相位
PSI   = np.pi / 4 # z 正弦波相位
pitch_min, pitch_max = PITCH_RANGE
lon1, lat1, lon2, lat2 = ABS_INIT
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
        # raise RuntimeError('File has already existed!')
    # 1. 根据起始点计算对应 UTM EPSG
    epsg = get_utm_epsg_from_lonlat(ABS_INIT[0], ABS_INIT[1])
    print(f"自动判定的 UTM EPSG: {epsg}")
    if epsg is None:
        raise ValueError("经纬度超出可处理范围，无法确定 UTM 带。")
    
    # 2. 将起始点从 WGS84 转换到 UTM 坐标系
    abs_utm_init = wgs84_to_utm(ABS_INIT[0], ABS_INIT[1], HEIGHT_INIT, epsg)
    
    # 3. 生成局部轨迹（相对于起始 UTM 坐标）的偏移
    bearing = calculate_bearing(lat1, lon1, lat2, lon2)
    ALPHA_DEG = (360 - bearing) % 360
    print("转换后的 ALPHA_DEG:", ALPHA_DEG)
    alpha = np.radians(ALPHA_DEG)
    # 原始 (x, y)
    y_range = np.linspace(0, TOTAL_LENGTH, NUM_POINTS)
    x0 = A * np.sin(K_NEW * y_range + PHI)
    y0 = y_range  # 还是 y 线性增

    # 对 (x0, y0) 做 2D 旋转 -> (X, Y)
    x = x0 * np.cos(alpha) - y0 * np.sin(alpha)
    y = x0 * np.sin(alpha) + y0 * np.cos(alpha)
    z = B * np.sin(M_NEW * y_range + PSI)

    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, c=x, cmap='viridis', s=5)
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')
    # plt.title('3D Points with Sinusoidal Projections')
    # plt.show()
    
    # 4. 计算轨迹上各点的航向角（基于 x,y 的微分）
    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)
    
    # 正弦在 [-1,1] 之间，先转为 [0,1]，再放大到 [0,45]
    t = np.linspace(0,1,NUM_POINTS)
    pitch_factor = 0.5*(np.sin(2*np.pi * 2 * t) + 1)   # -> [0,1]
    pitch_array = pitch_min + (pitch_max - pitch_min) * pitch_factor  # 映射到 [20, 30]

    # ROLL 保持 0 (或随机小扰动)
    # roll_degrees = np.random.normal(loc=0, scale=0.3, size=NUM_POINTS)  # 示例：roll随机±2度
    roll_degrees = 0.3 * np.sin(2 * np.pi * t)
    # 将 Y 轴正向设为 0 度，并顺时针旋转
    tangent_angles = np.degrees(np.arctan2(dy, dx))  # 注意: arctan2( dx, dy ), 不是 arctan2(dy, dx)
    yaw_degrees = (tangent_angles - 90) % 360#(360 - tangent_angles) % 360
    
    
    # 6. 计算所有轨迹点在 UTM 坐标系下的坐标，再转换回 WGS84
    utm_xyz_all = [
        [abs_utm_init[0] + x[i],
         abs_utm_init[1] + y[i],
         abs_utm_init[2] + z[i]]
        for i in range(NUM_POINTS)
    ]
    wgs84_coords = [utm_to_wgs84(p[0], p[1], p[2], epsg) for p in utm_xyz_all]
    
    # 7. 写入轨迹文件，每行格式：图片名称 pitch roll yaw lon lat alt
    lon_list, lat_list, alt_list = [], [], []
    with open(trajectory_file, 'w', encoding='utf-8') as f:
        for i in tqdm(range(NUM_POINTS)):
            euler_enu = [roll_degrees[i], pitch_array[i], yaw_degrees[i]]
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
