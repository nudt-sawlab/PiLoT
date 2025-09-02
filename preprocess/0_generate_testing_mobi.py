import math
import random
from turtle import hideturtle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pyproj import CRS, Transformer
import yaml
import os
import glob
from vis_fov_view import batch_fov_visualization

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
    folder_path = "/home/ubuntu/Documents/code/FPVLoc/src_open/preprocess/datasets/query_8"
    yaml_path_list = []
    for file in os.listdir(folder_path):
    # 检查文件后缀是否为 .yaml 或 .yml
        if file.endswith('.yaml'):
            file_path = os.path.join(folder_path, file)
            yaml_path_list.append(file_path)
    # 获取所有 .yaml 和 .yml 文件
    yaml_files = glob.glob(os.path.join(folder_path, '*.yaml')) + glob.glob(os.path.join(folder_path, '*.yml'))
    for yaml_path in yaml_files:
        TITLE = yaml_path.split('/')[-1].split('.')[0]

        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        params = cfg['params']
        # ----------------------- 配置部分 -----------------------
        ABS_INIT     = params['ABS_INIT']
        HEIGHT_INIT  = params['HEIGHT_INIT']
        ROLL         = params['ROLL']
        NUM_POINTS   = params['NUM_POINTS']  # 轨迹点数量
        TOTAL_LENGTH = params['TOTAL_LENGTH']  # # 飞行总距离（单位：米），用于 x 方向
        # 输出路径
        POSE_OUTPUT_DIR = params['POSE_OUTPUT_DIR']
        TRAJECTORY_OUTPUT_DIR        = params['TRAJECTORY_OUTPUT_DIR']
        # 正弦曲线生成轨迹时的参数（用于 y 和 z 的偏移）
        PITCH_AMPLITUDE = params['PITCH_AMPLITUDE']
        
        HEIGHT_CHOICE = params['HEIGHT_CHOICE']
        PHI   = 0         # y 正弦波相位
        PSI   = np.pi / 4 # z 正弦波相位
        lon1, lat1 = ABS_INIT
        # -------------------- 配置部分结束 ----------------------
        # 0. 构造输出文件名
        trajectory_file = POSE_OUTPUT_DIR + TITLE + '.txt'
        if os.path.exists(trajectory_file):
            continue
            # raise RuntimeError('File has already existed!')
        # 1. 根据起始点计算对应 UTM EPSG
        epsg = get_utm_epsg_from_lonlat(ABS_INIT[0], ABS_INIT[1])
        print(f"自动判定的 UTM EPSG: {epsg}")
        if epsg is None:
            raise ValueError("经纬度超出可处理范围，无法确定 UTM 带。")
        
        
        utm_xyz_all = []
        for height_init in HEIGHT_CHOICE:
            # 1. 将初始 WGS84 坐标转换为 UTM 坐标（假设函数 wgs84_to_utm 已定义）
            abs_utm_init = wgs84_to_utm(ABS_INIT[0], ABS_INIT[1], height_init+HEIGHT_INIT, epsg)

            # 2. 设置水平尺寸参数（单位：米）
            x_max = TOTAL_LENGTH        # 整体尺寸，例如 300 米
            a = x_max / 2      # a 作为主半径

            # 4. 生成参数 t 及相对轨迹点
            t = np.linspace(0, 2 * np.pi, NUM_POINTS)
            traj_rel_x = 0.3*a * np.sin(t)
            traj_rel_y =  a * np.sin(t) * np.cos(t)

            # 4. 添加高度变化：中间低、两边高
            # 根据每个点到中心点（0, 0）的水平距离来确定高度
            r = np.sqrt(traj_rel_x**2 + traj_rel_y**2)
            max_r = np.max(r)                   # 最大水平距离（用于归一化）
            height_amplitude = 50               # 最大高度差，单位：米
            traj_rel_z = height_amplitude * (r / max_r)  # 距离越远，高度越高；中心点处 r=0，高度最低

            # 5. 计算轨迹各点的梯度，用于后续求解切向角（这里仅计算水平平面内的切向角）
            dx = np.gradient(traj_rel_x)
            dy = np.gradient(traj_rel_y)
            dz = np.gradient(traj_rel_z)

            tangent_angles = np.arctan2(dy, dx)
            yaw_degrees = (np.degrees(tangent_angles) - 90) % 360

            t_norm = np.linspace(0, 1, NUM_POINTS)
            
            pitch_degrees = PITCH_AMPLITUDE * np.sin(2 * np.pi * t_norm) + PITCH_AMPLITUDE + 30 # 较高频率使滚转更剧烈)

            roll_amplitude = 1  # 例如 30 度的滚转角幅度
            roll_degrees = roll_amplitude * np.sin(np.pi * t_norm)  # 较高频率使滚转更剧烈

            # 6. 将相对轨迹点转换为全局 UTM 坐标（在 abs_utm_init 的基础上偏移），同时添加高度变化
            for i in range(NUM_POINTS):

                utm_xyz_all.append([abs_utm_init[0] + traj_rel_x[i],
                    abs_utm_init[1] + traj_rel_y[i],
                    abs_utm_init[2] + traj_rel_z[i]])
        
        wgs84_coords = [utm_to_wgs84(p[0], p[1], p[2], epsg) for p in utm_xyz_all]
            
        # 7. 写入轨迹文件，每行格式：图片名称 pitch roll yaw lon lat alt
        lon_list, lat_list, alt_list = [], [], []
        # if os.path.exists(trajectory_file):
        #     continue
        with open(trajectory_file, 'w', encoding='utf-8') as f:
            for i in tqdm(range(NUM_POINTS*len(HEIGHT_CHOICE))):
                euler_enu = [roll_degrees[i% NUM_POINTS] , pitch_degrees[i% NUM_POINTS], yaw_degrees[i% NUM_POINTS] ]
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
        
        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(traj_rel_x, traj_rel_y, traj_rel_z, c=traj_rel_x, cmap='viridis', s=5)
        # ax.set_xlabel('X (m)')
        # ax.set_ylabel('Y (m)')
        # ax.set_zlabel('Z (m)')
        # plt.title('Random FPV Flight Trajectory (Local UTM Offsets)')
        # plt.show()
