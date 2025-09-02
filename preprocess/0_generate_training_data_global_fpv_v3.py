import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pyproj import CRS, Transformer
from scipy.interpolate import CubicSpline
import yaml
from vis_fov_view import batch_fov_visualization
from transform import osg_to_ue
from math import sin, cos, radians, degrees, atan2
from noise import pnoise1   # Perlin 噪声 1D
from scipy.signal import butter, filtfilt
import scipy
# ----------------------- 配置部分 -----------------------
# 输出文件配置

yaml_path = '/home/ubuntu/Documents/code/FPVLoc/src_open/preprocess/datasets/query/switzerland_seq3@200@fpv.yaml'
TITLE = yaml_path.split('/')[-1].split('.')[0]

with open(yaml_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

params = cfg['params']

# 从配置中读取各参数
ABS_INIT     = params['ABS_INIT']
HEIGHT_INIT  = params['HEIGHT_INIT']
ROLL         = params['ROLL']
NUM_POINTS   = params['NUM_POINTS']  # 轨迹点数量
PITCH_RANGE = params['PITCH_RANGE']
ROLL_RANGE = params['ROLL_RANGE']
# INIT_EULER = params['INIT_EULER']
FLIGHT_DISTANCE = params['TOTAL_LENGTH']  # # 飞行总距离（单位：米），用于 x 方向
POSE_OUTPUT_DIR = params['POSE_OUTPUT_DIR']
NUM_CONTROL_POINTS = params['NUM_CONTROL_POINTS']    # 随机控制点数量（决定机动段数量）
TRAJECTORY_OUTPUT_DIR        = params['TRAJECTORY_OUTPUT_DIR']

LATERAL_MAX = FLIGHT_DISTANCE / 30   # 横向最大偏移（单位：米）
VERTICAL_MAX = FLIGHT_DISTANCE / 20  # 垂直最大偏移（单位：米）
# -------------------------------------------------------

# ----------------------- 坐标转换函数 -----------------------
def get_utm_epsg_from_lonlat(lon, lat):
    """
    根据经纬度 (lon, lat) 计算对应的 UTM 分带 EPSG 号。
    北半球返回 EPSG:326XX，南半球返回 EPSG:327XX。
    """
    if lon < -180 or lon > 180 or lat < -90 or lat > 90:
        return None
    zone = int(math.floor((lon + 180) / 6)) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone

def wgs84_to_utm(lon, lat, alt, epsg):
    """
    将 WGS84 坐标 (lon, lat, alt) 转换到对应 UTM 坐标系 (x, y, z)。
    这里 alt 直接平移。
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
# -------------------------------------------------------

if __name__ == "__main__":
    SEED            = 42             # 随机种子，便于复现
    # 1. 判定 UTM 带
    epsg = get_utm_epsg_from_lonlat(*ABS_INIT)
    if epsg is None:
        raise ValueError("经纬度超出可处理范围，无法确定 UTM 带。")

    # 2. 起点转 UTM
    # 速度随机
    SPEED_MIN, SPEED_MAX = 0.5, 1.5   # 相对平均速度
    SPEED_SMOOTH_SIGMA   = 20         # 高斯平滑窗口

    # ========= 1. 起点 → UTM =========
    abs_utm_init = wgs84_to_utm(ABS_INIT[0], ABS_INIT[1], HEIGHT_INIT, epsg)

    # ========= 2. 控制点 & 插值 =========
    t_ctrl = np.arange(NUM_CONTROL_POINTS)
    ctrl_x = np.linspace(0, FLIGHT_DISTANCE, NUM_CONTROL_POINTS)

    small_mult = 0.3
    ctrl_y = np.random.uniform(-LATERAL_MAX*small_mult,
                            LATERAL_MAX*small_mult,
                            NUM_CONTROL_POINTS)
    ctrl_z = np.random.uniform(-VERTICAL_MAX*small_mult,
                            VERTICAL_MAX*small_mult,
                            NUM_CONTROL_POINTS)

    cs_x, cs_y, cs_z = (CubicSpline(t_ctrl, v) for v in (ctrl_x, ctrl_y, ctrl_z))

    # ========= 3. 速度随机：不均匀采样 =========
    rng   = np.random.default_rng()
    v_rel = rng.uniform(SPEED_MIN, SPEED_MAX, NUM_POINTS)
    v_rel = scipy.ndimage.gaussian_filter1d(v_rel, sigma=SPEED_SMOOTH_SIGMA)
    v_rel /= v_rel.mean()

    s = np.cumsum(v_rel)
    s = (s - s[0]) / (s[-1] - s[0])

    t_uniform = np.linspace(0, 1, NUM_POINTS)
    t_vals    = np.interp(t_uniform, s, np.linspace(t_ctrl[0], t_ctrl[-1], NUM_POINTS))

    # ========= 4. 轨迹（无抖动） =========
    x, y, z = cs_x(t_vals), cs_y(t_vals), cs_z(t_vals)

    # ========= 5. 姿态角 =========
    dx_dt, dy_dt, dz_dt = np.gradient(x), np.gradient(y), np.gradient(z)

    yaw = (np.degrees(np.arctan2(dy_dt, dx_dt)) - 90) % 360
    yaw = np.where(yaw > 180, yaw-360, yaw)          # [-180,180]

    horizontal_speed = np.hypot(dx_dt, dy_dt)
    t_norm = np.linspace(0, 1, NUM_POINTS)
    pitch_amplitude = (PITCH_RANGE[1] - PITCH_RANGE[0]) / 2

    initial_phase_deg = np.random.uniform(PITCH_RANGE[0], PITCH_RANGE[1])
    initial_phase_rad = np.deg2rad(initial_phase_deg)
    # 2) 生成整条曲线
    t_norm        = np.linspace(0, 1, NUM_POINTS)
    pitch = pitch_amplitude * np.sin(2*np.pi*t_norm + initial_phase_rad) + pitch_amplitude

    # roll：用较平滑的正弦，幅度可调
    t = np.linspace(0,1,NUM_POINTS)
    roll = 1.0 * np.sin(2 * np.pi * t)#INIT_EULER[1] + np.random.normal(0, 5, NUM_POINTS)
    # roll = np.clip(roll, *ROLL_RANGE)

    # ========= 6. UTM → WGS84 =========
    utm_xyz_all = np.column_stack([abs_utm_init[0] + x,
                                abs_utm_init[1] + y,
                                abs_utm_init[2] + z])

    wgs84_coords = [utm_to_wgs84(*p, epsg) for p in utm_xyz_all]
    
    # 7. 写入轨迹文件，每行格式：图片名称 pitch roll yaw lon lat alt
    pose_file = POSE_OUTPUT_DIR + TITLE + '.txt'
    lon_list, lat_list, alt_list = [], [], []
    with open(pose_file, 'w', encoding='utf-8') as f:
        for i in tqdm(range(NUM_POINTS)):
            euler_enu = [roll[i], pitch[i], yaw[i]]
            # euler_enu_ue = osg_to_ue(euler_enu)
            # euler_enu_osg = ue_to_osg(euler_enu_ue)
            wgs84_coord = wgs84_coords[i]
            line_data = wgs84_coord + euler_enu
            name_str = str(i) + '.jpg'
            out_str = ' '.join(map(str, line_data))
            f.write(f'{name_str} {out_str}\n')
            
            lon_list.append(wgs84_coord[0])
            lat_list.append(wgs84_coord[1])
            alt_list.append(wgs84_coord[2])
    print(f"轨迹文件已写入: {pose_file}")


    kml_output = TRAJECTORY_OUTPUT_DIR + TITLE + '.kml'
    batch_fov_visualization(pose_file, kml_output)
    print("Done. Open '.kml' in Google Earth to see multiple FOV pyramids.")

    
    # # 9. 绘制 3D 轨迹（局部 UTM 坐标下）
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=x, cmap='viridis', s=5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.title('Random FPV Flight Trajectory (Local UTM Offsets)')
    plt.show()
    
    # # 可选：绘制 XY 投影及航向指示
    # plt.figure(figsize=(10, 7))
    # plt.plot(x, y, label='XY Trajectory', color='blue')
    # for i in range(0, NUM_POINTS, 200):
    #     angle = yaw_degrees[i]
    #     plt.arrow(x[i], y[i],
    #               10 * np.cos(np.radians(90 - angle)),
    #               10 * np.sin(np.radians(90 - angle)),
    #               head_width=5, head_length=7, fc='green', ec='green')
    # plt.xlabel('X (m)')
    # plt.ylabel('Y (m)')
    # plt.title('XY Projection with Yaw Directions')
    # plt.legend()
    # plt.grid()
    # plt.axis('equal')
    # plt.show()
