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
# ----------------------- 配置部分 -----------------------
# 输出文件配置

yaml_path = '/home/ubuntu/Documents/code/FPVLoc/src_open/preprocess/datasets/query/switzerland_seq2@250@fpv.yaml'
TITLE = yaml_path.split('/')[-1].split('.')[0]

with open(yaml_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

params = cfg['params']

# 从配置中读取各参数
ABS_INIT     = params['ABS_INIT']
HEIGHT_INIT  = params['HEIGHT_INIT']
ROLL         = params['ROLL']
NUM_POINTS   = params['NUM_POINTS']  # 轨迹点数量
INIT_EULER = params['INIT_EULER']
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
    # 1. 根据起始点确定 UTM EPSG
    epsg = get_utm_epsg_from_lonlat(ABS_INIT[0], ABS_INIT[1])
    print(f"自动判定的 UTM EPSG: {epsg}")
    if epsg is None:
        raise ValueError("经纬度超出可处理范围，无法确定 UTM 带。")

    # 2. 将起始点转换到 UTM 坐标系
    abs_utm_init = wgs84_to_utm(ABS_INIT[0], ABS_INIT[1], HEIGHT_INIT, epsg)

    # 3. 生成随机控制点构造轨迹（FPV 风格）
    # 为了获得更剧烈的动作，放大横向和垂直偏移的幅度
    LATERAL_MULTIPLIER = 3    # 横向放大倍数
    VERTICAL_MULTIPLIER = 3   # 垂直放大倍数

    # 控制点在局部 UTM 坐标下的 x 坐标仍均匀分布，从 0 到 FLIGHT_DISTANCE
    t_control = np.linspace(0, NUM_CONTROL_POINTS - 1, NUM_CONTROL_POINTS)
    control_x = np.linspace(0, FLIGHT_DISTANCE, NUM_CONTROL_POINTS)
    # 横向和垂直偏移在 [-LATERAL_MAX, LATERAL_MAX] 和 [-VERTICAL_MAX, VERTICAL_MAX] 内随机生成，
    # 这里乘以放大倍数以获得较大的动作
    
    small_multiplier = 0.3
    control_y = np.random.uniform(-LATERAL_MAX * small_multiplier, LATERAL_MAX * small_multiplier, NUM_CONTROL_POINTS)
    control_z = np.random.uniform(-VERTICAL_MAX * small_multiplier, VERTICAL_MAX * small_multiplier, NUM_CONTROL_POINTS)
    # 利用 CubicSpline 对控制点进行平滑插值
    t_vals = np.linspace(t_control[0], t_control[-1], NUM_POINTS)
    cs_x = CubicSpline(t_control, control_x)
    cs_y = CubicSpline(t_control, control_y)
    cs_z = CubicSpline(t_control, control_z)

    x = cs_x(t_vals)
    y = cs_y(t_vals)
    z = cs_z(t_vals)

    # 4. 根据曲线导数计算动态姿态角
    dx_dt = cs_x.derivative()(t_vals)
    dy_dt = cs_y.derivative()(t_vals)
    dz_dt = cs_z.derivative()(t_vals)  # 注意：这里对 z 的导数取了负号

    # 航向角 yaw：由水平速度方向决定（单位：度）
    tangent_angles = np.arctan2(dy_dt, dx_dt)
    yaw_degrees = (np.degrees(tangent_angles) - 90) % 360
    t_norm = np.linspace(0, 1, NUM_POINTS)
    # 俯仰角 pitch：由竖直分量与水平速度比值得到
    horizontal_speed = np.sqrt(dx_dt**2 + dy_dt**2)
    # pitch_degrees = np.degrees(np.arctan2(dz_dt, horizontal_speed)) + 90
    pitch_amplitude = 30  # 例如 30 度的滚转角幅度
    pitch_degrees = pitch_amplitude * np.sin(2 * np.pi * t_norm) + pitch_amplitude  # 较高频率使滚转更剧烈)

    # 滚转角 roll：增加更大的幅度和更快的变化，模拟 FPV 动作
    
    roll_amplitude = 5  # 例如 30 度的滚转角幅度
    roll_degrees = roll_amplitude * np.sin(2 * np.pi * t_norm)  # 较高频率使滚转更剧烈

    # 5. 将局部 UTM 坐标加到起始 UTM 坐标上，获得全局 UTM 坐标
    utm_xyz_all = [
        [abs_utm_init[0] + x[i],
        abs_utm_init[1] + y[i],
        abs_utm_init[2] + z[i]]
        for i in range(NUM_POINTS)
    ]
        
    # 6. 转换为 WGS84 坐标
    wgs84_coords = [utm_to_wgs84(p[0], p[1], p[2], epsg) for p in utm_xyz_all]
    
    # 7. 写入轨迹文件，每行格式：图片名称 pitch roll yaw lon lat alt
    pose_file = POSE_OUTPUT_DIR + TITLE + '.txt'
    lon_list, lat_list, alt_list = [], [], []
    with open(pose_file, 'w', encoding='utf-8') as f:
        for i in tqdm(range(NUM_POINTS)):
            euler_enu = [roll_degrees[i], pitch_degrees[i], yaw_degrees[i]]
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
