import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pyproj import CRS, Transformer
import yaml
import os
import glob
from scipy.spatial.transform import Rotation as R
from transform import osg_to_ue
from vis_fov_view import batch_fov_visualization

'''
1. 封装到yaml，给定轨迹起始点，生成螺旋轨迹pose文件
2. y改为100米范围，减小视野外的加载压力
3. roll和pitch角改为平滑变化
4. 矫正yaw角和机头飞行方向一致
5. osg转ue坐标
6. 添加沿轨迹方向飞行的初始化，增加模型缓冲时间
'''
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
    # ----------------------- 配置部分 -----------------------
    # 起始点（WGS84，经度, 纬度, 高程）
    folder_path = "/home/ubuntu/Documents/code/FPVLoc/src_open/preprocess/datasets/temp"
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
        # 0. 构造输出文件名
        trajectory_file = POSE_OUTPUT_DIR + TITLE + '.txt'
        if os.path.exists(trajectory_file):
            continue
            # raise RuntimeError('File has already existed!')
        # ===== 1. 起点转换为UTM坐标 =====
        epsg = get_utm_epsg_from_lonlat(ABS_INIT[0], ABS_INIT[1])
        print(f"自动判定的 UTM EPSG: {epsg}")
        if epsg is None:
            raise ValueError("经纬度超出可处理范围，无法确定 UTM 带。")
        abs_utm_init = wgs84_to_utm(ABS_INIT[0], ABS_INIT[1], HEIGHT_INIT, epsg)

        abs_utm_end = wgs84_to_utm(ABS_INIT[2], ABS_INIT[3], HEIGHT_INIT, epsg)

        # ===== 2. 计算起终点连线的航向角及旋转角 =====
        bearing = calculate_bearing(lat1, lon1, lat2, lon2)
        ALPHA_DEG = (360 - bearing) % 360
        print("转换后的 ALPHA_DEG:", ALPHA_DEG)
        alpha = np.radians(ALPHA_DEG)

        # ===== 3. 生成原始局部螺旋轨迹（相对于起点UTM的偏移） =====
        # y_range 为沿轨迹的线性进程
        y_range = np.linspace(0, TOTAL_LENGTH, NUM_POINTS)
        # 螺旋形在局部坐标下，x0 为正弦曲线，y0 线性增长
        x0 = A * np.sin(K_NEW * y_range + PHI)
        y0 = y_range

        # 对 (x0, y0) 作 2D 旋转，得到旋转后的 (x, y)
        x = x0 * np.cos(alpha) - y0 * np.sin(alpha)
        y = x0 * np.sin(alpha) + y0 * np.cos(alpha)
        z = B * np.sin(M_NEW * y_range + PSI)

        # ===== 4. 生成直线飞行段 =====
        # 1. 定义起飞点 P0（局部坐标下，假设起飞点为原点，海拔取 abs_utm_init[2]）
        # --- 计算飞行方向的单位向量（仅考虑水平面） ---
        dx = abs_utm_end[0] - abs_utm_init[0] 
        dy = abs_utm_end[1] - abs_utm_init[1]
        flight_direction_deg =  (np.degrees(np.arctan2(dy, dx))) % 360
        flight_direction_rad = np.radians(flight_direction_deg)
        unit_vector = np.array([np.cos(flight_direction_rad), np.sin(flight_direction_rad), 0])

        # --- 反向计算起飞点 P0 ---
        # 螺旋轨迹起始点 P1（由螺旋生成代码提供）
        P1 = np.array([x[0], y[0], z[0]])
        # 起飞点 = 螺旋起始点 - 直线飞行距离 * 飞行方向单位向量
        P0 = P1 - 1.0 * unit_vector
        
        # 2. 计算从 P0 到 P1 的位移向量、距离及方向
        diff = P1 - P0
        dist = np.linalg.norm(diff)
        direction = diff / dist

        # 3. 生成直线段：沿 P0 到 P1 的直线飞行
        N_straight = 10  # 直线段采样点数，可根据需要调整
        d_straight = np.linspace(0, dist, N_straight)
        straight_x = P0[0] + d_straight * direction[0]
        straight_y = P0[1] + d_straight * direction[1]
        straight_z = P0[2] + d_straight * direction[2]

        # 4. 计算直线段的航向角（yaw）
        # 航向角由 P1 - P0 的水平向量计算，采用 (atan2 - 90) 模 360 的方式
        straight_yaw = np.degrees(np.arctan2(direction[1], direction[0]))
        straight_yaw = (straight_yaw - 90) % 360

        # 5. 计算螺旋轨迹初始切向角（以螺旋段前两个点计算）
        if len(x) > 1:
            spiral_dx = x[1] - x[0]
            spiral_dy = y[1] - y[0]
            spiral_yaw = np.degrees(np.arctan2(spiral_dy, spiral_dx))
            spiral_yaw = (spiral_yaw - 90) % 360
        else:
            spiral_yaw = straight_yaw

        # 6. 对螺旋轨迹前 N_blend 个点的航向做平滑过渡，逐渐从直线段航向转向螺旋初始切向
        N_blend = 200
        t_blend = np.linspace(0, 1, N_blend)
        def interpolate_angle(a, b, t):
            # 保证采用最短转弯方向
            diff_angle = ((b - a + 180) % 360) - 180
            return (a + diff_angle * t) % 360
        yaw_blend = np.array([interpolate_angle(straight_yaw, spiral_yaw, t) for t in t_blend])

        # 7. 构造螺旋段的航向数组
        # 此处为了示例，后续螺旋段航向均取 spiral_yaw；实际应用中可根据位置梯度计算
        # spiral_yaw_array = np.full(len(x), spiral_yaw)
        # # 用过渡段覆盖螺旋轨迹初段的航向
        # spiral_yaw_array[:N_blend] = yaw_blend
        rotate_x = np.full(N_blend, x[0])
        rotate_y = np.full(N_blend, y[0])
        rotate_z = np.full(N_blend, z[0])
        # 8. 拼接完整轨迹
        # 直线段位置 + 螺旋轨迹（去除螺旋轨迹第一个点 P1，因为 P1 已在直线段末端）
        x_total = np.concatenate([straight_x, rotate_x, x])
        y_total = np.concatenate([straight_y, rotate_y, y])
        z_total = np.concatenate([straight_z, rotate_z, z])
        # 航向：直线段部分统一为 straight_yaw，螺旋段部分采用 spiral_yaw_array（但第一个点已在直线段中）
        
        # ===== 8. 可视化完整轨迹 =====
        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot(111, projection='3d')
        # sc = ax.scatter(x_total, y_total, z_total, c=x_total, cmap='viridis', s=5)
        # ax.set_xlabel('X (m)')
        # ax.set_ylabel('Y (m)')
        # ax.set_zlabel('Z (m)')
        # plt.title('直线段 + 混合过渡 + 螺旋飞行轨迹')
        # plt.show()

        # ===== 9. 后续处理（计算微分及航向、俯仰、横滚角） =====
        dx = np.gradient(x_total)
        dy = np.gradient(y_total)
        dz = np.gradient(z_total)

        # 计算俯仰角（此处示例使用正弦函数映射，可根据需求调整）
        t = np.linspace(0, 1, len(x_total))
        pitch_factor = 0.5 * (np.sin(2 * np.pi * 2 * t) + 1)  # 范围 [0, 1]
        pitch_array = pitch_min + (pitch_max - pitch_min) * pitch_factor  # 映射到 [pitch_min, pitch_max]

        # 横滚角：这里采用小幅正弦扰动
        roll_degrees = 0.3 * np.sin(2 * np.pi * t)

        # 根据 (dx, dy) 计算航向角，注意 arctan2 的参数顺序
        tangent_angles = np.degrees(np.arctan2(dy, dx))
        yaw_degrees_spiral = (tangent_angles - 90) % 360
        yaw_degrees = np.concatenate([np.full(N_straight, straight_yaw), yaw_blend, yaw_degrees_spiral[N_blend+N_straight:]])

        # ===== 10. 将局部轨迹转换回UTM，再转换为WGS84坐标 =====
        utm_xyz_all = [
            [abs_utm_init[0] + x_total[i],
            abs_utm_init[1] + y_total[i],
            abs_utm_init[2] + z_total[i]]  #
            for i in range(len(x_total))
        ]
        wgs84_coords = [utm_to_wgs84(p[0], p[1], p[2], epsg) for p in utm_xyz_all]
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=x, cmap='viridis', s=5)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.title('3D Points with Sinusoidal Projections')
        plt.show()   
        # 7. 写入轨迹文件，每行格式：图片名称 pitch roll yaw lon lat alt
        lon_list, lat_list, alt_list = [], [], []
        with open(trajectory_file, 'w', encoding='utf-8') as f:
            for i in tqdm(range( NUM_POINTS+N_blend+N_straight)):
                euler_enu = [roll_degrees[i], pitch_array[i], yaw_degrees[i]]
                # euler_enu_ue = osg_to_ue(euler_enu)
                wgs84_coord = wgs84_coords[i]
                line_data = wgs84_coord + euler_enu   # 拼接姿态和坐标  'Viewpoint_1.jpg' 135.400234 34.643935 150 0.0 0.0 0.0
                name_str = str(i) + '.jpg' #-(N_blend+N_straight)
                out_str = ' '.join(map(str, line_data))
                f.write(f'{name_str} {out_str}\n')
                
                lon_list.append(wgs84_coord[0])
                lat_list.append(wgs84_coord[1])
                alt_list.append(wgs84_coord[2])
        print(f"轨迹文件已写入: {trajectory_file}")

        kml_output = TRAJECTORY_OUTPUT_DIR + TITLE + '.kml'
        batch_fov_visualization(trajectory_file, kml_output)
        print("Done. Open '.kml' in Google Earth to see multiple FOV pyramids.")
