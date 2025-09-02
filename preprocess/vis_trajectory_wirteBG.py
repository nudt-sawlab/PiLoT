import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

###############################################################################
# 1) (lon,lat,alt) <-> ENU（近似）转换函数（仅用于小范围可视化）
###############################################################################
def lla_to_enu(lon_deg, lat_deg, alt_m, lon0_deg, lat0_deg, alt0_m):
    """
    将地理坐标(lon, lat, alt)转换成相对于(lon0, lat0, alt0)的 ENU (X东, Y北, Z上)坐标。
    注意仅适用于小范围近似。
    """
    R_EARTH_LAT = 111320.0  # 每度纬度约 111.32 km
    avg_lat_rad = math.radians(lat0_deg)
    east_m  = (lon_deg - lon0_deg) * R_EARTH_LAT * math.cos(avg_lat_rad)
    north_m = (lat_deg - lat0_deg) * R_EARTH_LAT
    up_m    = alt_m - alt0_m
    return np.array([east_m, north_m, up_m], dtype=float)

def enu_to_lla(x_m, y_m, z_m, lon0_deg, lat0_deg, alt0_m):
    """
    将 ENU 坐标(x, y, z)转换回地理坐标(lon, lat, alt)。
    """
    R_EARTH_LAT = 111320.0
    avg_lat_rad = math.radians(lat0_deg)
    d_lon = x_m / (R_EARTH_LAT * math.cos(avg_lat_rad))
    d_lat = y_m / R_EARTH_LAT
    lon_deg = lon0_deg + d_lon
    lat_deg = lat0_deg + d_lat
    alt_m   = alt0_m + z_m
    return (lon_deg, lat_deg, alt_m)


###############################################################################
# 2) 计算相机/机头 在 ENU 下的方向向量
#    （Yaw=0 => 朝北, 逆时针, Pitch=0 => 正向下, Roll 绕前向轴）
###############################################################################
def camera_direction_enu(yaw_deg, pitch_deg, roll_deg=0.0):
    """
    - Yaw=0 => 朝北(ENU的+Y), 逆时针增大(90=>向西)
    - Pitch=0 => 向下, 90=>水平
    - Roll=绕机头“前向”轴

    返回在 ENU (X东,Y北,Z上) 坐标系下的单位朝向向量
    """
    # pitch=0向下 => 这里先做 90 - pitch, 让 pitch=0 => +90° => 指向 -Z
    pitch_eff = 90.0 - pitch_deg
    # yaw=0 => 北 => ENU+Y，而 ZYX欧拉时 yaw=0通常=>+X => 做 yaw_eff= yaw - 90
    yaw_eff   = yaw_deg - 90.0

    # 先绕 z(yaw_eff), 再绕 y(pitch_eff), 再绕 x(roll_deg)
    rot = R.from_euler('ZYX', [yaw_eff, pitch_eff, roll_deg], degrees=True)
    v_body = np.array([1, 0, 0], dtype=float)  # 机体默认+X为前方
    v_enu  = rot.apply(v_body)
    v_enu  = v_enu / np.linalg.norm(v_enu)
    return v_enu

###############################################################################
# 3) 生成 “长方形底面 + 无人机位置顶点” 的四棱锥(FOV)在 ENU 中的 5 个点（顶点 + 4个底角）
###############################################################################
def get_fov_rect_pyramid_enu(
    apex_enu,
    yaw_deg, pitch_deg, roll_deg=0.0,
    dist_forward=100.0,
    width=40.0,
    height=20.0
):
    """
    根据 apex_enu（在 ENU 坐标系下的相机位置）和姿态（yaw, pitch, roll），
    计算四棱锥顶点(相机位置)与底面四角在 ENU 中的坐标，返回一个字典:
      {
        "apex": (x,y,z),
        "base": [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3), (x4,y4,z4)]
      }
    用于后续可视化。
    """
    dir_enu = camera_direction_enu(yaw_deg, pitch_deg, roll_deg)

    # 底面中心
    center_enu = apex_enu + dist_forward * dir_enu

    # 找到与 dir_enu 垂直的两个方向向量 u, v
    not_collinear = np.array([0,0,1], dtype=float)
    if abs(dir_enu.dot(not_collinear)) > 0.999:
        not_collinear = np.array([1,0,0], dtype=float)
    u = np.cross(dir_enu, not_collinear); u /= np.linalg.norm(u)
    v = np.cross(dir_enu, u); v /= np.linalg.norm(v)

    half_w = width/2.0
    half_h = height/2.0

    p1_enu = center_enu + half_w*u + half_h*v
    p2_enu = center_enu + half_w*u - half_h*v
    p3_enu = center_enu - half_w*u - half_h*v
    p4_enu = center_enu - half_w*u + half_h*v

    return {
        "apex": apex_enu,
        "base": [p1_enu, p2_enu, p3_enu, p4_enu]
    }

###############################################################################
# 4) 读取轨迹数据，并生成动画
###############################################################################
def animate_camera_fov(txt_path,
                       save_path,
                       dist_forward=100.0,
                       width=40.0,
                       height=20.0,
                       interval_ms=200):
    """
    从 txt_path 读取数据，每行: filename lon lat alt roll pitch yaw
    （与之前KML版本脚本格式相同或相似）。
    生成 3D 动画：相机视角锥在轨迹点上逐帧移动。
    """
    # 先读取全部数据
    # 这里你也可以改为只取少量点做演示
    data_list = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue

            filename = parts[0]
            lon   = float(parts[1])
            lat   = float(parts[2])
            alt   = float(parts[3])
            roll  = float(parts[4])
            pitch = float(parts[5])
            yaw   = float(parts[6])
            data_list.append((lon, lat, alt, roll, pitch, yaw))

    if not data_list:
        print("No valid data found in the text file.")
        return

    # 选择第一个点作为 ENU 原点(也可自行指定)
    lon0, lat0, alt0, _, _, _ = data_list[0]

    # 转成 ENU，并保存
    enu_points = []   # [(x, y, z, roll, pitch, yaw), ...]
    for (lon, lat, alt, roll, pitch, yaw) in data_list:
        xyz = lla_to_enu(lon, lat, alt, lon0, lat0, alt0)
        enu_points.append((xyz[0], xyz[1], xyz[2], roll, pitch, yaw))

    # 提前把轨迹线(ENU)存好，后面在动画里可作为整体背景
    traj_xyz = np.array([[p[0], p[1], p[2]] for p in enu_points])

    # Matplotlib 3D 初始化
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 画出整个轨迹 (仅一次，不随动画变化)
    ax.plot(traj_xyz[:,0], traj_xyz[:,1], traj_xyz[:,2], label="Trajectory")

    # 先准备一些容器，后续动画里更新
    # 我们用以下对象画“相机视角锥”的四条侧棱 + 底面矩形线
    # 注意：只是简单用线段表示，方便展示
    apex_line = ax.plot([], [], [], marker='o', linestyle="None", label="Camera")[0]

    # 侧面线(4条)
    side_lines = [ax.plot([], [], [], 'r-')[0] for _ in range(4)]
    # 底面(4条边)
    base_lines = [ax.plot([], [], [], 'g-')[0] for _ in range(4)]

    # 设置一下坐标轴范围（你可以根据实际数据自动缩放）
    # 这里让它根据轨迹的 min/max 来设置:
    min_x, max_x = np.min(traj_xyz[:,0]), np.max(traj_xyz[:,0])
    min_y, max_y = np.min(traj_xyz[:,1]), np.max(traj_xyz[:,1])
    min_z, max_z = np.min(traj_xyz[:,2]), np.max(traj_xyz[:,2])

    # 适当加一点边界
    margin = 0.1
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z
    ax.set_xlim(min_x - margin*range_x, max_x + margin*range_x)
    ax.set_ylim(min_y - margin*range_y, max_y + margin*range_y)
    ax.set_zlim(min_z - margin*range_z, max_z + margin*range_z)

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    ax.legend()

    def init():
        """init 函数在开始动画前被调用, 返回可更新的艺术家对象列表。"""
        # 不用做太多事，可空实现
        return [apex_line] + side_lines + base_lines

    def update(frame_idx):
        """
        update 函数: 在每帧被调用，用第 frame_idx 个点更新相机锥体位置。
        """
        # 获取当前帧的相机 ENU 位置/姿态
        x, y, z, roll, pitch, yaw = enu_points[frame_idx]
        apex_enu = np.array([x, y, z], dtype=float)

        # 计算视锥
        fov_dict = get_fov_rect_pyramid_enu(
            apex_enu,
            yaw, pitch, roll,
            dist_forward=dist_forward,
            width=width,
            height=height
        )
        apex = fov_dict["apex"]
        base4 = fov_dict["base"]

        # 更新 apex 的散点
        apex_line.set_data([apex[0]], [apex[1]])
        apex_line.set_3d_properties([apex[2]])

        # 更新四条侧棱
        for i in range(4):
            bx, by, bz = base4[i]
            side_lines[i].set_data([apex[0], bx], [apex[1], by])
            side_lines[i].set_3d_properties([apex[2], bz])

        # 更新底面(4条边)
        for i in range(4):
            pA = base4[i]
            pB = base4[(i+1)%4]
            base_lines[i].set_data([pA[0], pB[0]], [pA[1], pB[1]])
            base_lines[i].set_3d_properties([pA[2], pB[2]])

        # 返回更新的艺术家对象列表
        return [apex_line] + side_lines + base_lines

    ani = FuncAnimation(fig, update, frames=len(enu_points),
                        init_func=init, interval=interval_ms, blit=False)

    plt.show()
    # 如果想直接保存 gif 或 mp4，可使用:
    ani.save(save_path, writer="pillow", fps=25)
    # 或:
    # ani.save("camera_fov_animation.mp4", writer="ffmpeg", fps=5)

    

if __name__ == "__main__":
    # 假设你的文本路径是 'trajectory.txt'
    # 文件格式: filename lon lat alt roll pitch yaw
    txt_path = "/mnt/sda/MapScape/query/poses/taiwan_seq1@8@100.txt"
    save_path = os.path.join("/mnt/sda/MapScape/query/trajectory",txt_path.split('/')[-1].split('.')[0]+'.gif' )
    dist_forward = 5.0  # 视锥底面中心与相机点之间的距离
    width = 4.0          # 视锥底面宽度
    height = 4.0         # 视锥底面高度
    interval_ms = 40     # 每帧间隔毫秒
    animate_camera_fov(txt_path,
                       save_path,
                       dist_forward=dist_forward,
                       width=width,
                       height=height,
                       interval_ms=interval_ms)