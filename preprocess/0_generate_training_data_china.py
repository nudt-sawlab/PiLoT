import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm
from transform import qvec2rotmat, rotmat2qvec, cgcs2000towgs84_dev, cgcs2000towgs84, WGS84_to_ECEF, wgs84tocgcs2000
def get_epsg_from_longitude(longitude):
    """
    根据经度信息返回对应的 CGCS2000 EPSG 代号。

    :param longitude: 输入经度（float）
    :return: EPSG 代号（int），如果经度超出范围，返回 None
    """
    # 定义经度范围与对应的 EPSG 代号
    epsg_mapping = [
        (73.5, 76.5, 4534),
        (76.5, 79.5, 4535),
        (79.5, 82.5, 4536),
        (82.5, 85.5, 4537),
        (85.5, 88.5, 4538),
        (88.5, 91.5, 4539),
        (91.5, 94.5, 4540),
        (94.5, 97.5, 4541),
        (97.5, 100.5, 4542),
        (100.5, 103.5, 4543),
        (103.5, 106.5, 4544),
        (106.5, 109.5, 4545),
        (109.5, 112.5, 4546),
        (112.5, 115.5, 4547),
        (115.5, 118.5, 4548),
        (118.5, 121.5, 4549),
        (121.5, 124.5, 4550),
        (124.5, 127.5, 4551),
        (127.5, 130.5, 4552),
        (130.5, 133.5, 4553),
        (133.5, 136.5, 4554),
    ]

    # 查找对应的 EPSG 代号
    for min_lon, max_lon, epsg in epsg_mapping:
        if min_lon <= longitude < max_lon:
            return epsg

    # 如果经度超出范围，返回 None
    return None


abs_init = [8.54932, 47.37487, 90.523]
# pitch ,
pitch = 45
roll = 0
epsg = get_epsg_from_longitude(abs_init[0])
abs_cgcs_init = wgs84tocgcs2000(abs_init, epsg)
# 参数设置
num_points = 3000  # 点的数量
total_length = 500  # 总飞行距离，单位米qua
x_range = np.linspace(0, total_length, num_points)  # x 坐标范围

# name 
title = 'Citystar_seq'

trajectory_txt = '/mnt/sda/MapScape/pose/'  
idx = 3 #len(os.listdir(trajectory_txt))
# 调整幅值和周期
A = 100  # y 正弦函数幅值 (最大值减去最小值不超过100米)
B = 20   # z 正弦函数幅值 (最大值减去最小值不超过100米)
k_new = 2 * np.pi / 500  # y 正弦波周期 (周期约500米)
m_new = 2 * np.pi / 500  # z 正弦波周期 (周期约500米)
phi = 0  # y 正弦函数相位
psi = np.pi / 4  # z 正弦函数相位

# 生成无人机轨迹点
x = x_range
y = A * np.sin(k_new * x + phi)
z = B * np.sin(m_new * x + psi)


# 计算每个点的切线角度（相邻点的差异）
dx = np.gradient(x)
dy = np.gradient(y)
tangent_angles = np.arctan2(dy, dx)  # 切线角度（弧度）

# 转换为度数
tangent_angles_degrees = np.degrees(tangent_angles)

# 根据需求调整角度：以Y轴正方向为0度，顺时针旋转
angles_xy_degrees = (90 - tangent_angles_degrees) % 360  # 顺时针旋转，Y轴正方向为0度


file_name =title +str(idx+1)+'.txt' 
trajectory_txt += file_name
lon = []
lat = []
alt = []
name_list = []
cgcs2000_xyz = np.array([np.array(abs_cgcs_init) + np.array([x[i], y[i], z[i]]) for i in range(len(x))])
wgs84_coords = cgcs2000towgs84_dev(cgcs2000_xyz, epsg)

with open(trajectory_txt, 'w') as f:
    for i in tqdm(range(len(x))):
        name = str(i) +'.jpg'
        cgcs2000_xyz = [np.array(abs_cgcs_init) + np.array([x[i], y[i], z[i]])]
        pitch_noise = random.random() - 0.5 + pitch
        roll_noise = random.random() - 0.5
        euler_enu = [pitch_noise, roll_noise, angles_xy_degrees[i]]
        # wgs84_coord = cgcs2000towgs84(cgcs2000_xyz, epsg)
        # ecef_xyz = WGS84_to_ECEF(wgs84_coord)
        wgs84_coord = wgs84_coords[i].tolist()
        coord = ' '.join(map(str, euler_enu+ wgs84_coord))
        f.write(f'{name} {coord}\n')
        lon.append(wgs84_coord[1])
        lat.append(wgs84_coord[0])
        alt.append(wgs84_coord[2])
        name_list.append(i)


trajectory = np.vstack((lat, lon, alt)).T
trajectory_df = pd.DataFrame(trajectory, columns=["X", "Y", "Z"])
trajectory_df["name"] = i

# 打印数据
print(trajectory_df)
google_earth_trajectory_csv = "/mnt/sda/MapScape/trajectory/"+title+str(idx+1)+'.csv' 
# 如果需要保存到文件，可以使用以下代码：
trajectory_df.to_csv(google_earth_trajectory_csv, index=False)
# # 绘制三维点云
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=x, cmap='viridis', s=5)

# 设置标签
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.title('3D Points with Sinusoidal Projections')

plt.show()

# ---Plot 绘制XY投影和切线方向
# 绘制XY投影和切线方向
# plt.figure(figsize=(10, 7))

# # 绘制 XY 投影曲线
# plt.plot(x, y, label='XY Projection', color='blue')

# # 绘制切线方向（每隔一定间隔显示切线）
# for i in range(0, len(x), 200):  # 每隔 200 个点显示一个切线角度
#     angle = tangent_angles_degrees[i]
#     plt.arrow(x[i], y[i],
#               10 * np.cos(angle * np.pi / 180),  # 切线的 X 分量
#               10 * np.sin(angle * np.pi / 180),  # 切线的 Y 分量
#               head_width=5, head_length=7, fc='green', ec='green')

# # 设置标签和图例
# plt.xlabel('X-axis (meters)')
# plt.ylabel('Y-axis (meters)')
# plt.title('XY Projection with Tangent Directions')
# plt.legend(['XY Projection'])
# plt.grid()
# plt.axis('equal')

# plt.show()
