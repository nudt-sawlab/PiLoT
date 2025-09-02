import os
import json
import numpy as np
import json
import pyproj
import random
from scipy.spatial.transform import Rotation as R
from wrapper import Pose, Camera
from transform import visualize_points_on_images
import cv2
import torch
from tqdm import tqdm
from torch import nn
from get_depth import sample_points_with_valid_depth, get_3D_samples, transform_ecef_origin, get_points2D_ECEF_projection
from transform import get_rotation_enu_in_ecef, WGS84_to_ECEF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 绘图支持
# pose
lon, lat, alt, roll, pitch, yaw = [-87.621138, 41.875109, 200, 0, 0, 0]
euler_angles_ref = [pitch, roll, yaw]
translation_ref = [lon, lat, alt]
lon, lat, height = translation_ref
rot_pose_in_ned = R.from_euler('xyz', euler_angles_ref, degrees=True).as_matrix()  # ZXY 东北天  
rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
t_c2w = WGS84_to_ECEF(translation_ref)
ref_T = np.eye(4)
ref_T[:3, :3] = R_c2w
ref_T[:3, 3] = t_c2w
ref_T = ref_T.tolist()

# camera
K = [1600, 1200, 1931.7, 1931.7, 800.0, 600.0]
width, height = K[0], K[1]
  
cam_ref = {
'model': 'PINHOLE',
'width': width,
'height': height,
'params': [K[2], K[3], K[4], K[5]] #np.array(K[2:]
}   
origin = [0, 0, 0]     
rcamera = Camera.from_colmap(cam_ref)
pose_ref_origin = transform_ecef_origin(np.array(ref_T), origin=origin)
point3D_wgs84 = [-87.621138, 41.875109, 150.09313]
point3D_ECEF = WGS84_to_ECEF(point3D_wgs84)
point3D_ECEF = np.expand_dims(point3D_ECEF,0)
points2d_ref_rej, _, _, _ = get_points2D_ECEF_projection(pose_ref_origin, rcamera, point3D_ECEF, use_valid = False)
print(points2d_ref_rej)

object_WGS84 = [-87.621138, 41.875109, 150.09313]
object_euler_angles = [0, -0.001862, 180]
object_ECEF = np.array(WGS84_to_ECEF(object_WGS84), dtype=float)
object_lens = np.array([384.0,  160.0, 192.0])*0.02



# (f) 构造长方体 8 个顶点(相对于中心)的局部坐标
Lx, Ly, Lz = object_lens
dx = Lx / 2.0
dy = Ly / 2.0
dz = Lz / 2.0

# 8 个顶点在 “物体局部坐标” (假设该局部 xyz 未旋转时 x->长, y->宽, z->高)
# 这里的排列仅仅是示例
local_corners = np.array([
    [ dx,  dy,  dz],
    [ dx,  dy, -dz],
    [ dx, -dy,  dz],
    [ dx, -dy, -dz],
    [-dx,  dy,  dz],
    [-dx,  dy, -dz],
    [-dx, -dy,  dz],
    [-dx, -dy, -dz],
]).T  # 形状 (3,8)


rot_pose_in_enu = R.from_euler('xyz', object_euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
object_R_c2w = np.matmul(rot_enu_in_ecef, rot_pose_in_enu)

lon, lat, alt, roll, pitch, yaw = [-87.621138, 41.875109, 150.09313,  -0.001862, 90, 270]
euler_angles_ref = [pitch, roll, yaw]
translation_ref = [lon, lat, alt]
lon, lat, height = translation_ref
rot_pose_in_ned = R.from_euler('xyz', euler_angles_ref, degrees=True).as_matrix()  # ZXY 东北天  
rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
t_c2w = WGS84_to_ECEF(translation_ref)
ref_T = np.eye(4)
ref_T[:3, :3] = R_c2w
ref_T[:3, 3] = t_c2w
ref_T = ref_T.tolist()
pose_o2w_origin = transform_ecef_origin(np.array(ref_T), origin=origin)

a = pose_o2w_origin[:3, :3] @ local_corners
# (h) 再从 ENU 转到 ECEF
corners_in_ECEF = (object_ECEF.reshape(3,1) + pose_o2w_origin[:3, :3] @ local_corners).T
points2d_ref_rej, _, _, _ = get_points2D_ECEF_projection(pose_ref_origin, rcamera, corners_in_ECEF, use_valid = False)
# (i) 打印结果
print("中心点 ECEF (m):\n", object_ECEF)
print("\n长方体 8 个顶点 ECEF (m):")
for i in range(8):
    print(f"Corner {i}: {points2d_ref_rej[i]}")


img = cv2.imread("/home/ubuntu/Documents/code/FPVLoc/test_data/yan.png")
# 创建一个新的图形窗口
# 3. 将 3D 点的 X、Y 分量映射到图像像素坐标（仅为示例！）
#    这里先取点的前两维作为 (x, y) 坐标
pts_xy = points2d_ref_rej[:, :2]

# 获取图像尺寸（高度, 宽度）
img_h, img_w = img.shape[:2]

# 4. 在图像上绘制圆点和编号
for idx, (x, y) in enumerate(pts_xy.astype(int)):
    # 在图像上画红色圆点
    cv2.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
    # 添加编号文本（蓝色），文本位置略偏离圆点
    cv2.putText(img, f'{idx}', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness=1)

# 5. 将 BGR 图像转换为 RGB 以便用 Matplotlib 显示
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 6. 用 Matplotlib 打开图像，并显示带有叠加点的结果
plt.figure(figsize=(10, 8))
plt.imshow(img_rgb)
plt.axis('off')
plt.title("图像上叠加显示八个点")
plt.show()