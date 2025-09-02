import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from transform import WGS84_to_ECEF,ECEF_to_WGS84,get_rotation_enu_in_ecef, cgcs2000towgs84_batch
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['axes.labelweight'] = 'bold'
def euler_angles_to_matrix_ECEF_w2c(euler_angles, trans):
    lon, lat, _ = trans
    rot_pose_in_enu = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
    rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_enu_in_ecef, rot_pose_in_enu)
    t_c2w = WGS84_to_ECEF(trans)
    
    # R_w2c_in_ecef = R_c2w.transpose() # 和enu的差异是第二行和第三行取负号
    # t_w2c = -R_w2c_in_ecef.dot(t_c2w)

    # T_render_in_ECEF_w2c = np.eye(4)
    # T_render_in_ECEF_w2c[:3, :3] = R_w2c_in_ecef
    # T_render_in_ECEF_w2c[:3, 3] = t_w2c
    return R_c2w
# ✅ 配置参数
data_root = "/mnt/sda/MapScape/query/estimation/result_images"
methods_name = {
    "GT": "GT",
    "FPVLoc": "GeoPixel",
    "Pixloc": "PixLoc",
    "Render2loc": "Render2Loc",
    "ORB@per30": "Render2ORB",
    "Render2loc@raft": "Render2RAFT"
}
method_zorder = {
    "FPVLoc": 10,
    "Render2loc": 9,
    "Pixloc": 8,
    "ORB@per30": 7,
    "Render2loc@raft": 6
}
methods = {
    "FPVLoc": "#007F49",      # ✅ GeoPixel：深绿，不变
    "Pixloc": "#86AED5",       # 加深灰蓝
    "Render2loc": "#EF6C5D",   # 加深橘粉
    "ORB@per30": "#C79ACD",    # 加深淡紫
    "Render2loc@raft": "#FFE0B5"  # 奶油橙
}
methods_text_color = {
    "FPVLoc": "#007F49",      # ✅ GeoPixel：深绿，不变
    "Pixloc": "#86AED5",       # 加深灰蓝
    "Render2loc": "#EF6C5D",   # 加深橘粉
    "ORB@per30": "#C79ACD",    # 加深淡紫
    "Render2loc@raft": "#F7B84A"  # 奶油橙
}
method_display_order = {
    "FPVLoc": 0,
    "Render2loc": 0.0,
    "Pixloc": 0.02,
    "ORB@per30": 0,
    "Render2loc@raft": 0
}
start_frame = 0  # ✅ 可调
end_frame = 900

def load_pose_with_angle(file_path):
    xyz, angles, timestamps = [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                name = parts[0]
                if "_" in name:
                    frame_idx = int(name.split("_")[0])
                else:
                    continue
                lon, lat, alt = map(float, parts[1:4])
                roll, pitch, yaw = map(float, parts[4:7])
                e = [roll, pitch, yaw]
                t = [lon, lat, alt]
                if yaw < 0:
                    yaw += 360
                xyz.append(WGS84_to_ECEF([lon, lat, alt]))
                timestamps.append(frame_idx)
                R_c2w = euler_angles_to_matrix_ECEF_w2c(e, t)
                angles.append(e)
    return np.array(timestamps), np.array(xyz), np.array(angles)
def transform_points(points, scale, R, t):
    return scale * (R @ points.T).T + t

def umeyama_alignment(src, dst):
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    src_centered, dst_centered = src - mu_src, dst - mu_dst
    cov = dst_centered.T @ src_centered / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    scale = np.trace(np.diag(D)) / ((src_centered ** 2).sum() / src.shape[0])
    t = mu_dst - scale * R @ mu_src
    return scale, R, t

def load_gt_angle(file_path):
    angles = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                lon, lat, alt = map(float, parts[1:4])
                roll, pitch, yaw = map(float, parts[4:7])
                e = [pitch, roll, yaw]
                t = [lon, lat, alt]
                if yaw < 0:
                    yaw += 360
                R_c2w = euler_angles_to_matrix_ECEF_w2c(e, t)
                angles.append(R_c2w)
    return np.array(angles)
def load_pose_with_name(file_path):
    data, timestamps = [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                name = parts[0]
                if "_" in name:
                    frame_idx = int(name.split("_")[0])
                else:
                    continue
                lon, lat, alt = map(float, parts[1:4])
                data.append(WGS84_to_ECEF([lon, lat, alt]))
                timestamps.append(frame_idx)
    return np.array(timestamps), np.array(data)
# ✅ 主流程
def load_gt_pose(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                lon, lat, alt = map(float, parts[1:4])
                data.append(WGS84_to_ECEF([lon, lat, alt]))
    return np.array(data)
# ✅ 主流程（角度误差）
# seq = "USA_seq5@8@cloudy@300-100@200.txt"
seq = "USA_seq5@8@foggy@500-400@intensity3@500.txt"
print(f"�� 绘制角度误差随时间变化图：{seq}")
seq_name = seq.split('.')[0]

gt_angles_all = load_gt_angle(os.path.join(data_root, "GT", seq))
frame_ids_all = np.arange(len(gt_angles_all))
valid_mask = (frame_ids_all >= start_frame) & (frame_ids_all <= end_frame)
gt_angles = gt_angles_all[valid_mask]
frame_ids = frame_ids_all[valid_mask]
gt_xyz_all = load_gt_pose(os.path.join(data_root, "GT", seq))

fig, ax = plt.subplots(figsize=(10, 6))
max_y_limit = 0.5 # 角度误差上限（单位°）
method_medians = {}

file_path = "/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5/Render2ORB/USA_seq5@8@cloudy@300-100@200.txt"

est_frame_ids, _, est_angles = load_pose_with_angle(file_path)
est_frame_ids, est_xyz = load_pose_with_name(file_path)
scale, Rr, t = umeyama_alignment(est_xyz[0:300], gt_xyz_all[est_frame_ids[0:300]])
est_xyz = transform_points(est_xyz, scale, Rr, t)
wgs84 = []
for ecef in est_xyz:
    wgs84.append(ECEF_to_WGS84(ecef))
est_xyz = np.array(wgs84)

save_path = "/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5/Render2ORB/USA_seq5@8@cloudy@300-100@200_transformed.txt"
with open(save_path, 'w') as f:
    for i in range(len(est_xyz)):
        f.write(f"{est_frame_ids[i]}_0.png {est_xyz[i][0]} {est_xyz[i][1]} {est_xyz[i][2]} {est_angles[i][0]} {est_angles[i][1]} {est_angles[i][2]}\n")
        


    