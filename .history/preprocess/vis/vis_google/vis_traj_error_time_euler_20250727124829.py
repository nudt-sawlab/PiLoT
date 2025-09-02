import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from transform import WGS84_to_ECEF,wgs84tocgcs2000_batch,get_rotation_enu_in_ecef
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
                e = [pitch, roll, yaw]
                t = [lon, lat, alt]
                if yaw < 0:
                    yaw += 360
                xyz.append((lon, lat, alt))
                
                timestamps.append(frame_idx)
                R_c2w = euler_angles_to_matrix_ECEF_w2c(e, t)
                angles.append(R_c2w)
    xyz = wgs84tocgcs2000_batch(xyz, 4547)
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
# ✅ 主流程
seq_list = sorted([f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")])
def load_gt_pose(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                lon, lat, alt = map(float, parts[1:4])
                data.append((lon, lat, alt))
    xyz = wgs84tocgcs2000_batch(data, 4547)
    return np.array(xyz)
# ✅ 主流程（角度误差）
# seq = "USA_seq5@8@cloudy@300-100@200.txt"
seq = "switzerland_seq12@8@foggy@intensity2@200.txt"
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

for method, color in methods.items():
    file_path = os.path.join(data_root, method, seq)
    if not os.path.exists(file_path): continue

    est_frame_ids, _, est_angles = load_pose_with_angle(file_path)
    if 'ORB' in method:
        print('-')
    #     scale, R, t = umeyama_alignment(est_xyz[est_frame_ids[0:300]], gt_xyz[est_frame_ids[0:300]])
    #     est_xyz = transform_points(est_xyz, scale, R, t)

    ang_err = np.full_like(frame_ids, np.nan, dtype=np.float32)
    for i, fid in enumerate(frame_ids):
        if fid in est_frame_ids:
            idx = np.where(est_frame_ids == fid)[0][0]
            cos = np.clip((np.trace(np.dot(gt_angles[fid].T, est_angles[idx])) - 1) / 2, -1., 1.)
            e_R = np.rad2deg(np.abs(np.arccos(cos)))
            ang_err[i] = e_R
        else:
            ang_err[i] = max_y_limit  # 填一个上限

    ang_err_clipped = np.minimum(ang_err, max_y_limit)
    z = method_zorder.get(method, 5)
    ax.scatter(frame_ids, ang_err_clipped,
        color=color,
        label=methods_name[method],
        alpha=0.7,
        s=18,
        zorder=z)
    

    med = np.nanmedian(ang_err)
    method_medians[method] = med

    # 获取横轴范围
    x_min, x_max = ax.get_xlim()
    text_x = x_min + 0.9 * (x_max - x_min)  # 靠右但不出界

    # 中位线 glow 背景（在散点下方）
    ax.axhline(med, linestyle='-', color='white', linewidth=5, alpha=0.6, zorder=4)

    # 中位线主线（使用原色，zorder < scatter）
    ax.axhline(med, linestyle='-', color=color, linewidth=2.2, alpha=0.95, zorder=5)

    # 注释文本靠右侧自动对齐，不超图
    text_x = frame_ids[0] + 5        # 靠左，但不贴边
    text_y = med + method_display_order[method]              # 稍高于中位线
    if 'ORB' in method:
        text_label = f"{methods_name[method]}: N/A"
    else:
        text_label = f"{methods_name[method]}: {med:.2f}°"
    ax.text(text_x, text_y,
        text_label,
        color=methods_text_color[method],
        fontsize=9,
       family='serif',
        ha='left',    # ✅ 靠左对齐
        va='bottom',
        alpha=0.95,
        zorder=100,
        bbox=dict(boxstyle="round,pad=0.25", fc='white', ec='none', alpha=0.8))

ax.set_title(f"Angle Error vs Frame Index", fontsize=16, fontweight='bold', family='serif')
ax.set_xlabel("Frame Index", fontsize=13, fontweight='bold', family='serif')
ax.set_ylabel("Angle Error (°)", fontsize=13, fontweight='bold', family='serif')
ax.grid(True)
ax.set_ylim(0, max_y_limit+0.1)

handles, labels = ax.get_legend_handles_labels()
new_labels = []
for method, label in zip(methods.keys(), labels):
    if method in method_medians:
        new_labels.append(f"{label} (Med: {method_medians[method]:.2f}°)")
    else:
        new_labels.append(label)
legend = ax.legend(handles, labels, loc='upper right',
                    frameon=True, fontsize=11, handlelength=2)
for text in legend.get_texts():
    text.set_family('serif')
# ✅ 强制添加 legend 为顶层图层
ax.add_artist(legend)
legend.set_zorder(999)

fig.tight_layout()
output_path = os.path.join(data_root, "outputs", f"{seq_name}_angle_error_curve.png")
fig.savefig(output_path, dpi=300)
print(f"✅ 已保存至：{output_path}")
plt.close(fig)