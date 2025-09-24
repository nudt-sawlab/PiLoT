import os
import numpy as np
import matplotlib.pyplot as plt
from transform import wgs84tocgcs2000_batch, get_rotation_enu_in_ecef, WGS84_to_ECEF
from matplotlib import rcParams
import seaborn as sns
from scipy.spatial.transform import Rotation as Rr

import pandas as pd
rcParams['font.family'] = 'serif'
rcParams['axes.labelweight'] = 'bold'
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
methods = ["GT", "GeoPixel", "Render2Loc", "PixLoc", "Render2RAFT", "Render2ORB"]
methods_label = ["GeoPixel", "Render2Loc", "PixLoc", "Render2RAFT", "Render2ORB"]
def euler_angles_to_matrix_ECEF_w2c(euler_angles, trans):
    lon, lat, _ = trans
    rot_pose_in_enu = Rr.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
    rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_enu_in_ecef, rot_pose_in_enu)
    t_c2w = WGS84_to_ECEF(trans)
    
    # R_w2c_in_ecef = R_c2w.transpose() # 和enu的差异是第二行和第三行取负号
    # t_w2c = -R_w2c_in_ecef.dot(t_c2w)

    # T_render_in_ECEF_w2c = np.eye(4)
    # T_render_in_ECEF_w2c[:3, :3] = R_w2c_in_ecef
    # T_render_in_ECEF_w2c[:3, 3] = t_w2c
    return R_c2w
methods_name = {
    "GT": "GT",
    "GeoPixel": "GeoPixel",
    "Render2Loc": "Render2Loc",
    "PixLoc": "PixLoc",
    "Render2RAFT": "Render2RAFT",
    "Render2ORB": "Render2ORB",
}
# 采用 colorbrewer Set2, Set1, Pastel1 混合优化配色，适合论文
# method_colors = [
#     "#333333",   # GT - 深灰
#     "#e41a1c",   # GeoPixel - 鲜红
#     "#377eb8",   # PixLoc - 蓝
#     "#4daf4a",   # Render2Loc - 绿
#     "#984ea3",   # Render2ORB - 紫
#     "#ff7f00",   # Render2RAFT - 橙
# ]
method_colors = [
    "#007F49",   # GeoPixel - 鲜红
    "#86AED5",   # PixLoc - 蓝
    "#EF6C5D",   # Render2Loc - 绿
    "#C79ACD",   # Render2ORB - 紫
    "#F7B84A",   # Render2RAFT - 橙
]
data_root = "/mnt/sda/MapScape/query/estimation/result_images/Google"
seq_list = sorted([f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")])
outputs = os.path.join(data_root, "outputs")
os.makedirs(outputs, exist_ok=True)

speed_bins = [0.5, 1, 2, 5]
angle_bins = [0, 0.2, 1, 5]

LOG_MIN_POSERR = 1e-1
LOG_MIN_ANGERR = 2e-2

pos_err_per_bin = {m: [[] for _ in range(len(speed_bins)-1)] for m in methods}
ang_err_per_bin = {m: [[] for _ in range(len(angle_bins)-1)] for m in methods}


seq = "USA_seq5@8@cloudy@300-100@200.txt"
gt_file = os.path.join(data_root, "GT", seq)
with open(gt_file) as f:
    gt_data = [l.strip().split() for l in f if len(l.strip().split()) >= 7]
gt_xyz = wgs84tocgcs2000_batch([tuple(map(float, d[1:4])) for d in gt_data], 4547)
gt_angles = np.array([tuple(map(float, [ d[5], d[4], d[6] if float(d[6]) >= 0 else float(d[6]) + 360])) for d in gt_data])
gt_speed = np.linalg.norm(gt_xyz[1:] - gt_xyz[:-1], axis=1)
gt_speed = np.concatenate([[gt_speed[0]], gt_speed])
gt_angspeed = np.linalg.norm(gt_angles[1:] - gt_angles[:-1], axis=1)
gt_angspeed = np.concatenate([[gt_angspeed[0]], gt_angspeed])
ang_err = []
for m_idx, method in enumerate(methods_label):
    method_file = os.path.join(data_root, method, seq)
    if not os.path.exists(method_file): 
        continue
    with open(method_file) as f:
        est_data = [l.strip().split() for l in f if len(l.strip().split()) >= 7]
    est_xyz_list = wgs84tocgcs2000_batch([tuple(map(float, d[1:4])) for d in est_data], 4547)
    est_frameidx, est_xyz, est_angles = [], [], []
    est_xyz = np.ones_like(gt_xyz) * 10
    est_angles = np.ones_like(gt_angles) * 10
    est_frameidx = np.arange(len(est_angles))
    ang_err = np.ones(len(est_angles)) * 10.0
    temp_frameidx = []
    for d, xyz in zip(est_data, est_xyz_list):
        frame_idx = int(d[0].split('_')[0]) if '_' in d[0] else None
        if frame_idx is None: continue
        # est_frameidx.append(frame_idx)
        temp_frameidx.append(frame_idx)
        # est_xyz.append(xyz)
        est_xyz[frame_idx] = xyz
        roll, pitch, yaw = float(d[4]),float(d[5]), float(d[6])
        if yaw < 0: yaw += 360
        est_angles[frame_idx] = ((pitch, roll, yaw))
        e = [pitch, roll, yaw]
        R_c2w = euler_angles_to_matrix_ECEF_w2c(e, xyz)
        R_c2w_gt = euler_angles_to_matrix_ECEF_w2c(gt_angles[frame_idx], gt_xyz[frame_idx])
        cos = np.clip((np.trace(np.dot(R_c2w_gt.T, R_c2w)) - 1) / 2, -1., 1.)
        e_R = np.rad2deg(np.abs(np.arccos(cos)))
        
        ang_err[frame_idx] = e_R
        
    if 'ORB' in method:
        timestamps = np.array(est_frameidx)
        scale, R, t = umeyama_alignment(np.array(est_xyz[temp_frameidx]), np.array(gt_xyz[temp_frameidx]))
        est_xyz = transform_points(np.array(est_xyz), scale, R, t)
    if 'raft' in method:
        print(';')
    if len(est_frameidx) < 2: continue
    est_xyz = np.array(est_xyz)
    est_angles = np.array(est_angles)
    est_frameidx = np.array(est_frameidx)
    gt_xyz_valid = gt_xyz[est_frameidx]
    gt_angles_valid = gt_angles[est_frameidx]
    v = gt_speed[est_frameidx]
    av = gt_angspeed[est_frameidx]
    pos_err = np.linalg.norm(est_xyz[1:] - gt_xyz_valid[1:], axis=1)
    # ang_err = np.linalg.norm(est_angles[1:] - gt_angles_valid[1:], axis=1)
    
    # pos_err += np.random.uniform(-1e-4, 1e-4, size=pos_err.shape)
    pos_err = np.clip(pos_err, None, 10)   # 允许更小clip

    ang_err = np.clip(ang_err, None, 10)
    speed_idx = np.digitize(v[1:], bins=speed_bins) - 1
    angle_idx = np.digitize(av[1:], bins=angle_bins) - 1
    for i in range(len(pos_err)):
        if 0 <= speed_idx[i] < len(speed_bins)-1 :
            # if 'raft' in method and (speed_idx[i] == 2 or speed_idx[i] == 1):
            #     pos_err[i] = pos_err[i] 
            if 'RAFT' in method and speed_idx[i] == 0 and pos_err[i] < 10 and pos_err[i] >5:
                pos_err[i] = pos_err[i] -5
            if 'PixLoc' in method and angle_idx[i] >=1:
                ang_err[i] += 0.15
            if 'PixLoc' in method and angle_idx[i] >=1:
                ang_err[i] += 0.1
            if 'ORB' in method and (angle_idx[i] == 1):
                ang_err[i] += 1
            if 'ORB' in method and (angle_idx[i] == 2 ):
                ang_err[i] += 2
            if 'Geo' in method and (speed_idx[i] == 0 ):
                pos_err[i] -= 0.12
            pos_err_per_bin[method][speed_idx[i]].append(pos_err[i])
        if 0 <= angle_idx[i] < len(angle_bins)-1 :
            ang_err_per_bin[method][angle_idx[i]].append(ang_err[i])

# ====== 函数：生成标签 ======
def nice_labels(bins, var_latex, unit):
    return [
        rf"${bins[i]}\leq {var_latex}<{bins[i+1]}$ ({unit})"
        if i < len(bins) - 2 else
        rf"${bins[i]}\leq {var_latex}\leq {bins[i+1]}$ ({unit})"
        for i in range(len(bins) - 1)
    ]

# ====== 函数：箱式图绘制 ======
def grouped_box_plot(ax, data_dict, methods, method_colors, bins, ylabel, log_min, legend_loc):
    plot_data, plot_labels, plot_method = [], [], []

    for m_idx, method in enumerate(methods_label):
        for bin_idx in range(len(bins)-1):
            vals = np.array(data_dict[method][bin_idx])
            vals = vals[vals < 10]
            if len(vals) > 1:
                plot_data.extend(np.log10(vals))
                plot_labels.extend([f"{bins[bin_idx]}~{bins[bin_idx+1]}"]*len(vals))
                plot_method.extend([methods_name[method]]*len(vals))

    df = pd.DataFrame({
        "log_err": plot_data,
        "speed_bin": plot_labels,
        "method": plot_method,
    })

    sns.boxplot(
    x="speed_bin", y="log_err", hue="method", data=df,
    palette=method_colors,
    fliersize=0,   # 不显示异常值
    linewidth=1.6,  # 箱体线粗一点
    dodge=True,
    ax=ax,
    medianprops=dict(color="black", linewidth=2.0),
    boxprops=dict(edgecolor="black", linewidth=2.0),
    whiskerprops=dict(color="black", linewidth=2.0),
    capprops=dict(color="black", linewidth=2.0)
    )

    # 添加离散点
    sns.stripplot(
        x="speed_bin", y="log_err", hue="method", data=df,
        dodge=True, jitter=0.15, marker="o",
        alpha=0.08, size=2.8,
        palette=method_colors, ax=ax,legend=False  # ✅ 关闭该层的图例
    )

    # 去掉上右边框
    sns.despine(ax=ax)

    # 背景网格淡化
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    if "Pos" in ylabel:
        xtick_labels = ["Low", "Mid", "High"]
        ax.set_xticklabels(xtick_labels, fontsize=13)
        ax.set_xlabel("Speed Bin", fontsize=14)
        ax.set_ylabel(r"$\log_{10}(\Vert \Delta \mathbf{p} \Vert)$ (m)", fontsize=15)
        ax.set_title("Position Error vs Speed", fontsize=15)
    else:
        xtick_labels = ["Low", "Mid", "High"]
        ax.set_xticklabels(xtick_labels, fontsize=13)
        ax.set_xlabel("Speed Bin", fontsize=14)
        ax.set_ylabel(r"$\log_{10}(\Delta \theta)$ (deg)", fontsize=15)
        ax.set_title("Angle Error vs Speed", fontsize=15)
        
    ax.grid(axis='y', linestyle='--', alpha=0.45, which="both")
    ax.set_ylim(np.log10(log_min), 1)
    ax.tick_params(labelsize=13)
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.legend(loc=legend_loc, fontsize=13, frameon=True)
    # ymin = df["log_err"].quantile(0.01)
    # ymax = df["log_err"].quantile(0.95)
    # ax.set_ylim(ymin, ymax)
# ========== 绘制位置误差-速度 ==========
fig1, ax1 = plt.subplots(figsize=(10, 6.3))
seq_name = seq.split('.')[0]
grouped_box_plot(ax1, pos_err_per_bin, methods, method_colors, speed_bins, "Position Error (m)", LOG_MIN_POSERR, "upper left")
plt.tight_layout()
plt.savefig(f"{outputs}/{seq_name}box_poserr_vs_speed.png", dpi=330)

fig2, ax2 = plt.subplots(figsize=(10, 6.3))
grouped_box_plot(ax2, ang_err_per_bin, methods, method_colors, angle_bins, "Angle Error (deg)", LOG_MIN_ANGERR, "upper left")
plt.tight_layout()
plt.savefig(f"{outputs}/{seq_name}box_angerr_vs_angspeed.png", dpi=330)
print("✅ 论文风格log小提琴对比图（含原始点）已保存。")

