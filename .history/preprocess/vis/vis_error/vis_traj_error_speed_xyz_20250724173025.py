import os
import numpy as np
import matplotlib.pyplot as plt
from transform import wgs84tocgcs2000_batch
from matplotlib import rcParams
import seaborn as sns

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
methods = ["GT", "FPVLoc", "Render2loc", "Pixloc", "ORB@per30", "Render2loc@raft"]
methods_name = {
    "GT": "GT",
    "FPVLoc": "GeoPixel",
    "Render2loc": "Render2Loc",
    "Pixloc": "PixLoc",
    "ORB@per30": "Render2ORB",
    "Render2loc@raft": "Render2RAFT"
}
# 采用 colorbrewer Set2, Set1, Pastel1 混合优化配色，适合论文
method_colors = [
    "#333333",   # GT - 深灰
    "#e41a1c",   # GeoPixel - 鲜红
    "#377eb8",   # PixLoc - 蓝
    "#4daf4a",   # Render2Loc - 绿
    "#984ea3",   # Render2ORB - 紫
    "#ff7f00",   # Render2RAFT - 橙
]

data_root = "/mnt/sda/MapScape/query/estimation/result_images"
seq_list = sorted([f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")])
outputs = os.path.join(data_root, "outputs")
os.makedirs(outputs, exist_ok=True)

speed_bins = [0.5, 1, 2, 5]
angle_bins = [0, 0.2, 1, 5]

LOG_MIN_POSERR = 1e-2
LOG_MIN_ANGERR = 1e-3

pos_err_per_bin = {m: [[] for _ in range(len(speed_bins)-1)] for m in methods}
ang_err_per_bin = {m: [[] for _ in range(len(angle_bins)-1)] for m in methods}

for seq in seq_list:
    gt_file = os.path.join(data_root, "GT", seq)
    with open(gt_file) as f:
        gt_data = [l.strip().split() for l in f if len(l.strip().split()) >= 7]
    gt_xyz = wgs84tocgcs2000_batch([tuple(map(float, d[1:4])) for d in gt_data], 4547)
    gt_angles = np.array([tuple(map(float, [d[5], d[6] if float(d[6]) >= 0 else float(d[6]) + 360])) for d in gt_data])
    gt_speed = np.linalg.norm(gt_xyz[1:] - gt_xyz[:-1], axis=1)
    gt_speed = np.concatenate([[gt_speed[0]], gt_speed])
    gt_angspeed = np.linalg.norm(gt_angles[1:] - gt_angles[:-1], axis=1)
    gt_angspeed = np.concatenate([[gt_angspeed[0]], gt_angspeed])
    for m_idx, method in enumerate(methods):
        method_file = os.path.join(data_root, method, seq)
        if not os.path.exists(method_file): continue
        with open(method_file) as f:
            est_data = [l.strip().split() for l in f if len(l.strip().split()) >= 7]
        est_xyz_list = wgs84tocgcs2000_batch([tuple(map(float, d[1:4])) for d in est_data], 4547)
        est_frameidx, est_xyz, est_angles = [], [], []
        est_angles = np.ones_like(gt_angles) * 180
        est_xyz = np.ones_like(gt_xyz) * 99
        est_frameidx = np.arange(len(est_xyz))
        temp_frameidx = []
        for d, xyz in zip(est_data, est_xyz_list):
            frame_idx = int(d[0].split('_')[0]) if '_' in d[0] else None
            if frame_idx is None: continue
            temp_frameidx.append(frame_idx)
            # est_xyz.append(xyz)
            # pitch, yaw = float(d[5]), float(d[6])
            # if yaw < 0: yaw += 360
            # est_angles.append((pitch, yaw))
            est_frameidx[frame_idx] = int(frame_idx)
            est_xyz[frame_idx] = xyz
            pitch, yaw = float(d[5]), float(d[6])
            if yaw < 0: yaw += 360
            est_angles[frame_idx] = (pitch, yaw)
        if 'ORB' in method:
            timestamps = np.array(est_frameidx)
            scale, R, t = umeyama_alignment(np.array(est_xyz[temp_frameidx]), np.array(gt_xyz[temp_frameidx]))
            est_xyz = transform_points(np.array(est_xyz), scale, R, t)
        if len(est_frameidx) < 2: continue
        est_xyz = np.array(est_xyz)
        est_angles = np.array(est_angles)
        est_frameidx = np.array(est_frameidx)
        gt_xyz_valid = gt_xyz[est_frameidx]
        gt_angles_valid = gt_angles[est_frameidx]
        v = gt_speed[est_frameidx]
        av = gt_angspeed[est_frameidx]
        pos_err = np.linalg.norm(est_xyz[1:] - gt_xyz_valid[1:], axis=1)
        
        # pos_err += np.random.uniform(-1e-4, 1e-4, size=pos_err.shape)
        pos_err = np.clip(pos_err, None, 10)   # 允许更小clip
        ang_err = np.linalg.norm(est_angles[1:] - gt_angles_valid[1:], axis=1)
        # ang_err += np.random.uniform(-1e-4, 1e-4, size=ang_err.shape)
        ang_err = np.clip(ang_err, None, 10)
        speed_idx = np.digitize(v[1:], bins=speed_bins) - 1
        angle_idx = np.digitize(av[1:], bins=angle_bins) - 1
        for i in range(len(pos_err)):
            if 0 <= speed_idx[i] < len(speed_bins)-1:
                pos_err_per_bin[method][speed_idx[i]].append(pos_err[i])
            if 0 <= angle_idx[i] < len(angle_bins)-1:
                ang_err_per_bin[method][angle_idx[i]].append(ang_err[i])

def nice_labels(bins, unit):
    return [f"{bins[i]}~{bins[i+1]}{unit}" for i in range(len(bins)-1)]

def grouped_violin_plot(ax, data_dict, methods, method_colors, bins, ylabel, log_min, legend_loc):
    width = 0.12
    group_width = width * len(methods)
    positions = np.arange(len(bins)-1)
    # 按照分组做聚合数据以便seaborn画
    plot_data = []
    plot_labels = []
    plot_method = []
    plot_xs = []
    for m_idx, method in enumerate(methods):
        for bin_idx in range(len(bins)-1):
            vals = np.array(data_dict[method][bin_idx])
            if len(vals) > 1:
                plot_data.extend(np.log10(vals))  # log10更自然
                plot_labels.extend([f"{bins[bin_idx]}~{bins[bin_idx+1]}"]*len(vals))
                plot_method.extend([methods_name[method]]*len(vals))
                plot_xs.extend([bin_idx + m_idx/(len(methods)+1)]*len(vals)) # jitter in group

    import pandas as pd
    df = pd.DataFrame({
        "log_err": plot_data,
        "speed_bin": plot_labels,
        "method": plot_method,
        "x": plot_xs,
    })
    # 用seaborn小提琴
    sns.violinplot(
        x="speed_bin", y="log_err", hue="method", data=df,
        scale="width", inner="quartile", cut=0, linewidth=1.4,
        palette=method_colors,
        bw=0.3  # 0.3-0.5之间会更圆滑
    )
    # 叠加strip
    sns.stripplot(
        x="speed_bin", y="log_err", hue="method", data=df,
        dodge=True, palette=method_colors, marker=".", alpha=0.18, size=2.4, jitter=True
    )
    # 只保留一个legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc=legend_loc, fontsize=13, frameon=True)
    ax.set_xlabel("Speed Bin" if "Pos" in ylabel else "Angular Speed Bin", fontsize=14)
    ax.set_ylabel("log$_{10}$("+ylabel+")", fontsize=15)
    ax.grid(axis='y', linestyle='--', alpha=0.45, which="both")
    ax.set_ylim(np.log10(log_min), None)
    ax.set_title(ylabel + " Distribution across " + ("Speed Bins" if "Pos" in ylabel else "Angular Speed Bins"), fontsize=15)
    ax.tick_params(labelsize=13)
    plt.setp(ax.get_xticklabels(), rotation=12)

# ========== 绘制位置误差-速度 ==========
fig1, ax1 = plt.subplots(figsize=(10,6.3))
grouped_violin_plot(ax1, pos_err_per_bin, methods, method_colors, speed_bins, "Position Error (m)", LOG_MIN_POSERR, "upper right")
plt.tight_layout()
plt.savefig(f"{outputs}/nature_violin_log_poserr_vs_speed.png", dpi=330)
plt.show()

# ========== 绘制角度误差-角速度 ==========
fig2, ax2 = plt.subplots(figsize=(10,6.3))
grouped_violin_plot(ax2, ang_err_per_bin, methods, method_colors, angle_bins, "Angle Error (deg)", LOG_MIN_ANGERR, "upper right")
plt.tight_layout()
plt.savefig(f"{outputs}/nature_violin_log_angerr_vs_angspeed.png", dpi=330)
plt.show()

print("✅ 论文风格log小提琴对比图（含原始点）已保存。")

