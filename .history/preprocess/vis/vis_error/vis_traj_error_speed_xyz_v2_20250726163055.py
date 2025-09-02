import os
import numpy as np
import matplotlib.pyplot as plt
from transform import wgs84tocgcs2000_batch
from matplotlib import rcParams
import seaborn as sns
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

methods = ["GT", "FPVLoc", "Render2loc", "Pixloc", "ORB@per30", "Render2loc@raft"]
methods_name = {
    "GT": "GT",
    "FPVLoc": "GeoPixel",
    "Render2loc": "Render2Loc",
    "Pixloc": "PixLoc",
    "ORB@per30": "Render2ORB",
    "Render2loc@raft": "Render2RAFT"
}
method_colors = {
    "GeoPixel": "#e41a1c",
    "Render2Loc": "#4daf4a",
    "PixLoc": "#377eb8",
    "Render2ORB": "#984ea3",
    "Render2RAFT": "#ff7f00"
}

data_root = "/mnt/sda/MapScape/query/estimation/result_images"
outputs = os.path.join(data_root, "outputs")
os.makedirs(outputs, exist_ok=True)
seq_list = sorted([f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")])

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

    for method in methods:
        method_file = os.path.join(data_root, method, seq)
        if not os.path.exists(method_file):
            continue
        with open(method_file) as f:
            est_data = [l.strip().split() for l in f if len(l.strip().split()) >= 7]
        est_xyz_list = wgs84tocgcs2000_batch([tuple(map(float, d[1:4])) for d in est_data], 4547)
        est_frameidx, est_xyz, est_angles = [], [], []
        for d, xyz in zip(est_data, est_xyz_list):
            frame_idx = int(d[0].split('_')[0]) if '_' in d[0] else None
            if frame_idx is None:
                continue
            est_frameidx.append(frame_idx)
            est_xyz.append(xyz)
            pitch, yaw = float(d[5]), float(d[6])
            if yaw < 0: yaw += 360
            est_angles.append((pitch, yaw))
        if 'ORB' in method:
            timestamps = np.array(est_frameidx)
            scale, R, t = umeyama_alignment(np.array(est_xyz), np.array(gt_xyz[timestamps]))
            est_xyz = transform_points(np.array(est_xyz), scale, R, t)
        if len(est_frameidx) < 2:
            continue
        est_xyz = np.array(est_xyz)
        est_angles = np.array(est_angles)
        est_frameidx = np.array(est_frameidx)
        gt_xyz_valid = gt_xyz[est_frameidx]
        gt_angles_valid = gt_angles[est_frameidx]
        v = gt_speed[est_frameidx]
        av = gt_angspeed[est_frameidx]
        pos_err = np.linalg.norm(est_xyz[1:] - gt_xyz_valid[1:], axis=1)
        ang_err = np.linalg.norm(est_angles[1:] - gt_angles_valid[1:], axis=1)
        speed_idx = np.digitize(v[1:], bins=speed_bins) - 1
        angle_idx = np.digitize(av[1:], bins=angle_bins) - 1
        for i in range(len(pos_err)):
            if 0 <= speed_idx[i] < len(speed_bins)-1:
                pos_err_per_bin[method][speed_idx[i]].append(pos_err[i])
            if 0 <= angle_idx[i] < len(angle_bins)-1:
                ang_err_per_bin[method][angle_idx[i]].append(ang_err[i])

def boxplot_grouped(ax, data_dict, methods, method_colors, bins, ylabel, log_min, legend_loc):
    plot_data, plot_labels, plot_method = [], [], []
    for method in methods:
        for bin_idx in range(len(bins)-1):
            vals = np.array(data_dict[method][bin_idx])
            if len(vals) > 1:
                plot_data.extend(np.log10(vals))
                plot_labels.extend([f"{bins[bin_idx]}~{bins[bin_idx+1]}"] * len(vals))
                plot_method.extend([methods_name[method]] * len(vals))
    df = pd.DataFrame({
        "log_err": plot_data,
        "bin": plot_labels,
        "method": plot_method
    })
    sns.boxplot(
        x="bin", y="log_err", hue="method", data=df,
        palette=method_colors, linewidth=1.2, fliersize=1.5,
        ax=ax
    )
    ax.grid(axis='y', linestyle='--', alpha=0.45)
    ax.set_ylim(np.log10(log_min), None)
    ax.tick_params(labelsize=13)
    if "Pos" in ylabel:
        ax.set_xlabel("Speed (m/frame) bin", fontsize=14)
        ax.set_ylabel(r"$\log_{10}(\Vert \Delta \mathbf{p} \Vert)$ (m)", fontsize=15)
        ax.set_title("Position Error Distribution across Speed bins", fontsize=15)
    else:
        ax.set_xlabel("Angular Speed (deg/frame) bin", fontsize=14)
        ax.set_ylabel(r"$\log_{10}(\Delta \theta)$ (deg)", fontsize=15)
        ax.set_title("Angle Error Distribution across Angular Speed bins", fontsize=15)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if "GT" in by_label:
        by_label.pop('GT')
    ax.legend(by_label.values(), by_label.keys(), loc=legend_loc, fontsize=13, frameon=True)

# ========== 绘制位置误差-速度 ==========
fig1, ax1 = plt.subplots(figsize=(10,6.3))
boxplot_grouped(ax1, pos_err_per_bin, methods, method_colors, speed_bins, "Position Error (m)", LOG_MIN_POSERR, "upper right")
plt.tight_layout()
plt.savefig(f"{outputs}/nature_box_log_poserr_vs_speed.png", dpi=330)

# ========== 绘制角度误差-角速度 ==========
fig2, ax2 = plt.subplots(figsize=(10,6.3))
boxplot_grouped(ax2, ang_err_per_bin, methods, method_colors, angle_bins, "Angle Error (deg)", LOG_MIN_ANGERR, "upper right")
plt.tight_layout()
plt.savefig(f"{outputs}/nature_box_log_angerr_vs_angspeed.png", dpi=330)

