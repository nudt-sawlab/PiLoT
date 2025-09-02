import os
import numpy as np
import matplotlib.pyplot as plt
from transform import wgs84tocgcs2000_batch

# 高级配色：nature风格
methods = ["GT", "FPVLoc", "Pixloc", "Render2loc", "ORB@per30", "Render2loc@raft"]
method_colors = [
    "#222222",   # GT-深灰
    "#C44E52",   # FPVLoc-高级红
    "#4C72B0",   # Pixloc-深蓝
    "#55A868",   # Render2loc-高级绿
    "#8172B2",   # ORB@per30-紫
    "#CCB974"    # Render2loc@raft-棕
]

data_root = "/mnt/sda/MapScape/query/estimation/result_images"
seq_list = sorted([f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")])
data_root = "/mnt/sda/MapScape/query/estimation/result_images"
outputs = os.path.join(data_root, "outputs")
os.makedirs(outputs, exist_ok=True)
speed_bins = [0, 0.5, 2, 10]      # m/frame
angle_bins = [0, 0.2, 1, 10]      # deg/frame
POS_ERR_CLIP = 12                 # m
ANG_ERR_CLIP = 10                 # deg

pos_err_per_bin = {m: [[] for _ in range(len(speed_bins)-1)] for m in methods}
ang_err_per_bin = {m: [[] for _ in range(len(angle_bins)-1)] for m in methods}

for seq in seq_list:
    gt_file = os.path.join(data_root, "GT", seq)
    with open(gt_file) as f:
        gt_data = [l.strip().split() for l in f if len(l.strip().split()) >= 7]
    gt_xyz = wgs84tocgcs2000_batch([tuple(map(float, d[1:4])) for d in gt_data], 4547)
    gt_angles = np.array([tuple(map(float, [d[5], d[6] if float(d[6]) >= 0 else float(d[6]) + 360])) for d in gt_data])
    gt_speed = np.linalg.norm(gt_xyz[1:] - gt_xyz[:-1], axis=1)
    gt_angspeed = np.linalg.norm(gt_angles[1:] - gt_angles[:-1], axis=1)

    for m_idx, method in enumerate(methods):
        method_file = os.path.join(data_root, method, seq)
        if not os.path.exists(method_file): continue
        with open(method_file) as f:
            est_data = [l.strip().split() for l in f if len(l.strip().split()) >= 7]
        est_frameidx, est_xyz, est_angles = [], [], []
        for d in est_data:
            frame_idx = int(d[0].split('_')[0]) if '_' in d[0] else None
            if frame_idx is None: continue
            est_frameidx.append(frame_idx)
            est_xyz.append(tuple(map(float, d[1:4])))
            pitch, yaw = float(d[5]), float(d[6])
            if yaw < 0: yaw += 360
            est_angles.append((pitch, yaw))
        if len(est_frameidx) < 2: continue
        est_xyz = np.array(est_xyz)
        est_angles = np.array(est_angles)
        est_frameidx = np.array(est_frameidx)
        gt_xyz_valid = gt_xyz[est_frameidx]
        gt_angles_valid = gt_angles[est_frameidx]
        v = np.linalg.norm(gt_xyz_valid[1:] - gt_xyz_valid[:-1], axis=1)
        av = np.linalg.norm(gt_angles_valid[1:] - gt_angles_valid[:-1], axis=1)
        pos_err = np.linalg.norm(est_xyz[1:] - gt_xyz_valid[1:], axis=1)
        ang_err = np.linalg.norm(est_angles[1:] - gt_angles_valid[1:], axis=1)
        pos_err = np.clip(pos_err, 0, POS_ERR_CLIP)
        ang_err = np.clip(ang_err, 0, ANG_ERR_CLIP)
        speed_idx = np.digitize(v, bins=speed_bins) - 1
        angle_idx = np.digitize(av, bins=angle_bins) - 1
        for i in range(len(pos_err)):
            if 0 <= speed_idx[i] < len(speed_bins)-1:
                pos_err_per_bin[method][speed_idx[i]].append(pos_err[i])
            if 0 <= angle_idx[i] < len(angle_bins)-1:
                ang_err_per_bin[method][angle_idx[i]].append(ang_err[i])

def nice_labels(bins, unit):
    return [f"{bins[i]}~{bins[i+1]}{unit}" for i in range(len(bins)-1)]

def grouped_violin_plot(ax, data_dict, methods, method_colors, bins, ylabel, CLIP, legend_loc):
    width = 0.12
    group_width = width * len(methods)
    positions = np.arange(len(bins)-1)
    for m_idx, method in enumerate(methods):
        for bin_idx in range(len(bins)-1):
            vals = data_dict[method][bin_idx]
            if len(vals) > 1:
                viol = ax.violinplot([vals],
                    positions=[positions[bin_idx] + (m_idx - (len(methods)-1)/2)*width],
                    widths=width,
                    showmeans=False, showmedians=True, showextrema=False)
                for pc in viol['bodies']:
                    pc.set_facecolor(method_colors[m_idx])
                    pc.set_alpha(0.7)
                    pc.set_edgecolor("none")
                if 'cmedians' in viol:
                    viol['cmedians'].set_color('#222')
                    viol['cmedians'].set_linewidth(1.3)
    # x轴
    ax.set_xticks(positions)
    ax.set_xticklabels(nice_labels(bins, " m/帧" if "Pos" in ylabel else " °/帧"), fontsize=13)
    ax.set_xlabel("Speed Bin" if "Pos" in ylabel else "Angular Speed Bin", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_ylim(0, CLIP)
    ax.grid(axis='y', linestyle='--', alpha=0.45)
    legend_handles = [
        plt.Line2D([0], [0], color=method_colors[m_idx], lw=8, label=methods[m_idx]) for m_idx in range(len(methods))
    ]
    ax.legend(handles=legend_handles, loc=legend_loc, fontsize=13)
    ax.set_title(ylabel + " Distribution across " + ("Speed Bins" if "Pos" in ylabel else "Angular Speed Bins"), fontsize=15)

# ========== 绘制位置误差-速度 ==========
fig1, ax1 = plt.subplots(figsize=(11,6))
grouped_violin_plot(ax1, pos_err_per_bin, methods, method_colors, speed_bins, "Position Error (m)", POS_ERR_CLIP, "upper right")
plt.tight_layout()
plt.savefig(f"{outputs}/nature_violin_poserr_vs_speed.png", dpi=300)
plt.show()

# ========== 绘制角度误差-角速度 ==========
fig2, ax2 = plt.subplots(figsize=(11,6))
grouped_violin_plot(ax2, ang_err_per_bin, methods, method_colors, angle_bins, "Angle Error (deg)", ANG_ERR_CLIP, "upper right")
plt.tight_layout()
plt.savefig(f"{outputs}/nature_violin_angerr_vs_angspeed.png", dpi=300)
plt.show()

print("✅ 高级分组小提琴图已保存，已自动clip长尾，观感友好。")

