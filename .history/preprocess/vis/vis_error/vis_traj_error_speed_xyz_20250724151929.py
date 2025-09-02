import os
import numpy as np
import matplotlib.pyplot as plt
from transform import wgs84tocgcs2000_batch

# Nature-like配色方案
methods = ["GT", "FPVLoc", "Pixloc", "Render2loc", "ORB@per30", "Render2loc@raft"]
method_colors = {
    "GT": "#222222",           # 黑
    "FPVLoc": "#E64B35",       # 红
    "Pixloc": "#4DBBD5",       # 湖蓝
    "Render2loc": "#00A087",   # 青绿
    "ORB@per30": "#3C5488",    # 深蓝
    "Render2loc@raft": "#F39B7F"  # 橙
}

data_root = "/mnt/sda/MapScape/query/estimation/result_images"
seq_list = sorted([f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")])

# 速度/角速度分箱（单位：米/帧，度/帧）
speed_bins = [0, 0.5, 2, 10]
angle_bins = [0, 0.2, 1, 10]

# 箱式图结果：dict[(方法,分箱号)] = [误差,误差,误差...]
pos_err_per_bin = {m: [[] for _ in range(len(speed_bins)-1)] for m in methods}
ang_err_per_bin = {m: [[] for _ in range(len(angle_bins)-1)] for m in methods}
data_root = "/mnt/sda/MapScape/query/estimation/result_images"
outputs = os.path.join(data_root, "outputs")
os.makedirs(outputs, exist_ok=True)
for seq in seq_list:
    gt_file = os.path.join(data_root, "GT", seq)
    with open(gt_file) as f:
        gt_data = [l.strip().split() for l in f if len(l.strip().split()) >= 7]
    gt_xyz = wgs84tocgcs2000_batch([tuple(map(float, d[1:4])) for d in gt_data], 4547)
    gt_angles = np.array([tuple(map(float, [d[5], d[6] if float(d[6]) >= 0 else float(d[6]) + 360])) for d in gt_data])
    # 帧间变化
    gt_speed = np.linalg.norm(gt_xyz[1:] - gt_xyz[:-1], axis=1)
    gt_angspeed = np.linalg.norm(gt_angles[1:] - gt_angles[:-1], axis=1)

    for method in methods:
        method_file = os.path.join(data_root, method, seq)
        if not os.path.exists(method_file):
            continue
        # 读取算法估计
        with open(method_file) as f:
            est_data = [l.strip().split() for l in f if len(l.strip().split()) >= 7]
        est_frameidx = []
        est_xyz = []
        est_angles = []
        for d in est_data:
            frame_idx = int(d[0].split('_')[0]) if '_' in d[0] else None
            if frame_idx is None: continue
            est_frameidx.append(frame_idx)
            est_xyz.append(tuple(map(float, d[1:4])))
            pitch, yaw = float(d[5]), float(d[6])
            if yaw < 0: yaw += 360
            est_angles.append((pitch, yaw))
        if len(est_frameidx) < 2: continue

        # 对齐与误差
        est_xyz = np.array(est_xyz)
        est_angles = np.array(est_angles)
        est_frameidx = np.array(est_frameidx)
        # 只对有预测的帧进行统计
        gt_xyz_valid = gt_xyz[est_frameidx]
        gt_angles_valid = gt_angles[est_frameidx]
        # 帧间速度
        v = np.linalg.norm(gt_xyz_valid[1:] - gt_xyz_valid[:-1], axis=1)  # 用GT的真实速度
        av = np.linalg.norm(gt_angles_valid[1:] - gt_angles_valid[:-1], axis=1)
        # 帧间误差
        pos_err = np.linalg.norm(est_xyz[1:] - gt_xyz_valid[1:], axis=1)
        ang_err = np.linalg.norm(est_angles[1:] - gt_angles_valid[1:], axis=1)

        # 统计分箱
        speed_idx = np.digitize(v, bins=speed_bins) - 1
        angle_idx = np.digitize(av, bins=angle_bins) - 1
        for i in range(len(pos_err)):
            if 0 <= speed_idx[i] < len(speed_bins)-1:
                pos_err_per_bin[method][speed_idx[i]].append(pos_err[i])
            if 0 <= angle_idx[i] < len(angle_bins)-1:
                ang_err_per_bin[method][angle_idx[i]].append(ang_err[i])

# ============ 绘制箱式图 ==============
def nice_labels(bins, unit):
    return [f"{bins[i]}~{bins[i+1]}{unit}" for i in range(len(bins)-1)]

plt.figure(figsize=(11, 6))
box_width = 0.11  # 每个box的宽度
xticks = np.arange(len(speed_bins)-1)
for m_idx, method in enumerate(methods):
    positions = xticks + (m_idx - (len(methods)-1)/2) * box_width  # 算法错位分布
    data = [pos_err_per_bin[method][i] for i in range(len(speed_bins)-1)]
    plt.boxplot(data, positions=positions, patch_artist=True, showfliers=False,  # 不显示异常值
                boxprops=dict(facecolor=method_colors[method], alpha=0.68),
                medianprops=dict(color='k', linewidth=1.4),
                whiskerprops=dict(color=method_colors[method]),
                capprops=dict(color=method_colors[method]),
                widths=box_width, vert=True)
plt.xticks(xticks, nice_labels(speed_bins, " m/帧"))
plt.xlabel("Speed bin (per frame)", fontsize=14)
plt.ylabel("Position Error (m)", fontsize=14)
plt.ylim(0, 10)  # 上限裁剪
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(methods, loc='upper right')
plt.title("Position Error Distribution across Speed Bins")
plt.tight_layout()
plt.savefig(f"{outputs}/boxplot_poserr_vs_speed.png", dpi=300)
plt.show()

plt.figure(figsize=(11, 6))
for m_idx, method in enumerate(methods):
    positions = xticks + (m_idx - (len(methods)-1)/2) * box_width
    data = [ang_err_per_bin[method][i] for i in range(len(angle_bins)-1)]
    plt.boxplot(data, positions=positions, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=method_colors[method], alpha=0.68),
                medianprops=dict(color='k', linewidth=1.4),
                whiskerprops=dict(color=method_colors[method]),
                capprops=dict(color=method_colors[method]),
                widths=box_width, vert=True)
plt.xticks(xticks, nice_labels(angle_bins, " °/帧"))
plt.xlabel("Angular speed bin (per frame)", fontsize=14)
plt.ylabel("Angle Error (deg)", fontsize=14)
plt.ylim(0, 10)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(methods, loc='upper right')
plt.title("Angle Error Distribution across Angular Speed Bins")
plt.tight_layout()
plt.savefig(f"{outputs}/boxplot_angerr_vs_angspeed.png", dpi=300)
plt.show()

print("✅ 箱式图已保存，配色已美化，长尾抑制已完成")

