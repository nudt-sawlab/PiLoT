import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from transform import wgs84tocgcs2000_batch

rcParams['font.family'] = 'serif'
rcParams['axes.labelweight'] = 'bold'

methods = {
    "GT": "black",
    "FPVLoc": "red",
    "Pixloc": "blue",
    "Render2loc": "green",
    "ORB@per30": "purple",
    "Render2loc@raft": "orange"
}
method_names = list(methods.keys())
method_colors = [methods[k] for k in method_names]

data_root = "/mnt/sda/MapScape/query/estimation/result_images"
outputs = os.path.join(data_root, "outputs")
os.makedirs(outputs, exist_ok=True)

def load_pose_with_name(file_path):
    data, angles, timestamps = [], [], []
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
                if yaw < 0: yaw += 360
                data.append((lon, lat, alt))
                angles.append((pitch, yaw))
                timestamps.append(frame_idx)
    xyz = wgs84tocgcs2000_batch(data, 4547)
    return np.array(timestamps), np.array(xyz), np.array(angles)

def load_pose(file_path):
    data, angles = [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                lon, lat, alt = map(float, parts[1:4])
                roll, pitch, yaw = map(float, parts[4:7])
                if yaw < 0: yaw += 360
                data.append((lon, lat, alt))
                angles.append((pitch, yaw))
    xyz = wgs84tocgcs2000_batch(data, 4547)
    return np.array(xyz), np.array(angles)

def align_to_full_timestamps(full_timestamps, timestamps_est, values):
    aligned = np.full((len(full_timestamps), values.shape[1]), np.nan, dtype=np.float32)
    idx_map = {t: i for i, t in enumerate(full_timestamps)}
    for i, t in enumerate(timestamps_est):
        if t in idx_map:
            aligned[idx_map[t]] = values[i]
    return aligned

def compute_speed(poses):
    speed = np.linalg.norm(np.diff(poses, axis=0), axis=1) / 0.04
    return np.concatenate([[np.nan], speed])

def compute_ang_speed(angles):
    ang_speed = np.linalg.norm(np.diff(angles, axis=0), axis=1)/ 0.04
    return np.concatenate([[np.nan], ang_speed])

def bin_stats(bin_edges, x, y):
    inds = np.digitize(x, bin_edges) - 1
    binned_y = [y[inds == i] for i in range(len(bin_edges)-1)]
    return binned_y

# ===== 主流程 =====
seq_list = sorted([f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")])

# 建议聚合多序列/算法做
for seq in seq_list:
    print(f"📍 {seq}")
    seq_name = seq.split('.')[0]
    poses_gt, angles_gt = load_pose(os.path.join(data_root, "GT", seq))
    full_timestamps = np.arange(len(poses_gt))
    speed_gt = compute_speed(poses_gt)
    ang_speed_gt = compute_ang_speed(angles_gt)

    # 速度分箱
    speed_bins = np.array([0, 2, 4, 6, 8, 12, 20, 100])
    ang_speed_bins = np.array([0, 2, 4, 6, 8, 12, 20, 50])

    # 每个方法的分箱误差数据
    poserr_binned = [[] for _ in range(len(speed_bins)-1)]
    angerr_binned = [[] for _ in range(len(ang_speed_bins)-1)]
    labels = method_names

    # 收集每个算法的误差分箱
    for method in method_names:
        file_path = os.path.join(data_root, method, seq)
        if not os.path.exists(file_path):
            continue
        timestamps_est, poses_est, angles_est = load_pose_with_name(file_path)
        if len(poses_est) == 0:
            continue
        aligned_xyz = align_to_full_timestamps(full_timestamps, timestamps_est, poses_est)
        aligned_angles = align_to_full_timestamps(full_timestamps, timestamps_est, angles_est)

        pos_err = np.linalg.norm(aligned_xyz - poses_gt, axis=1)
        ang_err = np.linalg.norm(aligned_angles - angles_gt, axis=1)

        # 分箱
        for i in range(len(speed_bins)-1):
            mask = (speed_gt >= speed_bins[i]) & (speed_gt < speed_bins[i+1])
            poserr_binned[i].append(pos_err[mask & ~np.isnan(pos_err)])
        for i in range(len(ang_speed_bins)-1):
            mask = (ang_speed_gt >= ang_speed_bins[i]) & (ang_speed_gt < ang_speed_bins[i+1])
            angerr_binned[i].append(ang_err[mask & ~np.isnan(ang_err)])

    # 绘制位置误差-速度分箱箱式图
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    box_width = 0.10
    positions = np.arange(len(speed_bins)-1)
    for i, method in enumerate(labels):
        data = [poserr_binned[j][i] if i < len(poserr_binned[j]) else np.array([]) for j in range(len(speed_bins)-1)]
        pos = positions + (i - (len(labels)-1)/2)*box_width  # 横向偏移
        bp = ax1.boxplot(data, positions=pos, widths=box_width*0.85, patch_artist=True,
                    boxprops=dict(facecolor=method_colors[i], alpha=0.5), 
                    medianprops=dict(color='k', linewidth=2), showfliers=False)
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f"{speed_bins[i]}-{speed_bins[i+1]}" for i in range(len(speed_bins)-1)])
    ax1.set_xlabel("Speed (m/s)")
    ax1.set_ylabel("Position Error (m)")
    ax1.set_title(f"Position Error vs. Speed (All Methods)\n{seq_name}")
    ax1.grid(True, alpha=0.3)
    ax1.legend([plt.Line2D([0],[0], color=c, lw=6) for c in method_colors], labels)
    fig1.tight_layout()
    fig1.savefig(f"{outputs}/{seq_name}_poserr_vs_speed_grouped.png", dpi=300)
    plt.close(fig1)

    # 绘制角度误差-角速度分箱箱式图
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(ang_speed_bins)-1)
    for i, method in enumerate(labels):
        data = [angerr_binned[j][i] if i < len(angerr_binned[j]) else np.array([]) for j in range(len(ang_speed_bins)-1)]
        pos = positions + (i - (len(labels)-1)/2)*box_width
        bp = ax2.boxplot(data, positions=pos, widths=box_width*0.85, patch_artist=True,
                    boxprops=dict(facecolor=method_colors[i], alpha=0.5), 
                    medianprops=dict(color='k', linewidth=2), showfliers=False)
    ax2.set_xticks(positions)
    ax2.set_xticklabels([f"{ang_speed_bins[i]}-{ang_speed_bins[i+1]}" for i in range(len(ang_speed_bins)-1)])
    ax2.set_xlabel("Angular Speed (deg/s)")
    ax2.set_ylabel("Angle Error (deg)")
    ax2.set_title(f"Angle Error vs. Angular Speed (All Methods)\n{seq_name}")
    ax2.grid(True, alpha=0.3)
    ax2.legend([plt.Line2D([0],[0], color=c, lw=6) for c in method_colors], labels)
    fig2.tight_layout()
    fig2.savefig(f"{outputs}/{seq_name}_angerr_vs_angspeed_grouped.png", dpi=300)
    plt.close(fig2)
    print(f"✅ {seq_name} 箱式图已保存")

print("🎉 所有算法分组箱式图已完成")

