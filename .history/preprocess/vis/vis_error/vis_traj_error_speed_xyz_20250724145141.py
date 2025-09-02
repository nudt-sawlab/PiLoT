import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from transform import wgs84tocgcs2000_batch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

# 主流程
seq_list = sorted([f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")])

def compute_speed(poses):
    # 计算帧间欧氏速度
    speed = np.linalg.norm(np.diff(poses, axis=0), axis=1)
    # 为了和误差对齐，补1个nan（或重复首速度）
    return np.concatenate([[np.nan], speed])

def compute_ang_speed(angles):
    # 计算帧间欧拉角速度
    ang_speed = np.linalg.norm(np.diff(angles, axis=0), axis=1)
    return np.concatenate([[np.nan], ang_speed])

def bin_stats(bin_edges, x, y):
    # 根据x值分箱，返回每箱y的数组列表
    inds = np.digitize(x, bin_edges) - 1
    binned_y = [y[inds == i] for i in range(len(bin_edges)-1)]
    return binned_y

# ========= 主流程只截取核心分析部分 ===========

# ... 读入GT和算法结果代码略 ...

# 举例分析GT和FPVLoc
methods_to_plot = ["GT", "FPVLoc"]

for seq in seq_list:
    print(f"📍 Analyzing: {seq}")
    poses_gt, angles_gt = load_pose(os.path.join(data_root, "GT", seq))
    full_timestamps = np.arange(len(poses_gt))

    speed_gt = compute_speed(poses_gt)
    ang_speed_gt = compute_ang_speed(angles_gt)

    fig1, ax1 = plt.subplots(figsize=(9,5))
    fig2, ax2 = plt.subplots(figsize=(9,5))

    for method in methods_to_plot:
        file_path = os.path.join(data_root, method, seq)
        if not os.path.exists(file_path): continue
        timestamps_est, poses_est, angles_est = load_pose_with_name(file_path)
        if len(poses_est) == 0: continue

        aligned_xyz = align_to_full_timestamps(full_timestamps, timestamps_est, poses_est)
        aligned_angles = align_to_full_timestamps(full_timestamps, timestamps_est, angles_est)

        # 误差
        pos_err = np.linalg.norm(aligned_xyz - poses_gt, axis=1)
        ang_err = np.linalg.norm(aligned_angles - angles_gt, axis=1)
        
        # 速度/角速度
        speed = speed_gt
        ang_speed = ang_speed_gt

        # 定义速度分箱（自定义区间）
        speed_bins = np.array([0, 2, 4, 6, 8, 10, 20, 100])  # 单位 m/s，可自行调整
        ang_speed_bins = np.array([0, 2, 4, 6, 8, 12, 20, 50])  # 单位 deg/s，可自行调整

        pos_err_bins = bin_stats(speed_bins, speed, pos_err)
        ang_err_bins = bin_stats(ang_speed_bins, ang_speed, ang_err)

        # 箱式图
        ax1.boxplot(pos_err_bins, positions=np.arange(len(speed_bins)-1)+1, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='C0', alpha=0.4))
        ax1.set_xticks(np.arange(len(speed_bins)-1)+1)
        ax1.set_xticklabels([f"{speed_bins[i]}-{speed_bins[i+1]}" for i in range(len(speed_bins)-1)])
        ax1.set_xlabel("Speed (m/s)")
        ax1.set_ylabel("Position Error (m)")
        ax1.set_title(f"Position Error vs. Speed ({method})")
        ax1.grid(True, alpha=0.3)

        ax2.boxplot(ang_err_bins, positions=np.arange(len(ang_speed_bins)-1)+1, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='C1', alpha=0.4))
        ax2.set_xticks(np.arange(len(ang_speed_bins)-1)+1)
        ax2.set_xticklabels([f"{ang_speed_bins[i]}-{ang_speed_bins[i+1]}" for i in range(len(ang_speed_bins)-1)])
        ax2.set_xlabel("Angular Speed (deg/s)")
        ax2.set_ylabel("Angle Error (deg)")
        ax2.set_title(f"Angle Error vs. Angular Speed ({method})")
        ax2.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig2.tight_layout()
    seq_name = seq.split('.')[0]
    fig1.savefig(f"{outputs}/{seq_name}_poserr_vs_speed.png", dpi=300)
    fig2.savefig(f"{outputs}/{seq_name}_angerr_vs_angspeed.png", dpi=300)
    plt.close(fig1)
    plt.close(fig2)
    print(f"✅ 已保存 {seq} 误差-速度箱式图")

print("🎉 全部箱式图统计完毕")