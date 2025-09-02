import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.ndimage import uniform_filter1d
from transform import wgs84tocgcs2000_batch
from scipy.stats import gaussian_kde

# 设置LaTeX风格字体
rcParams['font.family'] = 'serif'
rcParams['axes.labelweight'] = 'bold'

# 方法与颜色
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

seq_list = sorted([f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")])

for seq in seq_list:
    print(f"📍 Processing: {seq}")
    seq_name = seq.split('.')[0]
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    ax_xy, ax_alt = axs[0, 0], axs[0, 1]
    ax_pitch, ax_yaw = axs[1, 0], axs[1, 1]
    ax_poserr, ax_angerr = axs[2, 0], axs[2, 1]

    poses_gt, angles_gt = load_pose(os.path.join(data_root, "GT", seq))
    full_timestamps = np.arange(len(poses_gt))

    # 平滑处理
    gt_speed = np.linalg.norm(np.diff(poses_gt, axis=0), axis=1)
    gt_ang_speed = np.linalg.norm(np.diff(angles_gt, axis=0), axis=1)
    speed_timestamps = full_timestamps[1:]
    gt_speed_smooth = uniform_filter1d(gt_speed, size=5)
    gt_ang_speed_smooth = uniform_filter1d(gt_ang_speed, size=5)

    ax_poserr.plot(speed_timestamps, gt_speed_smooth, color='gray', linewidth=2.0, label='GT Speed')
    ax_angerr.plot(speed_timestamps, gt_ang_speed_smooth, color='gray', linewidth=2.0, label='GT Angular Speed')

    for method, color in methods.items():
        file_path = os.path.join(data_root, method, seq)
        if not os.path.exists(file_path): continue
        timestamps_est, poses_est, angles_est = load_pose_with_name(file_path)
        if len(poses_est) == 0: continue

        aligned_xyz = align_to_full_timestamps(full_timestamps, timestamps_est, poses_est)
        aligned_angles = align_to_full_timestamps(full_timestamps, timestamps_est, angles_est)
        valid_mask = ~np.isnan(aligned_xyz[:, 0])
        rel = aligned_xyz.copy()
        rel[valid_mask] -= aligned_xyz[valid_mask][0]

        if 'GT' in method:
            gt_pts_cgcs = rel 
            ax_xy.plot(rel[:, 0], rel[:, 1], color=color, linewidth=2, label=method, linestyle='-')
            ax_alt.plot(full_timestamps, aligned_xyz[:, 2], color=color, linewidth=2, label=method)
            ax_pitch.plot(full_timestamps, aligned_angles[:, 0], color=color, linewidth=1.5, label=method)
            ax_yaw.plot(full_timestamps, aligned_angles[:, 1], color=color, linewidth=1.5, label=method)
            continue

        # 对齐并对 ORB 做位姿变换
        if 'ORB' in method:
            scale, R, t = umeyama_alignment(rel[timestamps_est[0:300]], gt_pts_cgcs[timestamps_est[0:300]])
            rel = transform_points(rel, scale, R, t)
            
        # 计算位置和角度误差
        pos_err = np.linalg.norm(aligned_xyz - poses_gt, axis=1)
        ang_err = np.linalg.norm(aligned_angles - angles_gt, axis=1)

        # 前四图中误差大的点改成scatter
        large_error = pos_err > 10
        
        ax_xy.plot(rel[~large_error, 0], rel[~large_error, 1], color=color, linewidth=1.8, label=method)
        ax_xy.scatter(rel[large_error, 0], rel[large_error, 1], color=color, s=10, alpha=0.6)

        ax_alt.plot(full_timestamps[~large_error], aligned_xyz[~large_error, 2], color=color, linewidth=1.8, label=method)
        ax_alt.scatter(full_timestamps[large_error], aligned_xyz[large_error, 2], color=color, s=10, alpha=0.6)

        ax_pitch.plot(full_timestamps[~large_error], aligned_angles[~large_error, 0], color=color, linewidth=1.5, label=method)
        ax_pitch.scatter(full_timestamps[large_error], aligned_angles[large_error, 0], color=color, s=10, alpha=0.6)

        ax_yaw.plot(full_timestamps[~large_error], aligned_angles[~large_error, 1], color=color, linewidth=1.5, label=method)
        ax_yaw.scatter(full_timestamps[large_error], aligned_angles[large_error, 1], color=color, s=10, alpha=0.6)

        # 后两图：误差scatter恢复
        ax_poserr.scatter(full_timestamps, pos_err, color=color, s=8, alpha=0.5, label=method)
        ax_angerr.scatter(full_timestamps, ang_err, color=color, s=8, alpha=0.5, label=method)

    # 图表美化
    ax_xy.set_title("Trajectory in XY Plane")
    ax_xy.set_xlabel("Relative X (m)")
    ax_xy.set_ylabel("Relative Y (m)")
    ax_xy.grid(True)
    ax_xy.legend()

    ax_alt.set_title("Altitude over Time")
    ax_alt.set_xlabel("Frame Index")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.grid(True)
    ax_alt.set_ylim(680, 780)
    ax_alt.legend()

    ax_pitch.set_title("Pitch Angle over Time")
    ax_pitch.set_xlabel("Frame Index")
    ax_pitch.set_ylabel("Pitch (°)")
    ax_pitch.grid(True)
    ax_pitch.set_ylim(25, 55)
    ax_pitch.legend()

    ax_yaw.set_title("Yaw Angle over Time")
    ax_yaw.set_xlabel("Frame Index")
    ax_yaw.set_ylabel("Yaw (°)")
    ax_yaw.grid(True)
    ax_yaw.legend()

    ax_poserr.set_title("Flight Speed & Position Error")
    ax_poserr.set_xlabel("Frame Index")
    ax_poserr.set_ylabel("Speed / Error (m/s, m)")
    ax_poserr.grid(True)
    ax_poserr.set_ylim(0, 5)
    ax_poserr.legend()

    ax_angerr.set_title("Angular Speed & Orientation Error")
    ax_angerr.set_xlabel("Frame Index")
    ax_angerr.set_ylabel("Angular Speed / Error (°/s, °)")
    ax_angerr.grid(True)
    ax_angerr.set_ylim(0, 5)
    ax_angerr.legend()

    fig.tight_layout()
    fig.savefig(f"{outputs}/{seq_name}_6plots.png", dpi=300)
    plt.close(fig)

print("✅ 已按要求完成三项改进并保存所有图像至 outputs/")

