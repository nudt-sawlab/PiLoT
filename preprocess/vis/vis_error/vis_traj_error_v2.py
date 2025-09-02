import os
import matplotlib.pyplot as plt
import numpy as np
from transform import wgs84tocgcs2000_batch
from scipy.ndimage import gaussian_filter1d

# 方法设定及颜色
methods = {
    "GT": "black",
    "FPVLoc": "red",
    "Pixloc": "blue",
    "Render2loc": "green",
    "ORB@per30": "purple",
    "Render2loc@raft": "orange"
}

data_root = "/mnt/sda/MapScape/query/estimation/result_images"
seq_list = sorted([
    f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")
])

def load_pose_with_name(file_path):
    data = []
    angles = []
    timestamps = []
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
                if yaw < 0:
                    yaw += 360
                data.append((lon, lat, alt))
                angles.append((pitch, yaw))
                timestamps.append(frame_idx)
    xyz = wgs84tocgcs2000_batch(data, 4547)
    return np.array(timestamps), np.array(xyz), np.array(angles)

def load_pose(file_path):
    data = []
    angles = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                lon, lat, alt = map(float, parts[1:4])
                roll, pitch, yaw = map(float, parts[4:7])
                if yaw < 0:
                    yaw += 360
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
    """从 src 到 dst 的相似变换：scale * R @ src + t = dst"""
    assert src.shape == dst.shape
    mu_src = src.mean(0)
    mu_dst = dst.mean(0)

    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    cov = dst_centered.T @ src_centered / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)

    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    var_src = (src_centered ** 2).sum() / src.shape[0]
    scale = np.trace(np.diag(D)) / var_src
    t = mu_dst - scale * R @ mu_src

    return scale, R, t
for seq in seq_list:
    print(f"📍 处理序列：{seq}")
    seq_name = seq.split('.')[0]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax_xy, ax_alt, ax_pitch, ax_yaw = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    poses_gt, angles_gt = load_pose(os.path.join(data_root, "GT", seq))
    full_timestamps = np.arange(len(poses_gt))

    for method, color in methods.items():
        file_path = os.path.join(data_root, method, seq)
        if not os.path.exists(file_path):
            print(f"[⚠️] {method} 缺少文件，跳过：{seq}")
            continue

        timestamps_est, poses_est, angles_est = load_pose_with_name(file_path)
        if len(poses_est) == 0:
            print(f"[❌] {method} 无有效位姿，跳过")
            continue

        coverage = len(timestamps_est) / len(full_timestamps)
        label = f"{method} ({coverage*100:.1f}%)"

        aligned_xyz = align_to_full_timestamps(full_timestamps, timestamps_est, poses_est)
        aligned_angles = align_to_full_timestamps(full_timestamps, timestamps_est, angles_est)
        
        # 相对轨迹（NaN保留断线）
        valid_mask = ~np.isnan(aligned_xyz[:, 0])
        rel = aligned_xyz.copy()
        rel[valid_mask] -= aligned_xyz[valid_mask][0]
        # rel[:, 0] = gaussian_filter1d(np.nan_to_num(rel[:, 0]), sigma=1.5)
        # rel[:, 1] = gaussian_filter1d(np.nan_to_num(rel[:, 1]), sigma=1.5)
        #================
        if 'GT' in method:
            gt_pts_cgcs = rel
        if 'ORB' in method:
            orb_pts_cgcs = rel
            scale, R, t = umeyama_alignment(orb_pts_cgcs[timestamps_est[0:300]], gt_pts_cgcs[timestamps_est[0:300]])
            orb_pts_cgcs = transform_points(orb_pts_cgcs, scale, R, t)
            rel = orb_pts_cgcs
        #==========
        ax_xy.plot(rel[:, 0], rel[:, 1], color=color, linewidth=2, label=label)
        ax_alt.plot(full_timestamps, aligned_xyz[:, 2], color=color, linewidth=2, label=label)

        ax_pitch.plot(full_timestamps, aligned_angles[:, 0], color=color, linewidth=1.5, label=label)
        # ax_pitch.scatter(timestamps_est, [ax_pitch.get_ylim()[0]] * len(timestamps_est),
        #                  marker='|', color=color, alpha=0.6)

        ax_yaw.plot(full_timestamps, aligned_angles[:, 1], color=color, linewidth=1.5, label=label)
        # ax_yaw.scatter(timestamps_est, [ax_yaw.get_ylim()[0]] * len(timestamps_est),
        #                marker='|', color=color, alpha=0.6)

    # 图表美化
    ax_xy.set_title("Trajectory in XY Plane")
    ax_xy.set_xlabel("Relative X (m)")
    ax_xy.set_ylabel("Relative Y (m)")
    ax_xy.grid(True, linestyle='--', linewidth=0.6)
    ax_xy.legend()

    ax_alt.set_title("Altitude over Time")
    ax_alt.set_xlabel("Frame Index")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.grid(True, linestyle='--', linewidth=0.6)
    ax_alt.legend()

    ax_pitch.set_title("Pitch Angle over Time")
    ax_pitch.set_xlabel("Frame Index")
    ax_pitch.set_ylabel("Pitch (°)")
    ax_pitch.grid(True, linestyle='--', linewidth=0.6)
    ax_pitch.legend()

    ax_yaw.set_title("Yaw Angle over Time")
    ax_yaw.set_xlabel("Frame Index")
    ax_yaw.set_ylabel("Yaw (°)")
    ax_yaw.grid(True, linestyle='--', linewidth=0.6)
    ax_yaw.legend()

    outputs = os.path.join(data_root, "outputs")
    os.makedirs(outputs, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{outputs}/{seq_name}_4plots.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

print("✅ 所有四宫格图像已保存至 outputs/")

