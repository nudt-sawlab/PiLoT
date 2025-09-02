import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib import rcParams
from scipy.stats import gaussian_kde
from scipy.ndimage import uniform_filter1d
from transform import wgs84tocgcs2000_batch

# LaTeX 风格字体设置
rcParams['font.family'] = 'serif'
rcParams['axes.labelweight'] = 'bold'

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

def plot_gradient_line(ax, x, y, cmap='viridis', linewidth=2.0):
    """绘制 GT 曲线渐变色"""
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap(cmap), norm=plt.Normalize(x.min(), x.max()))
    lc.set_array(x)
    lc.set_linewidth(linewidth)
    ax.add_collection(lc)

def plot_error_kde(ax, timestamps, errors, color, label):
    valid = ~np.isnan(errors)
    if np.sum(valid) < 10: return
    kde = gaussian_kde(timestamps[valid], weights=errors[valid])
    x_vals = np.linspace(timestamps[valid].min(), timestamps[valid].max(), 200)
    y_vals = kde(x_vals)
    ax.plot(x_vals, y_vals, color=color, linestyle='--', linewidth=1.2, label=label)

seq_list = sorted([f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")])

for seq in seq_list:
    print(f"📍 处理序列：{seq}")
    seq_name = seq.split('.')[0]

    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    ax_xy, ax_alt = axs[0, 0], axs[0, 1]
    ax_pitch, ax_yaw = axs[1, 0], axs[1, 1]
    ax_poserr, ax_angerr = axs[2, 0], axs[2, 1]

    poses_gt, angles_gt = load_pose(os.path.join(data_root, "GT", seq))
    full_timestamps = np.arange(len(poses_gt))

    gt_speed = np.linalg.norm(np.diff(poses_gt, axis=0), axis=1)
    gt_angular_speed = np.linalg.norm(np.diff(angles_gt, axis=0), axis=1)
    speed_timestamps = full_timestamps[1:]
    gt_speed_smooth = uniform_filter1d(gt_speed, size=5)
    gt_ang_speed_smooth = uniform_filter1d(gt_angular_speed, size=5)

    plot_gradient_line(ax_poserr, speed_timestamps, gt_speed_smooth, cmap='viridis')
    plot_gradient_line(ax_angerr, speed_timestamps, gt_ang_speed_smooth, cmap='plasma')

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

        if 'GT' in method: gt_pts_cgcs = rel
        if 'ORB' in method:
            orb_pts_cgcs = rel
            scale, R, t = umeyama_alignment(orb_pts_cgcs[timestamps_est[0:300]], gt_pts_cgcs[timestamps_est[0:300]])
            orb_pts_cgcs = transform_points(orb_pts_cgcs, scale, R, t)
            rel = orb_pts_cgcs

        if 'raft' in method.lower():
            ax_xy.scatter(rel[:, 0], rel[:, 1], color=color, s=5, label=method, alpha=0.7)
        else:
            ax_xy.plot(rel[:, 0], rel[:, 1], color=color, linewidth=2, label=method)

        ax_alt.plot(full_timestamps, aligned_xyz[:, 2], color=color, linewidth=2, label=method)
        ax_pitch.plot(full_timestamps, aligned_angles[:, 0], color=color, linewidth=1.5, label=method)
        ax_yaw.plot(full_timestamps, aligned_angles[:, 1], color=color, linewidth=1.5, label=method)

        if 'GT' in method: continue
        pos_err = np.linalg.norm(aligned_xyz - poses_gt, axis=1)
        ang_err = np.linalg.norm(aligned_angles - angles_gt, axis=1)
        plot_error_kde(ax_poserr, full_timestamps, pos_err, color, method)
        plot_error_kde(ax_angerr, full_timestamps, ang_err, color, method)

    ax_xy.set_title(r"\textbf{Trajectory in XY Plane}")
    ax_xy.set_xlabel(r"\textit{Relative X (m)}")
    ax_xy.set_ylabel(r"\textit{Relative Y (m)}")
    ax_xy.grid(True)
    ax_xy.legend()

    ax_alt.set_title(r"\textbf{Altitude over Time}")
    ax_alt.set_xlabel(r"\textit{Frame Index}")
    ax_alt.set_ylabel(r"\textit{Altitude (m)}")
    ax_alt.grid(True)
    ax_alt.legend()

    ax_pitch.set_title(r"\textbf{Pitch Angle over Time}")
    ax_pitch.set_xlabel(r"\textit{Frame Index}")
    ax_pitch.set_ylabel(r"\textit{Pitch (°)}")
    ax_pitch.grid(True)
    ax_pitch.legend()

    ax_yaw.set_title(r"\textbf{Yaw Angle over Time}")
    ax_yaw.set_xlabel(r"\textit{Frame Index}")
    ax_yaw.set_ylabel(r"\textit{Yaw (°)}")
    ax_yaw.grid(True)
    ax_yaw.legend()

    ax_poserr.set_title(r"\textbf{Flight Speed \& Position Error}")
    ax_poserr.set_xlabel(r"\textit{Frame Index}")
    ax_poserr.set_ylabel(r"\textit{Speed / Error (m/s, m)}")
    ax_poserr.grid(True)
    ax_poserr.set_ylim(0, 15)
    ax_poserr.legend()

    ax_angerr.set_title(r"\textbf{Angular Speed \& Orientation Error}")
    ax_angerr.set_xlabel(r"\textit{Frame Index}")
    ax_angerr.set_ylabel(r"\textit{Angular Speed / Error (°/s, °)}")
    ax_angerr.grid(True)
    ax_angerr.set_ylim(0, 30)
    ax_angerr.legend()

    fig.tight_layout()
    fig.savefig(f"{outputs}/{seq_name}_6plots.png", dpi=300)
    plt.close(fig)

print("✅ 所有六宫格图像已美化并保存至 outputs/")

