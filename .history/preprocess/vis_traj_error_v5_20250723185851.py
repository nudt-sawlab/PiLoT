import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.ndimage import uniform_filter1d
from transform import wgs84tocgcs2000_batch

# 设置字体
rcParams['font.family'] = 'serif'
rcParams['axes.labelweight'] = 'bold'

# 方法颜色定义
methods = {
    "GT": "black",
    "FPVLoc": "red",
    "Pixloc": "blue",
    "Render2loc": "green",
    "ORB@per30": "purple",
    "Render2loc@raft": "orange"
}

# 轨迹平移偏移量（方案一）
offset_dict = {
    "GT": (0.0, 0.0),
    "FPVLoc": (3.0, 2.0),
    "Pixloc": (-3.0, -2.0),
    "Render2loc": (2.0, -2.0),
    "ORB@per30": (-2.0, 2.0),
    "Render2loc@raft": (2.0, 3.0)
}

data_root = "/mnt/sda/MapScape/query/estimation/result_images"
outputs = os.path.join(data_root, "outputs")
os.makedirs(outputs, exist_ok=True)

# 加载估计结果带时间戳
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
                if yaw < 0:
                    yaw += 360
                data.append((lon, lat, alt))
                angles.append((pitch, yaw))
                timestamps.append(frame_idx)
    xyz = wgs84tocgcs2000_batch(data, 4547)
    return np.array(timestamps), np.array(xyz), np.array(angles)

# 加载 GT结果
def load_pose(file_path):
    data, angles = [], []
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

# 对齐到完整时间序列
def align_to_full_timestamps(full_tstamps, timestamps_est, values):
    aligned = np.full((len(full_tstamps), values.shape[1]), np.nan, dtype=np.float32)
    idx_map = {t: i for i, t in enumerate(full_tstamps)}
    for i, t in enumerate(timestamps_est):
        if t in idx_map:
            aligned[idx_map[t]] = values[i]
    return aligned

from numpy.linalg import svd
# Umeyama对齐

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

# 主流程
seq_list = sorted([f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")])
for seq in seq_list:
    print(f"📍 Processing: {seq}")
    seq_name = seq.split('.')[0]
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    ax_xy, ax_alt = axs[0,0], axs[0,1]
    ax_pitch, ax_yaw = axs[1,0], axs[1,1]
    ax_poserr, ax_angerr = axs[2,0], axs[2,1]

    # 加载GT
    poses_gt, angles_gt = load_pose(os.path.join(data_root, "GT", seq))
    full_tstamps = np.arange(len(poses_gt))

    # GT速度曲线
    gt_spd = np.linalg.norm(np.diff(poses_gt, axis=0), axis=1)
    gt_ang_spd = np.linalg.norm(np.diff(angles_gt, axis=0), axis=1)
    t_spd = full_tstamps[1:]
    ax_poserr.plot(t_spd, uniform_filter1d(gt_spd, 5), color='gray', linewidth=2, label='GT Speed')
    ax_angerr.plot(t_spd, uniform_filter1d(gt_ang_spd, 5), color='gray', linewidth=2, label='GT Angular')

    for method, color in methods.items():
        fp = os.path.join(data_root, method, seq)
        if not os.path.exists(fp):
            continue
        ts, xyz, ang = load_pose_with_name(fp)
        if len(xyz) == 0:
            continue

        # 对齐数据
        aligned_xyz = align_to_full_timestamps(full_tstamps, ts, xyz)
        aligned_ang = align_to_full_timestamps(full_tstamps, ts, ang)
        mask = ~np.isnan(aligned_xyz[:,0])
        rel = aligned_xyz.copy()
        rel[mask] -= aligned_xyz[mask][0]

        # 方案一：偏移
        dx, dy = offset_dict.get(method, (0.0, 0.0))
        rel[mask,0] += dx
        rel[mask,1] += dy

        # ORB对齐
        if 'ORB' in method:
            scale, Rm, tm = umeyama_alignment(rel[ts[:300]], rel[ts[:300]])
            rel = scale * (Rm @ rel.T).T + tm

        # 误差计算
        pos_err = np.linalg.norm(aligned_xyz - poses_gt, axis=1)
        pos_err[np.isnan(pos_err)] = np.nanmax(pos_err)
        ang_err = np.linalg.norm(aligned_ang - angles_gt, axis=1)
        ang_err[np.isnan(ang_err)] = np.nanmax(ang_err)
        large_p = pos_err > 20
        large_a = ang_err > 5

                # 方案三改：XY轨迹误差色带（透明色带，带宽随误差增大）
        from matplotlib.collections import LineCollection
        # 准备有效点序列
        xs = rel[mask, 0]
        ys = rel[mask, 1]
        errs = pos_err[mask]
        # 构建线段
        segments = [np.column_stack([xs[i:i+2], ys[i:i+2]]) for i in range(len(xs)-1)]
        # 根据误差设定线宽
        max_w = 8.0
        min_w = 0.5
        widths = min_w + (errs[:-1] / errs.max()) * (max_w - min_w)
        lc = LineCollection(segments, colors=color, linewidths=widths, alpha=0.6)
        ax_xy.add_collection(lc)

        # 高度、Pitch、Yaw 处理同前
        # Altitude
        ax_alt.plot(full_tstamps[~large_p], aligned_xyz[~large_p,2], color=color, linewidth=1.8)
        miss = np.isnan(aligned_xyz[:,2])
        i = 0
        while i < len(full_tstamps):
            if miss[i] or large_p[i]:
                start = i - 1
                while i < len(full_tstamps) and (miss[i] or large_p[i]):
                    i += 1
                end = i if i < len(full_tstamps) else len(full_tstamps)-1
                if 0 <= start < len(full_tstamps) and 0 <= end < len(full_tstamps):
                    ax_alt.plot([full_tstamps[start], full_tstamps[end]],
                                [aligned_xyz[start,2], aligned_xyz[end,2]],
                                color=color, linestyle=(0,(10,6)), linewidth=2, alpha=0.8)
            else:
                i += 1
        ax_alt.scatter(full_tstamps[large_p], aligned_xyz[large_p,2], color=color, s=10)

        # Pitch
        ax_pitch.plot(full_tstamps[~large_a], aligned_ang[~large_a,0], color=color, linewidth=1.5)
        miss_p = np.isnan(aligned_ang[:,0])
        i = 0
        while i < len(full_tstamps):
            if miss_p[i] or large_a[i]:
                start = i - 1
                while i < len(full_tstamps) and (miss_p[i] or large_a[i]):
                    i += 1
                end = i if i < len(full_tstamps) else len(full_tstamps)-1
                if 0 <= start < len(full_tstamps) and 0 <= end < len(full_tstamps):
                    ax_pitch.plot([full_tstamps[start], full_tstamps[end]],
                                  [aligned_ang[start,0], aligned_ang[end,0]],
                                  color=color, linestyle=(0,(10,6)), linewidth=2, alpha=0.8)
            else:
                i += 1
        ax_pitch.scatter(full_tstamps[large_a], aligned_ang[large_a,0], color=color, s=10)

        # Yaw
        ax_yaw.plot(full_tstamps[~large_a], aligned_ang[~large_a,1], color=color, linewidth=1.5)
        miss_y = np.isnan(aligned_ang[:,1])
        i = 0
        while i < len(full_tstamps):
            if miss_y[i] or large_a[i]:
                start = i - 1
                while i < len(full_tstamps) and (miss_y[i] or large_a[i]):
                    i += 1
                end = i if i < len(full_tstamps) else len(full_tstamps)-1
                if 0 <= start < len(full_tstamps) and 0 <= end < len(full_tstamps):
                    ax_yaw.plot([full_tstamps[start], full_tstamps[end]],
                                [aligned_ang[start,1], aligned_ang[end,1]],
                                color=color, linestyle=(0,(10,6)), linewidth=2, alpha=0.8)
            else:
                i += 1
        ax_yaw.scatter(full_tstamps[large_a], aligned_ang[large_a,1], color=color, s=10)

        # 误差散点
        ax_poserr.scatter(full_tstamps, pos_err, color=color, s=8, alpha=0.5, label=method)
        ax_angerr.scatter(full_tstamps, ang_err, color=color, s=8, alpha=0.5, label=method)

    # 绘图配置
    ax_xy.set_title("Trajectory Comparison (offset & marker-size)")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.grid(True)
    ax_xy.legend()

    ax_alt.set_title("Altitude over Time")
    ax_alt.set_xlabel("Frame Index")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.grid(True)
    ax_alt.set_ylim(680, 780)
    ax_alt.legend()

    ax_pitch.set_title("Pitch over Time")
    ax_pitch.set_xlabel("Frame Index")
    ax_pitch.set_ylabel("Pitch (°)")
    ax_pitch.grid(True)
    ax_pitch.set_ylim(25, 55)
    ax_pitch.legend()

    ax_yaw.set_title("Yaw over Time")
    ax_yaw.set_xlabel("Frame Index")
    ax_yaw.set_ylabel("Yaw (°)")
    ax_yaw.grid(True)
    ax_yaw.legend()

    ax_poserr.set_title("Position Error")
    ax_poserr.set_xlabel("Frame Index")
    ax_poserr.set_ylabel("Error (m)")
    ax_poserr.grid(True)
    ax_poserr.set_ylim(0, np.nanmax(pos_err) * 1.1)
    ax_poserr.legend()

    ax_angerr.set_title("Angular Error")
    ax_angerr.set_xlabel("Frame Index")
    ax_angerr.set_ylabel("Error (°)")
    ax_angerr.grid(True)
    ax_angerr.set_ylim(0, np.nanmax(ang_err) * 1.1)
    ax_angerr.legend()

    fig.tight_layout()
    fig.savefig(f"{outputs}/{seq_name}_comparison.png", dpi=300)
    plt.close(fig)

print("✅ Completed offset + marker-size error visualization, output saved.")

