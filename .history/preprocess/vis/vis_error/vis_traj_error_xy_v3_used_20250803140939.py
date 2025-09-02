import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from transform import wgs84tocgcs2000_batch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

rcParams['font.family'] = 'serif'
rcParams['axes.labelweight'] = 'bold'
methods_name = {
    "GT": "GT",
    "FPVLoc": "GeoPixel",
    "Pixloc": "PixLoc",
    "Render2loc": "Render2Loc",
    "ORB@per30": "Render2ORB",
    "Render2loc@raft": "Render2RAFT"
}
methods = {
    "GT": "black",
    "FPVLoc": "#007F49",      # ✅ GeoPixel：深绿，不变
    "Pixloc": "#86AED5",       # 加深灰蓝
    "Render2loc": "#EF6C5D",   # 加深橘粉
    "ORB@per30": "#C79ACD",    # 加深淡紫
    "Render2loc@raft": "#F7B84A"  # 奶油橙
}

offset_dict = {
    "GT": (0.0, 0.0),
    "FPVLoc": (0.0, 0.0),
    "Pixloc": (-1.0, -1.0),
    "Render2loc": (1.0, -0.5),
    "ORB@per30": (-1.0, 1.0),
    "Render2loc@raft": (1.0, 1.0)
}

data_root = "/mnt/sda/MapScape/query/estimation/result_images/feicuiwan"
outputs = os.path.join(data_root, "outputs")
os.makedirs(outputs, exist_ok=True)
from matplotlib.patches import Circle

def plot_trajectory_with_error_circle(ax, traj, pos_err, color, linewidth=2.5, alpha=0.4, scale=0.8):
    """
    在轨迹上绘制误差圆圈，透明度和大小随误差变化
    - traj: (N, 2)
    - pos_err: (N,), 单位: m
    """
    x, y = traj[:, 0], traj[:, 1]
    for i in range(len(x)):
        if np.isnan(x[i]) or np.isnan(pos_err[i]) or pos_err[i] <= 0:
            continue
        radius = pos_err[i] * scale  # 缩放比例调节圆大小
        circle = Circle((x[i], y[i]), radius=radius,
                        color=color, alpha=alpha, linewidth=0)
        ax.add_patch(circle)

    # 绘制主轨迹线（在误差圈上方）
    valid = ~np.isnan(x)
    ax.plot(x[valid], y[valid], color=color, linewidth=linewidth, zorder=10)

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

def plot_trajectory_with_gaps(ax, timestamps, traj, color, large_error=None, label=None, linewidth=2.2):
    x, y = traj[:, 0], traj[:, 1]
    isnan = np.isnan(x) | np.isnan(y)
    islarge = large_error if large_error is not None else np.zeros_like(x, dtype=bool)
    segments = []
    current = []
    for i in range(len(x)):
        if not isnan[i] and not islarge[i]:
            current.append((x[i], y[i]))
        else:
            if len(current) > 1:
                segments.append(np.array(current))
            current = []
    if len(current) > 1:
        segments.append(np.array(current))
    for i, seg in enumerate(segments):
        ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=linewidth, label=label if i == 0 else None)
    # 添加虚线连接缺失段首尾
    i = 0
    while i < len(y):
        if isnan[i]:
            start = i - 1
            while i < len(y) and (isnan[i]):
                i += 1
            end = i
            if 0 <= start < len(y) and end < len(y):
                if (not isnan[start]) and (not isnan[end]) and \
                   (not islarge[start]) and (not islarge[end]):
                    ax.plot([x[start], x[end]], [y[start], y[end]],
                            color=color, linestyle='dashed', linewidth=2.0, alpha=0.8)
        else:
            i += 1

# 主流程
seq_list = sorted([f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")])

# == 替换主流程部分（从 seq_list 开始）==
for seq in seq_list:
    # 仅测试一条可固定此行
    seq = 'DJI_20250612194903_0021_V.txt'
    print(f"📍 Processing: {seq}")
    seq_name = seq.split('.')[0]
    
    poses_gt, angles_gt = load_pose(os.path.join(data_root, "GT", seq))
    full_timestamps = np.arange(len(poses_gt))

    for method, color in methods.items():
        file_path = os.path.join(data_root, methods_name[method], seq)
        if not os.path.exists(file_path): continue

        timestamps_est, poses_est, angles_est = load_pose_with_name(file_path)
        if len(poses_est) == 0: continue

        aligned_xyz = align_to_full_timestamps(full_timestamps, timestamps_est, poses_est)
        valid_mask = ~np.isnan(aligned_xyz[:, 0])
        rel = aligned_xyz.copy()
        rel[valid_mask] -= aligned_xyz[valid_mask][0]

        dx, dy = offset_dict.get(method, (0.0, 0.0))
        rel[valid_mask, 0] += dx
        rel[valid_mask, 1] += dy
        if 'ORB' in method:
            print(';')
        if method != "GT":
            scale, R, t = umeyama_alignment(rel[timestamps_est[0:300]], poses_gt[timestamps_est[0:300]])
            rel = transform_points(rel, scale, R, t)
        def add_noise_to_trajectory(traj, std=0.5):
            """
            给轨迹加二维高斯扰动。
            - traj: shape=(N, 3) or (N, 2)，默认 XYZ 或 XY
            - std: 噪声标准差（单位：米）
            """
            noise = np.random.normal(loc=0.0, scale=std, size=traj[:, :2].shape)
            traj_noised = traj.copy()
            traj_noised[:, 0:2] += noise
            return traj_noised


        # rel = add_noise_to_trajectory(rel, std=0.3)
        # 创建图像
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off')                   # 不显示坐标轴
        ax.set_aspect('equal')          # 保持xy比例
        valid = ~np.isnan(rel[:, 0])
        # ax.plot(rel[valid, 0], rel[valid, 1], color=color, linewidth=2.5)
        ax.plot(rel[:600, 0], rel[:600, 1], color=color, linewidth=1)
        
        # pos_err = np.linalg.norm(aligned_xyz - poses_gt, axis=1)
        # pos_err[np.isnan(pos_err)] = 0  # 避免 NaN 导致绘图出错

        # plot_trajectory_with_error_circle(ax, rel, pos_err, color=color,
        #                                 linewidth=2.5, alpha=0.25, scale=0.4)
        # 保存路径
        save_path = os.path.join(outputs, f"{seq_name}_{methods_name[method]}.png")
        # fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True, facecolor='none')


        print(f"✅ Saved: {save_path}")
        plt.close(fig)

print("✅ 每个方法的纯轨迹图已分别保存。")


