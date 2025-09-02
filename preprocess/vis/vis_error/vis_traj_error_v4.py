import os 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.ndimage import uniform_filter1d
from transform import wgs84tocgcs2000_batch
import matplotlib.ticker as mticker
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
offset_dict = {
    "GT": (0.0, 0.0),
    "FPVLoc": (0.0, 0.0),
    "Pixloc": (-2.0, -1.0),
    "Render2loc": (1.0, -1.0),
    "ORB@per30": (-1.0, 1.0),
    "Render2loc@raft": (1.0, 1.0)
}
offset_height_dict = {
    "GT": (0.0),
    "FPVLoc": (0.0),
    "Pixloc": (1.0),
    "Render2loc": (0.5),
    "ORB@per30": (-0.5),
    "Render2loc@raft": (-0.5)
}
offset_angle_dict = {
    "GT": (0.0, 0.0),
    "FPVLoc": (0.0, 0.0),
    "Pixloc": (-0.2, -0.34),
    "Render2loc": (0.2, -0.1),
    "ORB@per30": (-0.2, 0.3),
    "Render2loc@raft": (0.1, 0.5)
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

def plot_with_virtual_lines(ax, x, y, color, large_error=None,label=None, linewidth=1.8,
                            ):
    """
    绘制数据序列中断实线段，并以虚线连接缺失段首尾。
    如果某帧 large_error==True，则跳过不绘制。
    """
    isnan = np.isnan(y)
    segments = []
    current_x, current_y = [], []

    for i in range(len(y)):
        if not isnan[i] and (large_error is None or not large_error[i]):
            current_x.append(x[i])
            current_y.append(y[i])
        else:
            if len(current_x) > 1:
                segments.append((current_x, current_y))
            current_x, current_y = [], []

    if len(current_x) > 1:
        segments.append((current_x, current_y))

    # 实线绘制有效段
    for i, (xs, ys) in enumerate(segments):
        # if color != 'black':
        #     ax.plot(xs, ys,  alpha=0.7, color=color,  linewidth=5,  label=label if i == 0 else None)
        # else:   
        ax.plot(xs, ys, color=color, linewidth=linewidth,  label=label if i == 0 else None)

    # 虚线连接缺失段首尾
    i = 0
    while i < len(y):
        if ( isnan[i]) :
            start = i - 1
            while i < len(y) and (isnan[i]):
                i += 1
            end = i
            if 0 <= start < len(y) and end < len(y):
                if (not isnan[start]) and (not isnan[end]) and \
                (large_error is None or (not large_error[start] and not large_error[end])):
                    ax.plot([x[start], x[end]], [y[start], y[end]],
                            color=color, linestyle='dashed', linewidth=2.0, alpha=0.8)
        else:
            i += 1
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
        if isnan[i] :
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

def plot_with_shaded_error(ax, x, y, err, label, color, scale=3.0):
    ax.plot(x, y, color=color, linewidth=1.5, label=label)
    ax.fill_between(x, y - err * scale, y + err * scale, color=color, alpha=0.2)

# 主流程
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
        
        dz = offset_height_dict.get(method, (0.0))
        # aligned_xyz+= dz
        
        dx, dy = offset_dict.get(method, (0.0, 0.0))
        # rel[valid_mask, 0] += dx
        # rel[valid_mask, 1] += dy
        
        dpitch, dyaw = offset_angle_dict.get(method, (0.0, 0.0))
        # aligned_angles[valid_mask, 0] += dpitch
        # aligned_angles[valid_mask, 1] += dyaw

        if 'GT' in method:
            gt_pts_cgcs = rel
            plot_trajectory_with_gaps(ax_xy, full_timestamps, rel, color, label=method)
            plot_with_virtual_lines(ax_alt, full_timestamps, aligned_xyz[:, 2], color, label=method)
            plot_with_virtual_lines(ax_pitch, full_timestamps, aligned_angles[:, 0], color, label=method)
            plot_with_virtual_lines(ax_yaw, full_timestamps, aligned_angles[:, 1], color, label=method)
            continue

        if 'ORB' in method:
            scale, R, t = umeyama_alignment(rel[timestamps_est[0:300]], gt_pts_cgcs[timestamps_est[0:300]])
            rel = transform_points(rel, scale, R, t)
        print('-----method', method)
        pos_err = np.linalg.norm(aligned_xyz - poses_gt, axis=1)
        pos_err[np.isnan(pos_err)] = 50
        ang_err = np.linalg.norm(aligned_angles - angles_gt, axis=1)
        ang_err[np.isnan(ang_err)] = 10

        large_error = pos_err > 20
        large_euler_error = ang_err > 5
        plot_trajectory_with_gaps(ax_xy, full_timestamps, rel, color, large_error, label=method)

        plot_with_virtual_lines(ax_alt, full_timestamps, aligned_xyz[:, 2], color, large_error)
        ax_alt.scatter(full_timestamps[large_error], aligned_xyz[large_error, 2], color=color, s=10, alpha=0.6)

        plot_with_virtual_lines(ax_pitch, full_timestamps, aligned_angles[:, 0], color, large_euler_error)
        ax_pitch.scatter(full_timestamps[large_error], aligned_angles[large_error, 0], color=color, s=10, alpha=0.6)

        plot_with_virtual_lines(ax_yaw, full_timestamps, aligned_angles[:, 1], color, large_euler_error)
        ax_yaw.scatter(full_timestamps[large_error], aligned_angles[large_error, 1], color=color, s=10, alpha=0.6)

        # aligned_angles[valid_mask, 0] -= dpitch
        # aligned_angles[valid_mask, 1] -= dyaw
        # aligned_xyz+= dz
        pos_err = np.linalg.norm(aligned_xyz - poses_gt, axis=1)
        pos_err[np.isnan(pos_err)] = 50
        ang_err = np.linalg.norm(aligned_angles - angles_gt, axis=1)
        ang_err[np.isnan(ang_err)] = 10
        
        ax_poserr.scatter(full_timestamps, pos_err, color=color, s=8, alpha=0.5, label=method)
        ax_angerr.scatter(full_timestamps, ang_err, color=color, s=8, alpha=0.5, label=method)

        # plot_with_shaded_error(ax_alt, full_timestamps, aligned_xyz[:, 2], np.abs(pos_err), method, color)
        # plot_with_shaded_error(ax_pitch, timestamps_est, angles_est[:, 0], np.abs(angles_est[:, 0] - angles_gt[timestamps_est][:, 0]), method, color)
        # plot_with_shaded_error(ax_yaw, timestamps_est, angles_est[:, 1], np.abs(angles_est[:, 1] - angles_gt[timestamps_est][:, 1]), method, color)

    ax_xy.set_title("Trajectory in XY Plane")
    ax_xy.set_xlabel("Relative X (m)")
    ax_xy.set_ylabel("Relative Y (m)")
    ax_xy.grid(True)
    ax_xy.legend()

    ax_alt.set_title("Altitude over Time")
    ax_alt.set_xlabel("Frame Index")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.grid(True)
    ax_alt.legend()
    # ax_alt.set_ylim(745, 755)
    # 
    # ax_alt.set_xlim(443, 446)
    # ax_alt.yaxis.set_major_locator(mticker.MultipleLocator(1))
    # ax_alt.xaxis.set_major_locator(mticker.MultipleLocator(1))
    

    ax_pitch.set_title("Pitch Angle over Time")
    ax_pitch.set_xlabel("Frame Index")
    ax_pitch.set_ylabel("Pitch (°)")
    ax_pitch.grid(True)
    ax_pitch.set_ylim(25, 55)
    ax_pitch.legend()

    ax_yaw.set_title("Yaw Angle over Time")
    ax_yaw.set_xlabel("Frame Index")
    ax_yaw.set_ylabel("Yaw (°)")
    # ax_yaw.scatter(full_timestamps, angles_gt[:, 1], color='black', s=100, label='GT')
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
    ax_angerr.legend()
    ax_angerr.set_ylim(0, 5)

    fig.tight_layout()
    fig.savefig(f"{outputs}/{seq_name}_5plots.png", dpi=300)
    plt.close(fig)

print("✅ 已完成轨迹及Yaw/Pitch/高度缺失段虚线连接绘制")
