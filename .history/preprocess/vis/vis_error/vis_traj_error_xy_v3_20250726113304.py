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
    "FPVLoc": "red",
    "Pixloc": "blue",
    "Render2loc": "green",
    "ORB@per30": "purple",
    "Render2loc@raft": "orange"
}
offset_dict = {
    "GT": (0.0, 0.0),
    "FPVLoc": (-0.3, 0.0),
    "Pixloc": (0.5, 0.0),
    "Render2loc": (0.3, 0),
    "ORB@per30": (0.0, 0.0),
    "Render2loc@raft": (0.0, 0.0)
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
from matplotlib.collections import LineCollection
def plot_colored_trajectory_fixed_color(ax, traj, errors, base_color='blue', linewidth=2.5, label=None):
    """
    同一方法使用固定颜色，误差通过透明度（alpha）体现。
    """
    points = traj[:, :2].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 将误差映射为alpha值：误差越小 alpha越高
    norm = plt.Normalize(vmin=np.nanmin(errors), vmax=np.nanmax(errors))
    alphas = 1.0 - norm(errors[:-1])  # 小误差对应高alpha

    # 构建颜色 + alpha 列表
    from matplotlib.colors import to_rgba
    rgba_base = np.array(to_rgba(base_color))
    colors = [rgba_base[:3].tolist() + [a] for a in alphas]

    # 创建LineCollection
    lc = LineCollection(segments, colors=colors, linewidth=linewidth)
    ax.add_collection(lc)

    # legend中画一个实线
    ax.plot([], [], color=base_color, label=label)
    return lc
# 主流程
seq_list = sorted([f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")])
start_frame = 50
end_frame = 100
for seq in seq_list:
    print(f"📍 Processing: {seq}")
    seq_name = seq.split('.')[0]
    fig, ax_xy = plt.subplots(figsize=(10, 8))   # 只保留xy轨迹子图

    poses_gt_full, angles_gt_full = load_pose(os.path.join(data_root, "GT", seq))
    full_timestamps_full = np.arange(len(poses_gt_full))

    # ✅ 只保留区间内的 GT 数据
    mask = (full_timestamps_full >= start_frame) & (full_timestamps_full <= end_frame)
    full_timestamps = full_timestamps_full[mask]
    poses_gt = poses_gt_full[mask]
    angles_gt = angles_gt_full[mask]
    trajs_for_magnifier = {}

    for method, color in methods.items():
        file_path = os.path.join(data_root, method, seq)
        if not os.path.exists(file_path): continue
        timestamps_est, poses_est, angles_est = load_pose_with_name(file_path)
        if len(poses_est) == 0: continue

         # ✅ 原始全序列对齐
        aligned_xyz_full = align_to_full_timestamps(full_timestamps_full, timestamps_est, poses_est)

        # ✅ 提取对应区间段
        aligned_xyz = aligned_xyz_full[mask]
        valid_mask = ~np.isnan(aligned_xyz[:, 0])
        rel = aligned_xyz.copy()
        rel[valid_mask] -= aligned_xyz[valid_mask][0]

        dx, dy = offset_dict.get(method, (0.0, 0.0))
        # rel[valid_mask, 0] += dx
        # rel[valid_mask, 1] += dy

        if 'GT' in method:
            gt_pts_cgcs = rel  # ✅ GT 轨迹就是 rel
            plot_trajectory_with_gaps(ax_xy, full_timestamps, rel, color, label=methods_name[method])
            trajs_for_magnifier[method] = rel
            continue

        if 'ORB' in method:
            # ✅ 先对原始 ORB 轨迹进行对齐（截断区间内的 timestamp）
            sel_mask = (timestamps_est >= start_frame) & (timestamps_est <= end_frame)
            sel_idx = timestamps_est[sel_mask] - start_frame  # 对齐 index

            if len(sel_idx) >= 5:  # 至少5个点对
                scale, R, t = umeyama_alignment(rel[sel_idx], gt_pts_cgcs[sel_idx])
                rel = transform_points(rel, scale, R, t)

        pos_err = np.linalg.norm(aligned_xyz - poses_gt, axis=1)
        pos_err[np.isnan(pos_err)] = 50
        large_error = pos_err > 20
        # 对应帧的误差值
        # 调用新函数画带误差着色的轨迹
        line = plot_colored_trajectory_fixed_color(ax_xy, rel, pos_err,
                                           base_color=color,
                                           label=methods_name[method])
        trajs_for_magnifier[method] = rel
        # plot_trajectory_with_gaps(ax_xy, full_timestamps, rel, color, large_error, label=methods_name[method])
        # valid_mask = ~np.isnan(rel[:, 0])
        # scatter = ax_xy.scatter(rel[valid_mask, 0],
        #                 rel[valid_mask, 1],
        #                 s=20,  # 点大小
        #                 c=color,
        #                 label=methods_name[method],
        #                 alpha=0.8,
        #                 edgecolors='none')
        # trajs_for_magnifier[method] = rel

    # ========= 交互放大镜 =========
    axins = inset_axes(ax_xy, width="28%", height="28%", loc='upper right', borderpad=2)
    axins.set_facecolor("white")
    # axins.set_xticks([])
    # axins.set_yticks([])

    for method, color in methods.items():
        if method in trajs_for_magnifier:
            rel = trajs_for_magnifier[method]
            valid_mask = ~np.isnan(rel[:, 0])
            axins.plot(rel[valid_mask, 0], rel[valid_mask, 1], color=color, linewidth=2.8, label=method)

    def on_move(event):
        if event.inaxes != ax_xy: return
        x0, y0 = event.xdata, event.ydata
        zoom = 10   # 控制放大镜区域（坐标跨度，可调）
        axins.set_xlim(x0 - zoom, x0 + zoom)
        axins.set_ylim(y0 - zoom, y0 + zoom)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)

    def on_key(event):
        if event.key.lower() == 's':
            save_path = f"{outputs}/{seq_name}_xy_with_magnifier.png"
            fig.canvas.draw()
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 按S已保存 XY 主图和放大镜至 {save_path}")

    fig.canvas.mpl_connect('key_press_event', on_key)

    ax_xy.set_title("Trajectory in XY Plane")
    ax_xy.set_xlabel("Relative X (m)")
    ax_xy.set_ylabel("Relative Y (m)")
    ax_xy.grid(True)
    ax_xy.legend()
    # ax_xy.set_ylim(-25, 25)
    # ax_xy.set_xlim(-40, 0)
    
    fig.tight_layout()
    plt.show()
    plt.close(fig)

print("✅ 只可视化xy轨迹主图，带动态放大镜，可S键保存")

