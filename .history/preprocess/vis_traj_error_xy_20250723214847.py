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

for seq in seq_list:
    print(f"📍 Processing: {seq}")
    seq_name = seq.split('.')[0]
    fig, ax_xy = plt.subplots(1, 1, figsize=(10, 8))

    poses_gt, angles_gt = load_pose(os.path.join(data_root, "GT", seq))
    full_timestamps = np.arange(len(poses_gt))
    all_xy_data = {}

    for method, color in methods.items():
        file_path = os.path.join(data_root, method, seq)
        if not os.path.exists(file_path): continue
        timestamps_est, poses_est, angles_est = load_pose_with_name(file_path)
        if len(poses_est) == 0: continue

        aligned_xyz = align_to_full_timestamps(full_timestamps, timestamps_est, poses_est)
        valid_mask = ~np.isnan(aligned_xyz[:, 0]) & ~np.isnan(aligned_xyz[:, 1])

        # 原点平移，使不同方法一致
        rel = aligned_xyz.copy()
        rel[valid_mask] -= aligned_xyz[valid_mask][0]

        ax_xy.plot(rel[valid_mask, 0], rel[valid_mask, 1], color=color, linewidth=2.0, label=method)
        ax_xy.scatter(rel[~valid_mask, 0], rel[~valid_mask, 1], color=color, s=10, alpha=0.6)
        all_xy_data[method] = (rel[:, 0], rel[:, 1], color, valid_mask)

    ax_xy.set_title("Trajectory in XY Plane")
    ax_xy.set_xlabel("Relative X (m)")
    ax_xy.set_ylabel("Relative Y (m)")
    ax_xy.grid(True)
    ax_xy.legend()
    fig.tight_layout()

    # ========== 动态放大镜 ========== #
    axins = inset_axes(ax_xy, width="28%", height="32%", loc='upper right', borderpad=2)
    axins.set_facecolor("white")
    axins.tick_params(axis='both', which='both', labelsize=8)

    def update_inset(x0, y0):
        zoom = 10  # 控制局部放大镜 xy 范围
        axins.clear()
        for method, (rx, ry, color, valid) in all_xy_data.items():
            axins.plot(rx[valid], ry[valid], color=color, linewidth=2.0, label=method)
        axins.set_xlim(x0 - zoom, x0 + zoom)
        axins.set_ylim(y0 - zoom, y0 + zoom)
        axins.set_facecolor("white")
        axins.tick_params(axis='both', which='both', labelsize=8)
        fig.canvas.draw_idle()

    # 鼠标移动主图自动放大
    def on_move(event):
        if event.inaxes != ax_xy: return
        x0, y0 = event.xdata, event.ydata
        if x0 is not None and y0 is not None:
            update_inset(x0, y0)

    fig.canvas.mpl_connect('motion_notify_event', on_move)

    # 按S键保存
    def on_key(event):
        if event.key.lower() == 's':
            save_path = f"{outputs}/{seq_name}_xy_with_magnifier.png"
            fig.canvas.draw()
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 按S已保存 XY轨迹主图+放大细节至 {save_path}")
    fig.canvas.mpl_connect('key_press_event', on_key)

    # 初始化时显示第一个点
    for method, (rx, ry, color, valid) in all_xy_data.items():
        first_valid = np.where(valid)[0]
        if len(first_valid) > 0:
            update_inset(rx[first_valid[0]], ry[first_valid[0]])
            break

    plt.show()
    plt.close(fig)

print("✅ XY轨迹主图+动态局部放大镜功能完成，支持S键保存")

