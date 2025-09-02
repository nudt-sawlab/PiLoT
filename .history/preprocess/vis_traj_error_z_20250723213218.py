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
    fig, ax_alt = plt.subplots(1, 1, figsize=(10, 6))

    poses_gt, angles_gt = load_pose(os.path.join(data_root, "GT", seq))
    full_timestamps = np.arange(len(poses_gt))

    all_alt_data = {}

    for method, color in methods.items():
        file_path = os.path.join(data_root, method, seq)
        if not os.path.exists(file_path): continue
        timestamps_est, poses_est, angles_est = load_pose_with_name(file_path)
        if len(poses_est) == 0: continue

        aligned_xyz = align_to_full_timestamps(full_timestamps, timestamps_est, poses_est)
        valid_mask = ~np.isnan(aligned_xyz[:, 2])

        ax_alt.plot(full_timestamps[valid_mask], aligned_xyz[valid_mask, 2], color=color, linewidth=2, label=method)
        ax_alt.scatter(full_timestamps[~valid_mask], aligned_xyz[~valid_mask, 2], color=color, s=10, alpha=0.6)

        all_alt_data[method] = (full_timestamps, aligned_xyz[:, 2], color)

    ax_alt.set_title("Altitude over Time")
    ax_alt.set_xlabel("Frame Index")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.grid(True)
    ax_alt.legend()
    fig.tight_layout()

    # ========== 动态放大镜功能 ==========
    axins = inset_axes(ax_alt, width="30%", height="38%", loc='upper right', borderpad=2)
    axins.set_facecolor("white")
    axins.set_xticks([])
    axins.set_yticks([])

    # 绘制全部方法主曲线到inset（初始化先画第一个区间）
    xlim, ylim = ax_alt.get_xlim(), ax_alt.get_ylim()
    def update_inset(x0):
        zoom = 30  # 横轴跨度
        axins.clear()
        for method, (ts, alt, color) in all_alt_data.items():
            valid = ~np.isnan(alt)
            axins.plot(ts[valid], alt[valid], color=color, linewidth=2, label=method)
        axins.set_xlim(x0 - zoom, x0 + zoom)
        # 自动范围或固定范围
        axins.set_ylim(ylim)
        axins.set_facecolor("white")
        axins.set_xticks([])
        axins.set_yticks([])
        fig.canvas.draw_idle()

    def on_move(event):
        if event.inaxes != ax_alt: return
        x0 = event.xdata
        if x0 is not None:
            update_inset(x0)

    fig.canvas.mpl_connect('motion_notify_event', on_move)

    # 保存图片快捷键
    def on_key(event):
        if event.key.lower() == 's':
            save_path = f"{outputs}/{seq_name}_altitude_with_magnifier.png"
            fig.canvas.draw()
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 按S已保存 高度随时间图+放大镜至 {save_path}")
    fig.canvas.mpl_connect('key_press_event', on_key)

    # 初始化时inset放大第一个点
    update_inset(full_timestamps[0] if len(full_timestamps) > 0 else 0)

    plt.show()
    plt.close(fig)

print("✅ 只可视化高度随时间主图（含交互放大镜，支持S键保存）")

