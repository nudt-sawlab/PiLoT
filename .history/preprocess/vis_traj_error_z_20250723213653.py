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

    # ========== 动态放大镜（y轴极限放大） ==========
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax_alt, width="28%", height="38%", loc='upper right', borderpad=2)
    axins.set_facecolor("white")
    axins.tick_params(axis='both', which='both', labelsize=8)

    def update_inset(x0):
        zoom = 8  # 横轴窗口±8
        axins.clear()
        miny, maxy = None, None
        for method, (ts, alt, color) in all_alt_data.items():
            valid = ~np.isnan(alt)
            axins.plot(ts[valid], alt[valid], color=color, linewidth=2, label=method)
            # 取本方法在窗口内有效的高度范围
            in_window = (ts >= x0 - zoom) & (ts <= x0 + zoom) & valid
            if np.any(in_window):
                thismin, thismax = np.min(alt[in_window]), np.max(alt[in_window])
                if miny is None or thismin < miny:
                    miny = thismin
                if maxy is None or thismax > maxy:
                    maxy = thismax
        # y方向极限放大（窗口实际高度±margin）
        if miny is not None and maxy is not None:
            margin = (maxy - miny) * 0.3 + 0.01  # 保证放大很大
            axins.set_ylim(miny - margin, maxy + margin)
        else:
            axins.set_ylim(ax_alt.get_ylim())
        axins.set_xlim(x0 - zoom, x0 + zoom)
        axins.set_facecolor("white")
        axins.tick_params(axis='both', which='both', labelsize=8)
        fig.canvas.draw_idle()

    def on_move(event):
        if event.inaxes != ax_alt: return
        x0 = event.xdata
        if x0 is not None:
            update_inset(x0)

    fig.canvas.mpl_connect('motion_notify_event', on_move)

    def on_key(event):
        if event.key.lower() == 's':
            save_path = f"{outputs}/{seq_name}_altitude_with_zoomed_magnifier.png"
            fig.canvas.draw()
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 按S已保存 高度主图+放大细节至 {save_path}")
    fig.canvas.mpl_connect('key_press_event', on_key)

    # 初始化：先在第一个点
    update_inset(full_timestamps[0] if len(full_timestamps) > 0 else 0)

    plt.show()
    plt.close(fig)

print("✅ y轴极限细节放大镜已生效，随鼠标横向跟踪")

