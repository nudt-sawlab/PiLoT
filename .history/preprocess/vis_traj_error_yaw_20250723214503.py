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
    fig, ax_pitch = plt.subplots(1, 1, figsize=(10, 6))

    poses_gt, angles_gt = load_pose(os.path.join(data_root, "GT", seq))
    full_timestamps = np.arange(len(poses_gt))

    all_pitch_data = {}

    for method, color in methods.items():
        file_path = os.path.join(data_root, method, seq)
        if not os.path.exists(file_path): continue
        timestamps_est, poses_est, angles_est = load_pose_with_name(file_path)
        if len(angles_est) == 0: continue

        aligned_angles = align_to_full_timestamps(full_timestamps, timestamps_est, angles_est)
        valid_mask = ~np.isnan(aligned_angles[:, 0])

        ax_pitch.plot(full_timestamps[valid_mask], aligned_angles[valid_mask, 0], color=color, linewidth=2)
        ax_pitch.scatter(full_timestamps[~valid_mask], aligned_angles[~valid_mask, 0], color=color, s=10, alpha=0.6)

        all_pitch_data[method] = (full_timestamps, aligned_angles[:, 0], color)

    ax_pitch.set_title("Pitch Angle over Time")
    ax_pitch.set_xlabel("Frame Index")
    ax_pitch.set_ylabel("Pitch (°)")
    ax_pitch.grid(True)
    ax_pitch.legend()
    fig.tight_layout()

    # ========== 动态放大镜（y轴极限放大） ==========
    axins = inset_axes(ax_pitch, width="28%", height="38%", loc='upper right', borderpad=2)
    axins.set_facecolor("white")
    axins.tick_params(axis='both', which='both', labelsize=8)

    def update_inset(x0):
        zoom = 8  # 横轴窗口±8
        axins.clear()
        miny, maxy = None, None
        for method, (ts, pitch, color) in all_pitch_data.items():
            valid = ~np.isnan(pitch)
            axins.plot(ts[valid], pitch[valid], color=color, linewidth=3)
            in_window = (ts >= x0 - zoom) & (ts <= x0 + zoom) & valid
            if np.any(in_window):
                thismin, thismax = np.min(pitch[in_window]), np.max(pitch[in_window])
                if miny is None or thismin < miny:
                    miny = thismin
                if maxy is None or thismax > maxy:
                    maxy = thismax
        # y方向极限放大（窗口实际Pitch±margin）
        if miny is not None and maxy is not None:
            margin = (maxy - miny) * 0.3 + 0.02
            axins.set_ylim(miny - margin, maxy + margin)
        else:
            axins.set_ylim(ax_pitch.get_ylim())
        axins.set_xlim(x0 - zoom, x0 + zoom)
        axins.set_facecolor("white")
        axins.tick_params(axis='both', which='both', labelsize=8)
        fig.canvas.draw_idle()

    def on_move(event):
        if event.inaxes != ax_pitch: return
        x0 = event.xdata
        if x0 is not None:
            update_inset(x0)

    fig.canvas.mpl_connect('motion_notify_event', on_move)

    def on_key(event):
        if event.key.lower() == 's':
            save_path = f"{outputs}/{seq_name}_pitch_with_zoomed_magnifier.png"
            fig.canvas.draw()
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 按S已保存 Pitch主图+放大细节至 {save_path}")
    fig.canvas.mpl_connect('key_press_event', on_key)

    # 初始化：先在第一个点
    update_inset(full_timestamps[0] if len(full_timestamps) > 0 else 0)

    plt.show()
    plt.close(fig)

print("✅ Pitch角随时间主图+动态极限细节放大镜完成，支持S键保存")

