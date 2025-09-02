import os
import matplotlib.pyplot as plt
import numpy as np
from transform import wgs84tocgcs2000_batch
# 设定方法和颜色
methods = {
    "GT": "black",
    "GeoPixel": "red",
    "PixLoc": "blue",
    "Render2Loc": "green",
    "Render2ORB": "purple",
    "Render2RAFT": "orange"
}
methods = {
    "GT": "black",
    "FPVLoc": "red",
    "Pixloc": "blue",
    "Render2loc": "green",
    "ORB@per30": "purple",
    "Render2loc@raft": "orange"
}


# 数据根目录
data_root = "/mnt/sda/MapScape/query/estimation/result_images"  # 可修改为你的路径
seq_list = sorted([
    f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")
])

def load_ecef(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                lon, lat, alt = map(float, parts[1:4])
                # x, y, z = wgs84tocgcs2000([lon, lat, alt], 4547)
                data.append((lon, lat, alt))
    data_cgcs = wgs84tocgcs2000_batch(data, 4547)
    return np.array(data_cgcs)

for seq in seq_list:
    fig_xy, ax_xy = plt.subplots(figsize=(8, 6))
    fig_alt, ax_alt = plt.subplots(figsize=(8, 6))

    for method, color in methods.items():
        file_path = os.path.join(data_root, method, seq)
        if not os.path.exists(file_path):
            print(f"[警告] {method} 没有 {seq} 文件，跳过")
            continue

        traj = load_ecef(file_path)
        traj_rel = traj - traj[0]
        if traj.shape[0] == 0:
            continue

        # 平滑 + 跳跃剔除 + 折线图
        MAX_JUMP = 10
        from scipy.ndimage import gaussian_filter1d

        traj_rel = traj - traj[0]
        traj_rel[:, 0] = gaussian_filter1d(traj_rel[:, 0], sigma=1.5)
        traj_rel[:, 1] = gaussian_filter1d(traj_rel[:, 1], sigma=1.5)

        x, y = traj_rel[:, 0], traj_rel[:, 1]
        for i in range(1, len(x)):
            if np.hypot(x[i] - x[i-1], y[i] - y[i-1]) < MAX_JUMP:
                ax_xy.plot([x[i-1], x[i]], [y[i-1], y[i]],
                        color=color, linewidth=1.5, alpha=0.9)
        
        timestamps = np.arange(traj.shape[0])
        import matplotlib.ticker as ticker

        interval = 10
        timestamps_ds = timestamps[::interval]
        altitude_ds = traj[::interval, 2]
        
        ax_alt.plot(timestamps_ds, altitude_ds, label=method, color=color, linewidth=2)
        
    ax_xy.set_title(f"Trajectory in the XY Plane", fontsize=12)
    ax_xy.set_xlabel("Relative X (m)", fontsize=11)
    ax_xy.set_ylabel("Relative Y (m)", fontsize=11)
    ax_xy.grid(True, linestyle='--', linewidth=0.5)
    ax_xy.legend(fontsize=10)

    ax_alt.set_title(f"Altitude over Time", fontsize=12)
    ax_alt.set_xlabel("Timestamp (Frame Index)", fontsize=11)
    ax_alt.set_ylabel("Altitude (Z, m)", fontsize=11)
    ax_alt.grid(True, linestyle='--', linewidth=0.5)
    ax_alt.legend(fontsize=10)
    ax_alt.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax_alt.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    outputs = "/mnt/sda/MapScape/query/estimation/result_images/outputs"
    os.makedirs(outputs, exist_ok=True)
    seq = seq.split('.')[0]
    fig_xy.savefig(outputs+f"/{seq}_xy.png")
    fig_alt.savefig(outputs+f"/{seq}_altitude.png")
    plt.close(fig_xy)
    plt.close(fig_alt)

print("✅ 所有轨迹图已保存至 outputs/")