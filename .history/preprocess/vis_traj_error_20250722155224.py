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
        if traj.shape[0] == 0:
            continue

        ax_xy.plot(traj[:, 0], traj[:, 1], label=method, color=color, linewidth=2)
        
        timestamps = np.arange(traj.shape[0])
        import matplotlib.ticker as ticker

        interval = 10
        timestamps_ds = timestamps[::interval]
        altitude_ds = traj[::interval, 2]
        
        ax_alt.plot(timestamps_ds, altitude_ds, label=method, color=color, linewidth=2)
        
        

    ax_xy.set_title(f"{seq} - XY轨迹图")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.grid(True)
    ax_xy.legend()

    ax_alt.set_title(f"{seq} - 高度随时间变化图")
    ax_alt.set_xlabel("时间戳（帧号）")
    ax_alt.set_ylabel("Z (海拔 m)")
    ax_alt.grid(True)
    ax_alt.legend()
    ax_alt.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax_alt.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    outputs = "/mnt/sda/MapScape/query/estimation/result_images/outputs"
    os.makedirs(outputs, exist_ok=True)
    fig_xy.savefig(outputs+f"/{seq}_xy.png")
    fig_alt.savefig(outputs+f"/{seq}_altitude.png")
    plt.close(fig_xy)
    plt.close(fig_alt)

print("✅ 所有轨迹图已保存至 outputs/")