import os
import matplotlib.pyplot as plt
import numpy as np

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
                x, y, z = map(float, parts[1:4])
                data.append((x, y, z))
    return np.array(data)

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
        ax_alt.plot(traj[:, 0], traj[:, 2], label=method, color=color, linewidth=2)

    ax_xy.set_title(f"{seq} - XY轨迹图")
    ax_xy.set_xlabel("ECEF X (m)")
    ax_xy.set_ylabel("ECEF Y (m)")
    ax_xy.grid(True)
    ax_xy.legend()

    ax_alt.set_title(f"{seq} - 高度变化图")
    ax_alt.set_xlabel("ECEF X (m)")
    ax_alt.set_ylabel("ECEF Z (高度, m)")
    ax_alt.grid(True)
    ax_alt.legend()
    outputs = "/mnt/sda/MapScape/query/estimation/result_images/outputs"
    os.makedirs(outputs, exist_ok=True)
    fig_xy.savefig(outputs+"/{seq}_xy.png")
    fig_alt.savefig(outputs+"/{seq}_altitude.png")
    plt.close(fig_xy)
    plt.close(fig_alt)

print("✅ 所有轨迹图已保存至 outputs/")