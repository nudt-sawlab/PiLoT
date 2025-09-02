import os
import matplotlib.pyplot as plt
import numpy as np
from transform import wgs84tocgcs2000_batch
from scipy.ndimage import gaussian_filter1d

# 方法设定及颜色
methods = {
    "GT": "black",
    "FPVLoc": "red",
    "Pixloc": "blue",
    "Render2loc": "green",
    "ORB@per30": "purple",
    "Render2loc@raft": "orange"
}

data_root = "/mnt/sda/MapScape/query/estimation/result_images"
seq_list = sorted([
    f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")
])
def load_pose_with_name(file_path):
    data = []
    angles = []
    timestamps = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                name = parts[0]
                if "_" in name:
                    frame_idx = int(name.split("_")[0])  # e.g., "12_0.png" → 12
                else:
                    continue  # 不符合命名规则
                lon, lat, alt = map(float, parts[1:4])
                roll, pitch, yaw = map(float, parts[4:7])
                data.append((lon, lat, alt))
                angles.append((pitch, yaw))
                timestamps.append(frame_idx)
    xyz = wgs84tocgcs2000_batch(data, 4547)
    return np.array(timestamps), np.array(xyz), np.array(angles)
# def load_pose(file_path):
#     data = []
#     angles = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) >= 7:
#                 lon, lat, alt = map(float, parts[1:4])
#                 roll, pitch, yaw = map(float, parts[4:7])
#                 if yaw < 0:
#                     yaw += 360
#                 data.append((lon, lat, alt))
#                 angles.append((pitch, yaw))
#     xyz = wgs84tocgcs2000_batch(data, 4547)
#     return np.array(xyz), np.array(angles)

for seq in seq_list:
    print(f"📍 处理序列：{seq}")
    seq_name = seq.split('.')[0]

    # 四宫格子图
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax_xy, ax_alt, ax_pitch, ax_yaw = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    poses_gt, angles_gt = load_pose(os.path.join(data_root, "GT", seq))

    for method, color in methods.items():
        file_path = os.path.join(data_root, method, seq)
        if not os.path.exists(file_path):
            print(f"[⚠️] {method} 缺少文件，跳过：{seq}")
            continue

        poses_est, angles_est = load_pose_with_name(file_path)
        # if len(poses_est) == 0 or len(poses_gt) != len(poses_est):
        #     print(f"[❌] {method} 数据不一致，跳过")
        #     continue

        timestamps_est, poses_est, angles_est = load_pose_with_name(file_path)
        if len(poses_est) == 0:
            print(f"[❌] {method} 无有效位姿，跳过")
            continue

        # 平滑轨迹
        rel = poses_est - poses_est[0]
        rel[:, 0] = gaussian_filter1d(rel[:, 0], sigma=1.5)
        rel[:, 1] = gaussian_filter1d(rel[:, 1], sigma=1.5)

        # 1. XY轨迹图
        x, y = rel[:, 0], rel[:, 1]
        for i in range(1, len(x)):
            if np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) < 10:
                ax_xy.plot([x[i - 1], x[i]], [y[i - 1], y[i]], color=color, linewidth=1.8)

        # 2. 高度随时间
        ax_alt.plot(timestamps_est, poses_est[:, 2], color=color, label=method, linewidth=2)

        # 3. Pitch 原始值
        ax_pitch.plot(timestamps_est, angles_est[:, 0], label=method, color=color, linewidth=1.5)

        # 4. Yaw 原始值
        ax_yaw.plot(timestamps_est, angles_est[:, 1], label=method, color=color, linewidth=1.5)
    # 图表美化
    ax_xy.set_title("Trajectory in XY Plane")
    ax_xy.set_xlabel("Relative X (m)")
    ax_xy.set_ylabel("Relative Y (m)")
    ax_xy.grid(True, linestyle='--', linewidth=0.6)
    ax_xy.legend()

    ax_alt.set_title("Altitude over Time")
    ax_alt.set_xlabel("Frame Index")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.grid(True, linestyle='--', linewidth=0.6)
    ax_alt.legend()

    ax_pitch.set_title("Pitch Angle over Time")
    ax_pitch.set_xlabel("Frame Index")
    ax_pitch.set_ylabel("Pitch (°)")
    ax_pitch.grid(True, linestyle='--', linewidth=0.6)
    ax_pitch.legend()

    ax_yaw.set_title("Yaw Angle over Time")
    ax_yaw.set_xlabel("Frame Index")
    ax_yaw.set_ylabel("Yaw (°)")
    ax_yaw.grid(True, linestyle='--', linewidth=0.6)
    ax_yaw.legend()

    # 保存
    outputs = os.path.join(data_root, "outputs")
    os.makedirs(outputs, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{outputs}/{seq_name}_4plots.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

print("✅ 所有四宫格图像已保存至 outputs/")

