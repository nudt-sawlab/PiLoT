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

def load_pose(file_path):
    data = []
    angles = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                lon, lat, alt = map(float, parts[1:4])
                roll, pitch, yaw = map(float, parts[4:7])
                data.append((lon, lat, alt))
                if yaw <0:
                    yaw += 360
                angles.append((pitch, yaw))
    xyz = wgs84tocgcs2000_batch(data, 4547)
    return np.array(xyz), np.array(angles)

for seq in seq_list:
    print(f"📍 处理序列：{seq}")
    seq_name = seq.split('.')[0]

    fig_xy, ax_xy = plt.subplots(figsize=(8, 6))
    fig_alt, ax_alt = plt.subplots(figsize=(8, 6))
    fig_err, ax_err = plt.subplots(figsize=(8, 6))
    fig_angle, ax_angle = plt.subplots(figsize=(8, 6))

    poses_gt, angles_gt = load_pose(os.path.join(data_root, "GT", seq))

    for method, color in methods.items():
        file_path = os.path.join(data_root, method, seq)
        if not os.path.exists(file_path):
            print(f"[⚠️] {method} 缺少文件，跳过：{seq}")
            continue

        poses_est, angles_est = load_pose(file_path)
        if len(poses_est) == 0 or len(poses_gt) != len(poses_est):
            print(f"[❌] {method} 数据不一致，跳过")
            continue

        # 平滑
        rel = poses_est - poses_est[0]
        # rel[:, 0] = gaussian_filter1d(rel[:, 0], sigma=1.5)
        # rel[:, 1] = gaussian_filter1d(rel[:, 1], sigma=1.5)

        x, y = rel[:, 0], rel[:, 1]
        for i in range(1, len(x)):
            if np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) < 10:
                ax_xy.plot([x[i - 1], x[i]], [y[i - 1], y[i]], color=color, linewidth=1.8)

        timestamps = np.arange(len(poses_gt))
        ax_alt.plot(timestamps, poses_est[:, 2], color=color, label=method, linewidth=2)

        # 坐标误差（米）
        pos_err = np.linalg.norm(poses_est - poses_gt, axis=1)
        ax_err.plot(timestamps, pos_err, label=method, color=color, linewidth=1.5)

        # 角度误差（°） pitch/yaw
        angle_err = np.abs(angles_est - angles_gt)
        pitch_err, yaw_err = angle_err[:, 0], angle_err[:, 1]
        ax_angle.plot(timestamps, pitch_err, label=f"{method} Pitch", color=color, linestyle='--', linewidth=1)
        ax_angle.plot(timestamps, yaw_err, label=f"{method} Yaw", color=color, linestyle='-', linewidth=1.5)

    for ax in [ax_xy, ax_alt, ax_err, ax_angle]:
        ax.grid(True, linestyle='--', linewidth=0.6)

    # 轨迹图
    ax_xy.set_title("Trajectory (XY Plane)", fontsize=13)
    ax_xy.set_xlabel("Relative X (m)")
    ax_xy.set_ylabel("Relative Y (m)")
    ax_xy.legend()

    # 高度图
    ax_alt.set_title("Altitude over Time", fontsize=13)
    ax_alt.set_xlabel("Frame Index")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.legend()

    # 误差图
    ax_err.set_title("Position Error over Time", fontsize=13)
    ax_err.set_xlabel("Frame Index")
    ax_err.set_ylabel("Error (m)")
    ax_err.legend()

    # 角度误差图
    ax_angle.set_title("Pitch & Yaw Error over Time", fontsize=13)
    ax_angle.set_xlabel("Frame Index")
    ax_angle.set_ylabel("Angular Error (°)")
    ax_angle.legend(ncol=2)

    outputs = os.path.join(data_root, "outputs")
    os.makedirs(outputs, exist_ok=True)
    fig_xy.savefig(f"{outputs}/{seq_name}_xy.png", dpi=300, bbox_inches='tight')
    fig_alt.savefig(f"{outputs}/{seq_name}_altitude.png", dpi=300, bbox_inches='tight')
    fig_err.savefig(f"{outputs}/{seq_name}_error.png", dpi=300, bbox_inches='tight')
    fig_angle.savefig(f"{outputs}/{seq_name}_angle_error.png", dpi=300, bbox_inches='tight')

    plt.close(fig_xy)
    plt.close(fig_alt)
    plt.close(fig_err)
    plt.close(fig_angle)

print("✅ 所有轨迹与误差图已保存至 outputs/")
