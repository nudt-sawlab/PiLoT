import os
import matplotlib.pyplot as plt
import numpy as np
from transform import wgs84tocgcs2000_batch

# ✅ 配置参数
data_root = "/mnt/sda/MapScape/query/estimation/result_images"
methods_name = {
    "GT": "GT",
    "FPVLoc": "GeoPixel",
    "Pixloc": "PixLoc",
    "Render2loc": "Render2Loc",
    "ORB@per30": "Render2ORB",
    "Render2loc@raft": "Render2RAFT"
}
methods = {
    "FPVLoc": "red",
    "Pixloc": "blue",
    "Render2loc": "green",
    "ORB@per30": "purple",
    "Render2loc@raft": "orange"
}

start_frame = 0  # ✅ 可调
end_frame = 900

def load_pose_with_name(file_path):
    data, timestamps = [], []
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
                data.append((lon, lat, alt))
                timestamps.append(frame_idx)
    xyz = wgs84tocgcs2000_batch(data, 4547)
    return np.array(timestamps), np.array(xyz)

def load_gt_pose(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                lon, lat, alt = map(float, parts[1:4])
                data.append((lon, lat, alt))
    xyz = wgs84tocgcs2000_batch(data, 4547)
    return np.array(xyz)

# ✅ 主流程
seq_list = sorted([f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")])

for seq in seq_list:
    print(f"�� 绘制误差随时间变化图：{seq}")
    seq_name = seq.split('.')[0]

    # === 加载 GT
    gt_xyz_all = load_gt_pose(os.path.join(data_root, "GT", seq))
    frame_ids_all = np.arange(len(gt_xyz_all))

    # ✅ 截取帧范围
    valid_mask = (frame_ids_all >= start_frame) & (frame_ids_all <= end_frame)
    gt_xyz = gt_xyz_all[valid_mask]
    frame_ids = frame_ids_all[valid_mask]

    # === 开始绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, color in methods.items():
        file_path = os.path.join(data_root, method, seq)
        if not os.path.exists(file_path):
            continue

        est_frame_ids, est_xyz = load_pose_with_name(file_path)

        # 对齐：每一帧误差与 GT 匹配
        pos_err = np.full_like(frame_ids, np.nan, dtype=np.float32)
        for i, fid in enumerate(frame_ids):
            if fid in est_frame_ids:
                idx = np.where(est_frame_ids == fid)[0][0]
                err = np.linalg.norm(est_xyz[idx] - gt_xyz[i])
                pos_err[i] = err

        # ✅ 绘制误差曲线
        ax.plot(frame_ids, pos_err, color=color, label=methods_name[method], linewidth=2)

    ax.set_title(f"Position Error vs Frame Index ({seq_name})")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Position Error (m)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    # ✅ 保存或展示
    output_path = os.path.join(data_root, "outputs", f"{seq_name}_pos_error_curve.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    print(f"✅ 已保存至：{output_path}")

    plt.close(fig)
