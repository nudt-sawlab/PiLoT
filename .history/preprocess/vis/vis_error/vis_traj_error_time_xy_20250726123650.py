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
    "FPVLoc": "#007F49",      # ✅ GeoPixel：深绿，不变
    "Pixloc": "#86AED5",       # 加深灰蓝
    "Render2loc": "#EF6C5D",   # 加深橘粉
    "ORB@per30": "#C79ACD",    # 加深淡紫
    "Render2loc@raft": "#FFE0B5"  # 奶油橙
}
method_zorder = {
    "FPVLoc": 10,
    "Render2loc": 9,
    "Pixloc": 8,
    "ORB@per30": 6,
    "Render2loc@raft": 7
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
def transform_points(points, scale, R, t):
    return scale * (R @ points.T).T + t

def umeyama_alignment(src, dst):
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    src_centered, dst_centered = src - mu_src, dst - mu_dst
    cov = dst_centered.T @ src_centered / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    scale = np.trace(np.diag(D)) / ((src_centered ** 2).sum() / src.shape[0])
    t = mu_dst - scale * R @ mu_src
    return scale, R, t

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
    max_y_limit = 5  # ✅ 最大误差限制，可根据需求调整
    method_medians = {}

    for method, color in methods.items():
        file_path = os.path.join(data_root, method, seq)
        if not os.path.exists(file_path):
            continue
        est_frame_ids, est_xyz = load_pose_with_name(file_path)
        if 'ORB' in method:
            scale, R, t = umeyama_alignment(est_xyz[est_frame_ids], gt_xyz[est_frame_ids])
            est_xyz = transform_points(est_xyz, scale, R, t)

        # 对齐
        pos_err = np.full_like(frame_ids, np.nan, dtype=np.float32)
        for i, fid in enumerate(frame_ids):
            if fid in est_frame_ids:
                idx = np.where(est_frame_ids == fid)[0][0]
                err = np.linalg.norm(est_xyz[idx] - gt_xyz[i])
                pos_err[i] = err
            else:
                pos_err[i] = 5
                

        # ✅ 裁剪极值（避免极端值压缩 y 轴）
        pos_err_clipped = np.minimum(pos_err, max_y_limit)
        z = method_zorder.get(method, 5)
        # ✅ 改为散点图
        ax.scatter(frame_ids, pos_err_clipped,
               color=color,
               label=methods_name[method],
               alpha=0.7,
                s=18,
                zorder=z)
        # ✅ 记录中位数
        med = np.nanmedian(pos_err)
        method_medians[method] = med

        # ✅ 添加中值误差参考线
        ax.axhline(med, linestyle='--', color=color, linewidth=1.2, alpha=0.4,
           zorder=z - 1)

    # === 图属性 ===
    ax.set_title(f"Position Error vs Frame Index ({seq_name})")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Position Error (m)")
    ax.grid(True)
    ax.set_ylim(0, max_y_limit + 1)  # ✅ 限制Y轴范围

    # ✅ 添加 legend（方法名 + 中位数）
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for method, label in zip(methods.keys(), labels):
        if method in method_medians:
            new_labels.append(f"{label} (Med: {method_medians[method]:.2f}m)")
        else:
            new_labels.append(label)
    ax.legend(handles, new_labels)

    fig.tight_layout()
    output_path = os.path.join(data_root, "outputs", f"{seq_name}_pos_error_curve.png")
    fig.savefig(output_path, dpi=300)
    print(f"✅ 已保存至：{output_path}")
    plt.close(fig)