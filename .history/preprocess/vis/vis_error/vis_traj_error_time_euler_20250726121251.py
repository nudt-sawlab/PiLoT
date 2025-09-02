import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from transform import WGS84_to_ECEF,wgs84tocgcs2000_batch,get_rotation_enu_in_ecef
def euler_angles_to_matrix_ECEF_w2c(euler_angles, trans):
    lon, lat, _ = trans
    rot_pose_in_enu = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
    rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_enu_in_ecef, rot_pose_in_enu)
    t_c2w = WGS84_to_ECEF(trans)
    
    # R_w2c_in_ecef = R_c2w.transpose() # 和enu的差异是第二行和第三行取负号
    # t_w2c = -R_w2c_in_ecef.dot(t_c2w)

    # T_render_in_ECEF_w2c = np.eye(4)
    # T_render_in_ECEF_w2c[:3, :3] = R_w2c_in_ecef
    # T_render_in_ECEF_w2c[:3, 3] = t_w2c
    return R_c2w
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

start_frame = 0  # ✅ 可调
end_frame = 900

def load_pose_with_angle(file_path):
    xyz, angles, timestamps = [], [], []
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
                e = [pitch, roll, yaw]
                t = [lon, lat, alt]
                if yaw < 0:
                    yaw += 360
                xyz.append((lon, lat, alt))
                
                timestamps.append(frame_idx)
                R_c2w = euler_angles_to_matrix_ECEF_w2c(e, t)
                angles.append(R_c2w)
    xyz = wgs84tocgcs2000_batch(xyz, 4547)
    return np.array(timestamps), np.array(xyz), np.array(angles)

def load_gt_angle(file_path):
    angles = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                lon, lat, alt = map(float, parts[1:4])
                roll, pitch, yaw = map(float, parts[4:7])
                e = [pitch, roll, yaw]
                t = [lon, lat, alt]
                if yaw < 0:
                    yaw += 360
                R_c2w = euler_angles_to_matrix_ECEF_w2c(e, t)
                angles.append(R_c2w)
    return np.array(angles)
# ✅ 主流程
seq_list = sorted([f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")])

# ✅ 主流程（角度误差）
for seq in seq_list:
    print(f"�� 绘制角度误差随时间变化图：{seq}")
    seq_name = seq.split('.')[0]

    gt_angles_all = load_gt_angle(os.path.join(data_root, "GT", seq))
    frame_ids_all = np.arange(len(gt_angles_all))
    valid_mask = (frame_ids_all >= start_frame) & (frame_ids_all <= end_frame)
    gt_angles = gt_angles_all[valid_mask]
    frame_ids = frame_ids_all[valid_mask]

    fig, ax = plt.subplots(figsize=(10, 6))
    max_y_limit = 2  # 角度误差上限（单位°）
    method_medians = {}

    for method, color in methods.items():
        file_path = os.path.join(data_root, method, seq)
        if not os.path.exists(file_path): continue

        est_frame_ids, _, est_angles = load_pose_with_angle(file_path)

        ang_err = np.full_like(frame_ids, np.nan, dtype=np.float32)
        for i, fid in enumerate(frame_ids):
            if fid in est_frame_ids:
                idx = np.where(est_frame_ids == fid)[0][0]
                cos = np.clip((np.trace(np.dot(gt_angles[fid].T, est_angles[idx])) - 1) / 2, -1., 1.)
                e_R = np.rad2deg(np.abs(np.arccos(cos)))
                ang_err[i] = e_R
            else:
                ang_err[i] = max_y_limit  # 填一个上限

        ang_err_clipped = np.minimum(ang_err, max_y_limit)

        ax.scatter(frame_ids, ang_err_clipped,
                   color=color,
                   label=methods_name[method],
                   alpha=0.7,
                   s=18,
                   zorder=10 if method == "FPVLoc" else 5)

        med = np.nanmedian(ang_err)
        method_medians[method] = med

        ax.axhline(med, linestyle='--', color=color, linewidth=1.2, alpha=0.4,
                   zorder=1 if method != "FPVLoc" else 9)

    ax.set_title(f"Angle Error vs Frame Index ({seq_name})")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Angle Error (°)")
    ax.grid(True)
    ax.set_ylim(0, max_y_limit + 1)

    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for method, label in zip(methods.keys(), labels):
        if method in method_medians:
            new_labels.append(f"{label} (Med: {method_medians[method]:.2f}°)")
        else:
            new_labels.append(label)
    ax.legend(handles, new_labels)

    fig.tight_layout()
    output_path = os.path.join(data_root, "outputs", f"{seq_name}_angle_error_curve.png")
    fig.savefig(output_path, dpi=300)
    print(f"✅ 已保存至：{output_path}")
    plt.close(fig)