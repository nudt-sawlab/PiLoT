import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import rcParams
from scipy.spatial.transform import Rotation as Rr

# 你项目里的坐标/投影工具
from transform import wgs84tocgcs2000_batch, get_rotation_enu_in_ecef, WGS84_to_ECEF

# ========== Matplotlib & Seaborn 样式 ==========
rcParams['font.family'] = 'serif'
rcParams['axes.labelweight'] = 'bold'
sns.set_context("talk")
sns.set_style("whitegrid")

# ========== 基础工具 ==========
def transform_points(points, scale, R, t):
    return scale * (R @ points.T).T + t

def umeyama_alignment(src, dst):
    """刚体+尺度配准（用于某些方法轨迹对齐，可按需启用）"""
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

def euler_angles_to_matrix_ECEF_c2w(euler_angles_deg, trans_llh):
    """与你之前一致：先机体系欧拉→ENU，再 ENU→ECEF，得到 R_c2w（注：后续用了相对旋转角）"""
    lon, lat, _ = trans_llh
    rot_pose_in_enu = Rr.from_euler('xyz', euler_angles_deg, degrees=True).as_matrix()
    rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = rot_enu_in_ecef @ rot_pose_in_enu
    return R_c2w

# ========== 方法与配色 ==========
# 注意：methods_label 要与 data_root 下的子文件夹名对应
methods_label = ["GeoPixel", "Render2Loc", "PixLoc", "Render2RAFT", "Render2ORB"]

methods_name = {
    "GT": "GT",
    "GeoPixel": "GeoPixel",
    "Render2Loc": "Render2Loc",
    "PixLoc": "PixLoc",
    "Render2RAFT": "Render2RAFT",
    "Render2ORB": "Render2ORB",
}
method_colors = [
    "#007F49",  # GeoPixel
    "#EF6C5D",  # Render2Loc
    "#86AED5",  # PixLoc
    "#F7B84A",  # Render2RAFT
    "#C79ACD",  # Render2ORB
]

# ========== 配置 ==========
data_root = "/mnt/sda/MapScape/query/estimation/result_images/Google"
seq = "USA_seq5@8@cloudy@300-100@200.txt"
outputs = os.path.join(data_root, "outputs")
os.makedirs(outputs, exist_ok=True)

# 误差对数显示的下界（避免 -inf）
LOG_MIN_POSERR = 1e-1   # m
LOG_MIN_ANGERR = 2e-2   # deg

# ========== 读取 GT ==========
gt_file = os.path.join(data_root, "GT", seq)
with open(gt_file) as f:
    gt_data = [l.strip().split() for l in f if len(l.strip().split()) >= 7]

# gt: 列约定 [name, lon, lat, alt, roll, pitch, yaw]
gt_llh_list = [tuple(map(float, d[1:4])) for d in gt_data]
gt_xyz = wgs84tocgcs2000_batch(gt_llh_list, 4547)  # 到 CGCS2000（或你常用的投影米系）
# 欧拉角顺序与你之前保持一致：[pitch, roll, yaw(>=0)]
gt_angles = np.array([
    (float(d[5]), float(d[4]), float(d[6]) if float(d[6]) >= 0 else float(d[6]) + 360.0)
    for d in gt_data
], dtype=np.float64)

# ========== 读取相邻帧 IoU 并三档分区 ==========
iou_csv_path = "/mnt/sda/MapScape/query/depth/USA_seq5@8@cloudy@300-100@200/pair_iou3d.csv"
pair_idx, depth0, depth1, ious = [], [], [], []
with open(iou_csv_path, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        pair_idx.append(int(row["pair_idx"]))
        depth0.append(row["depth0"])
        depth1.append(row["depth1"])
        ious.append(float(row["iou3d"]))
ious = np.array(ious, dtype=np.float64)

# 等宽三档：[e0,e1), [e1,e2), [e2,e3]（最后档右闭，含最大值）
vmin, vmax = float(np.min(ious)), float(np.max(ious))
if np.isclose(vmin, vmax, atol=1e-12):
    eps = 1e-6
    vmin, vmax = vmin - eps, vmax + eps
edges = np.linspace(vmin, vmax, 4)
e0, e1, e2, e3 = edges.tolist()
bin_ids = np.where(ious < e1, 0, np.where(ious < e2, 1, 2))
bin_ids[ious == vmax] = 2
# pair_idx -> bin 的映射（pair i 表示 i 与 i+1 帧）
pair_to_bin = {pair_idx[i]: int(bin_ids[i]) for i in range(len(pair_idx))}
iou_bin_labels = [f"[{e0:.2f},{e1:.2f})", f"[{e1:.2f},{e2:.2f})", f"[{e2:.2f},{e3:.2f}]"]

print("\nIoU3D 等宽三区间：")
print(f"Bin 1 低:  [{e0:.4f}, {e1:.4f}) -> {(bin_ids==0).sum()} 对")
print(f"Bin 2 中:  [{e1:.4f}, {e2:.4f}) -> {(bin_ids==1).sum()} 对")
print(f"Bin 3 高:  [{e2:.4f}, {e3:.4f}] -> {(bin_ids==2).sum()} 对")

# ========== 统计容器（每方法 × 3 桶）==========
NUM_IOU_BINS = 3
pos_err_per_bin = {m: [[] for _ in range(NUM_IOU_BINS)] for m in methods_label}
ang_err_per_bin = {m: [[] for _ in range(NUM_IOU_BINS)] for m in methods_label}

# ========== 逐方法读取估计轨迹，计算误差并按 IoU 桶入箱 ==========
for method in methods_label:
    method_file = os.path.join(data_root, method, seq)
    if not os.path.exists(method_file):
        print(f"[Warn] Missing result file for {method}: {method_file}")
        continue

    # 读取估计：列约定与 GT 相同
    with open(method_file) as f:
        est_data = [l.strip().split() for l in f if len(l.strip().split()) >= 7]
    est_llh_list = [tuple(map(float, d[1:4])) for d in est_data]

    # 框架：对齐到 GT 帧数（按帧号写入），空缺帧填 NaN，避免错位
    N = len(gt_data)
    est_xyz = np.full((N, 3), np.nan, dtype=np.float64)
    est_angles = np.full((N, 3), np.nan, dtype=np.float64)

    for d, llh in zip(est_data, est_llh_list):
        # 从文件名里取帧号，支持形如 "123_0.png" 或 "123"
        name = d[0]
        m = re.match(r"^(\d+)(?:_.*)?$", name)
        if not m:
            continue
        frame_idx = int(m.group(1))
        if frame_idx < 0 or frame_idx >= N:
            continue

        xyz = wgs84tocgcs2000_batch([llh], 4547)[0]
        est_xyz[frame_idx] = xyz

        roll, pitch, yaw = float(d[4]), float(d[5]), float(d[6])
        if yaw < 0: yaw += 360.0
        est_angles[frame_idx] = (pitch, roll, yaw)

    # 有些方法（如 ORB）可能需要对位姿做相似变换对齐，这里按需开关
    # 仅对有足够匹配帧的情况做 Umeyama
    if method.startswith("ORB"):
        valid_mask = ~np.isnan(est_xyz).any(axis=1)
        idxs = np.where(valid_mask)[0]
        if len(idxs) >= 10:
            scale, Rm, tm = umeyama_alignment(est_xyz[idxs], gt_xyz[idxs])
            est_xyz = transform_points(np.nan_to_num(est_xyz, nan=0.0), scale, Rm, tm)
            # 对齐仅影响平移误差；角度误差仍基于各自姿态计算

    # 计算误差（与 pair i 对齐使用 i→i+1 的 IoU 档）
    # 只统计相邻帧误差（1..N-1），并忽略缺失帧
    pos_err = []
    ang_err = []
    bins_for_this_method = []
    POS_ERR_MAX = 10.0  # 平移误差上限（m），缺失就记这个
    ANG_ERR_MAX = 10.0  # 角度误差上限（deg），缺失就记这个
    N = len(gt_data)
    for i in range(1, N):
        # 该误差样本对应的 IoU 桶（pair i-1）
        b = pair_to_bin.get(i-1, None)
        if b is None or not (0 <= b < NUM_IOU_BINS):
            continue  # 该对缺 IoU 的，跳过

        # 平移误差
        if np.any(np.isnan(gt_xyz[i])):
            continue  # GT 缺失直接跳过（一般不会）
        if np.any(np.isnan(est_xyz[i])):
            pe = POS_ERR_MAX  # 缺失→最大值
        else:
            pe = float(np.linalg.norm(est_xyz[i] - gt_xyz[i]))
            if not np.isfinite(pe):
                pe = POS_ERR_MAX
            pe = min(pe, POS_ERR_MAX)  # 上限裁剪

        # 角度误差
        llh_i = gt_llh_list[i]
        R_gt  = euler_angles_to_matrix_ECEF_c2w(gt_angles[i], llh_i)
        if np.any(np.isnan(est_angles[i])):
            ae = ANG_ERR_MAX  # 缺失→最大值
        else:
            try:
                R_est = euler_angles_to_matrix_ECEF_c2w(est_angles[i], llh_i)
                cosang = np.clip((np.trace(R_gt.T @ R_est) - 1.0) * 0.5, -1.0, 1.0)
                ae = float(np.degrees(np.arccos(cosang)))
            except Exception:
                ae = ANG_ERR_MAX
            if not np.isfinite(ae):
                ae = ANG_ERR_MAX
            ae = min(ae, ANG_ERR_MAX)

        # 你之前的特例微调（可选，保留原样或删除）
        # if 'raft' in method and (pe < POS_ERR_MAX) and (pe > 3):
        #     pe = max(pe - 5, LOG_MIN_POSERR)

        # 入桶
        pos_err_per_bin[method][b].append(max(pe, LOG_MIN_POSERR))
        ang_err_per_bin[method][b].append(max(ae, LOG_MIN_ANGERR))

    pos_err = np.asarray(pos_err, dtype=np.float64)
    ang_err = np.asarray(ang_err, dtype=np.float64)
    bins_for_this_method = np.asarray(bins_for_this_method)

    # 过滤异常与对数下界裁剪（绘图稳定）
    pos_err = np.clip(pos_err, LOG_MIN_POSERR, np.inf)
    ang_err = np.clip(ang_err, LOG_MIN_ANGERR, np.inf)

    for b in range(NUM_IOU_BINS):
        sel = (bins_for_this_method == b)
        pos_err_per_bin[method][b].extend(pos_err[sel].tolist())
        ang_err_per_bin[method][b].extend(ang_err[sel].tolist())

# ========== 绘图函数：箱线图 + 稀疏散点（对数 y 轴）==========
def plot_err_vs_iou(ax, data_dict, ylabel, log_min, method_colors):
    plot_data, plot_labels, plot_method = [], [], []
    for m_idx, method in enumerate(methods_label):
        pretty = methods_name.get(method, method)
        for b in range(NUM_IOU_BINS):
            vals = np.array(data_dict[method][b], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 2:
                plot_data.extend(np.log10(vals))
                plot_labels.extend([["Low","Mid","High"][b]] * len(vals))
                plot_method.extend([pretty] * len(vals))

    if len(plot_data) == 0:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", fontsize=14)
        ax.axis('off')
        return

    df = pd.DataFrame({"log_err": plot_data, "iou_bin": plot_labels, "method": plot_method})

    # boxplot
    sns.boxplot(
        x="iou_bin", y="log_err", hue="method", data=df,
        palette=method_colors, fliersize=0, linewidth=1.6, dodge=True, ax=ax,
        medianprops=dict(color="black", linewidth=2.0),
        boxprops=dict(edgecolor="black", linewidth=2.0),
        whiskerprops=dict(color="black", linewidth=2.0),
        capprops=dict(color="black", linewidth=2.0),
    )
    # jitter points
    sns.stripplot(
        x="iou_bin", y="log_err", hue="method", data=df,
        dodge=True, jitter=0.15, marker="o", alpha=0.08, size=2.8,
        palette=method_colors, ax=ax, legend=False
    )

    # 轴与标题
    ax.set_xlabel("IoU Bin", fontsize=15)
    if "Position" in ylabel:
        ax.set_ylabel(r"$\log_{10}(\Vert \Delta \mathbf{p} \Vert)$ (m)", fontsize=16)
        ax.set_title("Position Error vs IoU", fontsize=16)
    else:
        ax.set_ylabel(r"$\log_{10}(\Delta \theta)$ (deg)", fontsize=16)
        ax.set_title("Angle Error vs IoU", fontsize=16)

    ax.set_ylim(np.log10(log_min), None)
    ax.tick_params(labelsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    sns.despine(ax=ax)
    ax.legend(loc="upper left", fontsize=12, frameon=True)

# ========== 出图 ==========
seq_name = os.path.splitext(seq)[0]

fig1, ax1 = plt.subplots(figsize=(10, 6.2))
plot_err_vs_iou(ax1, pos_err_per_bin, "Position Error (m)", LOG_MIN_POSERR, method_colors)
plt.tight_layout()
plt.savefig(os.path.join(outputs, f"{seq_name}_poserr_vs_iou.png"), dpi=330)

fig2, ax2 = plt.subplots(figsize=(10, 6.2))
plot_err_vs_iou(ax2, ang_err_per_bin, "Angle Error (deg)", LOG_MIN_ANGERR, method_colors)
plt.tight_layout()
plt.savefig(os.path.join(outputs, f"{seq_name}_angerr_vs_iou.png"), dpi=330)

print("✅ Saved:",
      os.path.join(outputs, f"{seq_name}_poserr_vs_iou.png"),
      os.path.join(outputs, f"{seq_name}_angerr_vs_iou.png"))
