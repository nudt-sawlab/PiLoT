import os
import cv2
import numpy as np

# ========= 相机与投影工具 =========
def warp_ref_to_query(ref_img, ref_depth, ref_T_c2w, rcamera, qry_T_c2w, qcamera, stride=1):
    """
    用 ref 深度将 ref_img 前向重投影到 query 视图（双线性splat + z加权）。
    返回：warp_img (Hq,Wq,3, uint8), valid_mask (Hq,Wq, bool)
    rcamera/qcamera: [W, H, fx, fy, cx, cy]
    """
    Hr, Wr = ref_img.shape[:2]
    Wq, Hq, fx_q, fy_q, cx_q, cy_q = qcamera
    Wq, Hq = int(Wq), int(Hq)
    fx_r, fy_r, cx_r, cy_r = rcamera[2], rcamera[3], rcamera[4], rcamera[5]

    # 采样网格（可调 stride）
    us = np.arange(0, Wr, stride, dtype=np.float32)
    vs = np.arange(0, Hr, stride, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    uu_f = uu.ravel()
    vv_f = vv.ravel()

    z = ref_depth[vv_f.astype(np.int64), uu_f.astype(np.int64)].astype(np.float32)
    valid_z = z > 1e-6
    uu_f, vv_f, z = uu_f[valid_z], vv_f[valid_z], z[valid_z]

    # ref 相机坐标
    Xc = np.stack([(uu_f - cx_r) / fx_r * z,
                   (vv_f - cy_r) / fy_r * z,
                   z], axis=0)  # (3,N)

    Rr = ref_T_c2w[:3, :3].astype(np.float32)
    tr = ref_T_c2w[:3, 3].astype(np.float32).reshape(3,1)
    Rq = qry_T_c2w[:3, :3].astype(np.float32)
    tq = qry_T_c2w[:3, 3].astype(np.float32).reshape(3,1)

    # 相机->世界->Query相机
    Xw = Rr @ Xc + tr              # (3,N)
    Xq = Rq.T @ (Xw - tq)          # (3,N)

    Zq = Xq[2, :]
    front = Zq > 1e-6
    Xq, uu_f, vv_f, Zq = Xq[:, front], uu_f[front], vv_f[front], Zq[front]

    uq = fx_q * Xq[0, :] / Zq + cx_q
    vq = fy_q * Xq[1, :] / Zq + cy_q

    # 双线性 splat（目标像素四邻域）
    x0 = np.floor(uq).astype(np.int64)
    y0 = np.floor(vq).astype(np.int64)
    inb = (x0 >= 0) & (x0 < Wq-1) & (y0 >= 0) & (y0 < Hq-1)
    x0, y0, uq, vq, Zq = x0[inb], y0[inb], uq[inb], vq[inb], Zq[inb]

    dx = uq - x0
    dy = vq - y0

    # 源颜色（整数像素，若 stride=1 等价 NN；够用且快）
    src_cols = ref_img[vv_f[inb].astype(np.int64), uu_f[inb].astype(np.int64), :].astype(np.float32)

    out = np.zeros((Hq, Wq, 3), np.float32)
    wgt = np.zeros((Hq, Wq), np.float32)
    z_w = 1.0 / (Zq * Zq + 1e-6)  # 更近权重大

    def add_splat(xi, yi, w):
        for c in range(3):
            np.add.at(out[:,:,c], (yi, xi), src_cols[:, c] * w * z_w)
        np.add.at(wgt, (yi, xi), w * z_w)

    add_splat(x0,   y0,   (1-dx)*(1-dy))
    add_splat(x0+1, y0,   dx*(1-dy))
    add_splat(x0,   y0+1, (1-dx)*dy)
    add_splat(x0+1, y0+1, dx*dy)

    valid = wgt > 0
    out[valid] /= wgt[valid, None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, valid

def to_gray01(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32) / 255.0

def ssim_map(I1, I2, win=11, sigma=1.5):
    # I1/I2: gray [0,1]
    k = int(win) + (int(win) % 2 == 0)
    mu1 = cv2.GaussianBlur(I1, (k, k), sigma)
    mu2 = cv2.GaussianBlur(I2, (k, k), sigma)
    mu1_2, mu2_2, mu1mu2 = mu1*mu1, mu2*mu2, mu1*mu2
    s1_2 = cv2.GaussianBlur(I1*I1, (k,k), sigma) - mu1_2
    s2_2 = cv2.GaussianBlur(I2*I2, (k,k), sigma) - mu2_2
    s12  = cv2.GaussianBlur(I1*I2, (k,k), sigma) - mu1mu2
    C1, C2 = (0.01**2), (0.03**2)
    num = (2*mu1mu2 + C1) * (2*s12 + C2)
    den = (mu1_2 + mu2_2 + C1) * (s1_2 + s2_2 + C2) + 1e-12
    ssim = num / den
    return np.clip(ssim, -1, 1)  # 理论上[0,1]，此处稳妥裁剪

def ncc_map(I1, I2, win=11, sigma=1.5):
    k = int(win) + (int(win) % 2 == 0)
    mu1 = cv2.GaussianBlur(I1, (k,k), sigma)
    mu2 = cv2.GaussianBlur(I2, (k,k), sigma)
    s1_2 = cv2.GaussianBlur(I1*I1, (k,k), sigma) - mu1*mu1
    s2_2 = cv2.GaussianBlur(I2*I2, (k,k), sigma) - mu2*mu2
    s12  = cv2.GaussianBlur(I1*I2, (k,k), sigma) - mu1*mu2
    den = np.sqrt(np.maximum(s1_2, 0)) * np.sqrt(np.maximum(s2_2, 0)) + 1e-12
    ncc = s12 / den
    return np.clip(ncc, -1, 1)

def normalize01(x, mask=None, p_low=1, p_high=99):
    x = x.astype(np.float32)
    vals = x[mask] if mask is not None else x.ravel()
    if vals.size == 0:
        return np.zeros_like(x, np.float32)
    lo, hi = np.percentile(vals, p_low), np.percentile(vals, p_high)
    if hi - lo < 1e-6:
        return np.zeros_like(x, np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1)

def heatmap_with_contours(error01, valid_mask=None, levels=(0.1,0.3,0.5,0.7,0.9)):
    err = np.where(valid_mask, error01, 0.0) if valid_mask is not None else error01
    heat = cv2.applyColorMap((err * 255).astype(np.uint8), cv2.COLORMAP_TURBO)  # 暖色=大
    for lv in levels:
        th = (err >= lv).astype(np.uint8) * 255
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(heat, contours, -1, (255,255,255), 1, cv2.LINE_AA)
    return heat

def overlay_binary_mask(base_bgr, mask01, color=(0,0,255), alpha=0.5):
    out = base_bgr.copy()
    m = (mask01 > 0.5)
    overlay = np.zeros_like(out, np.uint8)
    overlay[m] = color
    out[m] = (out[m].astype(np.float32) * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)
    return out

# ========= 你的主循环里替换为下面这段 =========
# 路径/相机参数按你的实际
methods = ['PixLoc']  # 你也可以循环多个算法目录
points_npy = "/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif/vis_video_with_error/clicked_points.npy"

for method in methods:
    ref_dir   = "/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif/GT"         # Render/Ref
    query_dir = os.path.join("/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif/", method)  # Query
    output_path = os.path.join("/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif/vis/")
    os.makedirs(output_path, exist_ok=True)

    query_pose_file = os.path.join(query_dir, "USA_seq5@8@foggy@200.txt")
    ref_pose_file   = os.path.join(ref_dir,   "USA_seq5@8@cloudy@300-100@200.txt")

    # 你已实现的载入位姿函数：返回 c2w 齐次矩阵列表
    from transform import get_matrix, get_rotation_enu_in_ecef, WGS84_to_ECEF
    from scipy.spatial.transform import Rotation as R

    def euler_angles_to_matrix_ECEF_w2c(euler_angles, trans):
        lon, lat, _ = trans
        rot_pose_in_enu = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()
        rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
        R_c2w = np.matmul(rot_enu_in_ecef, rot_pose_in_enu)
        return R_c2w

    def load_pose(file_path):
        timestamps, T_list = [], []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    name = parts[0]
                    frame_idx = int(name.split("_")[0]) if "_" in name else len(timestamps)
                    lon, lat, alt = map(float, parts[1:4])
                    roll, pitch, yaw = map(float, parts[4:7])
                    e = [pitch, roll, yaw]  # 你原来的顺序
                    t = [lon, lat, alt]
                    T = get_matrix(t, e, origin=[0,0,0], mode='c2w')  # 相机到ECEF
                    timestamps.append(frame_idx)
                    T_list.append(T)
        return np.array(timestamps), np.array(T_list)

    gt_ts,  gt_T_list  = load_pose(ref_pose_file)
    es_ts,  es_T_list  = load_pose(query_pose_file)

    # 相机内参（按你代码）
    rcamera = [1920, 1080, 2317.6, 2317.6, 960.0, 540.0]
    qcamera = [960,  540, 1158.8, 1158.8, 480.0, 270.0]

    start_idx, end_idx = 775, 775
    # 重投影的采样步长（速度-质量权衡，1=最密，2=快很多）
    splat_stride = 1

    for i in range(start_idx, end_idx + 1):
        ref_path  = os.path.join(ref_dir,   f"{i}_0.png")
        qry_path  = os.path.join(query_dir, f"{i}_0.png")
        ref_img = cv2.imread(ref_path)
        qry_img = cv2.imread(qry_path)
        if ref_img is None or qry_img is None:
            print(f"[跳过] 帧 {i}：缺图")
            continue

        # 载入与 Ref 对应的深度（与ref_img同分辨率）
        ref_depth = np.load(ref_path[:-4] + ".npy").astype(np.float32)

        # 取对应位姿（c2w）
        try:
            ref_T = gt_T_list[np.where(gt_ts == i)][0]
            qry_T = es_T_list[np.where(es_ts == i)][0]
        except Exception as e:
            print(f"[跳过] 帧 {i}：位姿缺失 {e}")
            continue

        # === 1) 将 Render/Ref 重投影到 Query 视图 ===
        warp_ref, valid_mask = warp_ref_to_query(ref_img, ref_depth, ref_T, rcamera, qry_T, qcamera, stride=splat_stride)

        # === 2) 计算误差图（三种） ===
        g_q = to_gray01(cv2.resize(qry_img, (qcamera[0], qcamera[1])))
        g_w = to_gray01(warp_ref)

        # |I1 - warp(I2)|
        e_l1 = np.abs(g_q - g_w)
        e_l1 = normalize01(e_l1, mask=valid_mask)

        # NCC -> 误差： (1 - NCC)/2
        ncc = ncc_map(g_q, g_w, win=11, sigma=1.5)
        e_ncc = normalize01((1.0 - ncc) * 0.5, mask=valid_mask)

        # SSIM -> 误差： 1 - (ssim_norm) ，其中 ssim_norm=(ssim+1)/2
        ssim = ssim_map(g_q, g_w, win=11, sigma=1.5)
        e_ssim = normalize01(1.0 - (ssim + 1.0) * 0.5, mask=valid_mask)

        # === 3) 热力图 + 等值线（暖色=误差大） ===
        heat_l1  = heatmap_with_contours(e_l1,  valid_mask)
        heat_ncc = heatmap_with_contours(e_ncc, valid_mask)
        heat_ssi = heatmap_with_contours(e_ssim,valid_mask)

        # === 4) 误差>阈值 二值红色高亮（以 SSIM 误差为例） ===
        THRESH = 0.5  # 你可以改
        bin_mask = (e_ssim >= THRESH).astype(np.float32)
        red_highlight = overlay_binary_mask(cv2.resize(qry_img, (qcamera[0], qcamera[1])), bin_mask, color=(0,0,255), alpha=0.5)

        # 保存
        base = os.path.join(output_path, f"{i}_{method}")
        cv2.imwrite(base + "_warpRef2Qry.png", warp_ref)
        cv2.imwrite(base + "_heat_L1.png",     heat_l1)
        cv2.imwrite(base + "_heat_NCC.png",    heat_ncc)
        cv2.imwrite(base + "_heat_SSIM.png",   heat_ssi)
        cv2.imwrite(base + f"_mask_gt{THRESH:.2f}.png", red_highlight)

        print(f"✔ 帧 {i} 输出到 {base}_*.png")

print("✅ 全部完成")
