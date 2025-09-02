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

# ===== 固定尺度的像素误差可视化工具 =====
PIX_VMAX = 5.0  # 全局统一上限（像素）。改成你要的值，如 10.0
CONTOUR_LEVELS_PX = [1.0, 2.0, 3.0, 5.0]  # 固定像素等值线

def heatmap_fixed_scale_px(err_px, vmax_px, valid_mask=None, levels_px=(1,2,3,5)):
    """
    err_px:   像素误差（绝对量纲）
    vmax_px:  统一上限（像素）
    返回BGR热力图（暖色=大）并叠加“固定像素阈值”的等值线
    """
    vmax = float(max(vmax_px, 1e-6))
    # 固定尺度映射：0..vmax → 0..1，并裁到 [0,1]
    err01 = np.clip(err_px / vmax, 0.0, 1.0).astype(np.float32)

    heat = cv2.applyColorMap((err01 * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    # 可选：遮蔽无效区
    if valid_mask is not None:
        m = (~valid_mask).astype(np.uint8) * 255
        heat[m.astype(bool)] = (0, 0, 0)  # 无效区域设为黑色

    # 固定像素阈值的等值线
    for lv_px in levels_px:
        th = (err_px >= lv_px).astype(np.uint8) * 255
        if valid_mask is not None:
            th[~valid_mask] = 0
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(heat, contours, -1, (255,255,255), 1, cv2.LINE_AA)

    return heat

def draw_colorbar_legend(heat_img, vmax_px, ticks_px=(0,1,2,3,5), size=(220, 14), margin=20, loc='br'):
    """
    在图上画一条固定尺度的颜色条（OpenCV实现，不依赖matplotlib）
    loc: 'br'右下, 'tr'右上, 'bl'左下, 'tl'左上
    """
    H, W = heat_img.shape[:2]
    bar_w, bar_h = size
    # 渐变 [0..1]
    grad = np.linspace(0, 1, bar_w, dtype=np.float32)[None, :].repeat(bar_h, axis=0)
    bar = cv2.applyColorMap((grad * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    # 位置
    x0 = W - bar_w - margin if 'r' in loc else margin
    y0 = H - bar_h - margin if 'b' in loc else margin
    heat_img[y0:y0+bar_h, x0:x0+bar_w] = bar

    # 边框
    cv2.rectangle(heat_img, (x0, y0), (x0+bar_w, y0+bar_h), (255,255,255), 1, cv2.LINE_AA)

    # 刻度
    for t in ticks_px:
        x = int(x0 + np.clip(t / vmax_px, 0, 1) * bar_w + 0.5)
        cv2.line(heat_img, (x, y0+bar_h), (x, y0+bar_h+6), (255,255,255), 1, cv2.LINE_AA)
        label = f"{t:g}px"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tx = np.clip(x - tw//2, 0, W-tw-1)
        ty = min(y0+bar_h+2+th, H-3)
        cv2.putText(heat_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    # 标注最大值
    cap = f"0..{vmax_px:g}px"
    (tw, th), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    tx = x0 + bar_w - tw
    ty = y0 - 4 if y0 - 4 > th else y0 + bar_h + th + 6
    cv2.putText(heat_img, cap, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return heat_img
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
def _project_ref_to_query_uv(ref_depth, ref_T_c2w, rcamera, qry_T_c2w, qcamera, stride=1):
    """返回：uq, vq, Zq（长度为N的一维数组），仅包含正深度且在Query视野内的采样"""
    Hr, Wr = ref_depth.shape[:2]
    Wq, Hq, fx_q, fy_q, cx_q, cy_q = qcamera
    Wq, Hq = int(Wq), int(Hq)
    fx_r, fy_r, cx_r, cy_r = rcamera[2], rcamera[3], rcamera[4], rcamera[5]

    us = np.arange(0, Wr, stride, dtype=np.float32)
    vs = np.arange(0, Hr, stride, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    uu = uu.ravel(); vv = vv.ravel()

    z = ref_depth[vv.astype(np.int64), uu.astype(np.int64)].astype(np.float32)
    ok = z > 1e-6
    uu = uu[ok]; vv = vv[ok]; z = z[ok]

    # Ref 相机坐标
    Xc = np.stack([(uu - cx_r) / fx_r * z,
                   (vv - cy_r) / fy_r * z,
                   z], axis=0)  # (3,N)

    Rr = ref_T_c2w[:3, :3].astype(np.float32)
    tr = ref_T_c2w[:3, 3].astype(np.float32).reshape(3,1)
    Rq = qry_T_c2w[:3, :3].astype(np.float32)
    tq = qry_T_c2w[:3, 3].astype(np.float32).reshape(3,1)

    # 世界 -> Query 相机
    Xw = Rr @ Xc + tr
    Xq = Rq.T @ (Xw - tq)
    Zq = Xq[2, :]
    front = Zq > 1e-6
    Xq = Xq[:, front]; Zq = Zq[front]

    uq = fx_q * Xq[0, :] / Zq + cx_q
    vq = fy_q * Xq[1, :] / Zq + cy_q

    inb = (uq >= 0) & (uq < Wq-1) & (vq >= 0) & (vq < Hq-1)
    return uq[inb], vq[inb], Zq[inb]

def _splat_scalar_to_grid(u, v, z, val, W, H):
    """把标量 val（与u,v同长）双线性+近距加权 splat 到 (H,W) 网格"""
    out = np.zeros((H, W), np.float32)
    wgt = np.zeros((H, W), np.float32)

    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    dx = u - x0
    dy = v - y0

    z_w = 1.0 / (z * z + 1e-6)  # 近距权重
    w00 = (1-dx)*(1-dy) * z_w
    w10 = dx*(1-dy)   * z_w
    w01 = (1-dx)*dy   * z_w
    w11 = dx*dy       * z_w

    def add(xi, yi, w):
        np.add.at(out, (yi, xi), val * w)
        np.add.at(wgt, (yi, xi), w)

    add(x0,   y0,   w00)
    add(x0+1, y0,   w10)
    add(x0,   y0+1, w01)
    add(x0+1, y0+1, w11)

    valid = wgt > 0
    out[valid] /= (wgt[valid] + 1e-6)
    return out, valid
def dense_pixel_error_xy_from_ref_depth(ref_depth, ref_T, rcamera,
                                        qry_T_gt, qry_T_est, qcamera,
                                        stride=1):
    """
    返回：
      err_u_map, err_v_map, err_mag_map (Hq,Wq,float32)，valid_mask (Hq,Wq,bool)
    """
    Hr, Wr = ref_depth.shape[:2]
    Wq, Hq, fx_q, fy_q, cx_q, cy_q = qcamera
    Wq, Hq = int(Wq), int(Hq)
    fx_r, fy_r, cx_r, cy_r = rcamera[2], rcamera[3], rcamera[4], rcamera[5]

    # 统一采样：同一批 ref 像素
    us = np.arange(0, Wr, stride, dtype=np.float32)
    vs = np.arange(0, Hr, stride, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    uu = uu.ravel(); vv = vv.ravel()

    z = ref_depth[vv.astype(np.int64), uu.astype(np.int64)].astype(np.float32)
    ok = z > 1e-6
    uu = uu[ok]; vv = vv[ok]; z = z[ok]

    # ref 相机坐标
    Xc = np.stack([(uu - cx_r) / fx_r * z,
                   (vv - cy_r) / fy_r * z,
                   z], axis=0).astype(np.float32)  # (3,N)

    # 公共：世界坐标
    Rr = ref_T[:3, :3].astype(np.float32)
    tr = ref_T[:3, 3].astype(np.float32).reshape(3,1)
    Xw = Rr @ Xc + tr

    # 两个 query 位姿
    def proj_q(T_c2w):
        Rq = T_c2w[:3, :3].astype(np.float32)
        tq = T_c2w[:3, 3].astype(np.float32).reshape(3,1)
        Xq = Rq.T @ (Xw - tq)
        Z  = Xq[2, :]
        u  = fx_q * Xq[0, :] / (Z + 1e-6) + cx_q
        v  = fy_q * Xq[1, :] / (Z + 1e-6) + cy_q
        inb = (Z > 1e-6) & (u >= 0) & (u < Wq-1) & (v >= 0) & (v < Hq-1)
        return u[inb], v[inb], Z[inb], inb

    u_gt, v_gt, z_gt, m_gt   = proj_q(qry_T_gt)
    u_est, v_est, z_est, m_est = proj_q(qry_T_est)

    # 对齐同一批索引（两边都可见）
    # 注意：m_gt/m_est 是在 "ok" 采样子集上的掩码，需要取交集
    inter = m_gt & m_est
    if inter.sum() == 0:
        Hq, Wq = int(qcamera[1]), int(qcamera[0])
        return (np.zeros((Hq, Wq), np.float32),
                np.zeros((Hq, Wq), np.float32),
                np.zeros((Hq, Wq), np.float32),
                np.zeros((Hq, Wq), bool))

    # 将各数组按交集索引提取
    # 为了在同一子集上取值，需要重新投影一次或用布尔索引回取；这里简单做法：重新调用 proj_q 并拿完整值后再布尔筛
    u_gt, v_gt, z_gt, _   = proj_q(qry_T_gt)
    u_est, v_est, z_est, _= proj_q(qry_T_est)

    # 现在 inter 对应的是 ok 子集的掩码；我们需要把 inter 映射到 u_gt/u_est 的同一顺序。
    # 简化：上面 proj_q 每次都基于 Xw(=ok子集)顺序计算并各自 inb；所以用 inter 过滤时保证一一对应：
    u_gt  = u_gt[inter[m_gt]];  v_gt  = v_gt[inter[m_gt]];  z_gt  = z_gt[inter[m_gt]]
    u_est = u_est[inter[m_est]];v_est = v_est[inter[m_est]];z_est = z_est[inter[m_est]]

    # 分量误差（签名）
    du = (u_est - u_gt).astype(np.float32)
    dv = (v_est - v_gt).astype(np.float32)
    dmag = np.sqrt(du*du + dv*dv)

    # 软Z权重
    z_use = np.minimum(z_gt, z_est)
    z_w = 1.0 / (z_use * z_use + 1e-6)

    # 双线性 splat 到 Query 平面（锚点用 GT 坐标）
    def splat_scalar(u, v, val, w, W, H):
        out = np.zeros((H, W), np.float32)
        wgt = np.zeros((H, W), np.float32)
        x0 = np.floor(u).astype(np.int64)
        y0 = np.floor(v).astype(np.int64)
        dx = u - x0; dy = v - y0
        w00 = (1-dx)*(1-dy) * w
        w10 = dx*(1-dy)     * w
        w01 = (1-dx)*dy     * w
        w11 = dx*dy         * w
        def add(xi, yi, wi):
            np.add.at(out, (yi, xi), val * wi)
            np.add.at(wgt, (yi, xi), wi)
        add(x0,   y0,   w00)
        add(x0+1, y0,   w10)
        add(x0,   y0+1, w01)
        add(x0+1, y0+1, w11)
        valid = wgt > 0
        out[valid] /= (wgt[valid] + 1e-6)
        return out, valid

    Hq, Wq = int(qcamera[1]), int(qcamera[0])
    err_u_map, v_u = splat_scalar(u_gt, v_gt, du, z_w, Wq, Hq)
    err_v_map, v_v = splat_scalar(u_gt, v_gt, dv, z_w, Wq, Hq)
    err_mag_map, v_m= splat_scalar(u_gt, v_gt, dmag, z_w, Wq, Hq)
    valid_mask = v_u | v_v | v_m
    return err_u_map, err_v_map, err_mag_map, valid_mask
def dense_pixel_error_map_from_ref_depth(ref_depth, ref_T, rcamera,
                                         qry_T_gt, qry_T_est, qcamera,
                                         stride=1):
    """
    用 Ref 深度生成“GT Flow”和“EST Flow”，误差为它们的像素距离。
    返回：err_px_map (Hq,Wq, float32), valid_mask (Hq,Wq,bool)
    """
    Wq, Hq = int(qcamera[0]), int(qcamera[1])

    # Ref->Query（GT / EST）的投影坐标
    u_gt, v_gt, z_gt   = _project_ref_to_query_uv(ref_depth, ref_T, rcamera, qry_T_gt,  qcamera, stride)
    u_est, v_est, z_est= _project_ref_to_query_uv(ref_depth, ref_T, rcamera, qry_T_est, qcamera, stride)

    # 对齐长度（采样一致；若stride相同、深度同源，索引一一对应）
    n = min(len(u_gt), len(u_est))
    if n == 0:
        return np.zeros((Hq, Wq), np.float32), np.zeros((Hq, Wq), bool)
    u_gt, v_gt, z_gt   = u_gt[:n],  v_gt[:n],  z_gt[:n]
    u_est, v_est, z_est= u_est[:n], v_est[:n], z_est[:n]

    # 每个 3D 点在 Query 上的像素误差
    d_px = np.sqrt((u_est - u_gt)**2 + (v_est - v_gt)**2).astype(np.float32)

    # 用 GT 的深度权（也可用 min(z_gt,z_est)）
    z_use = np.minimum(z_gt, z_est)

    # splat 成 Query 分辨率的稠密误差图
    err_map, valid = _splat_scalar_to_grid(u_gt, v_gt, z_use, d_px, Wq, Hq)
    return err_map, valid
# ====== 组归一化（同一帧对所有算法统一动态范围） ======
P_LOW, P_HIGH = 1, 99  # 组归一化的稳健分位
CONTOUR_LEVELS = (0.2, 0.4, 0.6, 0.8)  # 在归一化后 [0,1] 空间的等值线，仅用于“着色对比”
SSIM_ERR_THRESH = 0.5  # 二值高亮仍用原始 e_ssim（绝对语义）

def group_percentile_range(maps, masks, p_low=1, p_high=99):
    """maps/masks 均为列表，拼接有效像素后取统一分位区间"""
    vecs = []
    for m, mk in zip(maps, masks):
        if mk is not None:
            vec = m[mk]
        else:
            vec = m.ravel()
        if vec.size > 0:
            vecs.append(vec)
    if not vecs:
        return 0.0, 1.0
    allv = np.concatenate(vecs).astype(np.float32)
    lo, hi = np.percentile(allv, [p_low, p_high])
    if hi - lo < 1e-6:
        # 退化保护
        return float(max(lo - 1e-3, 0.0)), float(min(hi + 1e-3, 1.0))
    return float(lo), float(hi)
def overlay_heat_on_query(query_bgr, heat_bgr, valid_mask, alpha=0.6, alpha_map=None):
    """
    将 heat_bgr 半透明叠加到 query_bgr 上（仅在 valid_mask==True 的区域）
    alpha: 常数透明度 [0,1]
    alpha_map: 可选的逐像素透明度（2D [0..1]），若提供会替代 alpha
    """
    Hq, Wq = query_bgr.shape[:2]
    if heat_bgr.shape[:2] != (Hq, Wq):
        heat_bgr = cv2.resize(heat_bgr, (Wq, Hq))
    if valid_mask.shape != (Hq, Wq):
        valid_mask = cv2.resize(valid_mask.astype(np.uint8), (Wq, Hq), interpolation=cv2.INTER_NEAREST).astype(bool)

    base = query_bgr.astype(np.float32)
    over = heat_bgr.astype(np.float32)
    m = valid_mask.astype(np.float32)[..., None]  # HxWx1

    if alpha_map is not None:
        if alpha_map.shape != (Hq, Wq):
            alpha_map = cv2.resize(alpha_map.astype(np.float32), (Wq, Hq))
        a = np.clip(alpha_map, 0.0, 1.0)[..., None] * m
    else:
        a = float(alpha) * m  # 常数透明度

    out = base * (1.0 - a) + over * a
    return np.clip(out, 0, 255).astype(np.uint8)
def norm_to_01(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)
# ========= 你的主循环里替换为下面这段 =========
# 路径/相机参数按你的实际
# methods = ['PixLoc']  # 你也可以循环多个算法目录
methods = ['GeoPixel', 'Render2Loc', 'Render2ORB', 'Render2RAFT', 'PixLoc']
points_npy = "/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif/vis_video_with_error/clicked_points.npy"

for method in methods:
    ref_dir   = "/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif/GT"         # Render/Ref
    query_dir = os.path.join("/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif/", method)  # Query
    output_path = os.path.join("/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif/vis/pix_error_heat_vis")
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

    start_idx, end_idx = 775, 779
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
        # g_q = to_gray01(cv2.resize(qry_img, (qcamera[0], qcamera[1])))
        # g_w = to_gray01(warp_ref)

        # # |I1 - warp(I2)|
        # e_l1 = np.abs(g_q - g_w)
        # e_l1 = normalize01(e_l1, mask=valid_mask)
        
        # # NCC -> 误差： (1 - NCC)/2
        # ncc = ncc_map(g_q, g_w, win=11, sigma=1.5)
        # e_ncc = normalize01((1.0 - ncc) * 0.5, mask=valid_mask)

        # # SSIM -> 误差： 1 - (ssim_norm) ，其中 ssim_norm=(ssim+1)/2
        # ssim = ssim_map(g_q, g_w, win=11, sigma=1.5)
        # e_ssim = normalize01(1.0 - (ssim + 1.0) * 0.5, mask=valid_mask)

        # # === 3) 热力图 + 等值线（暖色=误差大） ===
        # # heat_l1  = heatmap_with_contours(e_l1,  valid_mask)
        # heat_ncc = heatmap_with_contours(e_ncc, valid_mask)
        # heat_ssi = heatmap_with_contours(e_ssim,valid_mask)

        # # === 4) 误差>阈值 二值红色高亮（以 SSIM 误差为例） ===
        # THRESH = 0.5  # 你可以改
        # bin_mask = (e_ssim >= THRESH).astype(np.float32)
        # red_highlight = overlay_binary_mask(cv2.resize(qry_img, (qcamera[0], qcamera[1])), bin_mask, color=(0,0,255), alpha=0.5)

        # # 保存
        # base = os.path.join(output_path, f"{i}_{method}")
        # cv2.imwrite(base + "_warpRef2Qry.png", warp_ref)
        # cv2.imwrite(base + "_heat_L1.png",     heat_l1)
        # cv2.imwrite(base + "_heat_NCC.png",    heat_ncc)
        # cv2.imwrite(base + "_heat_SSIM.png",   heat_ssi)
        # cv2.imwrite(base + f"_mask_gt{THRESH:.2f}.png", red_highlight)
        
        
        err_u, err_v, err_mag, valid = dense_pixel_error_xy_from_ref_depth(
            ref_depth, ref_T, rcamera,
            qry_T_gt=ref_T,          # GT 的 Query 位姿
            qry_T_est=qry_T,       # 估计的 Query 位姿
            qcamera=qcamera,
            stride=1
        )

        # 可视化：模长热力图 + 等值线
        base = os.path.join(output_path, f"{i}_{method}")
        heat_mag = heatmap_fixed_scale_px(err_mag, vmax_px=PIX_VMAX,
                                  valid_mask=valid_mask,
                                  levels_px=CONTOUR_LEVELS_PX)
        heat_mag = draw_colorbar_legend(heat_mag, vmax_px=PIX_VMAX, ticks_px=[0,1,2,3,5], size=(220,14), loc='br')
        # cv2.imwrite(base + "_err_mag_heat.png", heat_mag)

        # 统一像素阈值的二值高亮
        THRESH_PX = 2.0  # 全算法统一
        mask_bin = (err_mag >= THRESH_PX).astype(np.float32)
        red_highlight = overlay_binary_mask(cv2.resize(qry_img, (qcamera[0], qcamera[1])), mask_bin, color=(0,0,255), alpha=0.5)
        # cv2.imwrite(base + f"_err_gt{THRESH_PX:.1f}px.png", red_highlight)

        # 计算像素误差图
        err_u, err_v, err_mag, valid = dense_pixel_error_xy_from_ref_depth(
            ref_depth, ref_T, rcamera,
            qry_T_gt=ref_T,          # GT 的 Query 位姿
            qry_T_est=qry_T,     # 估计位姿
            qcamera=qcamera,
            stride=1
        )

        base = os.path.join(output_path, f"{i}_{method}")

        # 固定尺度热力图（含等值线），用于叠加
        heat_mag = heatmap_fixed_scale_px(
            err_mag, vmax_px=PIX_VMAX, valid_mask=valid, levels_px=CONTOUR_LEVELS_PX
        )

        # 方式A：固定透明度叠加（例如 0.6）
        q_vis = cv2.resize(ref_img, (qcamera[0], qcamera[1]))
        overlay = overlay_heat_on_query(q_vis, heat_mag, valid_mask=valid, alpha=0.6)
        overlay = draw_colorbar_legend(overlay, vmax_px=PIX_VMAX,
                                    ticks_px=[0,1,2,3,5], size=(220,14), loc='br')
        cv2.imwrite(base + "_err_mag_on_query.png", overlay)

        # 方式B：按误差大小自适应透明度（更强调大误差）
        # alpha_map: 将误差归一到 [0,1]（按固定上限 PIX_VMAX），并做gamma增强
        # err01_for_alpha = np.clip(err_mag / float(PIX_VMAX), 0.0, 1.0) ** 0.7  # gamma 可调 0.6~0.8
        # overlay_var = overlay_heat_on_query(q_vis, heat_mag, valid_mask=valid, alpha_map=err01_for_alpha)
        # overlay_var = draw_colorbar_legend(overlay_var, vmax_px=PIX_VMAX,
        #                                 ticks_px=[0,1,2,3,5], size=(220,14), loc='br')
        # cv2.imwrite(base + "_err_mag_on_query_varalpha.png", overlay_var)
        

print("✅ 全部完成")
