#!/usr/bin/env python3
# visual_pyramid_simple.py
# --------------------------------------------
# 三层金字塔误差可视化（去掉 argparse）
# Author : ChatGPT (2025‑07)
# --------------------------------------------

import cv2, torch, numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
def feature_to_heatmap(feat: np.ndarray, method='mean', colormap=cv2.COLORMAP_VIRIDIS):
    """
    将 (C, H, W) 的特征图可视化为 RGB 彩图。
    - method: 'mean' | 'max' | 'pca'
    """
    assert feat.ndim == 3
    C, H, W = feat.shape
    feat2d = None

    if method == 'mean':
        feat2d = feat.mean(axis=0)  # (H, W)
    elif method == 'max':
        feat2d = feat.max(axis=0)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        feat_flat = feat.reshape(C, -1).T  # (H*W, C)
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(feat_flat).reshape(H, W)
        feat2d = pc1
    else:
        raise ValueError("method should be 'mean', 'max', or 'pca'")

    # normalize
    feat2d = (feat2d - feat2d.min()) / (feat2d.max() - feat2d.min() + 1e-6)
    feat2d = (feat2d * 255).astype(np.uint8)
    color = cv2.applyColorMap(feat2d, colormap)  # shape: (H, W, 3)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)  # 转为 RGB 顺序
# ================================================================
# 0. 配置区 —— 只改这里就够了
# ================================================================
PT_FILE      = "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/0_feature.pt"
QUERY_IMG    = None#"/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/query_8.png"  # 若没有查询图，置 None
SAVE_PATH    = "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/vis_0.png"          # 若想直接弹窗而不保存，置 None
SHOW_WINDOW  = False                # True -> 显示窗口；False -> 直接保存
DPI          = 150
ARROW_SCALE  = 1.0                  # 全局箭头长度缩放

# ================================================================
# 1. 读数据
# ================================================================
data = torch.load(PT_FILE, map_location="cpu")
p2d_r  = data["p2d_render"         ].cpu().numpy()   # (3,L,N,2) 或 (3,N,2)
p2d_q0 = data["p2d_query"          ].cpu().numpy()
p2d_q1 = data["p2d_query_refined"  ].cpu().numpy()
n_lvl  = p2d_r.shape[0]

assert n_lvl == 3, "脚本默认正好 3 层金字塔！"

# ================================================================
# 2. 背景图像（可选）
# ================================================================
qimg = None
if QUERY_IMG:
    qimg = cv2.cvtColor(cv2.imread(QUERY_IMG), cv2.COLOR_BGR2RGB)

H, W = (qimg.shape[:2] if qimg is not None else (720, 1280))

# ================================================================
# 3. 画布
# ================================================================
fig = plt.figure(figsize=(6, 10), dpi=DPI)
gs  = GridSpec(nrows=3, ncols=1, height_ratios=[1, 1, 1], hspace=0.03)
cmap = plt.get_cmap("viridis")

# 映射：层号 → 缩放倍数
scales = [0.25, 0.5, 1.0]

def draw_layer(ax, lvl, scale):
    # ----- 背景 -----
    # 背景使用特征图可视化
    Fq = data["f_q"].cpu().numpy()  # shape: (3, C, H, W)
    feat_map = Fq[lvl]              # 当前层 (C, H, W)
    bg_feat  = feature_to_heatmap(feat_map, method='mean')  # shape (H, W, 3)

    # resize 到原图分辨率
    bg = cv2.resize(bg_feat, (W, H), interpolation=cv2.INTER_NEAREST)
    ax.imshow(bg)
    ax.axis("off")

    pr, q0, q1 = p2d_r[lvl], p2d_q0[lvl], p2d_q1[lvl]
    res0       = np.linalg.norm(pr - q0, axis=1)
    res1       = np.linalg.norm(pr - q1, axis=1)
    rmse0, rmse1 = np.sqrt((res0**2).mean()), np.sqrt((res1**2).mean())

    vmax     = max(res0.max(), res1.max()) + 1e-3
    colors0  = cmap(res0 / vmax)
    colors1  = cmap(res1 / vmax)

    sz_pt = 20 / scale
    lw    = 2  / scale

    ax.scatter(pr[:, 0], pr[:, 1], c=colors0, s=sz_pt, marker='o')
    ax.scatter(q0[:, 0], q0[:, 1], c=colors0, s=sz_pt, marker='x')
    ax.scatter(q1[:, 0], q1[:, 1], c=colors1, s=sz_pt, marker='+', alpha=0.4)

    for (x0, y0), (x1, y1), (x2, y2), c0, c1 in zip(pr, q0, q1, colors0, colors1):
        ax.arrow(x0, y0, (x1 - x0) * ARROW_SCALE, (y1 - y0) * ARROW_SCALE,
                 head_width=6/scale, head_length=8/scale, color=c0,
                 length_includes_head=True, lw=lw)
        ax.arrow(x0, y0, (x2 - x0) * ARROW_SCALE, (y2 - y0) * ARROW_SCALE,
                 head_width=4/scale, head_length=6/scale, color=c1,
                 length_includes_head=True, lw=lw/1.5, alpha=0.4)

    ax.text(10, 30, f"L{lvl}  (×{scale:.2f})\nRMSE: {rmse0:.2f} → {rmse1:.2f}px",
            color="white", fontsize=11, weight="bold",
            bbox=dict(fc="black", alpha=0.5, pad=3))

# 绘制三层
for lvl, sc in enumerate(scales):
    ax = fig.add_subplot(gs[lvl])
    draw_layer(ax, lvl, sc)

# ================================================================
# 4. 输出
# ================================================================
if SAVE_PATH and not SHOW_WINDOW:
    fig.savefig(SAVE_PATH, bbox_inches="tight")
    print("✅ 结果已保存到", SAVE_PATH)
else:
    plt.show()
