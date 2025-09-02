import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from matplotlib.gridspec import GridSpec

# 特征图转彩色图
def feature_to_heatmap(feat: np.ndarray, method='mean', colormap=cv2.COLORMAP_VIRIDIS):
    assert feat.ndim == 3  # (C, H, W)
    C, H, W = feat.shape

    if method == 'mean':
        feat2d = feat.mean(axis=0)
    elif method == 'max':
        feat2d = feat.max(axis=0)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        feat_flat = feat.reshape(C, -1).T  # (H*W, C)
        pc1 = PCA(n_components=1).fit_transform(feat_flat).reshape(H, W)
        feat2d = pc1
    else:
        raise ValueError("method must be 'mean', 'max', or 'pca'")

    feat2d = (feat2d - feat2d.min()) / (feat2d.max() - feat2d.min() + 1e-6)
    feat2d = (feat2d * 255).astype(np.uint8)
    color = cv2.applyColorMap(feat2d, colormap)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

# 主可视化函数（合并3层在一个图中）
def visualize_pyramid_three_levels(pt_paths, save_path):
    assert len(pt_paths) == 3, "必须提供3个金字塔层.pt文件"

    # 创建3层画布
    fig = plt.figure(figsize=(6, 10))
    gs = GridSpec(nrows=3, ncols=1, height_ratios=[1, 1, 1], hspace=0.03)

    for lvl, pt_path in enumerate(pt_paths):
        data = torch.load(pt_path, map_location="cpu")
        p2d_r = data["p2d_render"].numpy()
        p2d_q0 = data["p2d_query"][0].numpy()
        p2d_q1 = data["p2d_query_refined"][0].numpy()
        feat_map = data["f_q"].numpy()  # shape: (H, W)

        C, H, W = feat_map.shape
        bg = feature_to_heatmap(feat_map)

        pr = p2d_r
        pq0 = p2d_q0
        pq1 = p2d_q1

        res0 = np.linalg.norm(pr - pq0, axis=1)
        res1 = np.linalg.norm(pr - pq1, axis=1)
        rmse0 = np.sqrt((res0 ** 2).mean())
        rmse1 = np.sqrt((res1 ** 2).mean())
        vmax = max(res0.max(), res1.max()) + 1e-3
        cmap = plt.cm.get_cmap("viridis")

        ax = fig.add_subplot(gs[lvl])
        ax.imshow(bg)
        ax.axis("off")

        for (x0, y0), (x1, y1), (x2, y2), r0, r1 in zip(pr, pq0, pq1, res0, res1):
            color0 = cmap(r0 / vmax)
            color1 = cmap(r1 / vmax)
            ax.add_patch(FancyArrow(x0, y0, x1 - x0, y1 - y0,
                                    width=0.5, color=color0, alpha=0.9))
            ax.add_patch(FancyArrow(x0, y0, x2 - x0, y2 - y0,
                                    width=0.3, color=color1, alpha=0.4))

        ax.set_title(f"L{lvl}  RMSE: {rmse0:.2f} → {rmse1:.2f}", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"✅ 已保存可视化图: {save_path}")

# 路径配置
pt_files = [
    "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/0_feature.pt",
    "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/1_feature.pt",
    "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/2_feature.pt",
]
save_path = "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs/pyramid_all_levels.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 执行绘图
visualize_pyramid_three_levels(pt_files, save_path)
