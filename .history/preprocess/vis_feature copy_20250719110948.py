import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrow
from sklearn.decomposition import PCA
import pickle
import torch
# 模拟 torch.load，避免使用 PyTorch（此版本不依赖 torch）
def load_pt_as_dict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
# 特征图转热力图
def feature_to_heatmap(feat: np.ndarray, method='mean', colormap=cv2.COLORMAP_VIRIDIS):
    assert feat.ndim == 3
    C, H, W = feat.shape
    if method == 'mean':
        feat2d = feat.mean(axis=0)
    elif method == 'max':
        feat2d = feat.max(axis=0)
    elif method == 'pca':
        feat_flat = feat.reshape(C, -1).T
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(feat_flat).reshape(H, W)
        feat2d = pc1
    else:
        raise ValueError("Invalid method")

    feat2d = (feat2d - feat2d.min()) / (feat2d.max() - feat2d.min() + 1e-6)
    feat2d = (feat2d * 255).astype(np.uint8)
    color = cv2.applyColorMap(feat2d, colormap)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

# 可视化主函数
def visualize_feature_alignment_np(pt_path, save_path):
    data = torch.load(pt_path, map_location="cpu")

    p2d_r = data["p2d_render"].numpy()
    p2d_q0 = data["p2d_query"].numpy()
    p2d_q1 = data["p2d_query_refined"].numpy()
    f_q = data["f_q"].numpy()  # shape: (3, C, H, W)

    levels = p2d_r.shape[0]
    fig = plt.figure(figsize=(6, 10))
    gs = GridSpec(nrows=levels, ncols=1, height_ratios=[1]*levels, hspace=0.03)

    for lvl in range(levels):
        pr = p2d_r[lvl]
        pq0 = p2d_q0[lvl]
        pq1 = p2d_q1[lvl]
        feat_map = f_q[lvl]  # (C, H, W)
        H, W = feat_map.shape

        bg = feature_to_heatmap(feat_map, method='mean')
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_NEAREST)

        ax = fig.add_subplot(gs[lvl])
        ax.imshow(bg)
        ax.axis("off")

        res0 = np.linalg.norm(pr - pq0, axis=1)
        res1 = np.linalg.norm(pr - pq1, axis=1)
        rmse0, rmse1 = np.sqrt((res0**2).mean()), np.sqrt((res1**2).mean())
        vmax = max(res0.max(), res1.max()) + 1e-3
        cmap = plt.cm.get_cmap("viridis")

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
    return save_path

# 批量执行
pt_files = [
    "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/0_feature.pt",
    "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/1_feature.pt",
    "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/2_feature.pt",
]
save_dir = "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs"
os.makedirs(save_dir, exist_ok=True)

output_images = []
for pt_path in pt_files:
    name = os.path.basename(pt_path).replace("_feature.pt", "_vis_feat.png")
    out_path = os.path.join(save_dir, name)
    visualize_feature_alignment_np(pt_path, out_path)
    output_images.append(out_path)

import pandas as pd
import IPython.display as display
df = pd.DataFrame({"Output Image": output_images})
display.display(df)

