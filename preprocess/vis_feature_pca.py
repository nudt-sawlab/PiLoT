import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

def feature_to_heatmap(feat: np.ndarray, method='mean', colormap=cv2.COLORMAP_VIRIDIS):
    assert feat.ndim == 3  # (C, H, W)
    C, H, W = feat.shape
    if method == 'mean':
        feat2d = feat.mean(axis=0)
    elif method == 'max':
        feat2d = feat.max(axis=0)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        feat_flat = feat.reshape(C, -1).T
        pc1 = PCA(n_components=1).fit_transform(feat_flat).reshape(H, W)
        feat2d = pc1
    else:
        raise ValueError("method must be 'mean', 'max', or 'pca'")
    feat2d = (feat2d - feat2d.min()) / (feat2d.max() - feat2d.min() + 1e-6)
    feat2d = (feat2d * 255).astype(np.uint8)
    color = cv2.applyColorMap(feat2d, colormap)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

def crop_from_origin_16_9(img):
    H, W, _ = img.shape
    target_h = int(W * 9 / 16)
    if target_h > H:
        return img
    return img[0:target_h, :, :]

def visualize_and_save_single_layer(pt_path, save_path, level_name="L0"):
    data = torch.load(pt_path, map_location="cpu")
    p2d_r = data["p2d_render"].numpy()
    p2d_q0 = data["p2d_query"][0].numpy()
    p2d_q1 = data["p2d_query_refined"][0].numpy()
    feat_map = data["f_q"].numpy()
    w_q_map = data.get("w_q", torch.ones_like(data["f_q"][:1])).squeeze().numpy()

    C, H, W = feat_map.shape
    bg = feature_to_heatmap(feat_map)

    # 仅裁剪图像，不再乘以 alpha，避免变暗
    bg = crop_from_origin_16_9(bg)
    w_q_map = crop_from_origin_16_9(w_q_map[..., None])[:, :, 0]
    orig_h, orig_w = bg.shape[:2]
    target_size = (512, 288)
    scale_x = target_size[0] / orig_w
    scale_y = target_size[1] / orig_h
    bg = cv2.resize(bg, target_size, interpolation=cv2.INTER_AREA)
    w_q_map_resized = cv2.resize(w_q_map, target_size, interpolation=cv2.INTER_LINEAR)

    # 用更柔和的 alpha 融合灰色遮罩，提升可视性（非直接变暗）
    overlay = np.full_like(bg, 128)
    alpha = (1.0 - w_q_map_resized) * 0.4  # 不透明度最多 0.4
    alpha = np.clip(alpha, 0, 0.4)[..., None]
    bg = bg.astype(np.float32)
    bg = bg * (1 - alpha) + overlay * alpha
    bg = np.clip(bg, 0, 255).astype(np.uint8)

    # 坐标缩放
    pq0 = p2d_q0[:50] * np.array([scale_x, scale_y])
    pq1 = p2d_q1[:50] * np.array([scale_x, scale_y])
    pr = p2d_r[:50] * np.array([scale_x, scale_y])

    res0 = np.linalg.norm(pr - pq0, axis=1)
    res1 = np.linalg.norm(pr - pq1, axis=1)
    rmse0 = np.sqrt((res0 ** 2).mean())
    rmse1 = np.sqrt((res1 ** 2).mean())

    W_img, H_img  = target_size
    fig, ax = plt.subplots(figsize=(5.12, 2.88), dpi=100)  # 精确到 512×288
    ax.imshow(bg)
    ax.axis("off")

    # 绘图：所有元素限制在图像边界内
    def in_bounds(x, y):
        return 0 <= x < W_img and 0 <= y < H_img

    for (x0, y0), (x1, y1) in zip(pq0, pq1):
        if in_bounds(x0, y0) and in_bounds(x1, y1):
            ax.add_patch(FancyArrow(x0, y0,
                                    x1 - x0, y1 - y0,
                                    width=0.4, color='cyan', alpha=0.8))

    for i in range(10):
        x2, y2 = pq1[i]
        if in_bounds(x2, y2):
            ax.plot(x2, y2, 'g+', markersize=4)

    for i, (x, y) in enumerate(pq1):
        ix, iy = int(round(x)), int(round(y))
        if in_bounds(ix, iy):
            local_uncertainty = w_q_map_resized[iy, ix]
            radius = 2.0 + local_uncertainty * 8.0
            ax.add_patch(plt.Circle((x, y), radius=radius,
                                    color='orange', alpha=0.4, linewidth=1.0, fill=True))

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ 已保存 {level_name}: {save_path}")

# ==== 主调用 ====
pt_files = [
    "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/0_feature.pt",
    "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/1_feature.pt",
    "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/2_feature.pt",
]

save_dir = "/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs"
os.makedirs(save_dir, exist_ok=True)

for i, pt_path in enumerate(pt_files):
    save_path = os.path.join(save_dir, f"pyramid_level{i}.png")
    visualize_and_save_single_layer(pt_path, save_path, level_name=f"L{i}")
