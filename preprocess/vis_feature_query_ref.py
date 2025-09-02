import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------- 工具函数 ----------
def feature_to_heatmap(feat: np.ndarray, method='mean', colormap=cv2.COLORMAP_VIRIDIS):
    """(C,H,W) -> 彩色热力图 (H,W,3)"""
    assert feat.ndim == 3
    C, H, W = feat.shape
    if method == 'mean':
        feat2d = feat.mean(axis=0)
    elif method == 'max':
        feat2d = feat.max(axis=0)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        feat_flat = feat.reshape(C, -1).T
        feat2d = PCA(n_components=1).fit_transform(feat_flat).reshape(H, W)
    else:
        raise ValueError("method must be 'mean', 'max', or 'pca'")
    feat2d = (feat2d - feat2d.min()) / (feat2d.max() - feat2d.min() + 1e-6)
    feat2d = (feat2d * 255).astype(np.uint8)
    # feat2d = (1.0 - feat2d) * 255   # 关键：反转
    # feat2d = feat2d.astype(np.uint8)
    color = cv2.applyColorMap(feat2d, colormap)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

def crop_from_origin_16_9(img):
    """从上边开始，裁出 16:9，超出则不裁"""
    H, W, _ = img.shape
    target_h = int(W * 9 / 16)
    if target_h > H:
        return img
    return img[0:target_h, :, :]

def resize_to(img, size=(512, 288)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def get_first_existing(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"找不到任何可用的特征键：{keys}")

# ---------- 可视化：一行两列（Query | Ref） ----------
def visualize_ref_query_pair(pt_path, save_path, level_name="L0",
                             reduce='mean', out_size=(960, 540)):
    """
    读取单个层级的 .pt 文件，生成一张 1x2 图：
      左列：Query 特征热力图
      右列：Ref/Render 特征热力图
    """
    data = torch.load(pt_path, map_location="cpu")

    # 兼容不同命名
    f_q = get_first_existing(data, ["f_q", "feat_q", "q_feat", "query_feat"]).numpy()
    f_r = get_first_existing(data, ["f_r", "f_ref", "f_render", "feat_r", "r_feat", "ref_feat", "render_feat"]).numpy()

    # 各自转热力图
    q_hm = feature_to_heatmap(f_q, method=reduce)
    f_r = f_r / np.linalg.norm(f_r, axis=0, keepdims=True)
    r_hm = feature_to_heatmap(f_r, method=reduce)

    # 统一裁剪与尺寸
    q_hm = resize_to(crop_from_origin_16_9(q_hm), out_size)
    r_hm = resize_to(crop_from_origin_16_9(r_hm), out_size)

    # 画布：一行两列
    fig, axes = plt.subplots(1, 2, figsize=(out_size[0]*2/100, out_size[1]/100), dpi=100)
    for ax, img, title in zip(
        axes,
        [q_hm, r_hm],
        [f"Query {level_name}", f"Reference {level_name}"]
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    fig.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"✅ 已保存 {level_name}: {save_path}")

# ---------- 主调用 ----------
seq_name = "switzerland_seq4@8@rainy@200"
pt_files = [
    os.path.join("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs",seq_name ,"0_feature.pt"),
    os.path.join("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs",seq_name ,"1_feature.pt"),
    os.path.join("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs",seq_name ,"2_feature.pt")
    
]
save_dir = os.path.join("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs/", seq_name)
os.makedirs(save_dir, exist_ok=True)

for i, pt_path in enumerate(pt_files):
    save_path = os.path.join(save_dir, f"feat_pair_L{i}.png")
    visualize_ref_query_pair(pt_path, save_path, level_name=f"L{i}", reduce='mean', out_size=(960, 540))

