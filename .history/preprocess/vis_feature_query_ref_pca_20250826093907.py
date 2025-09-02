import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Optional
# ---------- 工具 ----------
def crop_from_origin_16_9(img):
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

def minmax01(x, eps=1e-6):
    return (x - x.min()) / (x.max() - x.min() + eps)

# ---------- 原有：顺序/灰度色带 ----------
def feature_to_heatmap(feat: np.ndarray, method='mean', colormap=cv2.COLORMAP_VIRIDIS):
    """(C,H,W) -> 彩色热力图 (H,W,3)  （保留你原来的接口）"""
    assert feat.ndim == 3
    C, H, W = feat.shape
    if method == 'mean':
        feat2d = feat.mean(axis=0)
    elif method == 'max':
        feat2d = feat.max(axis=0)
    elif method == 'pca':
        # 单通道的 PCA 可视化（灰度）——保留原有行为
        feat_flat = feat.reshape(C, -1).T
        pc1 = PCA(n_components=1).fit_transform(feat_flat).reshape(H, W)
        feat2d = pc1
    else:
        raise ValueError("method must be 'mean', 'max', or 'pca'")
    feat2d = minmax01(feat2d)
    feat2d = (feat2d * 255).astype(np.uint8)
    color = cv2.applyColorMap(feat2d, colormap)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

# ---------- 新增：PCA → RGB （三主成分映射到 R/G/B） ----------
def pca_rgb(feat: np.ndarray, pca_model: Optional[PCA] = None):
    """
    (C,H,W) -> (H,W,3) RGB，用前三个主成分作为RGB通道。
    若 pca_model 提供，则使用给定PCA（用于跨图一致着色）。
    返回：rgb(uint8), pca_model
    """
    assert feat.ndim == 3
    C, H, W = feat.shape
    X = feat.reshape(C, -1).T.astype(np.float32)  # (N, C)

    # 中心化
    X = X - X.mean(axis=0, keepdims=True)

    if pca_model is None:
        pca_model = PCA(n_components=min(3, C), svd_solver='auto', whiten=False)
        Y = pca_model.fit_transform(X)  # (N, 3)
    else:
        Y = pca_model.transform(X)

    # 符号对齐：让每个PC的均值为正，避免随机翻转导致颜色颠倒
    for i in range(Y.shape[1]):
        if Y[:, i].mean() < 0:
            Y[:, i] *= -1

    # 稳健归一到 [0,1]：按分位数 1%-99% 裁剪，避免极端值压扁色彩
    rgb = np.zeros((Y.shape[0], 3), dtype=np.float32)
    for i in range(Y.shape[1]):
        lo, hi = np.percentile(Y[:, i], [1, 99])
        if hi - lo < 1e-6:
            rgb[:, i] = 0.5
        else:
            rgb[:, i] = np.clip((Y[:, i] - lo) / (hi - lo), 0.0, 1.0)

    rgb = (rgb.reshape(H, W, 3) * 255).astype(np.uint8)
    return rgb, pca_model

# ---------- 可视化：一行两列（Query | Ref），加入 PCA→RGB 选项 ----------
def visualize_ref_query_pair(
    pt_path, save_path, level_name="L0",
    mode='colormap',                       # 'colormap' | 'pca_rgb'
    reduce='mean',                         # 仅在 mode='colormap' 时使用
    out_size=(512, 288),
    colormap=cv2.COLORMAP_VIRIDIS,
    pca_fit_mode='joint',                  # 'joint' | 'ref' | 'separate'
    l2norm_ref=True
):
    """
    mode='pca_rgb' 时：将多通道特征映射为 RGB（PCA前三主成分）。
    pca_fit_mode:
      - 'joint': 用 (Query ∪ Ref) 一起拟合，再分别transform（颜色一致，推荐）
      - 'ref':   只用 Ref 拟合，把同一PCA应用于 Query（保持对齐）
      - 'separate': 各自拟合（颜色不完全可比，但强调局部差异）
    """
    data = torch.load(pt_path, map_location="cpu")
    f_q = get_first_existing(data, ["f_q", "feat_q", "q_feat", "query_feat"]).numpy()
    f_r = get_first_existing(data, ["f_r", "f_ref", "f_render", "feat_r", "r_feat", "ref_feat", "render_feat"]).numpy()
    f_r = f_r / np.linalg.norm(f_r, axis=0, keepdims=True)
    if l2norm_ref:
        f_r = f_r / (np.linalg.norm(f_r, axis=0, keepdims=True) + 1e-6)

    if mode == 'pca_rgb':
        # 选择PCA拟合策略
        pca_model = None
        if pca_fit_mode == 'joint':
            # 拼接后拟合
            Cq, Hq, Wq = f_q.shape
            Cr, Hr, Wr = f_r.shape
            assert Cq == Cr, "Q/R 通道数需一致以做 joint PCA"
            X_joint = np.concatenate([f_q.reshape(Cq, -1).T, f_r.reshape(Cr, -1).T], axis=0)
            X_joint = X_joint - X_joint.mean(axis=0, keepdims=True)
            pca_model = PCA(n_components=min(3, Cq), svd_solver='auto', whiten=False).fit(X_joint)

        elif pca_fit_mode == 'ref':
            Cr = f_r.shape[0]
            X_ref = f_r.reshape(Cr, -1).T
            X_ref = X_ref - X_ref.mean(axis=0, keepdims=True)
            pca_model = PCA(n_components=min(3, Cr), svd_solver='auto', whiten=False).fit(X_ref)

        # 应用到Q/R
        q_rgb, _ = pca_rgb(f_q, pca_model=pca_model if pca_fit_mode in ['joint', 'ref'] else None)
        r_rgb, _ = pca_rgb(f_r, pca_model=pca_model if pca_fit_mode in ['joint', 'ref'] else None)

        q_img, r_img = q_rgb, r_rgb

    else:
        # 普通色带模式（与你原先一致）
        q_img = feature_to_heatmap(f_q, method=reduce, colormap=colormap)
        r_img = feature_to_heatmap(f_r, method=reduce, colormap=colormap)

    # 统一裁剪与尺寸
    q_img = resize_to(crop_from_origin_16_9(q_img), out_size)
    r_img = resize_to(crop_from_origin_16_9(r_img), out_size)

    # 画布：一行两列
    fig, axes = plt.subplots(1, 2, figsize=(out_size[0]*2/100, out_size[1]/100), dpi=100)
    titles = [f"Query {level_name}", f"Reference {level_name}"]
    if mode == 'pca_rgb':
        titles = [t + f" (PCA→RGB, {pca_fit_mode})" for t in titles]
    else:
        # colormap 名称以便你记录
        cmap_name = str(colormap) if isinstance(colormap, int) else 'custom'
        titles = [t + f" (cmap)" for t in titles]

    for ax, img, title in zip(axes, [q_img, r_img], titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    fig.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"✅ 已保存 {level_name}: {save_path}")

# ---------- 主调用 ----------
seq_name = "DJI_20241113180128_0042_D"
pt_files = [
    os.path.join("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs", seq_name, "0_feature.pt"),
    os.path.join("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs", seq_name, "1_feature.pt"),
    os.path.join("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs", seq_name, "2_feature.pt")
]
save_dir = os.path.join("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs", seq_name)
os.makedirs(save_dir, exist_ok=True)

for i, pt_path in enumerate(pt_files):
    # 1) 试 PCA→RGB（颜色统一）：joint
    save_path = os.path.join(save_dir, f"feat_pair_L{i}_pcaRGB_joint.png")
    visualize_ref_query_pair(
        pt_path, save_path, level_name=f"L{i}",
        mode='pca_rgb', pca_fit_mode='joint', out_size=(512, 288), l2norm_ref=True
    )

    # 2) 对比：仅用Ref拟合，再应用到Query
    save_path = os.path.join(save_dir, f"feat_pair_L{i}_pcaRGB_ref.png")
    visualize_ref_query_pair(
        pt_path, save_path, level_name=f"L{i}",
        mode='pca_rgb', pca_fit_mode='ref', out_size=(512, 288), l2norm_ref=True
    )

    # 3) 对比：各自独立拟合（局部对比强，但跨图不可比）
    save_path = os.path.join(save_dir, f"feat_pair_L{i}_pcaRGB_separate.png")
    visualize_ref_query_pair(
        pt_path, save_path, level_name=f"L{i}",
        mode='pca_rgb', pca_fit_mode='separate', out_size=(512, 288), l2norm_ref=True
    )
