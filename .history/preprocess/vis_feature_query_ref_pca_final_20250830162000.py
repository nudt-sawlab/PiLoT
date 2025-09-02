import os, cv2, numpy as np, torch, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Optional

# ---------- 小工具 ----------
def crop_from_origin_16_9(img):
    H, W, _ = img.shape
    target_h = int(W * 9 / 16)
    return img if target_h > H else img[0:target_h, :, :]

def resize_to(img, size=(512, 288)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def get_first_existing(d, keys):
    for k in keys:
        if k in d: return d[k]
    raise KeyError(f"找不到任何可用的特征键：{keys}")

def pca_rgb(feat: np.ndarray, pca_model: Optional[PCA] = None,
            ref_sign: Optional[np.ndarray] = None):
    """
    (C,H,W)->(H,W,3) RGB。若提供 pca_model，则仅 transform；
    ref_sign 是长度为3的向量，指定每个主成分的符号(±1)，用于锁定颜色方向。
    返回：rgb(uint8), pca_model, sign(3,)
    """
    C, H, W = feat.shape
    X = feat.reshape(C, -1).T.astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)

    if pca_model is None:
        pca_model = PCA(n_components=min(3, C), svd_solver='auto', whiten=False)
        Y = pca_model.fit_transform(X)
    else:
        Y = pca_model.transform(X)

    # 计算/应用符号：以 ref_sign 为准；否则按当前数据均值正向
    if ref_sign is None:
        sign = np.ones(Y.shape[1], dtype=np.int8)
        for i in range(Y.shape[1]):
            if Y[:, i].mean() < 0: sign[i] = -1
    else:
        sign = np.asarray(ref_sign, dtype=np.int8)
    Y = Y * sign  # 锁定方向

    # 稳健归一(1–99分位) -> [0,1]
    rgb = np.zeros((Y.shape[0], 3), dtype=np.float32)
    for i in range(Y.shape[1]):
        lo, hi = np.percentile(Y[:, i], [1, 99])
        if hi - lo < 1e-6: rgb[:, i] = 0.5
        else:              rgb[:, i] = np.clip((Y[:, i] - lo) / (hi - lo), 0, 1)

    rgb = (rgb.reshape(H, W, 3) * 255).astype(np.uint8)
    return rgb, pca_model, sign

# ---------- 主可视化：固定用 Ref 的 PCA 模型显示 Q/R ----------
def visualize_ref_query_pair_refPCA(
    pt_path, save_path, level_name="L0",
    out_size=(512, 288), l2norm_ref=True
):
    data = torch.load(pt_path, map_location="cpu")
    f_q = get_first_existing(data, ["f_q","feat_q","q_feat","query_feat"]).numpy()
    f_r = get_first_existing(data, ["f_r","f_ref","f_render","feat_r","r_feat","ref_feat","render_feat"]).numpy()

    if l2norm_ref:
        f_r = f_r / (np.linalg.norm(f_r, axis=0, keepdims=True) + 1e-6)

    # 1) 仅用 Ref 拟合 PCA，并记录符号
    r_rgb, pca_ref, ref_sign = pca_rgb(f_r, pca_model=None, ref_sign=None)
    # 2) 用同一 PCA 模型 + 同一符号应用到 Query（保证颜色坐标系一致）
    q_rgb, _, _ = pca_rgb(f_q, pca_model=pca_ref, ref_sign=ref_sign)

    q_img = resize_to(crop_from_origin_16_9(q_rgb), out_size)
    r_img = resize_to(crop_from_origin_16_9(r_rgb), out_size)

    fig, axes = plt.subplots(1, 2, figsize=(out_size[0]*2/100, out_size[1]/100), dpi=100)
    titles = [f"Query {level_name} (PCA→RGB, ref-fit)", f"Reference {level_name} (PCA→RGB, ref-fit)"]
    for ax, img, title in zip(axes, [q_img, r_img], titles):
        ax.imshow(img); ax.set_title(title, fontsize=10); ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    fig.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"✅ 已保存 {level_name}: {save_path}")

# ---------- 调用示例 ----------
seq_name = "switzerland_seq4@8@rainy@200"
pt_files = [
    os.path.join("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs", seq_name, "0_feature.pt"),
    os.path.join("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs", seq_name, "1_feature.pt"),
    os.path.join("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs", seq_name, "2_feature.pt"),
]
save_dir = os.path.join("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs", seq_name)
os.makedirs(save_dir, exist_ok=True)

for i, pt_path in enumerate(pt_files):
    save_path = os.path.join(save_dir, f"feat_pair_L{i}_pcaRGB_refFit.png")
    visualize_ref_query_pair_refPCA(pt_path, save_path, level_name=f"L{i}", out_size=(960, 540))
