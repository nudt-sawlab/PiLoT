import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm as mplcm

# ---------- 颜色映射工具 ----------
_OPENCV_CMAPS = {
    "viridis":  cv2.COLORMAP_VIRIDIS,
    "plasma":   cv2.COLORMAP_PLASMA,
    "inferno":  cv2.COLORMAP_INFERNO,
    "magma":    cv2.COLORMAP_MAGMA,
    "cividis":  cv2.COLORMAP_CIVIDIS,
    "turbo":    cv2.COLORMAP_TURBO,
    "jet":      cv2.COLORMAP_JET,
    "bone":     cv2.COLORMAP_BONE,
}

_MPL_DIVERGING = {"bwr", "seismic", "coolwarm"}  # 需要零中心归一化

def _apply_opencv_colormap(norm01: np.ndarray, name: str) -> np.ndarray:
    """norm01 in [0,1] -> RGB uint8 via OpenCV colormap"""
    norm255 = np.clip(norm01 * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm255, _OPENCV_CMAPS[name])
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

def _apply_mpl_colormap(norm01: np.ndarray, name: str) -> np.ndarray:
    """norm01 in [0,1] -> RGB uint8 via matplotlib colormap"""
    cmap = mplcm.get_cmap(name)
    rgba = cmap(np.clip(norm01, 0, 1))  # (H,W,4) float
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb

def _normalize_minmax(arr2d: np.ndarray, eps=1e-6) -> np.ndarray:
    mn = float(arr2d.min())
    mx = float(arr2d.max())
    return (arr2d - mn) / (mx - mn + eps)

def _normalize_symmetric_zero_center(arr2d: np.ndarray, eps=1e-6) -> np.ndarray:
    """把数据缩放到 [-1,1] 再映射到 [0,1]，用于发散色带"""
    m = float(np.max(np.abs(arr2d)))
    if m < eps:
        return np.full_like(arr2d, 0.5, dtype=np.float32)
    z = arr2d / m  # [-1,1]
    return (z + 1.0) * 0.5  # [0,1]

# ---------- 核心：特征到彩色图 ----------
def feature_to_heatmap(
    feat: np.ndarray,
    method: str = 'mean',
    vis_style: str = 'viridis',  # 支持：viridis/plasma/magma/inferno/turbo/jet/cividis/gray/diverging_bwr|seismic|coolwarm/gray_edge
    canny_thresh: tuple = (100, 200)
) -> np.ndarray:
    """
    (C,H,W) -> (H,W,3) RGB
    vis_style:
      - 'viridis'/'plasma'/'magma'/'inferno'/'turbo'/'jet'/'cividis'：常用顺序色带
      - 'gray'：灰度
      - 'gray_edge'：灰度 + Canny 边缘(红)
      - 'diverging_bwr'/'diverging_seismic'/'diverging_coolwarm'：发散色带（零中心）
    """
    assert feat.ndim == 3
    C, H, W = feat.shape

    # 降维
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

    feat2d = feat2d.astype(np.float32)

    # 选择风格
    if vis_style == 'gray':
        g = (_normalize_minmax(feat2d) * 255).astype(np.uint8)
        return np.dstack([g, g, g])

    if vis_style == 'gray_edge':
        g = (_normalize_minmax(feat2d) * 255).astype(np.uint8)
        gray_rgb = np.dstack([g, g, g])
        edges = cv2.Canny(g, canny_thresh[0], canny_thresh[1])
        edge_rgb = np.zeros_like(gray_rgb)
        edge_rgb[edges > 0] = [255, 0, 0]  # 红色边缘
        out = cv2.addWeighted(gray_rgb, 1.0, edge_rgb, 1.0, 0)
        return out

    if vis_style.startswith('diverging_'):
        # 发散色带，零中心归一（[-max,+max] 映射到 [0,1]）
        cmap_name = vis_style.split('_', 1)[1]  # bwr / seismic / coolwarm
        if cmap_name not in _MPL_DIVERGING:
            raise ValueError(f"Unsupported diverging cmap: {cmap_name}")
        norm01 = _normalize_symmetric_zero_center(feat2d)
        return _apply_mpl_colormap(norm01, cmap_name)

    # 其余顺序色带
    if vis_style in _OPENCV_CMAPS:
        norm01 = _normalize_minmax(feat2d)
        return _apply_opencv_colormap(norm01, vis_style)

    # 兜底：用 matplotlib 的名字
    try:
        norm01 = _normalize_minmax(feat2d)
        return _apply_mpl_colormap(norm01, vis_style)
    except Exception as e:
        raise ValueError(f"Unknown vis_style: {vis_style}") from e

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
def visualize_ref_query_pair(
    pt_path, save_path, level_name="L0",
    reduce='mean', out_size=(512, 288),
    vis_style_q: str = 'viridis',
    vis_style_r: str = 'viridis',
    canny_thresh: tuple = (100, 200),
    l2norm_ref: bool = True
):
    """
    读取单个层级的 .pt 文件，生成一张 1x2 图：
      左列：Query 特征
      右列：Ref/Render 特征
    vis_style_* 见 feature_to_heatmap 的说明
    """
    data = torch.load(pt_path, map_location="cpu")

    # 兼容不同命名
    f_q = get_first_existing(data, ["f_q", "feat_q", "q_feat", "query_feat"]).numpy()
    f_r = get_first_existing(data, ["f_r", "f_ref", "f_render", "feat_r", "r_feat", "ref_feat", "render_feat"]).numpy()

    # 可选：在通道维做L2归一，便于跨通道比较
    if l2norm_ref:
        f_r = f_r / (np.linalg.norm(f_r, axis=0, keepdims=True) + 1e-6)

    # 各自转彩图
    q_hm = feature_to_heatmap(f_q, method=reduce, vis_style=vis_style_q, canny_thresh=canny_thresh)
    r_hm = feature_to_heatmap(f_r, method=reduce, vis_style=vis_style_r, canny_thresh=canny_thresh)

    # 统一裁剪与尺寸
    q_hm = resize_to(crop_from_origin_16_9(q_hm), out_size)
    r_hm = resize_to(crop_from_origin_16_9(r_hm), out_size)

    # 画布：一行两列
    fig, axes = plt.subplots(1, 2, figsize=(out_size[0]*2/100, out_size[1]/100), dpi=100)
    for ax, img, title in zip(
        axes,
        [q_hm, r_hm],
        [f"Query {level_name} ({vis_style_q})", f"Reference {level_name} ({vis_style_r})"]
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    fig.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"✅ 已保存 {level_name}: {save_path}")

# ---------- 主调用（按你现有路径） ----------
seq_name = "DJI_20250612194903_0021_V"
pt_files = [
    os.path.join("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs", seq_name, "0_feature.pt"),
    os.path.join("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs", seq_name, "1_feature.pt"),
    os.path.join("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs", seq_name, "2_feature.pt")
]
save_dir = os.path.join("/home/ubuntu/Documents/code/github/FPV/FPV-Test-512-cuda/feature_vis_outputs", seq_name)
os.makedirs(save_dir, exist_ok=True)

# 示例：你可以改 vis_style_* 来对比
styles_to_try = [
    ("viridis", "viridis"),
    ("plasma", "plasma"),
    ("magma", "magma"),
    ("inferno", "inferno"),
    ("turbo", "turbo"),
    ("jet", "jet"),
    ("cividis", "cividis"),
    ("gray", "gray"),
    ("gray_edge", "gray_edge"),
    ("diverging_bwr", "diverging_bwr"),
    ("diverging_seismic", "diverging_seismic"),
    ("diverging_coolwarm", "diverging_coolwarm"),
]

# for i, pt_path in enumerate(pt_files):
#     # 挑一个你想测试的风格（也可以循环 styles_to_try 输出多套）
#     vis_q, vis_r = "viridis", "gray_edge"   # 举例：左 viridis，右 灰度+边缘
#     save_path = os.path.join(save_dir, f"feat_pair_L{i}_{vis_q}_{vis_r}.png")
#     visualize_ref_query_pair(
#         pt_path, save_path, level_name=f"L{i}", reduce='pca', out_size=(512, 288),
#         vis_style_q=vis_q, vis_style_r=vis_r, canny_thresh=(80, 160), l2norm_ref=True
#     )

# 如果要批量对比所有风格，可把上面循环换成下面这段：
for i, pt_path in enumerate(pt_files):
    for (vis_q, vis_r) in styles_to_try:
        save_path = os.path.join(save_dir, f"feat_pair_L{i}_{vis_q}_{vis_r}.png")
        visualize_ref_query_pair(
            pt_path, save_path, level_name=f"L{i}", reduce='pca', out_size=(512, 288),
            vis_style_q=vis_q, vis_style_r=vis_r, canny_thresh=(80, 160), l2norm_ref=True
        )
