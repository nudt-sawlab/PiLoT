"""Structure-aware point sampling for DOM/DSM rendered RGB and depth."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


def _normalize_weight(weight: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    weight = np.nan_to_num(weight.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if valid is not None:
        weight = np.where(valid, weight, 0.0)
    positive = weight[weight > 0]
    if positive.size == 0:
        return np.zeros_like(weight, dtype=np.float32)
    lo = float(np.percentile(positive, 1))
    hi = float(np.percentile(positive, 99))
    if hi <= lo:
        hi = float(positive.max())
        lo = float(positive.min())
    norm = (weight - lo) / max(hi - lo, 1e-6)
    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)
    if valid is not None:
        norm[~valid] = 0.0
    return norm


def compute_dom_gradient_weight(render_rgb: np.ndarray) -> np.ndarray:
    """Compute normalized image-gradient structure weights from rendered RGB."""
    if render_rgb.ndim != 3 or render_rgb.shape[2] != 3:
        raise ValueError("render_rgb must be HxWx3 RGB")
    gray = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return _normalize_weight(mag)


def compute_dsm_depth_gradient_weight(depth: np.ndarray) -> np.ndarray:
    """Compute normalized depth-gradient weights on valid depth pixels."""
    if depth.ndim != 2:
        raise ValueError("depth must be HxW")
    depth_f = depth.astype(np.float32)
    valid = np.isfinite(depth_f) & (depth_f > 0)
    if not np.any(valid):
        return np.zeros_like(depth_f, dtype=np.float32)
    filled = depth_f.copy()
    filled[~valid] = float(np.median(depth_f[valid]))
    gx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return _normalize_weight(mag, valid)


def compute_combined_structure_weight(
    render_rgb: np.ndarray,
    depth: np.ndarray,
    alpha_dom: float = 0.7,
    alpha_depth: float = 0.3,
) -> np.ndarray:
    """Combine rendered RGB gradients and depth gradients into one HxW weight map."""
    if render_rgb.shape[:2] != depth.shape[:2]:
        raise ValueError("render_rgb and depth must have matching HxW")
    valid = np.isfinite(depth) & (depth > 0)
    dom_w = compute_dom_gradient_weight(render_rgb)
    depth_w = compute_dsm_depth_gradient_weight(depth)
    combined = float(alpha_dom) * dom_w + float(alpha_depth) * depth_w
    return _normalize_weight(combined, valid)


def sample_points_from_weight(
    weight: np.ndarray,
    num_samples: int,
    device: str = "cuda",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Sample [x, y] pixel coordinates from a non-negative HxW weight map."""
    if weight.ndim != 2:
        raise ValueError("weight must be HxW")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    h, w = weight.shape
    flat = np.nan_to_num(weight.reshape(-1).astype(np.float64), nan=0.0)
    valid_idx = np.flatnonzero(flat > 0)
    rng = np.random.default_rng(seed)
    if valid_idx.size == 0:
        valid_idx = np.arange(h * w)
        probs = None
    else:
        probs = flat[valid_idx]
        total = probs.sum()
        probs = None if total <= 0 else probs / total
    replace = valid_idx.size < num_samples
    chosen = rng.choice(valid_idx, size=num_samples, replace=replace, p=probs)
    ys, xs = np.divmod(chosen, w)
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    return torch.as_tensor(points, device=device)


def sample_domdsm_points(
    render_rgb: np.ndarray,
    depth: np.ndarray,
    num_samples: int = 500,
    mode: str = "combined",
    device: str = "cuda",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Sample DOM/DSM 2D points in [x, y] order for PixLoc back-projection."""
    valid = (np.isfinite(depth) & (depth > 0)).astype(np.float32)
    if mode == "uniform":
        weight = valid
    elif mode == "dom_gradient":
        weight = compute_dom_gradient_weight(render_rgb) * valid
    elif mode == "depth_gradient":
        weight = compute_dsm_depth_gradient_weight(depth)
    elif mode == "combined":
        weight = compute_combined_structure_weight(render_rgb, depth)
    else:
        raise ValueError(f"Unknown DOM/DSM sampling mode: {mode}")
    return sample_points_from_weight(weight, num_samples, device=device, seed=seed)


def save_sampling_debug(
    path: Path,
    render_rgb: np.ndarray,
    depth: np.ndarray,
    weight: np.ndarray,
    points2d: torch.Tensor,
) -> None:
    """Save a visual overlay of sampling weights and sampled points."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weight_u8 = np.clip(weight * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(weight_u8, cv2.COLORMAP_TURBO)
    base = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)
    vis = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    pts = points2d.detach().cpu().numpy().astype(np.int32)
    h, w = weight.shape
    for x, y in pts[:2000]:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(vis, (int(x), int(y)), 1, (255, 255, 255), -1)
    cv2.imwrite(str(path), vis)
