"""PyTorch-only feature residual diagnostics for DOM/DSM PiLoT experiments."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def extract_pilot_features(localizer_or_extractor: Any, image: np.ndarray, name: str) -> Dict[str, Any]:
    """Extract PiLoT dense features without training or optimizer updates."""
    if name not in {"query", "render"}:
        raise ValueError("name must be 'query' or 'render'")
    with torch.no_grad():
        if hasattr(localizer_or_extractor, "refiner"):
            feats, scales = localizer_or_extractor.refiner.dense_feature_extraction(image)
            confidences = None
        elif hasattr(localizer_or_extractor, "dense_feature_extraction"):
            feats, scales = localizer_or_extractor.dense_feature_extraction(image)
            confidences = None
        else:
            feats, scales, confidences = localizer_or_extractor(image)
    return {"name": name, "features": feats, "scales": scales, "confidences": confidences}


def project_points_to_image(points_3d: torch.Tensor, pose_w2c: Any, camera: Any) -> Dict[str, torch.Tensor]:
    """Project local 3D points into an image with PixLoc Pose/Camera conventions."""
    points_cam = pose_w2c * points_3d
    points2d, valid = camera.world2image(points_cam)
    depth = points_cam[..., 2]
    valid = valid & torch.isfinite(depth) & (depth > camera.eps)
    return {"points2d": points2d, "valid": valid, "depth": depth}


def _scale_tuple(scale: Any) -> Tuple[float, float]:
    if isinstance(scale, (int, float)):
        return float(scale), float(scale)
    if torch.is_tensor(scale):
        vals = scale.detach().cpu().numpy().reshape(-1).tolist()
    else:
        vals = list(scale)
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    return float(vals[0]), float(vals[1])


def sample_feature_map(feature: torch.Tensor, points2d: torch.Tensor, scale: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a CxHxW feature map at original-image pixel coordinates."""
    if feature.ndim != 3:
        raise ValueError("feature must be CxHxW")
    sx, sy = _scale_tuple(scale)
    c, h, w = feature.shape
    pts = points2d.to(feature.device, dtype=feature.dtype)
    pts_feat = pts * pts.new_tensor([sx, sy])
    valid = (
        torch.isfinite(pts_feat).all(dim=-1)
        & (pts_feat[:, 0] >= 0)
        & (pts_feat[:, 0] <= w - 1)
        & (pts_feat[:, 1] >= 0)
        & (pts_feat[:, 1] <= h - 1)
    )
    if w <= 1 or h <= 1:
        raise ValueError("feature map too small for grid_sample")
    grid_x = pts_feat[:, 0] / (w - 1) * 2.0 - 1.0
    grid_y = pts_feat[:, 1] / (h - 1) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        feature.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).squeeze(0).squeeze(-1).transpose(0, 1)
    return sampled, valid


def _strip_confidence(features: Sequence[torch.Tensor], use_confidence: bool) -> List[torch.Tensor]:
    out = []
    for feat in features:
        if not use_confidence and feat.shape[0] > 1:
            out.append(feat[:-1])
        else:
            out.append(feat)
    return out


def compute_feature_residual_loss(
    features_query: Sequence[torch.Tensor],
    features_render: Sequence[torch.Tensor],
    scales_query: Sequence[Any],
    scales_render: Sequence[Any],
    points_3d: torch.Tensor,
    T_query_w2c: Any,
    T_render_w2c: Any,
    query_camera: Any,
    render_camera: Any,
    levels: Optional[Sequence[int]] = None,
    robust: str = "l2",
    use_confidence: bool = False,
) -> Dict[str, Any]:
    """Compute feature residual loss for fixed poses without CUDA extension or optimizer update."""
    if levels is None:
        levels = list(range(min(len(features_query), len(features_render))))
    fq = _strip_confidence(features_query, use_confidence)
    fr = _strip_confidence(features_render, use_confidence)
    proj_q = project_points_to_image(points_3d, T_query_w2c, query_camera)
    proj_r = project_points_to_image(points_3d, T_render_w2c, render_camera)
    losses = []
    num_valid = []
    valid_ratios = []
    residual_accum = torch.zeros(points_3d.shape[0], device=points_3d.device, dtype=torch.float32)
    residual_count = torch.zeros(points_3d.shape[0], device=points_3d.device, dtype=torch.float32)
    valid_union = torch.zeros(points_3d.shape[0], device=points_3d.device, dtype=torch.bool)
    for level in levels:
        feat_q = F.normalize(fq[level].to(points_3d.device), dim=0)
        feat_r = F.normalize(fr[level].to(points_3d.device), dim=0)
        q_sample, q_valid = sample_feature_map(feat_q, proj_q["points2d"], scales_query[level])
        r_sample, r_valid = sample_feature_map(feat_r, proj_r["points2d"], scales_render[level])
        valid = proj_q["valid"] & proj_r["valid"] & q_valid & r_valid
        if valid.any():
            diff = q_sample[valid] - r_sample[valid]
            residual = torch.linalg.norm(diff, dim=-1)
            if robust == "l1":
                level_loss = residual.mean()
            elif robust == "charbonnier":
                level_loss = torch.sqrt(residual * residual + 1e-6).mean()
            else:
                level_loss = (residual * residual).mean()
            losses.append(level_loss)
            idx = valid.nonzero(as_tuple=False).view(-1)
            residual_accum[idx] += residual.detach().float()
            residual_count[idx] += 1.0
            valid_union |= valid
            num_valid.append(int(valid.sum().item()))
            valid_ratios.append(float(valid.float().mean().item()))
        else:
            losses.append(torch.tensor(float("nan"), device=points_3d.device))
            num_valid.append(0)
            valid_ratios.append(0.0)
    finite_losses = [loss for loss in losses if torch.isfinite(loss)]
    loss_total = torch.stack(finite_losses).mean() if finite_losses else torch.tensor(float("inf"), device=points_3d.device)
    residual_per_point = torch.zeros_like(residual_accum)
    has_res = residual_count > 0
    residual_per_point[has_res] = residual_accum[has_res] / residual_count[has_res]
    return {
        "loss_total": float(loss_total.detach().cpu().item()),
        "loss_by_level": [float(x.detach().cpu().item()) if torch.isfinite(x) else None for x in losses],
        "num_valid_by_level": num_valid,
        "valid_ratio_by_level": valid_ratios,
        "residual_per_point": residual_per_point.detach().cpu().numpy(),
        "points_query": proj_q["points2d"].detach().cpu().numpy(),
        "points_render": proj_r["points2d"].detach().cpu().numpy(),
        "valid_mask": valid_union.detach().cpu().numpy(),
    }


def save_residual_debug_visualization(
    output_path: Path,
    query_rgb: np.ndarray,
    render_rgb: np.ndarray,
    points_query: np.ndarray,
    points_render: np.ndarray,
    residual_per_point: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    """Save residual overlays, histogram, and point JSON for a candidate."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    valid = valid_mask.astype(bool)
    residual = np.nan_to_num(residual_per_point.astype(np.float32), nan=0.0)
    valid_res = residual[valid]
    vmax = float(np.percentile(valid_res, 95)) if valid_res.size else 1.0
    vmax = max(vmax, 1e-6)

    def draw(base_rgb: np.ndarray, pts: np.ndarray, name: str) -> None:
        vis = cv2.cvtColor(base_rgb.copy(), cv2.COLOR_RGB2BGR)
        h, w = base_rgb.shape[:2]
        for p, r, ok in zip(pts, residual, valid):
            if not ok:
                continue
            x, y = int(round(float(p[0]))), int(round(float(p[1])))
            if 0 <= x < w and 0 <= y < h:
                t = min(float(r) / vmax, 1.0)
                color = (int(255 * t), int(255 * (1.0 - t)), 40)
                cv2.circle(vis, (x, y), 2, color, -1)
        cv2.imwrite(str(output_path / name), vis)

    draw(query_rgb, points_query, "query_residual_overlay.png")
    draw(render_rgb, points_render, "render_residual_overlay.png")
    hist = np.zeros((220, 360, 3), dtype=np.uint8) + 255
    if valid_res.size:
        bins = np.linspace(0, vmax, 32)
        counts, _ = np.histogram(np.clip(valid_res, 0, vmax), bins=bins)
        max_count = max(int(counts.max()), 1)
        for i, count in enumerate(counts):
            x0 = 20 + i * 10
            y0 = 200
            y1 = 200 - int(170 * count / max_count)
            cv2.rectangle(hist, (x0, y1), (x0 + 8, y0), (40, 80, 220), -1)
    cv2.imwrite(str(output_path / "residual_histogram.png"), hist)
    pts_json = []
    for i, (pq, pr, r, ok) in enumerate(zip(points_query, points_render, residual, valid)):
        if ok:
            pts_json.append({"index": int(i), "query_xy": [float(pq[0]), float(pq[1])], "render_xy": [float(pr[0]), float(pr[1])], "residual": float(r)})
    (output_path / "residual_points.json").write_text(__import__("json").dumps(pts_json, indent=2) + "\n", encoding="utf-8")
