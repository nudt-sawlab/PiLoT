#!/usr/bin/env python3
"""Check nvdiffrast texture-v and output-y orientation with a quadrant texture."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


COLORS = {
    "red": np.asarray([255.0, 0.0, 0.0], dtype=np.float32),
    "green": np.asarray([0.0, 255.0, 0.0], dtype=np.float32),
    "blue": np.asarray([0.0, 0.0, 255.0], dtype=np.float32),
    "yellow": np.asarray([255.0, 255.0, 0.0], dtype=np.float32),
}
EXPECTED = {
    "top_left": "red",
    "top_right": "green",
    "bottom_left": "blue",
    "bottom_right": "yellow",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


def _make_quadrant_texture(height: int = 256, width: int = 256) -> np.ndarray:
    texture = np.zeros((height, width, 3), dtype=np.float32)
    mid_y = height // 2
    mid_x = width // 2
    texture[:mid_y, :mid_x] = COLORS["red"] / 255.0
    texture[:mid_y, mid_x:] = COLORS["green"] / 255.0
    texture[mid_y:, :mid_x] = COLORS["blue"] / 255.0
    texture[mid_y:, mid_x:] = COLORS["yellow"] / 255.0
    return texture


def _save_depth_png(path: Path, depth: np.ndarray) -> None:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        depth_vis = np.zeros(depth.shape, dtype=np.uint8)
    else:
        d_min = float(valid.min())
        d_max = float(valid.max())
        depth_vis = np.clip((depth - d_min) / max(d_max - d_min, 1.0e-6) * 255.0, 0, 255)
        depth_vis[depth <= 0] = 0
        depth_vis = depth_vis.astype(np.uint8)
    cv2.imwrite(os.fspath(path), depth_vis)


def _classify_color(rgb: np.ndarray) -> Tuple[str, Dict[str, float]]:
    distances = {
        name: float(np.linalg.norm(rgb.astype(np.float32) - value))
        for name, value in COLORS.items()
    }
    return min(distances, key=distances.get), distances


def _classify_corners(rgb: np.ndarray) -> Dict[str, Any]:
    h, w = rgb.shape[:2]
    patch = max(8, min(h, w) // 10)
    slices = {
        "top_left": np.s_[:patch, :patch],
        "top_right": np.s_[:patch, w - patch :],
        "bottom_left": np.s_[h - patch :, :patch],
        "bottom_right": np.s_[h - patch :, w - patch :],
    }
    corners: Dict[str, Any] = {}
    labels: Dict[str, str] = {}
    for name, slc in slices.items():
        mean_rgb = rgb[slc].reshape(-1, 3).mean(axis=0)
        label, distances = _classify_color(mean_rgb)
        labels[name] = label
        corners[name] = {
            "mean_rgb": mean_rgb,
            "label": label,
            "distances": distances,
        }
    return {
        "corners": corners,
        "labels": labels,
        "matches_expected": labels == EXPECTED,
    }


def _classify_depth_orientation(depth: np.ndarray) -> Dict[str, Any]:
    h = depth.shape[0]
    patch = max(8, h // 10)
    top = depth[:patch]
    bottom = depth[h - patch :]
    top_valid = top[np.isfinite(top) & (top > 0)]
    bottom_valid = bottom[np.isfinite(bottom) & (bottom > 0)]
    top_mean = float(top_valid.mean()) if top_valid.size else None
    bottom_mean = float(bottom_valid.mean()) if bottom_valid.size else None
    matches_expected = (
        top_mean is not None
        and bottom_mean is not None
        and top_mean < bottom_mean
    )
    return {
        "depth_top_mean": top_mean,
        "depth_bottom_mean": bottom_mean,
        "depth_expected_top_lt_bottom": True,
        "depth_matches_expected": matches_expected,
    }


def _render_case(
    output_dir: Path,
    width: int,
    height: int,
    texture_v_flip: bool,
    output_y_flip: bool,
) -> Dict[str, Any]:
    import nvdiffrast.torch as dr

    device = torch.device("cuda")
    ctx = dr.RasterizeCudaContext()
    pos_clip = torch.tensor(
        [
            [-1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [-1.0, -1.0, 0.0, 1.0],
            [1.0, -1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    triangles = torch.tensor([[0, 2, 3], [0, 3, 1]], dtype=torch.int32, device=device)
    uv_raster = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    uv = uv_raster.clone()
    if texture_v_flip:
        uv[:, 1] = 1.0 - uv[:, 1]

    camera_z_m = torch.tensor([[10.0], [10.0], [20.0], [20.0]], dtype=torch.float32, device=device)
    texture = torch.as_tensor(_make_quadrant_texture(), dtype=torch.float32, device=device)[None, ...]

    rast, _ = dr.rasterize(ctx, pos_clip[None, ...], triangles, resolution=[height, width])
    hit_mask = rast[..., 3:4] > 0
    uv_img, _ = dr.interpolate(uv[None, ...], rast, triangles)
    z_img, _ = dr.interpolate(camera_z_m[None, ...], rast, triangles)
    color_img = dr.texture(texture.contiguous(), uv_img.contiguous(), filter_mode="nearest")
    color_img = torch.where(hit_mask, color_img, torch.zeros_like(color_img))
    depth_img = torch.where(hit_mask, z_img, torch.zeros_like(z_img))

    color_np = (torch.clamp(color_img[0], 0.0, 1.0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    depth_np = depth_img[0, :, :, 0].detach().cpu().numpy().astype(np.float32)
    if output_y_flip:
        color_np = np.flipud(color_np).copy()
        depth_np = np.flipud(depth_np).copy()

    case_dir = output_dir / f"texture_v_flip_{int(texture_v_flip)}__output_y_flip_{int(output_y_flip)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(os.fspath(case_dir / "gpu_rgb.png"), cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.fspath(case_dir / "gpu_rgb_flipud.png"), cv2.cvtColor(np.flipud(color_np), cv2.COLOR_RGB2BGR))
    _save_depth_png(case_dir / "gpu_depth.png", depth_np)
    _save_depth_png(case_dir / "gpu_depth_flipud.png", np.flipud(depth_np))
    valid_mask = ((depth_np > 0) & np.isfinite(depth_np)).astype(np.uint8) * 255
    cv2.imwrite(os.fspath(case_dir / "valid_mask.png"), valid_mask)

    report = {
        "texture_v_flip": texture_v_flip,
        "output_y_flip": output_y_flip,
        "width": width,
        "height": height,
        "valid_depth_ratio": float(np.count_nonzero(valid_mask) / max(valid_mask.size, 1)),
        **_classify_corners(color_np),
        **_classify_depth_orientation(depth_np),
    }
    report["matches_expected"] = bool(report["matches_expected"] and report["depth_matches_expected"])
    _write_json(case_dir / "orientation_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="docs/experiments/dom_dsm_prepare/gpu_texture_orientation_check",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nvdiffrast orientation check")

    reports = []
    for texture_v_flip in (False, True):
        for output_y_flip in (False, True):
            reports.append(
                _render_case(
                    output_dir,
                    args.width,
                    args.height,
                    texture_v_flip=texture_v_flip,
                    output_y_flip=output_y_flip,
                )
            )

    passing = [
        {
            "texture_v_flip": r["texture_v_flip"],
            "output_y_flip": r["output_y_flip"],
        }
        for r in reports
        if r["matches_expected"]
    ]
    summary = {
        "expected_corners": EXPECTED,
        "passing_combinations": passing,
        "reports": reports,
    }
    _write_json(output_dir / "orientation_summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if len(passing) == 1 else 2


if __name__ == "__main__":
    raise SystemExit(main())
