#!/usr/bin/env python3
"""Renderer-only DOM/DSM acceptance for one query image."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pixloc.utils.dom_dsm import DOMDSMRenderer


def read_pose_file(path: Path, image_name: str) -> Tuple[list, list]:
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0] != image_name:
            continue
        if len(parts) != 7:
            raise ValueError(f"Invalid pose row in {path}: {line}")
        lon, lat, alt, roll, pitch, yaw = map(float, parts[1:])
        return [lon, lat, alt], [pitch, roll, yaw]
    raise KeyError(f"{image_name} not found in {path}")


def scaled_camera(cam_cfg: Dict, render_width: int) -> np.ndarray:
    source_w = float(cam_cfg["width"])
    source_h = float(cam_cfg["height"])
    scale = render_width / source_w
    render_height = int(round(source_h * scale))
    fx, fy, cx, cy = map(float, cam_cfg["params"])
    return np.array(
        [
            render_width,
            render_height,
            cx * scale,
            cy * scale,
            fx * scale,
            fy * scale,
        ],
        dtype=np.float64,
    )


def depth_to_vis(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0
    if not np.any(valid):
        return np.zeros_like(depth, dtype=np.uint8)
    d_min = float(depth[valid].min())
    d_max = float(depth[valid].max())
    scaled = (depth - d_min) / max(d_max - d_min, 1e-6)
    scaled[~valid] = 0
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def make_overlay(query_rgb: np.ndarray, render_rgb: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(query_rgb, 0.5, render_rgb, 0.5, 0)


def make_edge_overlay(query_rgb: np.ndarray, render_rgb: np.ndarray) -> np.ndarray:
    query_gray = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2GRAY)
    render_gray = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2GRAY)
    query_gray = cv2.GaussianBlur(query_gray, (5, 5), 0)
    render_gray = cv2.GaussianBlur(render_gray, (5, 5), 0)
    query_edges = cv2.Canny(query_gray, 120, 240)
    render_edges = cv2.Canny(render_gray, 120, 240)
    base = make_overlay(query_rgb, render_rgb)
    out = base.copy()
    out[render_edges > 0] = [255, 40, 40]
    out[query_edges > 0] = [40, 255, 40]
    both = (query_edges > 0) & (render_edges > 0)
    out[both] = [255, 255, 40]
    return out


def make_checkerboard(query_rgb: np.ndarray, render_rgb: np.ndarray, tile: int = 32) -> np.ndarray:
    h, w = query_rgb.shape[:2]
    yy, xx = np.indices((h, w))
    mask = ((xx // tile) + (yy // tile)) % 2 == 0
    out = render_rgb.copy()
    out[mask] = query_rgb[mask]
    return out


def write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/caiwangcun_domdsm.yaml", type=Path)
    parser.add_argument("--image", default="data_caiwangcun/query/images/exif_test/0000.jpg", type=Path)
    parser.add_argument("--pose-file", default="data_caiwangcun/query/poses/exif_test.txt", type=Path)
    parser.add_argument("--output-dir", default="outputs/exif_test_single_512", type=Path)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--checker-tile", default=32, type=int)
    parser.add_argument("--near-m", default=300.0, type=float)
    parser.add_argument("--far-m", default=500.0, type=float)
    parser.add_argument("--ray-step-m", default=2.0, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    render_config = dict(config["render_config"])
    render_config["dom_dsm"] = dict(render_config["dom_dsm"])
    render_config["dom_dsm"]["debug_dir"] = None
    render_config["dom_dsm"]["near_m"] = args.near_m
    render_config["dom_dsm"]["far_m"] = args.far_m
    render_config["dom_dsm"]["ray_step_m"] = args.ray_step_m
    render_config["render_camera"] = scaled_camera(
        config["default_confs"]["cam_query"], args.width
    )

    trans, euler = read_pose_file(args.pose_file, args.image.name)
    query_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if query_bgr is None:
        raise FileNotFoundError(args.image)

    width = int(render_config["render_camera"][0])
    height = int(render_config["render_camera"][1])
    query_bgr = cv2.resize(query_bgr, (width, height), interpolation=cv2.INTER_AREA)
    query_rgb = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = DOMDSMRenderer(render_config)

    t0 = time.perf_counter()
    render_rgb, depth = renderer.render(trans, euler)
    render_time = time.perf_counter() - t0

    depth_vis = depth_to_vis(depth)
    overlay = make_overlay(query_rgb, render_rgb)
    edge_overlay = make_edge_overlay(query_rgb, render_rgb)
    checkerboard = make_checkerboard(query_rgb, render_rgb, args.checker_tile)

    write_rgb(args.output_dir / "rendered_rgb.png", render_rgb)
    cv2.imwrite(str(args.output_dir / "rendered_depth.png"), depth_vis)
    write_rgb(args.output_dir / "query_render_overlay.png", overlay)
    write_rgb(args.output_dir / "edge_overlay.png", edge_overlay)
    write_rgb(args.output_dir / "checkerboard_overlay.png", checkerboard)

    valid = depth > 0
    stats = {
        "valid_depth_ratio": float(valid.mean()),
        "depth_min": float(depth[valid].min()) if np.any(valid) else None,
        "depth_max": float(depth[valid].max()) if np.any(valid) else None,
        "render_time_sec": render_time,
        "render_params": {
            "near_m": args.near_m,
            "far_m": args.far_m,
            "ray_step_m": args.ray_step_m,
        },
        "pose": {
            "translation_lon_lat_alt": trans,
            "euler_pitch_roll_yaw": euler,
            "source_pose_format": "image_name lon lat alt roll pitch yaw",
        },
        "image_size": {"width": width, "height": height},
    }
    (args.output_dir / "render_stats_512.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
