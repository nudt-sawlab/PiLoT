#!/usr/bin/env python3
"""Analyze DJI yaw to DOMDSMRenderer yaw convention candidates.

This script only uses DOMDSMRenderer. It does not run RenderLocalizer or import
the CUDA refinement extension.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test_16x9/0000.jpg"
DEFAULT_OUTPUT_DIR = (
    "docs/experiments/dom_dsm_prepare/orientation_convention_results"
)
DEFAULT_TRANS = [114.4368608916, 30.3913609745, 391.462]
DEFAULT_DJI_YAW = -29.2
FIXED_PITCH = 0.0
FIXED_ROLL = 180.0


YawFn = Callable[[float], float]


YAW_CANDIDATES: List[Tuple[str, str, YawFn]] = [
    ("01_dji_yaw", "dji_yaw", lambda y: y),
    ("02_neg_dji_yaw", "-dji_yaw", lambda y: -y),
    ("03_90_minus_dji_yaw", "90 - dji_yaw", lambda y: 90.0 - y),
    ("04_dji_yaw_minus_90", "dji_yaw - 90", lambda y: y - 90.0),
    ("05_dji_yaw_plus_90", "dji_yaw + 90", lambda y: y + 90.0),
    ("06_dji_yaw_plus_180", "dji_yaw + 180", lambda y: y + 180.0),
    ("07_180_minus_dji_yaw", "180 - dji_yaw", lambda y: 180.0 - y),
    ("08_minus_90_minus_dji_yaw", "-90 - dji_yaw", lambda y: -90.0 - y),
]


def scaled_camera(cam_cfg: Dict[str, Any], render_width: int) -> np.ndarray:
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


def read_query_rgb(path: Path, width: int, height: int) -> np.ndarray:
    query_bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if query_bgr is None:
        raise FileNotFoundError(path)
    query_bgr = cv2.resize(query_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    cv2.imwrite(os.fspath(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def make_overlay(query_rgb: np.ndarray, render_rgb: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(query_rgb, 0.5, render_rgb, 0.5, 0)


def edges(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, 120, 240) > 0


def symmetric_chamfer(query_edges: np.ndarray, render_edges: np.ndarray) -> float:
    if not np.any(query_edges) or not np.any(render_edges):
        return float("inf")
    dist_to_render = cv2.distanceTransform(
        (~render_edges).astype(np.uint8), cv2.DIST_L2, 3
    )
    dist_to_query = cv2.distanceTransform(
        (~query_edges).astype(np.uint8), cv2.DIST_L2, 3
    )
    q_to_r = float(dist_to_render[query_edges].mean())
    r_to_q = float(dist_to_query[render_edges].mean())
    return (q_to_r + r_to_q) / 2.0


def make_edge_overlay(
    query_rgb: np.ndarray,
    render_rgb: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    query_edges = edges(query_rgb)
    render_edges = edges(render_rgb)

    kernel = np.ones((3, 3), dtype=np.uint8)
    query_dilated = cv2.dilate(query_edges.astype(np.uint8), kernel, iterations=1) > 0
    render_dilated = cv2.dilate(render_edges.astype(np.uint8), kernel, iterations=1) > 0
    overlap = (query_edges & render_dilated) | (render_edges & query_dilated)

    out = make_overlay(query_rgb, render_rgb)
    out[render_edges] = [255, 40, 40]
    out[query_edges] = [40, 255, 40]
    out[overlap] = [255, 255, 40]

    query_count = int(query_edges.sum())
    render_count = int(render_edges.sum())
    overlap_count = int(overlap.sum())
    return out, {
        "edge_overlap_ratio": float(overlap_count / max(min(query_count, render_count), 1)),
        "edge_chamfer": symmetric_chamfer(query_edges, render_edges),
        "query_edge_count": query_count,
        "render_edge_count": render_count,
        "edge_overlap_count": overlap_count,
    }


def make_checkerboard(
    query_rgb: np.ndarray,
    render_rgb: np.ndarray,
    tile: int,
) -> np.ndarray:
    height, width = query_rgb.shape[:2]
    yy, xx = np.indices((height, width))
    mask = ((xx // tile) + (yy // tile)) % 2 == 0
    out = render_rgb.copy()
    out[mask] = query_rgb[mask]
    return out


def depth_stats(depth: np.ndarray) -> Dict[str, Any]:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return {
            "valid_depth_ratio": 0.0,
            "depth_min": None,
            "depth_max": None,
        }
    return {
        "valid_depth_ratio": float(valid.mean()),
        "depth_min": float(depth[valid].min()),
        "depth_max": float(depth[valid].max()),
    }


def parse_trans(value: str) -> List[float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--trans must contain lon,lat,alt")
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--trans", default=",".join(map(str, DEFAULT_TRANS)), type=parse_trans)
    parser.add_argument("--dji-yaw", default=DEFAULT_DJI_YAW, type=float)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--checker-tile", default=32, type=int)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)

    config_path = (REPO_ROOT / args.config).resolve()
    query_image_path = (REPO_ROOT / args.query_image).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    render_config = dict(config["render_config"])
    render_config["dom_dsm"] = dict(render_config["dom_dsm"])
    render_config["dom_dsm"]["debug_dir"] = None
    render_config["render_camera"] = scaled_camera(
        config["default_confs"]["cam_query"],
        args.width,
    )

    width = int(render_config["render_camera"][0])
    height = int(render_config["render_camera"][1])
    query_rgb = read_query_rgb(query_image_path, width, height)
    renderer = DOMDSMRenderer(render_config)

    summary = {
        "config_path": os.fspath(args.config),
        "query_image_path": os.fspath(args.query_image),
        "output_dir": os.fspath(args.output_dir),
        "trans_lon_lat_alt": args.trans,
        "fixed_pitch_roll": [FIXED_PITCH, FIXED_ROLL],
        "dji_yaw": args.dji_yaw,
        "image_size": {"width": width, "height": height},
        "candidates": [],
    }

    for name, expression, yaw_fn in YAW_CANDIDATES:
        yaw = float(yaw_fn(args.dji_yaw))
        euler = [FIXED_PITCH, FIXED_ROLL, yaw]
        candidate_dir = output_dir / name
        candidate_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        render_rgb, depth = renderer.render(args.trans, euler)
        render_time = time.perf_counter() - t0

        overlay = make_overlay(query_rgb, render_rgb)
        edge_overlay, edge_metrics = make_edge_overlay(query_rgb, render_rgb)
        checkerboard = make_checkerboard(query_rgb, render_rgb, args.checker_tile)

        write_rgb(candidate_dir / "rendered_rgb.png", render_rgb)
        write_rgb(candidate_dir / "overlay.png", overlay)
        write_rgb(candidate_dir / "edge_overlay.png", edge_overlay)
        write_rgb(candidate_dir / "checkerboard.png", checkerboard)

        metrics = {
            "candidate": name,
            "yaw_expression": expression,
            "render_yaw": yaw,
            "euler_pitch_roll_yaw": euler,
            "translation_lon_lat_alt": args.trans,
            "render_time_sec": render_time,
            **depth_stats(depth),
            **edge_metrics,
        }
        (candidate_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary["candidates"].append(metrics)
        print(json.dumps(metrics, indent=2, sort_keys=True))

    summary["best_by_edge_overlap_ratio"] = max(
        summary["candidates"],
        key=lambda item: item["edge_overlap_ratio"],
    )["candidate"]
    summary["best_by_edge_chamfer"] = min(
        summary["candidates"],
        key=lambda item: item["edge_chamfer"],
    )["candidate"]
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
