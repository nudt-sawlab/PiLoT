#!/usr/bin/env python3
"""Visual validation for initial vs refined DOM+DSM pose renders."""

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.pixlib.geometry import Camera
from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from pixloc.utils.get_depth import generate_render_camera


DEFAULT_CONFIG = "configs/caiwangcun_domdsm.yaml"
DEFAULT_RUN_LOG = "outputs/exif_test_single_full/run_log.json"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test/0000.jpg"
DEFAULT_OUTPUT_DIR = "outputs/exif_test_single_full/visual_check"


def _write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    cv2.imwrite(os.fspath(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def _setup_render_camera(config: Dict[str, Any]) -> np.ndarray:
    cam_cfg = copy.deepcopy(config["default_confs"]["cam_query"])
    query_resize_ratio = cam_cfg["width"] / cam_cfg["max_size"]
    fx, _fy, _cx, _cy = cam_cfg["params"]
    width, height = cam_cfg["width"], cam_cfg["height"]
    render_camera = np.array(
        [width, height, width / 2, height / 2, fx, fx],
        dtype=np.float64,
    )
    render_camera = render_camera / query_resize_ratio

    # Keep this equivalent to the full single-image script and main.py setup.
    config["render_config"]["render_camera"] = render_camera
    _ = Camera.from_colmap(
        {
            "model": cam_cfg["model"],
            "width": cam_cfg["width"] / query_resize_ratio,
            "height": cam_cfg["height"] / query_resize_ratio,
            "params": np.asarray(cam_cfg["params"], dtype=np.float64)
            / query_resize_ratio,
        }
    )
    _ = generate_render_camera(render_camera).float()
    return render_camera


def _pose_from_log(entry: Dict[str, Any]) -> Tuple[list, list]:
    trans = entry["translation_lon_lat_alt"]
    euler = entry["euler_pitch_roll_yaw"]
    return trans, euler


def _read_query_rgb(path: Path, width: int, height: int) -> np.ndarray:
    query_bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if query_bgr is None:
        raise FileNotFoundError(path)
    query_bgr = cv2.resize(query_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)


def _make_overlay(query_rgb: np.ndarray, render_rgb: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(query_rgb, 0.5, render_rgb, 0.5, 0)


def _edges(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, 120, 240) > 0


def _edge_overlay(query_rgb: np.ndarray, render_rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    query_edges = _edges(query_rgb)
    render_edges = _edges(render_rgb)

    kernel = np.ones((3, 3), dtype=np.uint8)
    query_dilated = cv2.dilate(query_edges.astype(np.uint8), kernel, iterations=1) > 0
    render_dilated = cv2.dilate(render_edges.astype(np.uint8), kernel, iterations=1) > 0
    overlap = (query_edges & render_dilated) | (render_edges & query_dilated)

    out = _make_overlay(query_rgb, render_rgb)
    out[render_edges] = [255, 40, 40]
    out[query_edges] = [40, 255, 40]
    out[overlap] = [255, 255, 40]

    query_count = int(query_edges.sum())
    render_count = int(render_edges.sum())
    overlap_count = int(overlap.sum())
    edge_overlap_ratio = float(
        overlap_count / max(min(query_count, render_count), 1)
    )

    chamfer = _symmetric_chamfer(query_edges, render_edges)
    return out, {
        "edge_overlap_ratio": edge_overlap_ratio,
        "edge_chamfer": chamfer,
        "query_edge_count": query_count,
        "render_edge_count": render_count,
        "edge_overlap_count": overlap_count,
    }


def _symmetric_chamfer(query_edges: np.ndarray, render_edges: np.ndarray) -> float:
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


def _checkerboard(
    query_rgb: np.ndarray,
    render_rgb: np.ndarray,
    tile: int = 32,
) -> np.ndarray:
    height, width = query_rgb.shape[:2]
    yy, xx = np.indices((height, width))
    mask = ((xx // tile) + (yy // tile)) % 2 == 0
    out = render_rgb.copy()
    out[mask] = query_rgb[mask]
    return out


def _depth_stats(depth: np.ndarray) -> Dict[str, Any]:
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


def _render_and_write(
    label: str,
    renderer: DOMDSMRenderer,
    query_rgb: np.ndarray,
    trans: list,
    euler: list,
    output_dir: Path,
    checker_tile: int,
) -> Dict[str, Any]:
    t0 = time.time()
    render_rgb, depth = renderer.render(trans, euler)
    render_time = time.time() - t0

    overlay = _make_overlay(query_rgb, render_rgb)
    edge_overlay, edge_metrics = _edge_overlay(query_rgb, render_rgb)
    checkerboard = _checkerboard(query_rgb, render_rgb, checker_tile)

    _write_rgb(output_dir / f"{label}_rendered_rgb.png", render_rgb)
    _write_rgb(output_dir / f"{label}_overlay.png", overlay)
    _write_rgb(output_dir / f"{label}_edge_overlay.png", edge_overlay)
    _write_rgb(output_dir / f"{label}_checkerboard.png", checkerboard)

    stats = _depth_stats(depth)
    return {
        "render_time_sec": render_time,
        "translation_lon_lat_alt": trans,
        "euler_pitch_roll_yaw": euler,
        **stats,
        **edge_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("--run-log", default=DEFAULT_RUN_LOG, type=Path)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--checker-tile", default=32, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_log = json.loads(args.run_log.read_text(encoding="utf-8"))
    if "refined_pose" not in run_log:
        raise KeyError(f"refined_pose missing from {args.run_log}")

    render_camera = _setup_render_camera(config)
    width = int(render_camera[0])
    height = int(render_camera[1])
    query_rgb = _read_query_rgb(args.query_image, width, height)

    render_config = config["render_config"]
    render_config["dom_dsm"] = dict(render_config["dom_dsm"])
    render_config["dom_dsm"]["debug_dir"] = None

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = DOMDSMRenderer(render_config)
    initial_trans, initial_euler = _pose_from_log(run_log["initial_pose"])
    refined_trans, refined_euler = _pose_from_log(run_log["refined_pose"])

    initial = _render_and_write(
        "initial",
        renderer,
        query_rgb,
        initial_trans,
        initial_euler,
        output_dir,
        args.checker_tile,
    )
    refined = _render_and_write(
        "refined",
        renderer,
        query_rgb,
        refined_trans,
        refined_euler,
        output_dir,
        args.checker_tile,
    )

    metrics = {
        "config_path": os.fspath(args.config),
        "run_log_path": os.fspath(args.run_log),
        "query_image_path": os.fspath(args.query_image),
        "output_dir": os.fspath(output_dir),
        "image_size": {"width": width, "height": height},
        "initial_valid_depth_ratio": initial["valid_depth_ratio"],
        "refined_valid_depth_ratio": refined["valid_depth_ratio"],
        "initial_edge_overlap_ratio": initial["edge_overlap_ratio"],
        "refined_edge_overlap_ratio": refined["edge_overlap_ratio"],
        "initial_edge_chamfer": initial["edge_chamfer"],
        "refined_edge_chamfer": refined["edge_chamfer"],
        "initial": initial,
        "refined": refined,
    }
    (output_dir / "visual_compare_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
