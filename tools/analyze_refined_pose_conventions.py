#!/usr/bin/env python3
"""Analyze whether refined_pose needs conversion before DOMDSMRenderer render."""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_RUN_LOG = "outputs/exif_test_16x9_yawfix_single_full/run_log.json"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test_16x9/0000.jpg"
DEFAULT_OUTPUT_DIR = (
    "docs/experiments/dom_dsm_prepare/refined_pose_convention_results"
)


def scaled_camera(cam_cfg: Dict[str, Any], render_width: int) -> np.ndarray:
    source_w = float(cam_cfg["width"])
    source_h = float(cam_cfg["height"])
    scale = render_width / source_w
    render_height = int(round(source_h * scale))
    fx, fy, cx, cy = map(float, cam_cfg["params"])
    return np.array(
        [render_width, render_height, cx * scale, cy * scale, fx * scale, fy * scale],
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
    return float((dist_to_render[query_edges].mean() + dist_to_query[render_edges].mean()) / 2.0)


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


def make_checkerboard(query_rgb: np.ndarray, render_rgb: np.ndarray, tile: int) -> np.ndarray:
    height, width = query_rgb.shape[:2]
    yy, xx = np.indices((height, width))
    mask = ((xx // tile) + (yy // tile)) % 2 == 0
    out = render_rgb.copy()
    out[mask] = query_rgb[mask]
    return out


def depth_stats(depth: np.ndarray) -> Dict[str, Any]:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return {"valid_depth_ratio": 0.0, "depth_min": None, "depth_max": None}
    return {
        "valid_depth_ratio": float(valid.mean()),
        "depth_min": float(depth[valid].min()),
        "depth_max": float(depth[valid].max()),
    }


def pose_from_log(entry: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    return (
        list(map(float, entry["translation_lon_lat_alt"])),
        list(map(float, entry["euler_pitch_roll_yaw"])),
    )


def equivalent_downward_form(refined_euler: List[float]) -> List[float]:
    pitch, roll, yaw = refined_euler
    return [pitch - 180.0, roll + 180.0, yaw + 174.0]


def build_candidates(run_log: Dict[str, Any]) -> List[Dict[str, Any]]:
    initial_trans, initial_euler = pose_from_log(run_log["initial_pose"])
    refined_trans, refined_euler = pose_from_log(run_log["refined_pose"])
    pitch, roll, yaw = refined_euler
    eq_euler = equivalent_downward_form(refined_euler)
    return [
        {
            "name": "01_initial_baseline",
            "description": "Yawfix initial pose baseline.",
            "translation_lon_lat_alt": initial_trans,
            "euler_pitch_roll_yaw": initial_euler,
        },
        {
            "name": "02_direct_refined",
            "description": "Use refined euler and translation exactly as logged.",
            "translation_lon_lat_alt": refined_trans,
            "euler_pitch_roll_yaw": refined_euler,
        },
        {
            "name": "03_neg_refined_yaw",
            "description": "Use refined pitch/roll, but negate refined yaw.",
            "translation_lon_lat_alt": refined_trans,
            "euler_pitch_roll_yaw": [pitch, roll, -yaw],
        },
        {
            "name": "04_refined_yaw_plus_180",
            "description": "Use refined pitch/roll, yaw = refined_yaw + 180.",
            "translation_lon_lat_alt": refined_trans,
            "euler_pitch_roll_yaw": [pitch, roll, yaw + 180.0],
        },
        {
            "name": "05_refined_yaw_minus_180",
            "description": "Use refined pitch/roll, yaw = refined_yaw - 180.",
            "translation_lon_lat_alt": refined_trans,
            "euler_pitch_roll_yaw": [pitch, roll, yaw - 180.0],
        },
        {
            "name": "06_swap_roll_pitch",
            "description": "Swap refined pitch and roll.",
            "translation_lon_lat_alt": refined_trans,
            "euler_pitch_roll_yaw": [roll, pitch, yaw],
        },
        {
            "name": "07_equivalent_downward_form",
            "description": "Convert near [180,0,-144.8] to near [0,180,29.2].",
            "translation_lon_lat_alt": refined_trans,
            "euler_pitch_roll_yaw": eq_euler,
        },
        {
            "name": "08_keep_initial_rotation_refined_translation",
            "description": "Use refined translation with initial euler.",
            "translation_lon_lat_alt": refined_trans,
            "euler_pitch_roll_yaw": initial_euler,
        },
        {
            "name": "09_keep_initial_translation_refined_rotation",
            "description": "Use initial translation with direct refined euler.",
            "translation_lon_lat_alt": initial_trans,
            "euler_pitch_roll_yaw": refined_euler,
        },
        {
            "name": "10_equiv_initial_translation",
            "description": "Use initial translation with equivalent downward refined euler.",
            "translation_lon_lat_alt": initial_trans,
            "euler_pitch_roll_yaw": eq_euler,
        },
        {
            "name": "11_equiv_refined_yaw_negated",
            "description": "Use equivalent downward form with yaw sign flipped.",
            "translation_lon_lat_alt": refined_trans,
            "euler_pitch_roll_yaw": [eq_euler[0], eq_euler[1], -eq_euler[2]],
        },
        {
            "name": "12_refined_translation_initial_yaw_only",
            "description": "Use refined translation and initial downward rotation/yaw.",
            "translation_lon_lat_alt": refined_trans,
            "euler_pitch_roll_yaw": [0.0, 180.0, initial_euler[2]],
        },
    ]


def render_candidate(
    candidate: Dict[str, Any],
    renderer: DOMDSMRenderer,
    query_rgb: np.ndarray,
    output_dir: Path,
    checker_tile: int,
) -> Dict[str, Any]:
    candidate_dir = output_dir / candidate["name"]
    candidate_dir.mkdir(parents=True, exist_ok=True)
    trans = candidate["translation_lon_lat_alt"]
    euler = candidate["euler_pitch_roll_yaw"]

    t0 = time.perf_counter()
    render_rgb, depth = renderer.render(trans, euler)
    render_time = time.perf_counter() - t0

    overlay = make_overlay(query_rgb, render_rgb)
    edge_overlay, edge_metrics = make_edge_overlay(query_rgb, render_rgb)
    checkerboard = make_checkerboard(query_rgb, render_rgb, checker_tile)

    write_rgb(candidate_dir / "rendered_rgb.png", render_rgb)
    write_rgb(candidate_dir / "overlay.png", overlay)
    write_rgb(candidate_dir / "edge_overlay.png", edge_overlay)
    write_rgb(candidate_dir / "checkerboard.png", checkerboard)

    metrics = {
        "candidate": candidate["name"],
        "description": candidate["description"],
        "translation_lon_lat_alt": trans,
        "euler_pitch_roll_yaw": euler,
        "render_time_sec": render_time,
        **depth_stats(depth),
        **edge_metrics,
    }
    (candidate_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("--run-log", default=DEFAULT_RUN_LOG, type=Path)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--checker-tile", default=32, type=int)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    run_log = json.loads((REPO_ROOT / args.run_log).read_text(encoding="utf-8"))
    render_config = dict(config["render_config"])
    render_config["dom_dsm"] = dict(render_config["dom_dsm"])
    render_config["dom_dsm"]["debug_dir"] = None
    render_config["render_camera"] = scaled_camera(
        config["default_confs"]["cam_query"],
        args.width,
    )
    width = int(render_config["render_camera"][0])
    height = int(render_config["render_camera"][1])
    query_rgb = read_query_rgb((REPO_ROOT / args.query_image), width, height)
    renderer = DOMDSMRenderer(render_config)

    candidates = build_candidates(run_log)
    metrics = []
    for candidate in candidates:
        item = render_candidate(candidate, renderer, query_rgb, output_dir, args.checker_tile)
        metrics.append(item)
        print(json.dumps(item, indent=2, sort_keys=True))

    sorted_by_overlap = sorted(metrics, key=lambda item: item["edge_overlap_ratio"], reverse=True)
    sorted_by_chamfer = sorted(metrics, key=lambda item: item["edge_chamfer"])
    summary = {
        "config_path": os.fspath(args.config),
        "run_log_path": os.fspath(args.run_log),
        "query_image_path": os.fspath(args.query_image),
        "output_dir": os.fspath(args.output_dir),
        "image_size": {"width": width, "height": height},
        "initial_pose": run_log["initial_pose"],
        "refined_pose": run_log["refined_pose"],
        "candidates": metrics,
        "sorted_by_edge_overlap_ratio": sorted_by_overlap,
        "sorted_by_edge_chamfer": sorted_by_chamfer,
        "best_by_edge_overlap_ratio": sorted_by_overlap[0]["candidate"],
        "best_by_edge_chamfer": sorted_by_chamfer[0]["candidate"],
    }
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
