#!/usr/bin/env python3
"""Render selected PiLoT/P13 optimized poses at native DOM/DSM resolution."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_yawfix_refinement_update import (
    _checkerboard,
    _edge_overlay,
    _make_overlay,
    _safe_jsonable,
    _write_rgb,
)
from tools.render_initial_domdsm_exif_test_fullres import FastArrayDOMDSMRenderer
from tools.run_dom_dsm_single_full import _depth_stats, _setup_camera


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_DIR = "data_caiwangcun/query/images/exif_test"
DEFAULT_SELECTED_DIR = "docs/experiments/dom_dsm_prepare/visual_objective_optimizer_p13"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/initial_domdsm_fullres_exif_test"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_query_rgb(path: Path, width: int, height: int) -> np.ndarray:
    bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    if bgr.shape[1] != width or bgr.shape[0] != height:
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_selected_pose(selected_dir: Path, stem: str) -> Dict[str, Any]:
    summary_path = selected_dir / stem / "summary_metrics.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary["selected"]


def _render_one(
    image_path: Path,
    selected: Dict[str, Any],
    renderer: FastArrayDOMDSMRenderer,
    output_dir: Path,
    width: int,
    height: int,
    checker_tile: int,
) -> Dict[str, Any]:
    out_dir = output_dir / image_path.stem / "pilot_optimized"
    out_dir.mkdir(parents=True, exist_ok=True)
    query_rgb = _read_query_rgb(image_path, width, height)
    trans = [float(x) for x in selected["translation_lon_lat_alt"]]
    euler = [float(x) for x in selected["euler_pitch_roll_yaw"]]

    t0 = time.perf_counter()
    render_rgb, depth = renderer.render(trans, euler)
    render_time = time.perf_counter() - t0

    overlay = _make_overlay(query_rgb, render_rgb)
    edge_overlay, edge_metrics = _edge_overlay(query_rgb, render_rgb)
    checkerboard = _checkerboard(query_rgb, render_rgb, checker_tile)

    _write_rgb(out_dir / "query_resized_to_render_camera.png", query_rgb)
    _write_rgb(out_dir / "rendered_rgb.png", render_rgb)
    _write_rgb(out_dir / "overlay.png", overlay)
    _write_rgb(out_dir / "edge_overlay.png", edge_overlay)
    _write_rgb(out_dir / "checkerboard.png", checkerboard)

    metrics = {
        "image": image_path.name,
        "candidate": selected.get("candidate", "pilot_optimized"),
        "source": "visual_objective_optimizer_p13_selected_pose",
        "query_image_path": str(image_path),
        "output_dir": str(out_dir),
        "render_width": int(width),
        "render_height": int(height),
        "translation_lon_lat_alt": trans,
        "euler_pitch_roll_yaw": euler,
        "offset_east_m": selected.get("offset_east_m"),
        "offset_north_m": selected.get("offset_north_m"),
        "offset_alt_m": selected.get("offset_alt_m", 0.0),
        "yaw_offset_deg": selected.get("yaw_offset_deg"),
        "render_time_sec": float(render_time),
        **_depth_stats(depth),
        **edge_metrics,
    }
    _write_json(out_dir / "metrics.json", metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    parser.add_argument("--selected-dir", default=DEFAULT_SELECTED_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--images", nargs="+", default=["0000.jpg", "0001.JPG", "0002.JPG"])
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--checker-tile", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    native_width = int(config["default_confs"]["cam_query"]["width"])
    render_width = int(args.width or native_width)
    config["default_confs"]["cam_query"]["max_size"] = render_width
    _, _, render_camera_gs, _, _ = _setup_camera(config)
    width = int(render_camera_gs[0])
    height = int(render_camera_gs[1])

    renderer = FastArrayDOMDSMRenderer(config["render_config"])
    query_dir = (REPO_ROOT / args.query_dir).resolve()
    selected_dir = (REPO_ROOT / args.selected_dir).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()

    rows: List[Dict[str, Any]] = []
    for image_name in args.images:
        image_path = query_dir / image_name
        selected = _load_selected_pose(selected_dir, image_path.stem)
        optimized = _render_one(image_path, selected, renderer, output_dir, width, height, args.checker_tile)
        initial_path = output_dir / image_path.stem / "initial" / "metrics.json"
        if initial_path.exists():
            initial = json.loads(initial_path.read_text(encoding="utf-8"))
            optimized["initial_edge_chamfer"] = initial.get("edge_chamfer")
            optimized["initial_edge_overlap_ratio"] = initial.get("edge_overlap_ratio")
            optimized["chamfer_improvement_vs_initial_fullres"] = float(initial["edge_chamfer"]) - float(optimized["edge_chamfer"])
            optimized["overlap_improvement_vs_initial_fullres"] = float(optimized["edge_overlap_ratio"]) - float(optimized["edge_overlap_ratio"])
            optimized["overlap_improvement_vs_initial_fullres"] = float(optimized["edge_overlap_ratio"]) - float(initial["edge_overlap_ratio"])
            optimized["safe_output_worse_than_initial_fullres"] = (
                float(optimized["edge_chamfer"]) > float(initial["edge_chamfer"]) + 1e-9
                or float(optimized["edge_overlap_ratio"]) + 1e-9 < float(initial["edge_overlap_ratio"])
            )
            _write_json(output_dir / image_path.stem / "pilot_optimized" / "metrics.json", optimized)
        rows.append(optimized)

    csv_path = output_dir / "pilot_optimized_render_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "image",
            "candidate",
            "render_width",
            "render_height",
            "edge_chamfer",
            "edge_overlap_ratio",
            "initial_edge_chamfer",
            "initial_edge_overlap_ratio",
            "chamfer_improvement_vs_initial_fullres",
            "overlap_improvement_vs_initial_fullres",
            "safe_output_worse_than_initial_fullres",
            "valid_depth_ratio",
            "query_edge_count",
            "render_edge_count",
            "edge_overlap_count",
            "render_time_sec",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "full-resolution DOM/DSM render of selected optimized poses",
        "selected_pose_source": args.selected_dir,
        "output_dir": args.output_dir,
        "render_width": width,
        "render_height": height,
        "num_images_rendered": len(rows),
        "metrics_mean": {
            "edge_chamfer": float(np.mean([r["edge_chamfer"] for r in rows])) if rows else None,
            "edge_overlap_ratio": float(np.mean([r["edge_overlap_ratio"] for r in rows])) if rows else None,
            "chamfer_improvement_vs_initial_fullres": float(np.mean([r["chamfer_improvement_vs_initial_fullres"] for r in rows if "chamfer_improvement_vs_initial_fullres" in r])) if rows else None,
            "overlap_improvement_vs_initial_fullres": float(np.mean([r["overlap_improvement_vs_initial_fullres"] for r in rows if "overlap_improvement_vs_initial_fullres" in r])) if rows else None,
        },
        "images": rows,
    }
    _write_json(output_dir / "pilot_optimized_summary_metrics.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
