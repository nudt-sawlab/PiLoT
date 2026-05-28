#!/usr/bin/env python3
"""Render EXIF yawfix initial DOM/DSM views at native project resolution."""

import argparse
import csv
import json
import os
import shutil
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

from pixloc.utils.dom_dsm.candidate_scorer import load_query_poses_from_file
from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from pixloc.utils.dom_dsm.pose_adapter import make_downward_euler_from_yaw
from tools.diagnose_yawfix_refinement_update import (
    _checkerboard,
    _edge_overlay,
    _get_raster_transformers,
    _make_overlay,
    _safe_jsonable,
    _write_rgb,
)
from tools.run_dom_dsm_single_full import _depth_stats, _setup_camera


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_DIR = "data_caiwangcun/query/images/exif_test"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/initial_domdsm_fullres_exif_test"


class FastArrayDOMDSMRenderer(DOMDSMRenderer):
    """DOMDSMRenderer with in-memory nearest-neighbor raster sampling.

    The render loop is unchanged. Only raster sampling is replaced so full
    project-resolution diagnostic renders do not spend hours in rasterio.sample.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._dsm_array = self.dsm.read(1)
        self._dom_array = np.moveaxis(self.dom.read(), 0, -1)
        self._dsm_transform = self.dsm.transform
        self._dom_transform = self.dom.transform

    @staticmethod
    def _xy_to_rowcol(transform: Any, xs: np.ndarray, ys: np.ndarray) -> Any:
        cols = np.floor((xs - transform.c) / transform.a).astype(np.int64)
        rows = np.floor((ys - transform.f) / transform.e).astype(np.int64)
        return rows, cols

    def _sample_dsm(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        rows, cols = self._xy_to_rowcol(self._dsm_transform, xs, ys)
        out = np.full(xs.shape, np.nan, dtype=np.float32)
        valid = (
            (rows >= 0)
            & (rows < self._dsm_array.shape[0])
            & (cols >= 0)
            & (cols < self._dsm_array.shape[1])
        )
        if np.any(valid):
            out[valid] = self._dsm_array[rows[valid], cols[valid]].astype(np.float32)
        return out

    def _sample_dom(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        rows, cols = self._xy_to_rowcol(self._dom_transform, xs, ys)
        out = np.zeros((len(xs), 3), dtype=np.uint8)
        valid = (
            (rows >= 0)
            & (rows < self._dom_array.shape[0])
            & (cols >= 0)
            & (cols < self._dom_array.shape[1])
        )
        if np.any(valid):
            sample = self._dom_array[rows[valid], cols[valid]]
            if sample.ndim == 1:
                sample = np.repeat(sample[:, None], 3, axis=1)
            out[valid] = np.clip(sample[:, :3], 0, 255).astype(np.uint8)
        return out


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_query_rgb(path: Path, width: int, height: int) -> Any:
    bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    resized = False
    if bgr.shape[1] != width or bgr.shape[0] != height:
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        resized = True
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), resized


def _render_one(
    image_path: Path,
    pose: Dict[str, Any],
    renderer: DOMDSMRenderer,
    output_dir: Path,
    width: int,
    height: int,
    checker_tile: int,
) -> Dict[str, Any]:
    image_out = output_dir / image_path.stem / "initial"
    image_out.mkdir(parents=True, exist_ok=True)
    query_rgb, query_resized = _read_query_rgb(image_path, width, height)
    trans = [float(x) for x in pose["translation_lon_lat_alt"]]
    base_yaw = float(pose["base_yaw"])
    euler = make_downward_euler_from_yaw(base_yaw)

    t0 = time.perf_counter()
    render_rgb, depth = renderer.render(trans, euler)
    render_time = time.perf_counter() - t0

    overlay = _make_overlay(query_rgb, render_rgb)
    edge_overlay, edge_metrics = _edge_overlay(query_rgb, render_rgb)
    checkerboard = _checkerboard(query_rgb, render_rgb, checker_tile)

    _write_rgb(image_out / "query_resized_to_render_camera.png", query_rgb)
    _write_rgb(image_out / "rendered_rgb.png", render_rgb)
    _write_rgb(image_out / "overlay.png", overlay)
    _write_rgb(image_out / "edge_overlay.png", edge_overlay)
    _write_rgb(image_out / "checkerboard.png", checkerboard)

    metrics = {
        "image": image_path.name,
        "candidate": "initial",
        "query_image_path": str(image_path),
        "output_dir": str(image_out),
        "render_width": int(width),
        "render_height": int(height),
        "source_query_width": int(cv2.imread(os.fspath(image_path), cv2.IMREAD_COLOR).shape[1]),
        "source_query_height": int(cv2.imread(os.fspath(image_path), cv2.IMREAD_COLOR).shape[0]),
        "query_resized_for_overlay": bool(query_resized),
        "translation_lon_lat_alt": trans,
        "euler_pitch_roll_yaw": euler,
        "render_time_sec": float(render_time),
        **_depth_stats(depth),
        **edge_metrics,
    }
    _write_json(image_out / "metrics.json", metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=None, help="Render width. Defaults to config cam_query.width.")
    parser.add_argument("--checker-tile", type=int, default=128)
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--use-fast-array-renderer", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    native_width = int(config["default_confs"]["cam_query"]["width"])
    native_height = int(config["default_confs"]["cam_query"]["height"])
    render_width = int(args.width or native_width)
    config["default_confs"]["cam_query"]["max_size"] = render_width
    _, _, render_camera_gs, _, _ = _setup_camera(config)
    width = int(render_camera_gs[0])
    height = int(render_camera_gs[1])

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer_cls = FastArrayDOMDSMRenderer if args.use_fast_array_renderer else DOMDSMRenderer
    renderer = renderer_cls(config["render_config"])
    _, _, raster_crs = _get_raster_transformers(config)
    pose_map = load_query_poses_from_file(str(REPO_ROOT / args.pose_file))
    query_dir = (REPO_ROOT / args.query_dir).resolve()
    images = sorted([p for p in query_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}])

    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for image_path in images:
        key = image_path.name if image_path.name in pose_map else image_path.name.lower()
        if key not in pose_map:
            skipped.append({"image": image_path.name, "reason": "missing pose"})
            continue
        metrics_path = output_dir / image_path.stem / "initial" / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
            continue
        rows.append(_render_one(image_path, pose_map[key], renderer, output_dir, width, height, args.checker_tile))

    with (output_dir / "initial_render_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "image",
            "render_width",
            "render_height",
            "source_query_width",
            "source_query_height",
            "edge_chamfer",
            "edge_overlap_ratio",
            "query_edge_count",
            "render_edge_count",
            "edge_overlap_count",
            "valid_depth_ratio",
            "depth_min",
            "depth_max",
            "render_time_sec",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "initial DOM/DSM full-resolution EXIF yawfix render",
        "config": args.config,
        "query_dir": args.query_dir,
        "pose_file": args.pose_file,
        "output_dir": args.output_dir,
        "native_config_width": native_width,
        "native_config_height": native_height,
        "render_width": width,
        "render_height": height,
        "raster_crs": raster_crs,
        "renderer": renderer_cls.__name__,
        "num_images_found": len(images),
        "num_images_rendered": len(rows),
        "skipped_images": skipped,
        "metrics_mean": {
            "edge_chamfer": float(np.mean([r["edge_chamfer"] for r in rows])) if rows else None,
            "edge_overlap_ratio": float(np.mean([r["edge_overlap_ratio"] for r in rows])) if rows else None,
            "valid_depth_ratio": float(np.mean([r["valid_depth_ratio"] for r in rows])) if rows else None,
        },
        "images": rows,
    }
    _write_json(output_dir / "summary_metrics.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
