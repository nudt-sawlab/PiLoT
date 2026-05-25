#!/usr/bin/env python3
"""Compare XML ray renderer against DOMDSMRenderer gpu_mesh for one XML photo."""

from __future__ import annotations

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
from pyproj import Transformer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from tools.diagnose_yawfix_refinement_update import (
    _edge_overlay,
    _make_overlay,
    _safe_jsonable,
    _write_rgb,
)
from tools.render_contextcapture_xml_domdsm_initial import (
    ContextCaptureDOMDSMRenderer,
    _load_pose_file_projected,
    _match_photos,
    _parse_xml,
)
from tools.run_dom_dsm_single_full import _save_depth_png


DEFAULT_XML = "data_caiwangcun/CaiWangCun.xml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test/0000.jpg"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_CONFIG = "configs/caiwangcun_domdsm.yaml"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/xml_ray_vs_gpu_mesh_renderer_0000"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _depth_diff_vis(ray_depth: np.ndarray, gpu_depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    vis = np.zeros(ray_depth.shape, dtype=np.uint8)
    if not np.any(valid):
        return vis
    diff = np.abs(gpu_depth.astype(np.float32) - ray_depth.astype(np.float32))
    vals = diff[valid]
    hi = max(float(np.percentile(vals, 99)), 1.0e-6)
    vis[valid] = np.clip(diff[valid] / hi * 255.0, 0, 255).astype(np.uint8)
    return vis


def _rgb_absdiff(ray_rgb: np.ndarray, gpu_rgb: np.ndarray) -> np.ndarray:
    diff = np.abs(ray_rgb.astype(np.int16) - gpu_rgb.astype(np.int16))
    return np.clip(diff, 0, 255).astype(np.uint8)


def _valid_stats(ray_depth: np.ndarray, gpu_depth: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    ray_valid = np.isfinite(ray_depth) & (ray_depth > 0)
    gpu_valid = np.isfinite(gpu_depth) & (gpu_depth > 0)
    intersection = ray_valid & gpu_valid
    union = ray_valid | gpu_valid
    total = float(ray_depth.size)
    return intersection, {
        "ray_valid_ratio": float(np.count_nonzero(ray_valid) / total),
        "gpu_valid_ratio": float(np.count_nonzero(gpu_valid) / total),
        "valid_intersection_ratio": float(np.count_nonzero(intersection) / total),
        "valid_union_ratio": float(np.count_nonzero(union) / total),
        "valid_iou": float(np.count_nonzero(intersection) / max(np.count_nonzero(union), 1)),
    }


def _comparison_stats(
    ray_rgb: np.ndarray,
    ray_depth: np.ndarray,
    gpu_rgb: np.ndarray,
    gpu_depth: np.ndarray,
    valid: np.ndarray,
) -> Dict[str, Any]:
    if not np.any(valid):
        return {
            "rgb_mae": None,
            "rgb_p95": None,
            "rgb_max": None,
            "depth_mae_m": None,
            "depth_p50_m": None,
            "depth_p95_m": None,
            "depth_max_m": None,
            "depth_signed_mean_m": None,
        }
    rgb_abs = np.abs(ray_rgb.astype(np.float32) - gpu_rgb.astype(np.float32))[valid]
    depth_signed = (gpu_depth.astype(np.float32) - ray_depth.astype(np.float32))[valid]
    depth_abs = np.abs(depth_signed)
    return {
        "rgb_mae": float(np.mean(rgb_abs)),
        "rgb_p95": float(np.percentile(rgb_abs, 95)),
        "rgb_max": float(np.max(rgb_abs)),
        "depth_mae_m": float(np.mean(depth_abs)),
        "depth_p50_m": float(np.percentile(depth_abs, 50)),
        "depth_p95_m": float(np.percentile(depth_abs, 95)),
        "depth_max_m": float(np.max(depth_abs)),
        "depth_signed_mean_m": float(np.mean(depth_signed)),
    }


def _make_gpu_config(config: Dict[str, Any], intr: Any) -> Dict[str, Any]:
    render_config = copy.deepcopy(config["render_config"])
    render_config["render_camera"] = [
        int(intr.width),
        int(intr.height),
        float(intr.cx),
        float(intr.cy),
        float(intr.fx),
        float(intr.fy),
    ]
    render_config["dom_dsm"]["render_backend"] = "gpu_mesh"
    render_config["dom_dsm"]["gpu_renderer"] = "nvdiffrast"
    render_config["dom_dsm"]["texture_v_flip"] = False
    render_config["dom_dsm"]["output_y_flip"] = True
    render_config["dom_dsm"]["debug_texture_mode"] = "none"
    render_config["dom_dsm"]["debug_every"] = 0
    return render_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ray-step-m", type=float, default=2.0)
    parser.add_argument("--ray-refine-iters", type=int, default=10)
    parser.add_argument("--chunk-rows", type=int, default=192)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    xml_path = (REPO_ROOT / args.xml).resolve()
    query_path = (REPO_ROOT / args.query_image).resolve()
    pose_path = (REPO_ROOT / args.pose_file).resolve()
    xml_srs, photos = _parse_xml(xml_path)
    pose_records = _load_pose_file_projected(pose_path, xml_srs)
    matches, match_report = _match_photos([query_path], pose_records, photos)
    item = matches[query_path.name]
    photo = item["photo"]
    intr = photo.intrinsics

    ray_renderer = ContextCaptureDOMDSMRenderer(config["render_config"], xml_srs, args.chunk_rows)
    t0 = time.perf_counter()
    ray_rgb, ray_depth, ray_debug = ray_renderer.render(
        photo,
        "cam_to_world_correct",
        distortion_enabled=False,
        axis_transform=(1.0, 1.0, 1.0),
        principal_point_mode="xml",
        sampling_mode="bilinear",
        ray_step_m=args.ray_step_m,
        render_scale=1.0,
        dsm_sampling_mode="bilinear",
        dom_sampling_mode="bilinear",
        ray_refine_iters=args.ray_refine_iters,
    )
    ray_time_sec = time.perf_counter() - t0

    gpu_config = _make_gpu_config(config, intr)
    gpu_renderer = DOMDSMRenderer(gpu_config)
    to_wgs84 = Transformer.from_crs(xml_srs, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(photo.center_xml[0], photo.center_xml[1])
    trans = [float(lon), float(lat), float(photo.center_xml[2])]
    R_camera_to_world = photo.rotation.T
    t0 = time.perf_counter()
    gpu_rgb, gpu_depth = gpu_renderer.render_matrix(trans, R_camera_to_world)
    gpu_time_sec = time.perf_counter() - t0

    if ray_rgb.shape != gpu_rgb.shape or ray_depth.shape != gpu_depth.shape:
        raise ValueError(
            f"Renderer output shape mismatch: ray_rgb={ray_rgb.shape} gpu_rgb={gpu_rgb.shape} "
            f"ray_depth={ray_depth.shape} gpu_depth={gpu_depth.shape}"
        )

    valid, valid_metrics = _valid_stats(ray_depth, gpu_depth)
    compare_metrics = _comparison_stats(ray_rgb, ray_depth, gpu_rgb, gpu_depth, valid)
    overlay = _make_overlay(ray_rgb, gpu_rgb)
    edge_overlay, edge_metrics = _edge_overlay(ray_rgb, gpu_rgb)
    rgb_diff = _rgb_absdiff(ray_rgb, gpu_rgb)
    depth_diff = _depth_diff_vis(ray_depth, gpu_depth, valid)
    ray_valid = (np.isfinite(ray_depth) & (ray_depth > 0))
    gpu_valid = (np.isfinite(gpu_depth) & (gpu_depth > 0))
    valid_xor = np.logical_xor(ray_valid, gpu_valid).astype(np.uint8) * 255

    _write_rgb(output_dir / "ray_render_rgb.png", ray_rgb)
    _write_rgb(output_dir / "gpu_mesh_render_rgb.png", gpu_rgb)
    _save_depth_png(output_dir / "ray_depth.png", ray_depth)
    _save_depth_png(output_dir / "gpu_mesh_depth.png", gpu_depth)
    _write_rgb(output_dir / "rgb_absdiff.png", rgb_diff)
    cv2.imwrite(os.fspath(output_dir / "depth_absdiff.png"), depth_diff)
    cv2.imwrite(os.fspath(output_dir / "valid_mask_xor.png"), valid_xor)
    _write_rgb(output_dir / "overlay_ray_gpu.png", overlay)
    _write_rgb(output_dir / "edge_overlay_ray_gpu.png", edge_overlay)

    metrics: Dict[str, Any] = {
        "experiment": "XML ray renderer vs DOMDSMRenderer gpu_mesh",
        "xml": args.xml,
        "xml_srs": xml_srs,
        "query_image_path": args.query_image,
        "pose_file_path": args.pose_file,
        "config": args.config,
        "output_dir": args.output_dir,
        "xml_photo_id": photo.photo_id,
        "xml_image_path": photo.image_path,
        "camera_center_xml_srs": photo.center_xml,
        "camera_center_lon_lat_alt": trans,
        "intrinsics": intr.as_dict(),
        "render_camera": gpu_config["render_camera"],
        "ray_renderer": {
            "class": "ContextCaptureDOMDSMRenderer",
            "distortion_enabled": False,
            "principal_point_mode": "xml",
            "dsm_sampling_mode": "bilinear",
            "dom_sampling_mode": "bilinear",
            "ray_step_m": args.ray_step_m,
            "ray_refine_iters": args.ray_refine_iters,
            "render_ray_rotation": "R_camera_to_world = R_xml.T",
            "render_time_sec": ray_time_sec,
            **ray_debug,
        },
        "gpu_renderer": {
            "class": "DOMDSMRenderer gpu_mesh",
            "render_ray_rotation": "R_camera_to_world = R_xml.T",
            "render_time_sec": gpu_time_sec,
            "metadata": gpu_renderer.last_render_metadata,
        },
        "width": int(ray_rgb.shape[1]),
        "height": int(ray_rgb.shape[0]),
        **valid_metrics,
        **compare_metrics,
        **edge_metrics,
        "geometry_alignment_good": bool(
            valid_metrics["valid_iou"] > 0.98
            and compare_metrics.get("depth_p95_m") is not None
            and float(compare_metrics["depth_p95_m"]) < 1.0
        ),
        "renderer_bug_likely": bool(
            valid_metrics["valid_iou"] < 0.95
            or (
                compare_metrics.get("depth_p95_m") is not None
                and float(compare_metrics["depth_p95_m"]) > 2.0
            )
        ),
        "camera_match_report": match_report,
    }
    _write_json(output_dir / "renderer_alignment_metrics.json", metrics)
    print(json.dumps(_safe_jsonable(metrics), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
