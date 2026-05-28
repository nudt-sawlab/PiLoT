#!/usr/bin/env python3
"""Synthetic self-check for DOM/DSM render consistency with one XML pose."""

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
    _checkerboard,
    _edge_overlay,
    _make_overlay,
    _safe_jsonable,
    _write_rgb,
)
from tools.render_contextcapture_xml_domdsm_initial import _parse_xml


DEFAULT_XML = "data_caiwangcun/CaiWangCun.xml"
DEFAULT_CONFIG = "configs/caiwangcun_domdsm.yaml"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/synthetic_self_render_consistency_0000"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _camera_matrix(intr: Any) -> np.ndarray:
    return np.asarray(
        [
            [float(intr.fx), 0.0, float(intr.cx)],
            [0.0, float(intr.fy), float(intr.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


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


def _select_photo(photos: Any, photo_id: str) -> Any:
    for photo in photos:
        if str(photo.photo_id) == str(photo_id):
            return photo
    raise ValueError(f"Photo id {photo_id} not found in XML")


def _trans_wgs84(xml_srs: str, photo: Any) -> Tuple[float, float, float]:
    to_wgs84 = Transformer.from_crs(xml_srs, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(photo.center_xml[0], photo.center_xml[1])
    return float(lon), float(lat), float(photo.center_xml[2])


def _render_with_renderer(renderer: DOMDSMRenderer, trans: Tuple[float, float, float], photo: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], float]:
    t0 = time.perf_counter()
    color, depth = renderer.render_matrix(list(trans), photo.rotation.T)
    elapsed = time.perf_counter() - t0
    return color, depth, dict(renderer.last_render_metadata), float(elapsed)


def _valid_iou(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    va = np.isfinite(a) & (a > 0)
    vb = np.isfinite(b) & (b > 0)
    inter = va & vb
    union = va | vb
    return {
        "valid_a_ratio": float(va.mean()),
        "valid_b_ratio": float(vb.mean()),
        "valid_intersection_ratio": float(inter.mean()),
        "valid_union_ratio": float(union.mean()),
        "valid_iou": float(inter.sum() / max(int(union.sum()), 1)),
    }


def _compare_rgb(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return {
        "rgb_mae": float(diff.mean()),
        "rgb_p50": float(np.percentile(diff, 50)),
        "rgb_p95": float(np.percentile(diff, 95)),
        "rgb_max": float(diff.max()),
    }


def _compare_depth(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if not np.any(valid):
        return {
            "depth_mae_m": None,
            "depth_p50_m": None,
            "depth_p95_m": None,
            "depth_max_m": None,
            "depth_signed_mean_m": None,
        }
    signed = a[valid].astype(np.float64) - b[valid].astype(np.float64)
    abs_diff = np.abs(signed)
    return {
        "depth_mae_m": float(abs_diff.mean()),
        "depth_p50_m": float(np.percentile(abs_diff, 50)),
        "depth_p95_m": float(np.percentile(abs_diff, 95)),
        "depth_max_m": float(abs_diff.max()),
        "depth_signed_mean_m": float(signed.mean()),
    }


def _comparison_metrics(a_rgb: np.ndarray, b_rgb: np.ndarray, a_depth: np.ndarray, b_depth: np.ndarray) -> Dict[str, Any]:
    _edge_img, edge_metrics = _edge_overlay(a_rgb, b_rgb)
    return {
        **_compare_rgb(a_rgb, b_rgb),
        **_compare_depth(a_depth, b_depth),
        **_valid_iou(a_depth, b_depth),
        **edge_metrics,
        "width": int(a_rgb.shape[1]),
        "height": int(a_rgb.shape[0]),
        "same_shape": bool(a_rgb.shape == b_rgb.shape and a_depth.shape == b_depth.shape),
    }


def _write_absdiff_rgb(path: Path, a: np.ndarray, b: np.ndarray) -> None:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).clip(0, 255).astype(np.uint8)
    _write_rgb(path, diff)


def _write_depth_absdiff(path: Path, a: np.ndarray, b: np.ndarray) -> None:
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    diff = np.zeros_like(a, dtype=np.float32)
    diff[valid] = np.abs(a[valid].astype(np.float32) - b[valid].astype(np.float32))
    if np.any(valid):
        scale = np.percentile(diff[valid], 99)
        scale = float(scale) if scale > 1e-9 else 1.0
    else:
        scale = 1.0
    vis = np.clip(diff / scale * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(os.fspath(path), vis)


def _write_visuals(output_dir: Path, synthetic_rgb: np.ndarray, repeat_rgb: np.ndarray, synthetic_depth: np.ndarray, repeat_depth: np.ndarray, checker_tile: int) -> Dict[str, Any]:
    edge_overlay, edge_metrics = _edge_overlay(synthetic_rgb, repeat_rgb)
    _write_rgb(output_dir / "overlay_synthetic_render.png", _make_overlay(synthetic_rgb, repeat_rgb))
    _write_rgb(output_dir / "edge_overlay_synthetic_render.png", edge_overlay)
    _write_rgb(output_dir / "checkerboard_synthetic_render.png", _checkerboard(synthetic_rgb, repeat_rgb, checker_tile))
    _write_absdiff_rgb(output_dir / "rgb_absdiff.png", synthetic_rgb, repeat_rgb)
    _write_depth_absdiff(output_dir / "depth_absdiff.png", synthetic_depth, repeat_depth)
    return edge_metrics


def _pass_rgb(metrics: Dict[str, Any]) -> bool:
    return bool(float(metrics.get("rgb_mae", 1e9)) <= 0.01 and float(metrics.get("rgb_p95", 1e9)) <= 0.0 and float(metrics.get("rgb_max", 1e9)) <= 1.0)


def _pass_depth(metrics: Dict[str, Any]) -> bool:
    depth_p95 = metrics.get("depth_p95_m")
    return bool(depth_p95 is not None and float(depth_p95) <= 1e-4 and float(metrics.get("valid_iou", 0.0)) > 0.999)


def _pass_check(metrics: Dict[str, Any]) -> bool:
    return _pass_rgb(metrics) and _pass_depth(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--photo-id", default="7")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checker-tile", type=int, default=128)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(REPO_ROOT / args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    xml_srs, photos = _parse_xml((REPO_ROOT / args.xml).resolve())
    photo = _select_photo(photos, args.photo_id)
    intr = photo.intrinsics
    K_xml = _camera_matrix(intr)
    trans = _trans_wgs84(xml_srs, photo)
    render_config = _make_gpu_config(config, intr)

    renderer = DOMDSMRenderer(render_config)
    synthetic_rgb, synthetic_depth, meta_a, time_a = _render_with_renderer(renderer, trans, photo)
    repeat_rgb, repeat_depth, meta_b, time_b = _render_with_renderer(renderer, trans, photo)

    _write_rgb(output_dir / "synthetic_query.png", synthetic_rgb)
    _write_rgb(output_dir / "render_repeat.png", repeat_rgb)

    synthetic_roundtrip = _read_rgb(output_dir / "synthetic_query.png")
    synthetic_query_resized = bool(synthetic_roundtrip.shape != synthetic_rgb.shape)

    adapter_renderer = DOMDSMRenderer(render_config)
    adapter_rgb, adapter_depth, meta_adapter, time_adapter = _render_with_renderer(adapter_renderer, trans, photo)

    repeat_metrics = _comparison_metrics(synthetic_rgb, repeat_rgb, synthetic_depth, repeat_depth)
    roundtrip_metrics = _comparison_metrics(synthetic_roundtrip, repeat_rgb, synthetic_depth, repeat_depth)
    adapter_metrics = _comparison_metrics(synthetic_roundtrip, adapter_rgb, synthetic_depth, adapter_depth)
    visual_edge_metrics = _write_visuals(output_dir, synthetic_roundtrip, adapter_rgb, synthetic_depth, adapter_depth, args.checker_tile)

    all_sizes = {
        "synthetic_query": [int(synthetic_rgb.shape[1]), int(synthetic_rgb.shape[0])],
        "render_repeat": [int(repeat_rgb.shape[1]), int(repeat_rgb.shape[0])],
        "synthetic_roundtrip": [int(synthetic_roundtrip.shape[1]), int(synthetic_roundtrip.shape[0])],
        "pipeline_adapter_render": [int(adapter_rgb.shape[1]), int(adapter_rgb.shape[0])],
    }
    checks = {
        "repeat_gpu_same_call": {
            "passed": _pass_check(repeat_metrics),
            "metrics": repeat_metrics,
            "metadata_a": meta_a,
            "metadata_b": meta_b,
            "render_time_sec_a": time_a,
            "render_time_sec_b": time_b,
        },
        "saved_synthetic_roundtrip": {
            "passed": _pass_check(roundtrip_metrics) and not synthetic_query_resized,
            "metrics": roundtrip_metrics,
            "synthetic_query_resized": synthetic_query_resized,
        },
        "pipeline_adapter_check": {
            "passed": _pass_check(adapter_metrics),
            "metrics": adapter_metrics,
            "metadata": meta_adapter,
            "render_time_sec": time_adapter,
        },
    }

    if not checks["repeat_gpu_same_call"]["passed"]:
        conclusion = "repeat_gpu_same_call failed; inspect renderer determinism, cache, texture sampling, or GPU output path"
    elif not checks["saved_synthetic_roundtrip"]["passed"]:
        conclusion = "saved_synthetic_roundtrip failed; inspect PNG RGB/BGR, dtype, image read/write, or resize handling"
    elif not checks["pipeline_adapter_check"]["passed"]:
        conclusion = "pipeline_adapter_check failed; inspect XML pose adapter, K injection, dimensions, coordinate conversion, or render path"
    else:
        conclusion = "synthetic self-check passed; continue attributing real query-render error to real imaging differences, DOM phase, texture, or metric reliability"

    metrics = {
        "experiment": "Synthetic self render consistency",
        "xml": args.xml,
        "xml_srs": xml_srs,
        "photo_id": photo.photo_id,
        "xml_image_path": photo.image_path,
        "camera_center_xml_srs": photo.center_xml,
        "camera_center_lon_lat_alt": trans,
        "render_ray_rotation": "R_camera_to_world = R_xml.T",
        "config": args.config,
        "render_camera": [intr.width, intr.height, intr.cx, intr.cy, intr.fx, intr.fy],
        "K_xml": K_xml,
        "intrinsics": intr.as_dict(),
        "output_dir": args.output_dir,
        "image_sizes": all_sizes,
        "backend_used": meta_adapter.get("backend_used"),
        "fallback_reason": meta_adapter.get("fallback_reason"),
        "synthetic_query_resized": synthetic_query_resized,
        "checks": checks,
        "visual_edge_metrics": visual_edge_metrics,
        "all_checks_passed": bool(all(item["passed"] for item in checks.values())),
        "conclusion": conclusion,
    }
    _write_json(output_dir / "synthetic_self_check_metrics.json", metrics)
    print(json.dumps(_safe_jsonable(metrics), indent=2, sort_keys=True))
    return 0 if metrics["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
