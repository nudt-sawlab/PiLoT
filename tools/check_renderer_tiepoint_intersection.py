#!/usr/bin/env python3
"""Compare renderer ray/DSM intersections against XML tie point 3D positions."""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.check_contextcapture_tiepoint_domdsm_consistency import _crop_patch, _patch_mad, _read_query
from tools.check_contextcapture_xml_reprojection import PROJECTION_CONVENTIONS, _parse_tiepoints, _project
from tools.check_query_domdsm_point_consistency import _intersect_dsm, _ray_from_pixel
from tools.diagnose_yawfix_refinement_update import _safe_jsonable, _write_rgb
from tools.render_contextcapture_xml_domdsm_initial import (
    ContextCaptureDOMDSMRenderer,
    RENDER_RAY_ROTATION,
    XML_PROJECTION_ROTATION,
    _load_convention_file,
    _load_pose_file_projected,
    _match_photos,
    _parse_xml,
)


DEFAULT_XML = "data_caiwangcun/CaiWangCun.xml"
DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_DIR = "data_caiwangcun/query/images/exif_test"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_CONVENTION = "docs/experiments/dom_dsm_prepare/contextcapture_xml_camera_convention_diagnosis/best_convention.json"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/renderer_tiepoint_intersection_check"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else None


def _percentile(values: Sequence[Optional[float]], q: float) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.percentile(np.asarray(finite, dtype=np.float64), q)) if finite else None


def _mad(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = np.asarray(
        [float(v) for v in values if v is not None and np.isfinite(float(v))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return None
    med = np.median(finite)
    return float(np.median(np.abs(finite - med)))


def _std(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = np.asarray(
        [float(v) for v in values if v is not None and np.isfinite(float(v))],
        dtype=np.float64,
    )
    return float(np.std(finite)) if finite.size else None


def _vector_norm(east: float, north: float) -> float:
    return float(math.hypot(float(east), float(north)))


def _offset_stats(rows: List[Dict[str, Any]], prefix: str = "") -> Dict[str, Any]:
    hit_rows = [
        r
        for r in rows
        if r.get("hit")
        and r.get("offset_east_m") is not None
        and r.get("offset_north_m") is not None
        and np.isfinite(float(r["offset_east_m"]))
        and np.isfinite(float(r["offset_north_m"]))
    ]
    if not hit_rows:
        return {
            f"{prefix}num_offset_samples": 0,
            f"{prefix}direction_consistency": None,
            f"{prefix}stable_horizontal_offset_likely": False,
        }
    east = np.asarray([float(r["offset_east_m"]) for r in hit_rows], dtype=np.float64)
    north = np.asarray([float(r["offset_north_m"]) for r in hit_rows], dtype=np.float64)
    norm = np.asarray([float(r["offset_norm_m"]) for r in hit_rows], dtype=np.float64)
    east_mean = float(east.mean())
    north_mean = float(north.mean())
    mean_norm = _vector_norm(east_mean, north_mean)
    mean_sample_norm = float(norm.mean())
    direction_consistency = mean_norm / max(mean_sample_norm, 1.0e-9)
    east_median = float(np.median(east))
    north_median = float(np.median(north))
    median_norm = _vector_norm(east_median, north_median)
    return {
        f"{prefix}num_offset_samples": int(len(hit_rows)),
        f"{prefix}offset_east_m_mean": east_mean,
        f"{prefix}offset_east_m_median": east_median,
        f"{prefix}offset_east_m_std": float(np.std(east)),
        f"{prefix}offset_east_m_mad": float(np.median(np.abs(east - east_median))),
        f"{prefix}offset_east_m_p10": float(np.percentile(east, 10)),
        f"{prefix}offset_east_m_p90": float(np.percentile(east, 90)),
        f"{prefix}offset_north_m_mean": north_mean,
        f"{prefix}offset_north_m_median": north_median,
        f"{prefix}offset_north_m_std": float(np.std(north)),
        f"{prefix}offset_north_m_mad": float(np.median(np.abs(north - north_median))),
        f"{prefix}offset_north_m_p10": float(np.percentile(north, 10)),
        f"{prefix}offset_north_m_p90": float(np.percentile(north, 90)),
        f"{prefix}offset_norm_m_mean": mean_sample_norm,
        f"{prefix}offset_norm_m_median": float(np.median(norm)),
        f"{prefix}offset_norm_m_p90": float(np.percentile(norm, 90)),
        f"{prefix}dominant_offset_east_m": east_mean,
        f"{prefix}dominant_offset_north_m": north_mean,
        f"{prefix}dominant_offset_norm_m": mean_norm,
        f"{prefix}median_vector_offset_east_m": east_median,
        f"{prefix}median_vector_offset_north_m": north_median,
        f"{prefix}median_vector_offset_norm_m": median_norm,
        f"{prefix}direction_consistency": float(direction_consistency),
        f"{prefix}stable_horizontal_offset_likely": bool(
            3.0 <= float(np.median(norm)) <= 6.0 and direction_consistency > 0.7
        ),
    }


def _robust_offset_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    hit_rows = [r for r in rows if r.get("hit") and r.get("offset_norm_m") is not None]
    if not hit_rows:
        return {"inlier_count": 0, "outlier_count": 0}
    east = np.asarray([float(r["offset_east_m"]) for r in hit_rows], dtype=np.float64)
    north = np.asarray([float(r["offset_north_m"]) for r in hit_rows], dtype=np.float64)
    center_east = float(np.median(east))
    center_north = float(np.median(north))
    residual = np.hypot(east - center_east, north - center_north)
    threshold = max(2.0, float(np.percentile(residual, 75)))
    inlier_rows = [row for row, keep in zip(hit_rows, residual <= threshold) if bool(keep)]
    return {
        "robust_center_east_m": center_east,
        "robust_center_north_m": center_north,
        "robust_inlier_threshold_m": threshold,
        "inlier_count": int(len(inlier_rows)),
        "outlier_count": int(len(hit_rows) - len(inlier_rows)),
        **_offset_stats(inlier_rows, prefix="inlier_"),
    }


def _write_html(path: Path, rows: List[Dict[str, Any]], per_image: List[Dict[str, Any]]) -> None:
    parts = [
        "<html><head><meta charset='utf-8'><style>body{font-family:sans-serif} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:6px} img{max-width:200px}</style></head><body>",
        "<h1>Renderer Tie Point Intersection Check</h1>",
        "<h2>Per-image summary</h2>",
        "<table><tr><th>Image</th><th>Samples</th><th>Hits</th><th>Mean XY err m</th><th>P90 XY err m</th><th>Mean Z err m</th><th>P90 Z err m</th></tr>",
    ]
    for row in per_image:
        parts.append(
            "<tr>"
            f"<td>{row['query_image']}</td>"
            f"<td>{row['num_samples']}</td>"
            f"<td>{row['num_hits']}</td>"
            f"<td>{row.get('hit_xy_error_m_mean')}</td>"
            f"<td>{row.get('hit_xy_error_m_p90')}</td>"
            f"<td>{row.get('hit_z_vs_tiepoint_error_m_mean')}</td>"
            f"<td>{row.get('hit_z_vs_tiepoint_error_m_p90')}</td>"
            "</tr>"
        )
    parts.extend(
        [
            "</table>",
            "<h2>Patch triplets</h2>",
            "<table><tr><th>#</th><th>Image</th><th>Hit</th><th>XY err m</th><th>East m</th><th>North m</th><th>Norm m</th><th>DOM px col,row</th><th>Z err m</th><th>Query</th><th>Tie point DOM</th><th>Ray-hit DOM</th></tr>",
        ]
    )
    for row in rows:
        parts.append(
            "<tr>"
            f"<td>{row['index']}</td>"
            f"<td>{row['query_image']}</td>"
            f"<td>{row['hit']}</td>"
            f"<td>{row.get('hit_xy_error_m')}</td>"
            f"<td>{row.get('offset_east_m')}</td>"
            f"<td>{row.get('offset_north_m')}</td>"
            f"<td>{row.get('offset_norm_m')}</td>"
            f"<td>{row.get('dom_pixel_offset_col')}, {row.get('dom_pixel_offset_row')}</td>"
            f"<td>{row.get('hit_z_vs_tiepoint_error_m')}</td>"
            f"<td><img src='{row['query_patch_rel']}'></td>"
            f"<td><img src='{row['tiepoint_dom_patch_rel']}'></td>"
            f"<td><img src='{row['hit_dom_patch_rel']}'></td>"
            "</tr>"
        )
    parts.append("</table></body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[row["query_image"]].append(row)
    per_image = []
    for image, image_rows in sorted(by_image.items()):
        per_image.append(
            {
                "query_image": image,
                "num_samples": len(image_rows),
                "num_hits": sum(1 for r in image_rows if r["hit"]),
                "hit_xy_error_m_mean": _mean([r.get("hit_xy_error_m") for r in image_rows]),
                "hit_xy_error_m_median": _percentile([r.get("hit_xy_error_m") for r in image_rows], 50),
                "hit_xy_error_m_p90": _percentile([r.get("hit_xy_error_m") for r in image_rows], 90),
                **_offset_stats(image_rows),
                "hit_z_vs_tiepoint_error_m_mean": _mean([r.get("hit_z_vs_tiepoint_error_m") for r in image_rows]),
                "hit_z_vs_tiepoint_error_m_median": _percentile([r.get("hit_z_vs_tiepoint_error_m") for r in image_rows], 50),
                "hit_z_vs_tiepoint_error_m_p90": _percentile([r.get("hit_z_vs_tiepoint_error_m") for r in image_rows], 90),
                "hit_depth_vs_projection_depth_m_mean": _mean([r.get("hit_depth_vs_projection_depth_m") for r in image_rows]),
                "tiepoint_dom_vs_hit_dom_mad_mean": _mean([r.get("tiepoint_dom_vs_hit_dom_mad") for r in image_rows]),
            }
        )
    return {
        "per_image": per_image,
        "hit_xy_error_m_mean": _mean([r.get("hit_xy_error_m") for r in rows]),
        "hit_xy_error_m_median": _percentile([r.get("hit_xy_error_m") for r in rows], 50),
        "hit_xy_error_m_p90": _percentile([r.get("hit_xy_error_m") for r in rows], 90),
        **_offset_stats(rows),
        **_robust_offset_stats(rows),
        "hit_z_vs_tiepoint_error_m_mean": _mean([r.get("hit_z_vs_tiepoint_error_m") for r in rows]),
        "hit_z_vs_tiepoint_error_m_median": _percentile([r.get("hit_z_vs_tiepoint_error_m") for r in rows], 50),
        "hit_z_vs_tiepoint_error_m_p90": _percentile([r.get("hit_z_vs_tiepoint_error_m") for r in rows], 90),
    }


def _write_offset_plots(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    hit_rows = [r for r in rows if r.get("hit") and r.get("offset_norm_m") is not None]
    if not hit_rows:
        return
    east = np.asarray([float(r["offset_east_m"]) for r in hit_rows], dtype=np.float64)
    north = np.asarray([float(r["offset_north_m"]) for r in hit_rows], dtype=np.float64)
    norm = np.asarray([float(r["offset_norm_m"]) for r in hit_rows], dtype=np.float64)
    x = np.asarray([float(r["tiepoint_raster_x"]) for r in hit_rows], dtype=np.float64)
    y = np.asarray([float(r["tiepoint_raster_y"]) for r in hit_rows], dtype=np.float64)

    plt.figure(figsize=(7, 7))
    plt.scatter(east, north, c=norm, cmap="viridis", s=35)
    plt.axhline(0.0, color="0.7", linewidth=1)
    plt.axvline(0.0, color="0.7", linewidth=1)
    plt.xlabel("offset east m")
    plt.ylabel("offset north m")
    plt.colorbar(label="offset norm m")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_dir / "offset_scatter.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(east, bins=20, alpha=0.6, label="east")
    plt.hist(north, bins=20, alpha=0.6, label="north")
    plt.hist(norm, bins=20, alpha=0.5, label="norm")
    plt.xlabel("meters")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "offset_histogram.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.quiver(x, y, east, north, angles="xy", scale_units="xy", scale=1.0, width=0.003)
    plt.scatter(x, y, s=10, color="black")
    plt.xlabel("raster x")
    plt.ylabel("raster y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_dir / "offset_quiver.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--convention-file", default=DEFAULT_CONVENTION)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--images", nargs="+", default=None)
    parser.add_argument("--sample-limit", type=int, default=100, help="Maximum accepted reliable tie point measurements per query image.")
    parser.add_argument("--max-total-samples", type=int, default=None)
    parser.add_argument("--patch-radius", type=int, default=64)
    parser.add_argument("--max-reprojection-error-px", type=float, default=2.0)
    parser.add_argument("--ray-step-m", type=float, default=None)
    parser.add_argument("--ray-max-m", type=float, default=500.0)
    parser.add_argument("--dsm-sampling-mode", choices=["nearest", "bilinear"], default=None)
    parser.add_argument("--dom-sampling-mode", choices=["nearest", "bilinear"], default=None)
    parser.add_argument("--ray-refine-iters", type=int, default=None)
    parser.add_argument("--chunk-rows", type=int, default=192)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    convention = _load_convention_file(args.convention_file)
    step_m = float(args.ray_step_m or convention.get("ray_step_m", 2.0))

    xml_path = (REPO_ROOT / args.xml).resolve()
    xml_srs, photos = _parse_xml(xml_path)
    photo_by_id = {p.photo_id: p for p in photos}
    tiepoints = _parse_tiepoints(xml_path)

    query_dir = (REPO_ROOT / args.query_dir).resolve()
    if args.images:
        image_paths = [query_dir / name for name in args.images]
    else:
        image_paths = sorted([p for p in query_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}])
    requested_images = sorted({p.name for p in image_paths})
    pose_records = _load_pose_file_projected((REPO_ROOT / args.pose_file).resolve(), xml_srs)
    _matches, match_report = _match_photos(image_paths, pose_records, photos)
    image_by_photo_id = {
        str(m["matched_photo_id"]): query_dir / m["image"]
        for m in match_report
        if m.get("status") == "ok" and m.get("matched_photo_id")
    }

    renderer = ContextCaptureDOMDSMRenderer(config["render_config"], xml_srs, args.chunk_rows)
    dsm_sampling_mode = str(args.dsm_sampling_mode or convention.get("dsm_sampling_mode", renderer.dsm_sampling_mode))
    dom_sampling_mode = str(args.dom_sampling_mode or convention.get("dom_sampling_mode", renderer.dom_sampling_mode))
    ray_refine_iters = int(args.ray_refine_iters if args.ray_refine_iters is not None else convention.get("ray_refine_iters", renderer.ray_refine_iters))
    primary_projection = next(c for c in PROJECTION_CONVENTIONS if c["is_primary_acceptance"])
    query_cache: Dict[Path, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    samples_by_image: Dict[str, int] = defaultdict(int)
    counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for tp in tiepoints:
        if args.max_total_samples is not None and len(rows) >= args.max_total_samples:
            break
        if requested_images and all(samples_by_image[image] >= args.sample_limit for image in requested_images):
            break
        point = tp["position"]
        tie_x, tie_y = renderer.xml_to_raster.transform(point[0], point[1])
        tie_dom_rows, tie_dom_cols = renderer._xy_to_rowcol(
            renderer.dom_transform,
            np.asarray([tie_x], dtype=np.float64),
            np.asarray([tie_y], dtype=np.float64),
        )
        tie_dom_row = int(tie_dom_rows[0])
        tie_dom_col = int(tie_dom_cols[0])
        if not (0 <= tie_dom_row < renderer.dom_array.shape[0] and 0 <= tie_dom_col < renderer.dom_array.shape[1]):
            continue
        tie_dsm_height = renderer._sample_dsm(np.asarray([tie_x]), np.asarray([tie_y]), dsm_sampling_mode)[0]
        if not np.isfinite(tie_dsm_height):
            continue

        for meas in tp["measurements"]:
            if args.max_total_samples is not None and len(rows) >= args.max_total_samples:
                break
            photo = photo_by_id.get(meas["photo_id"])
            query_path = image_by_photo_id.get(meas["photo_id"])
            if photo is None or query_path is None:
                continue
            query_image = query_path.name
            if samples_by_image[query_image] >= args.sample_limit:
                continue
            counters[query_image]["visited"] += 1
            proj = _project(point, photo, primary_projection)
            if proj is None:
                continue
            projected_x, projected_y, projection_depth = proj
            reproj_error = math.hypot(projected_x - meas["x"], projected_y - meas["y"])
            if projection_depth <= 0:
                counters[query_image]["depth_filtered"] += 1
                continue
            if reproj_error > args.max_reprojection_error_px:
                counters[query_image]["reprojection_filtered"] += 1
                continue

            origin, ray, ray_debug = _ray_from_pixel(renderer, photo, meas["x"], meas["y"], convention, "cam_to_world_correct")
            hit = _intersect_dsm(
                renderer,
                origin,
                ray,
                step_m,
                args.ray_max_m,
                dsm_sampling_mode,
                dom_sampling_mode,
                ray_refine_iters,
            )
            if query_path not in query_cache:
                query_cache[query_path] = _read_query(query_path)
            query_rgb = query_cache[query_path]
            query_patch = _crop_patch(query_rgb, meas["x"], meas["y"], args.patch_radius)
            tiepoint_dom_patch = _crop_patch(renderer.dom_array, tie_dom_col, tie_dom_row, args.patch_radius)

            idx = len(rows)
            image_stem = query_path.stem
            q_path = patches_dir / image_stem / f"query_{idx:04d}.png"
            tie_path = patches_dir / image_stem / f"tiepoint_dom_{idx:04d}.png"
            hit_path = patches_dir / image_stem / f"hit_dom_{idx:04d}.png"
            _write_rgb(q_path, query_patch)
            _write_rgb(tie_path, tiepoint_dom_patch)

            row: Dict[str, Any] = {
                "index": idx,
                "tiepoint_index": tp["tiepoint_index"],
                "photo_id": photo.photo_id,
                "query_image": query_image,
                "measurement_x": float(meas["x"]),
                "measurement_y": float(meas["y"]),
                "projected_x": float(projected_x),
                "projected_y": float(projected_y),
                "reprojection_error_px": float(reproj_error),
                "projection_depth": float(projection_depth),
                "tiepoint_xml_x": float(point[0]),
                "tiepoint_xml_y": float(point[1]),
                "tiepoint_xml_z": float(point[2]),
                "tiepoint_raster_x": float(tie_x),
                "tiepoint_raster_y": float(tie_y),
                "tiepoint_dom_row": tie_dom_row,
                "tiepoint_dom_col": tie_dom_col,
                "tiepoint_dsm_height": float(tie_dsm_height),
                "tiepoint_dsm_minus_xml_z_m": float(float(tie_dsm_height) - float(point[2])),
                "query_vs_tiepoint_dom_mad": _patch_mad(query_patch, tiepoint_dom_patch),
                "hit": hit is not None,
                "query_patch_rel": os.path.relpath(q_path, output_dir).replace("\\", "/"),
                "tiepoint_dom_patch_rel": os.path.relpath(tie_path, output_dir).replace("\\", "/"),
                **ray_debug,
            }
            if hit is not None:
                hit_dom_patch = _crop_patch(renderer.dom_array, hit["dom_col"], hit["dom_row"], args.patch_radius)
                _write_rgb(hit_path, hit_dom_patch)
                offset_east = float(hit["x_raster"]) - float(tie_x)
                offset_north = float(hit["y_raster"]) - float(tie_y)
                hit_xy_error = math.hypot(offset_east, offset_north)
                dom_pixel_offset_col = int(hit["dom_col"]) - int(tie_dom_col)
                dom_pixel_offset_row = int(hit["dom_row"]) - int(tie_dom_row)
                row.update(
                    {
                        "hit_depth": hit["depth"],
                        "hit_raster_x": hit["x_raster"],
                        "hit_raster_y": hit["y_raster"],
                        "hit_z_ray": hit["z_ray"],
                        "hit_dsm_height": hit["dsm_height"],
                        "hit_dom_row": hit["dom_row"],
                        "hit_dom_col": hit["dom_col"],
                        "offset_east_m": float(offset_east),
                        "offset_north_m": float(offset_north),
                        "offset_norm_m": float(hit_xy_error),
                        "offset_angle_deg": float(math.degrees(math.atan2(offset_north, offset_east))),
                        "dom_pixel_offset_col": int(dom_pixel_offset_col),
                        "dom_pixel_offset_row": int(dom_pixel_offset_row),
                        "hit_xy_error_m": float(hit_xy_error),
                        "hit_z_vs_tiepoint_error_m": float(abs(float(hit["z_ray"]) - float(point[2]))),
                        "hit_dsm_vs_tiepoint_dsm_error_m": float(abs(float(hit["dsm_height"]) - float(tie_dsm_height))),
                        "hit_dsm_minus_tiepoint_dsm_m": float(float(hit["dsm_height"]) - float(tie_dsm_height)),
                        "hit_depth_vs_projection_depth_m": float(abs(float(hit["depth"]) - float(projection_depth))),
                        "tiepoint_dom_vs_hit_dom_mad": _patch_mad(tiepoint_dom_patch, hit_dom_patch),
                        "hit_dom_patch_rel": os.path.relpath(hit_path, output_dir).replace("\\", "/"),
                    }
                )
            else:
                counters[query_image]["missed_ray_hit"] += 1
                zero_patch = np.zeros_like(tiepoint_dom_patch)
                _write_rgb(hit_path, zero_patch)
                row["hit_dom_patch_rel"] = os.path.relpath(hit_path, output_dir).replace("\\", "/")
            rows.append(row)
            samples_by_image[query_image] += 1

    with (output_dir / "renderer_tiepoint_intersection_results.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in rows for k in row.keys() if not isinstance(row.get(k), (list, dict))})
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary_stats = _summarize(rows)
    _write_offset_plots(output_dir, rows)
    _write_html(output_dir / "renderer_tiepoint_intersection_check.html", rows, summary_stats["per_image"])
    _write_json(
        output_dir / "renderer_tiepoint_intersection_summary.json",
        {
            "xml": args.xml,
            "xml_srs": xml_srs,
            "query_dir": args.query_dir,
            "config": args.config,
            "convention_file": args.convention_file,
            "xml_projection_rotation": XML_PROJECTION_ROTATION,
            "render_ray_rotation": RENDER_RAY_ROTATION,
            "primary_projection_convention": primary_projection,
            "sample_limit_per_image": args.sample_limit,
            "max_total_samples": args.max_total_samples,
            "max_reprojection_error_px": args.max_reprojection_error_px,
            "ray_step_m": step_m,
            "ray_max_m": args.ray_max_m,
            "dsm_sampling_mode": dsm_sampling_mode,
            "dom_sampling_mode": dom_sampling_mode,
            "ray_refine_iters": ray_refine_iters,
            "patch_radius": args.patch_radius,
            "num_samples_written": len(rows),
            "num_hits": sum(1 for r in rows if r["hit"]),
            "per_image_counters": {image: dict(vals) for image, vals in counters.items()},
            "interpretation_rule": {
                "ray_direction_issue": "Large hit_xy_error_m with otherwise reliable tie point projection.",
                "dsm_expression_issue": "Small hit_xy_error_m but large hit_z_vs_tiepoint_error_m or hit_dsm_vs_tiepoint_dsm_error_m.",
                "sampling_issue": "Small XY/Z errors but tiepoint_dom_vs_hit_dom_mad or patch framing looks wrong.",
            },
            **summary_stats,
            "results": rows,
        },
    )
    _write_json(
        output_dir / "tiepoint_spatial_offset_summary.json",
        {
            "xml": args.xml,
            "xml_srs": xml_srs,
            "query_dir": args.query_dir,
            "config": args.config,
            "convention_file": args.convention_file,
            "sample_limit_per_image": args.sample_limit,
            "max_reprojection_error_px": args.max_reprojection_error_px,
            "num_samples_written": len(rows),
            "num_hits": sum(1 for r in rows if r["hit"]),
            "per_image": summary_stats["per_image"],
            "stable_horizontal_offset_likely": summary_stats.get("stable_horizontal_offset_likely"),
            "recommended_east_offset_m": summary_stats.get("median_vector_offset_east_m"),
            "recommended_north_offset_m": summary_stats.get("median_vector_offset_north_m"),
            "recommended_offset_norm_m": summary_stats.get("median_vector_offset_norm_m"),
            **summary_stats,
        },
    )
    with (output_dir / "tiepoint_spatial_offset_results.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in rows for k in row.keys() if not isinstance(row.get(k), (list, dict))})
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _write_html(output_dir / "tiepoint_spatial_offset_report.html", rows, summary_stats["per_image"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
