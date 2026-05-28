#!/usr/bin/env python3
"""Check XML tie point measurements against DOM/DSM product patches."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.check_contextcapture_xml_reprojection import PROJECTION_CONVENTIONS, _parse_tiepoints, _project
from tools.diagnose_yawfix_refinement_update import _safe_jsonable, _write_rgb
from tools.render_contextcapture_xml_domdsm_initial import (
    ContextCaptureDOMDSMRenderer,
    RENDER_RAY_ROTATION,
    XML_PROJECTION_ROTATION,
    _load_pose_file_projected,
    _match_photos,
    _parse_xml,
)


DEFAULT_XML = "data_caiwangcun/CaiWangCun.xml"
DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_DIR = "data_caiwangcun/query/images/exif_test"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/contextcapture_tiepoint_domdsm_consistency_check"
MANUAL_LABEL_CHOICES = (
    "unlabeled",
    "same_object",
    "similar_but_texture_changed",
    "geometry_or_scale_mismatch",
    "not_same_object",
    "uncertain",
)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_query(path: Path) -> np.ndarray:
    bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _crop_patch(image: np.ndarray, x: float, y: float, radius: int) -> np.ndarray:
    h, w = image.shape[:2]
    xi = int(round(x))
    yi = int(round(y))
    x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
    y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
    patch = image[y0:y1, x0:x1].copy()
    if patch.size == 0:
        return np.zeros((2 * radius + 1, 2 * radius + 1, 3), dtype=np.uint8)
    return patch


def _patch_mad(query_patch: np.ndarray, dom_patch: np.ndarray) -> Optional[float]:
    if query_patch.size == 0 or dom_patch.size == 0:
        return None
    dom_resized = cv2.resize(dom_patch, (query_patch.shape[1], query_patch.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.mean(np.abs(query_patch.astype(np.float32) - dom_resized.astype(np.float32))))


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else None


def _percentile(values: Sequence[Optional[float]], q: float) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.percentile(np.asarray(finite, dtype=np.float64), q)) if finite else None


def _label_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    return {label: sum(1 for r in rows if r.get("manual_label") == label) for label in MANUAL_LABEL_CHOICES}


def _summarize_rows(rows: List[Dict[str, Any]], counters: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[row["query_image"]].append(row)
    per_image = []
    for image in sorted(set(counters.keys()) | set(by_image.keys())):
        image_rows = by_image.get(image, [])
        labels = _label_counts(image_rows)
        unresolved = labels["not_same_object"] + labels["uncertain"]
        positive = labels["same_object"] + labels["similar_but_texture_changed"]
        per_image.append(
            {
                "query_image": image,
                "num_samples_written": len(image_rows),
                "num_candidate_measurements_visited": counters[image]["visited"],
                "num_dom_out_of_bounds": counters[image]["out_of_dom"],
                "num_dsm_nodata": counters[image]["nodata"],
                "num_reprojection_filtered": counters[image]["reprojection_filtered"],
                "num_depth_filtered": counters[image]["depth_filtered"],
                "patch_mean_absdiff_rgb_mean": _mean([r.get("patch_mean_absdiff_rgb") for r in image_rows]),
                "patch_mean_absdiff_rgb_median": _percentile([r.get("patch_mean_absdiff_rgb") for r in image_rows], 50),
                "patch_mean_absdiff_rgb_p90": _percentile([r.get("patch_mean_absdiff_rgb") for r in image_rows], 90),
                "manual_label_counts": labels,
                "manual_positive_count": positive,
                "manual_unresolved_count": unresolved,
            }
        )
    labels = _label_counts(rows)
    return {
        "per_image": per_image,
        "manual_label_counts": labels,
        "manual_positive_count": labels["same_object"] + labels["similar_but_texture_changed"],
        "manual_unresolved_count": labels["not_same_object"] + labels["uncertain"],
        "patch_mean_absdiff_rgb_mean": _mean([r.get("patch_mean_absdiff_rgb") for r in rows]),
        "patch_mean_absdiff_rgb_median": _percentile([r.get("patch_mean_absdiff_rgb") for r in rows], 50),
        "patch_mean_absdiff_rgb_p90": _percentile([r.get("patch_mean_absdiff_rgb") for r in rows], 90),
    }


def _write_html(path: Path, rows: List[Dict[str, Any]], per_image: List[Dict[str, Any]]) -> None:
    parts = [
        "<html><head><meta charset='utf-8'><style>body{font-family:sans-serif} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:6px} img{max-width:220px}</style></head><body>",
        "<h1>ContextCapture Tie Point DOM/DSM Consistency</h1>",
        "<h2>Per-image summary</h2>",
        "<table><tr><th>Image</th><th>Samples</th><th>Visited</th><th>DOM OOB</th><th>DSM nodata</th><th>Mean RGB MAD</th><th>Manual labels</th></tr>",
    ]
    for row in per_image:
        parts.append(
            "<tr>"
            f"<td>{row['query_image']}</td>"
            f"<td>{row['num_samples_written']}</td>"
            f"<td>{row['num_candidate_measurements_visited']}</td>"
            f"<td>{row['num_dom_out_of_bounds']}</td>"
            f"<td>{row['num_dsm_nodata']}</td>"
            f"<td>{row.get('patch_mean_absdiff_rgb_mean')}</td>"
            f"<td>{json.dumps(row['manual_label_counts'], ensure_ascii=False)}</td>"
            "</tr>"
        )
    parts.extend(
        [
            "</table>",
            "<h2>Patch pairs</h2>",
            "<table><tr><th>#</th><th>Image</th><th>Measurement</th><th>Reprojection error</th><th>DOM/DSM</th><th>RGB MAD</th><th>Manual label</th><th>Query patch</th><th>DOM patch</th></tr>",
        ]
    )
    label_hint = " / ".join(MANUAL_LABEL_CHOICES)
    parts.append(f"<p>Manual label choices: {label_hint}</p>")
    for row in rows:
        parts.append(
            "<tr>"
            f"<td>{row['index']}</td>"
            f"<td>{row['query_image']}</td>"
            f"<td>({row['measurement_x']:.1f}, {row['measurement_y']:.1f})</td>"
            f"<td>{row['reprojection_error_px']:.3f}px</td>"
            f"<td>row={row.get('dom_row')} col={row.get('dom_col')} dsm={row.get('dsm_height')}</td>"
            f"<td>{row.get('patch_mean_absdiff_rgb')}</td>"
            f"<td>{row.get('manual_label', 'unlabeled')}</td>"
            f"<td><img src='{row['query_patch_rel']}'></td>"
            f"<td><img src='{row['dom_patch_rel']}'></td>"
            "</tr>"
        )
    parts.append("</table></body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_per_image_html(output_dir: Path, rows: List[Dict[str, Any]], per_image: List[Dict[str, Any]]) -> None:
    for image_summary in per_image:
        image = image_summary["query_image"]
        image_rows = [r for r in rows if r["query_image"] == image]
        _write_html(output_dir / f"tiepoint_domdsm_check_{Path(image).stem}.html", image_rows, [image_summary])


def _write_manual_label_template(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = ["index", "query_image", "tiepoint_index", "photo_id", "manual_label", "manual_note"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--images", nargs="+", default=None)
    parser.add_argument("--sample-limit", type=int, default=200, help="Maximum accepted reliable tie point measurements per query image.")
    parser.add_argument("--max-total-samples", type=int, default=None)
    parser.add_argument("--patch-radius", type=int, default=64)
    parser.add_argument("--max-reprojection-error-px", type=float, default=2.0)
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
    xml_path = (REPO_ROOT / args.xml).resolve()
    xml_srs, photos = _parse_xml(xml_path)
    photo_by_id = {p.photo_id: p for p in photos}
    tiepoints = _parse_tiepoints(xml_path)

    query_dir = (REPO_ROOT / args.query_dir).resolve()
    if args.images:
        image_paths = [query_dir / name for name in args.images]
    else:
        image_paths = sorted([p for p in query_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}])
    pose_records = _load_pose_file_projected((REPO_ROOT / args.pose_file).resolve(), xml_srs)
    _matches, match_report = _match_photos(image_paths, pose_records, photos)
    image_by_photo_id = {
        str(m["matched_photo_id"]): query_dir / m["image"]
        for m in match_report
        if m.get("status") == "ok" and m.get("matched_photo_id")
    }
    requested_images = sorted({p.name for p in image_paths})

    renderer = ContextCaptureDOMDSMRenderer(config["render_config"], xml_srs, args.chunk_rows)
    primary_projection = next(c for c in PROJECTION_CONVENTIONS if c["is_primary_acceptance"])
    query_cache: Dict[Path, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    samples_by_image: Dict[str, int] = defaultdict(int)

    for tp in tiepoints:
        if args.max_total_samples is not None and len(rows) >= args.max_total_samples:
            break
        if requested_images and all(samples_by_image[image] >= args.sample_limit for image in requested_images):
            break
        point = tp["position"]
        x_raster, y_raster = renderer.xml_to_raster.transform(point[0], point[1])

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
            projected_x, projected_y, depth = proj
            reproj_error = math.hypot(projected_x - meas["x"], projected_y - meas["y"])
            if depth <= 0:
                counters[query_image]["depth_filtered"] += 1
                continue
            if reproj_error > args.max_reprojection_error_px:
                counters[query_image]["reprojection_filtered"] += 1
                continue
            dom_rows, dom_cols = renderer._xy_to_rowcol(
                renderer.dom_transform,
                np.asarray([x_raster], dtype=np.float64),
                np.asarray([y_raster], dtype=np.float64),
            )
            dom_row = int(dom_rows[0])
            dom_col = int(dom_cols[0])
            in_dom = 0 <= dom_row < renderer.dom_array.shape[0] and 0 <= dom_col < renderer.dom_array.shape[1]
            if not in_dom:
                counters[query_image]["out_of_dom"] += 1
                continue
            dsm_height = renderer._sample_dsm(np.asarray([x_raster]), np.asarray([y_raster]), "nearest")[0]
            if not np.isfinite(dsm_height):
                counters[query_image]["nodata"] += 1
                continue
            if query_path not in query_cache:
                query_cache[query_path] = _read_query(query_path)
            query_rgb = query_cache[query_path]
            query_patch = _crop_patch(query_rgb, meas["x"], meas["y"], args.patch_radius)
            dom_patch = _crop_patch(renderer.dom_array, dom_col, dom_row, args.patch_radius)
            idx = len(rows)
            image_stem = query_path.stem
            q_path = patches_dir / image_stem / f"query_{idx:04d}.png"
            d_path = patches_dir / image_stem / f"dom_{idx:04d}.png"
            _write_rgb(q_path, query_patch)
            _write_rgb(d_path, dom_patch)
            rows.append(
                {
                    "index": idx,
                    "tiepoint_index": tp["tiepoint_index"],
                    "photo_id": photo.photo_id,
                    "query_image": query_image,
                    "measurement_x": float(meas["x"]),
                    "measurement_y": float(meas["y"]),
                    "projected_x": float(projected_x),
                    "projected_y": float(projected_y),
                    "reprojection_error_px": float(reproj_error),
                    "projection_depth": float(depth),
                    "tiepoint_xml_x": float(point[0]),
                    "tiepoint_xml_y": float(point[1]),
                    "tiepoint_xml_z": float(point[2]),
                    "tiepoint_raster_x": float(x_raster),
                    "tiepoint_raster_y": float(y_raster),
                    "dom_row": dom_row,
                    "dom_col": dom_col,
                    "dsm_height": float(dsm_height),
                    "patch_mean_absdiff_rgb": _patch_mad(query_patch, dom_patch),
                    "manual_label": "unlabeled",
                    "manual_note": "",
                    "manual_label_choices": "|".join(MANUAL_LABEL_CHOICES),
                    "query_patch_rel": os.path.relpath(q_path, output_dir).replace("\\", "/"),
                    "dom_patch_rel": os.path.relpath(d_path, output_dir).replace("\\", "/"),
                }
            )
            samples_by_image[query_image] += 1

    with (output_dir / "tiepoint_domdsm_results.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in rows for k in row.keys() if not isinstance(row.get(k), (list, dict))})
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary_stats = _summarize_rows(rows, counters)
    _write_html(output_dir / "tiepoint_domdsm_check.html", rows, summary_stats["per_image"])
    _write_per_image_html(output_dir, rows, summary_stats["per_image"])
    _write_manual_label_template(output_dir / "manual_label_template.csv", rows)
    _write_json(
        output_dir / "tiepoint_domdsm_summary.json",
        {
            "xml": args.xml,
            "xml_srs": xml_srs,
            "query_dir": args.query_dir,
            "config": args.config,
            "xml_projection_rotation": XML_PROJECTION_ROTATION,
            "render_ray_rotation_note": f"Not used for this check; render rays use {RENDER_RAY_ROTATION}.",
            "primary_projection_convention": primary_projection,
            "num_tiepoints_total": len(tiepoints),
            "num_candidate_measurements_visited": int(sum(v["visited"] for v in counters.values())),
            "num_samples_written": len(rows),
            "sample_limit_per_image": args.sample_limit,
            "max_total_samples": args.max_total_samples,
            "num_dom_out_of_bounds": int(sum(v["out_of_dom"] for v in counters.values())),
            "num_dsm_nodata": int(sum(v["nodata"] for v in counters.values())),
            "max_reprojection_error_px": args.max_reprojection_error_px,
            "patch_radius": args.patch_radius,
            "manual_label_choices": MANUAL_LABEL_CHOICES,
            "decision_rule": {
                "dom_dsm_product_likely_main_cause": "Most reliable tie point patches are manually labeled not_same_object or uncertain.",
                "renderer_precision_followup": "Most reliable tie point patches are manually labeled same_object or similar_but_texture_changed.",
            },
            **summary_stats,
            "results": rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
