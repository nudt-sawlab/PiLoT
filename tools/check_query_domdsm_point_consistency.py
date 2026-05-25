#!/usr/bin/env python3
"""Check query pixel to DOM/DSM point consistency with ContextCapture XML camera."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_yawfix_refinement_update import _safe_jsonable, _write_rgb
from tools.render_contextcapture_xml_domdsm_initial import (
    AXIS_TRANSFORMS,
    ContextCaptureDOMDSMRenderer,
    RAY_ROTATION_CONVENTIONS,
    XML_PROJECTION_ROTATION,
    _legacy_rotation_convention,
    _load_convention_file,
    _load_pose_file_projected,
    _match_photos,
    _parse_xml,
    _ray_rotation_description,
)


DEFAULT_XML = "data_caiwangcun/CaiWangCun.xml"
DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_CONVENTION = "docs/experiments/dom_dsm_prepare/contextcapture_xml_camera_convention_diagnosis/best_convention.json"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/query_domdsm_point_consistency_check/0000"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_query(path: Path) -> np.ndarray:
    bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _default_points(width: int, height: int) -> List[Dict[str, float]]:
    pts = [
        (0.50, 0.50),
        (0.25, 0.25),
        (0.75, 0.25),
        (0.25, 0.75),
        (0.75, 0.75),
        (0.50, 0.25),
        (0.50, 0.75),
        (0.25, 0.50),
        (0.75, 0.50),
        (0.50, 0.40),
    ]
    return [{"x": float(round(px * (width - 1))), "y": float(round(py * (height - 1)))} for px, py in pts]


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


def _ray_from_pixel(
    renderer: ContextCaptureDOMDSMRenderer,
    photo: Any,
    xpix: float,
    ypix: float,
    convention_cfg: Dict[str, Any],
    ray_convention: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    intr = photo.intrinsics
    principal_point_mode = convention_cfg.get("principal_point_mode", "xml")
    cy_value = intr.cy if principal_point_mode == "xml" else (intr.height - intr.cy)
    xd = np.asarray([(xpix - intr.cx) / intr.fx], dtype=np.float32)
    yd = np.asarray([(ypix - cy_value) / intr.fy], dtype=np.float32)
    undistort = bool(convention_cfg.get("distortion_enabled", True))
    x, y = renderer._undistort_normalized(xd, yd, intr, undistort)
    ray_cam = np.asarray([float(x[0]), float(y[0]), 1.0], dtype=np.float64)
    axis = np.asarray(convention_cfg.get("axis_transform", AXIS_TRANSFORMS.get(convention_cfg.get("axis_transform_key", "ppp"), [1.0, 1.0, 1.0])), dtype=np.float64)
    ray_cam *= axis
    R_world_to_camera = photo.rotation
    R_camera_to_world = photo.rotation.T
    ray_rotation = R_camera_to_world if ray_convention == "cam_to_world_correct" else R_world_to_camera
    ray_world = ray_rotation @ ray_cam
    cx_xml, cy_xml, alt = photo.center_xml
    cam_x, cam_y = renderer.xml_to_raster.transform(cx_xml, cy_xml)
    origin = np.asarray([float(cam_x), float(cam_y), float(alt)], dtype=np.float64)
    debug = {
        "xml_projection_rotation": XML_PROJECTION_ROTATION,
        "render_ray_convention": ray_convention,
        "render_ray_rotation": _ray_rotation_description(ray_convention),
        "ray_rotation_matrix_source": "photo.rotation.T" if ray_convention == "cam_to_world_correct" else "photo.rotation",
        "legacy_rotation_convention": _legacy_rotation_convention(ray_convention),
        "rotation_convention": _legacy_rotation_convention(ray_convention),
        "ray_cam": ray_cam.tolist(),
        "ray_world": ray_world.tolist(),
        "origin_raster": origin.tolist(),
    }
    return origin, ray_world, debug


def _intersect_dsm(
    renderer: ContextCaptureDOMDSMRenderer,
    origin: np.ndarray,
    ray: np.ndarray,
    step_m: float,
    max_m: float,
    dsm_sampling_mode: str = "bilinear",
    dom_sampling_mode: str = "bilinear",
    ray_refine_iters: int = 10,
) -> Optional[Dict[str, Any]]:
    depth_values = np.arange(renderer.near_m, max_m + step_m, step_m, dtype=np.float32)
    hit_x, hit_y, depths = renderer._intersect_rays_with_dsm(
        origin,
        ray.reshape(1, 3),
        depth_values,
        dsm_sampling_mode,
        ray_refine_iters,
    )
    if depths[0] <= 0:
        return None
    hit = origin + ray * float(depths[0])
    terrain = renderer._sample_dsm(np.asarray([hit[0]]), np.asarray([hit[1]]), dsm_sampling_mode)[0]
    rows, cols = renderer._xy_to_rowcol(renderer.dom_transform, np.asarray([hit_x[0]]), np.asarray([hit_y[0]]))
    color = renderer._sample_dom(np.asarray([hit_x[0]]), np.asarray([hit_y[0]]), dom_sampling_mode)[0].tolist()
    return {
        "hit": True,
        "depth": float(depths[0]),
        "x_raster": float(hit_x[0]),
        "y_raster": float(hit_y[0]),
        "z_ray": float(hit[2]),
        "dsm_height": float(terrain),
        "dom_col": int(cols[0]),
        "dom_row": int(rows[0]),
        "dom_rgb": color,
        "dsm_sampling_mode": dsm_sampling_mode,
        "dom_sampling_mode": dom_sampling_mode,
        "ray_refine_iters": int(ray_refine_iters),
    }
    return None


def _write_html(path: Path, rows: List[Dict[str, Any]]) -> None:
    parts = [
        "<html><head><meta charset='utf-8'><style>body{font-family:sans-serif} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:6px} img{max-width:220px}</style></head><body>",
        "<h1>Query DOM/DSM Point Consistency Check</h1>",
        "<table><tr><th>#</th><th>Query pixel</th><th>Hit</th><th>DSM/DOM</th><th>Query patch</th><th>DOM patch</th></tr>",
    ]
    for row in rows:
        parts.append(
            "<tr>"
            f"<td>{row['index']}</td>"
            f"<td>({row['query_x']:.1f}, {row['query_y']:.1f})</td>"
            f"<td>{row['hit']}</td>"
            f"<td>row={row.get('dom_row')} col={row.get('dom_col')} z={row.get('dsm_height')}</td>"
            f"<td><img src='{row['query_patch_rel']}'></td>"
            f"<td><img src='{row['dom_patch_rel']}'></td>"
            "</tr>"
        )
    parts.append("</table></body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--query-image", required=True)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--convention-file", default=DEFAULT_CONVENTION)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--points-json", default=None)
    parser.add_argument("--patch-radius", type=int, default=64)
    parser.add_argument("--ray-step-m", type=float, default=None)
    parser.add_argument("--ray-max-m", type=float, default=500.0)
    parser.add_argument("--ray-convention", choices=["both", *RAY_ROTATION_CONVENTIONS], default="both")
    parser.add_argument("--dsm-sampling-mode", choices=["nearest", "bilinear"], default=None)
    parser.add_argument("--dom-sampling-mode", choices=["nearest", "bilinear"], default=None)
    parser.add_argument("--ray-refine-iters", type=int, default=None)
    return parser.parse_args()


def _run_point_check(
    output_dir: Path,
    query_rgb: np.ndarray,
    points: List[Dict[str, float]],
    renderer: ContextCaptureDOMDSMRenderer,
    photo: Any,
    convention: Dict[str, Any],
    ray_convention: str,
    patch_radius: int,
    step_m: float,
    ray_max_m: float,
    dsm_sampling_mode: str,
    dom_sampling_mode: str,
    ray_refine_iters: int,
) -> List[Dict[str, Any]]:
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for idx, pt in enumerate(points):
        xpix, ypix = float(pt["x"]), float(pt["y"])
        origin, ray, ray_debug = _ray_from_pixel(renderer, photo, xpix, ypix, convention, ray_convention)
        hit = _intersect_dsm(
            renderer,
            origin,
            ray,
            step_m,
            ray_max_m,
            dsm_sampling_mode,
            dom_sampling_mode,
            ray_refine_iters,
        )
        query_patch = _crop_patch(query_rgb, xpix, ypix, patch_radius)
        q_path = patches_dir / f"query_{idx:02d}.png"
        _write_rgb(q_path, query_patch)
        row: Dict[str, Any] = {
            "index": idx,
            "query_x": xpix,
            "query_y": ypix,
            "hit": hit is not None,
            **ray_debug,
        }
        if hit is not None:
            row.update(hit)
            dom_patch = _crop_patch(renderer.dom_array, hit["dom_col"], hit["dom_row"], patch_radius)
        else:
            dom_patch = np.zeros_like(query_patch)
        d_path = patches_dir / f"dom_{idx:02d}.png"
        _write_rgb(d_path, dom_patch)
        row["query_patch_rel"] = os.path.relpath(q_path, output_dir).replace("\\", "/")
        row["dom_patch_rel"] = os.path.relpath(d_path, output_dir).replace("\\", "/")
        rows.append(row)
    with (output_dir / "point_results.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in rows for k in row.keys() if not isinstance(row.get(k), (list, dict))})
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _write_html(output_dir / "point_check.html", rows)
    return rows


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    xml_srs, photos = _parse_xml((REPO_ROOT / args.xml).resolve())
    query_path = (REPO_ROOT / args.query_image).resolve()
    pose_records = _load_pose_file_projected((REPO_ROOT / args.pose_file).resolve(), xml_srs)
    matches, match_report = _match_photos([query_path], pose_records, photos)
    item = matches[query_path.name]
    photo = item["photo"]
    convention = _load_convention_file(args.convention_file)
    renderer = ContextCaptureDOMDSMRenderer(config["render_config"], xml_srs, chunk_rows=192)
    query_rgb = _read_query(query_path)
    points = json.loads(Path(args.points_json).read_text(encoding="utf-8")) if args.points_json else _default_points(query_rgb.shape[1], query_rgb.shape[0])
    step_m = float(args.ray_step_m or convention.get("ray_step_m", 2.0))
    dsm_sampling_mode = str(args.dsm_sampling_mode or convention.get("dsm_sampling_mode", renderer.dsm_sampling_mode))
    dom_sampling_mode = str(args.dom_sampling_mode or convention.get("dom_sampling_mode", renderer.dom_sampling_mode))
    ray_refine_iters = int(args.ray_refine_iters if args.ray_refine_iters is not None else convention.get("ray_refine_iters", renderer.ray_refine_iters))
    ray_conventions = list(RAY_ROTATION_CONVENTIONS) if args.ray_convention == "both" else [args.ray_convention]
    results: Dict[str, List[Dict[str, Any]]] = {}
    for ray_convention in ray_conventions:
        mode_dir = output_dir / f"point_check_{ray_convention}"
        rows = _run_point_check(
            mode_dir,
            query_rgb,
            points,
            renderer,
            photo,
            convention,
            ray_convention,
            args.patch_radius,
            step_m,
            args.ray_max_m,
            dsm_sampling_mode,
            dom_sampling_mode,
            ray_refine_iters,
        )
        _write_json(
            mode_dir / "point_check_summary.json",
            {
                "query_image": args.query_image,
                "xml": args.xml,
                "xml_srs": xml_srs,
                "matched_photo": item["match"],
                "convention_file": convention,
                "xml_projection_rotation": XML_PROJECTION_ROTATION,
                "render_ray_convention": ray_convention,
                "render_ray_rotation": _ray_rotation_description(ray_convention),
                "ray_rotation_matrix_source": "photo.rotation.T" if ray_convention == "cam_to_world_correct" else "photo.rotation",
                "dsm_sampling_mode": dsm_sampling_mode,
                "dom_sampling_mode": dom_sampling_mode,
                "ray_refine_iters": ray_refine_iters,
                "legacy_rotation_convention": _legacy_rotation_convention(ray_convention),
                "rotation_convention": _legacy_rotation_convention(ray_convention),
                "num_points": len(rows),
                "num_hits": sum(1 for r in rows if r["hit"]),
                "points": rows,
            },
        )
        results[ray_convention] = rows
    _write_json(
        output_dir / "point_check_summary.json",
        {
            "query_image": args.query_image,
            "xml": args.xml,
            "xml_srs": xml_srs,
            "matched_photo": item["match"],
            "convention_file": convention,
            "xml_projection_rotation": XML_PROJECTION_ROTATION,
            "ray_conventions": [
                {
                    "render_ray_convention": ray_convention,
                    "render_ray_rotation": _ray_rotation_description(ray_convention),
                    "legacy_rotation_convention": _legacy_rotation_convention(ray_convention),
                    "dsm_sampling_mode": dsm_sampling_mode,
                    "dom_sampling_mode": dom_sampling_mode,
                    "ray_refine_iters": ray_refine_iters,
                    "num_points": len(rows),
                    "num_hits": sum(1 for r in rows if r["hit"]),
                    "output_dir": f"point_check_{ray_convention}",
                }
                for ray_convention, rows in results.items()
            ],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
