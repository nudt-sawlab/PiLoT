#!/usr/bin/env python3
"""Diagnose query image domain differences against XML-pose GPU mesh render."""

from __future__ import annotations

import argparse
import copy
import csv
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
from tools.render_contextcapture_xml_domdsm_initial import (
    _load_pose_file_projected,
    _match_photos,
    _parse_xml,
)


DEFAULT_XML = "data_caiwangcun/CaiWangCun.xml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test/0000.jpg"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_CONFIG = "configs/caiwangcun_domdsm.yaml"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/query_image_domain_vs_render_0000"
PP_OFFSETS = [-30, -20, -10, 0, 10, 20, 30]


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _camera_matrix(intr: Any, cx_offset: float = 0.0, cy_offset: float = 0.0, pp_mode: str = "xml") -> np.ndarray:
    cy_base = float(intr.cy) if pp_mode == "xml" else float(intr.height - intr.cy)
    return np.asarray(
        [
            [float(intr.fx), 0.0, float(intr.cx) + float(cx_offset)],
            [0.0, float(intr.fy), cy_base + float(cy_offset)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _dist_coeffs(intr: Any) -> np.ndarray:
    return np.asarray([intr.k1, intr.k2, intr.p1, intr.p2, intr.k3], dtype=np.float64)


def _make_gpu_config(config: Dict[str, Any], width: int, height: int, K: np.ndarray) -> Dict[str, Any]:
    render_config = copy.deepcopy(config["render_config"])
    render_config["render_camera"] = [
        int(width),
        int(height),
        float(K[0, 2]),
        float(K[1, 2]),
        float(K[0, 0]),
        float(K[1, 1]),
    ]
    render_config["dom_dsm"]["render_backend"] = "gpu_mesh"
    render_config["dom_dsm"]["gpu_renderer"] = "nvdiffrast"
    render_config["dom_dsm"]["texture_v_flip"] = False
    render_config["dom_dsm"]["output_y_flip"] = True
    render_config["dom_dsm"]["debug_texture_mode"] = "none"
    render_config["dom_dsm"]["debug_every"] = 0
    return render_config


def _render_gpu(config: Dict[str, Any], width: int, height: int, K: np.ndarray, xml_srs: str, photo: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], float]:
    render_config = _make_gpu_config(config, width, height, K)
    renderer = DOMDSMRenderer(render_config)
    to_wgs84 = Transformer.from_crs(xml_srs, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(photo.center_xml[0], photo.center_xml[1])
    trans = [float(lon), float(lat), float(photo.center_xml[2])]
    t0 = time.perf_counter()
    rgb, depth = renderer.render_matrix(trans, photo.rotation.T)
    return rgb, depth, renderer.last_render_metadata, time.perf_counter() - t0


def _scaled_camera_matrix(K: np.ndarray, scale: float, cx_offset_px: float = 0.0, cy_offset_px: float = 0.0) -> np.ndarray:
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale
    K_scaled[1, 1] *= scale
    K_scaled[0, 2] = (K[0, 2] + float(cx_offset_px)) * scale
    K_scaled[1, 2] = (K[1, 2] + float(cy_offset_px)) * scale
    return K_scaled


def _undistort_query(query_rgb: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    return cv2.undistort(query_rgb, K, D, None, K)


def _distort_pinhole_render_to_raw(render_rgb: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    h, w = render_rgb.shape[:2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    xd = ((u - K[0, 2]) / K[0, 0]).astype(np.float32)
    yd = ((v - K[1, 2]) / K[1, 1]).astype(np.float32)
    distorted_points = np.stack([xd, yd], axis=-1).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(distorted_points, np.eye(3, dtype=np.float64), D, P=None)
    xu = undistorted[:, 0, 0].reshape(h, w)
    yu = undistorted[:, 0, 1].reshape(h, w)
    map_x = (K[0, 0] * xu + K[0, 2]).astype(np.float32)
    map_y = (K[1, 1] * yu + K[1, 2]).astype(np.float32)
    return cv2.remap(render_rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _write_candidate(out_dir: Path, name: str, query_rgb: np.ndarray, render_rgb: np.ndarray, checker_tile: int) -> Dict[str, Any]:
    cand = out_dir / name
    cand.mkdir(parents=True, exist_ok=True)
    overlay = _make_overlay(query_rgb, render_rgb)
    edge_overlay, metrics = _edge_overlay(query_rgb, render_rgb)
    checker = _checkerboard(query_rgb, render_rgb, checker_tile)
    _write_rgb(cand / "overlay.png", overlay)
    _write_rgb(cand / "edge_overlay.png", edge_overlay)
    _write_rgb(cand / "checkerboard.png", checker)
    return {"candidate": name, "output_dir": os.fspath(cand), **metrics}


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row.keys() if not isinstance(row.get(k), (dict, list))})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_offsets(text: str) -> List[float]:
    offsets = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not offsets:
        raise ValueError("--pp-offsets cannot be empty")
    return offsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checker-tile", type=int, default=128)
    parser.add_argument("--scan-scale", type=float, default=0.25, help="Scale used only for principal point scan renders.")
    parser.add_argument(
        "--pp-offsets",
        default="-30,0,30",
        help="Comma-separated full-resolution principal point offsets. Use '-30,-20,-10,0,10,20,30' for the full grid.",
    )
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
    xml_srs, photos = _parse_xml((REPO_ROOT / args.xml).resolve())
    query_path = (REPO_ROOT / args.query_image).resolve()
    pose_records = _load_pose_file_projected((REPO_ROOT / args.pose_file).resolve(), xml_srs)
    matches, match_report = _match_photos([query_path], pose_records, photos)
    photo = matches[query_path.name]["photo"]
    intr = photo.intrinsics
    K_xml = _camera_matrix(intr, 0.0, 0.0, "xml")
    D = _dist_coeffs(intr)

    raw_query = _read_rgb(query_path)
    if raw_query.shape[:2] != (intr.height, intr.width):
        raise ValueError(f"Query shape {raw_query.shape[:2]} does not match XML {(intr.height, intr.width)}")

    pinhole_rgb, _depth, gpu_meta, render_time = _render_gpu(config, intr.width, intr.height, K_xml, xml_srs, photo)
    undistorted_query = _undistort_query(raw_query, K_xml, D)
    distorted_render = _distort_pinhole_render_to_raw(pinhole_rgb, K_xml, D)

    _write_rgb(output_dir / "raw_query.png", raw_query)
    _write_rgb(output_dir / "gpu_pinhole_render.png", pinhole_rgb)
    _write_rgb(output_dir / "query_undistorted.png", undistorted_query)
    _write_rgb(output_dir / "gpu_render_distorted_to_raw_domain.png", distorted_render)

    candidates: List[Dict[str, Any]] = []
    candidates.append(_write_candidate(output_dir, "pinhole_render_vs_raw_query", raw_query, pinhole_rgb, args.checker_tile))
    candidates.append(_write_candidate(output_dir, "pinhole_render_vs_undistorted_query", undistorted_query, pinhole_rgb, args.checker_tile))
    candidates.append(_write_candidate(output_dir, "distorted_render_vs_raw_query", raw_query, distorted_render, args.checker_tile))

    baseline = next(c for c in candidates if c["candidate"] == "pinhole_render_vs_raw_query")
    distorted = next(c for c in candidates if c["candidate"] == "distorted_render_vs_raw_query")

    if not (0.0 < float(args.scan_scale) <= 1.0):
        raise ValueError(f"--scan-scale must be in (0, 1], got {args.scan_scale}")
    scan_scale = float(args.scan_scale)
    scan_width = max(1, int(round(intr.width * scan_scale)))
    scan_height = max(1, int(round(intr.height * scan_scale)))
    scan_query = cv2.resize(raw_query, (scan_width, scan_height), interpolation=cv2.INTER_AREA)
    scan_D = D.copy()
    pp_offsets = _parse_offsets(args.pp_offsets)

    scan_rows: List[Dict[str, Any]] = []
    for pp_mode in ("xml", "flip_y"):
        K_scan_base_fullres = _camera_matrix(intr, 0.0, 0.0, pp_mode)
        for cx_offset in pp_offsets:
            for cy_offset in pp_offsets:
                K = _scaled_camera_matrix(K_scan_base_fullres, scan_scale, cx_offset, cy_offset)
                rgb, _d, meta, t_render = _render_gpu(config, scan_width, scan_height, K, xml_srs, photo)
                warped = _distort_pinhole_render_to_raw(rgb, K, scan_D)
                _edge_img, metrics = _edge_overlay(scan_query, warped)
                row = {
                    "principal_point_mode": pp_mode,
                    "cx_offset_px": float(cx_offset),
                    "cy_offset_px": float(cy_offset),
                    "scan_scale": scan_scale,
                    "scan_width": int(scan_width),
                    "scan_height": int(scan_height),
                    "render_time_sec": float(t_render),
                    "backend_used": meta.get("backend_used"),
                    "fallback_reason": meta.get("fallback_reason"),
                    **metrics,
                }
                scan_rows.append(row)
    ranked_scan = sorted(scan_rows, key=lambda r: (float(r["edge_chamfer"]), -float(r["edge_overlap_ratio"])))
    best_scan = ranked_scan[0]
    zero_scan = next(r for r in scan_rows if r["principal_point_mode"] == "xml" and r["cx_offset_px"] == 0.0 and r["cy_offset_px"] == 0.0)
    _write_csv(output_dir / "principal_point_scan.csv", scan_rows)
    scan_summary = {
        "scan_scale": scan_scale,
        "scan_width": int(scan_width),
        "scan_height": int(scan_height),
        "pp_offsets": pp_offsets,
        "best": best_scan,
        "zero_xml": zero_scan,
        "top10": ranked_scan[:10],
        "principal_point_shift_likely": bool(
            (abs(float(best_scan["cx_offset_px"])) >= 10.0 or abs(float(best_scan["cy_offset_px"])) >= 10.0)
            and float(best_scan["edge_chamfer"]) < float(zero_scan["edge_chamfer"]) - 2.0
        ),
        "recommended_cx_offset_px": float(best_scan["cx_offset_px"]),
        "recommended_cy_offset_px": float(best_scan["cy_offset_px"]),
    }
    _write_json(output_dir / "principal_point_scan_summary.json", scan_summary)

    best_candidate = sorted(candidates, key=lambda r: (float(r["edge_chamfer"]), -float(r["edge_overlap_ratio"])))[0]
    metrics = {
        "experiment": "Query image domain vs XML GPU mesh render",
        "xml": args.xml,
        "xml_srs": xml_srs,
        "query_image_path": args.query_image,
        "pose_file_path": args.pose_file,
        "config": args.config,
        "output_dir": args.output_dir,
        "xml_photo_id": photo.photo_id,
        "xml_image_path": photo.image_path,
        "intrinsics": intr.as_dict(),
        "K_xml": K_xml,
        "D": D,
        "gpu_metadata": gpu_meta,
        "gpu_render_time_sec": render_time,
        "candidates": candidates,
        "best_candidate": best_candidate["candidate"],
        "distorted_render_improves_raw_query": bool(
            float(distorted["edge_chamfer"]) < float(baseline["edge_chamfer"]) - 3.0
            and float(distorted["edge_overlap_ratio"]) > float(baseline["edge_overlap_ratio"])
        ),
        "principal_point_shift_likely": scan_summary["principal_point_shift_likely"],
        "recommended_cx_offset_px": scan_summary["recommended_cx_offset_px"],
        "recommended_cy_offset_px": scan_summary["recommended_cy_offset_px"],
        "principal_point_scan_best": best_scan,
        "principal_point_scan_zero_xml": zero_scan,
        "camera_match_report": match_report,
    }
    _write_json(output_dir / "image_domain_metrics.json", metrics)
    print(json.dumps(_safe_jsonable(metrics), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
