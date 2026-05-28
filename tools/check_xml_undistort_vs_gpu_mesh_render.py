#!/usr/bin/env python3
"""Compare XML-distortion-undistorted query against GPU mesh pinhole render."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

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
from tools.run_dom_dsm_single_full import _depth_stats, _save_depth_png


DEFAULT_XML = "data_caiwangcun/CaiWangCun.xml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test/0000.jpg"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_CONFIG = "configs/caiwangcun_domdsm.yaml"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/xml_undistort_gpu_mesh_compare_0000"
BASELINE_EDGE_CHAMFER = 28.398826599121094
BASELINE_EDGE_OVERLAP_RATIO = 0.2476509463651486


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_query_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _undistort_query(query_rgb: np.ndarray, intr: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = np.asarray(
        [
            [float(intr.fx), 0.0, float(intr.cx)],
            [0.0, float(intr.fy), float(intr.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    D = np.asarray(
        [float(intr.k1), float(intr.k2), float(intr.p1), float(intr.p2), float(intr.k3)],
        dtype=np.float64,
    )
    undistorted = cv2.undistort(query_rgb, K, D, None, K)
    return undistorted, K, D


def _make_renderer_config(config: Dict[str, Any], intr: Any) -> Dict[str, Any]:
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
    match = item["match"]
    intr = photo.intrinsics

    query_rgb = _read_query_rgb(query_path)
    expected_shape = (int(intr.height), int(intr.width))
    if query_rgb.shape[:2] != expected_shape:
        raise ValueError(
            f"Query shape {query_rgb.shape[:2]} does not match XML camera {expected_shape}"
        )

    query_undistorted, K, D = _undistort_query(query_rgb, intr)

    render_config = _make_renderer_config(config, intr)
    renderer = DOMDSMRenderer(render_config)
    to_wgs84 = Transformer.from_crs(xml_srs, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(photo.center_xml[0], photo.center_xml[1])
    trans = [float(lon), float(lat), float(photo.center_xml[2])]
    R_camera_to_world = photo.rotation.T

    t0 = time.perf_counter()
    render_rgb, depth = renderer.render_matrix(trans, R_camera_to_world)
    render_time_sec = time.perf_counter() - t0
    if render_rgb.shape[:2] != query_undistorted.shape[:2]:
        raise ValueError(
            "Undistorted query/render shape mismatch: "
            f"query={query_undistorted.shape[:2]} render={render_rgb.shape[:2]}"
        )

    overlay = _make_overlay(query_undistorted, render_rgb)
    edge_overlay, edge_metrics = _edge_overlay(query_undistorted, render_rgb)
    checkerboard = _checkerboard(query_undistorted, render_rgb, args.checker_tile)

    _write_rgb(output_dir / "query_undistorted.png", query_undistorted)
    _write_rgb(output_dir / "render_pinhole.png", render_rgb)
    _write_rgb(output_dir / "overlay_undistorted_query_render.png", overlay)
    _write_rgb(output_dir / "edge_overlay_undistorted_query_render.png", edge_overlay)
    _write_rgb(output_dir / "checkerboard_undistorted_query_render.png", checkerboard)
    _save_depth_png(output_dir / "rendered_depth.png", depth)

    chamfer = float(edge_metrics["edge_chamfer"])
    overlap = float(edge_metrics["edge_overlap_ratio"])
    metrics: Dict[str, Any] = {
        "experiment": "XML undistorted query vs GPU mesh pinhole render",
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
        "matched_exif_distance_xy_m": match.get("match_distance_xy_m"),
        "matched_exif_distance_z_m": match.get("match_distance_z_m"),
        "intrinsics": intr.as_dict(),
        "K": K,
        "D": D,
        "undistort_method": "cv2.undistort(query, K, D, None, K)",
        "render_camera": render_config["render_camera"],
        "render_ray_rotation": "R_camera_to_world = R_xml.T",
        "xml_projection_rotation": "R_world_to_camera = R_xml",
        "query_resized_for_overlay": False,
        "render_width": int(render_rgb.shape[1]),
        "render_height": int(render_rgb.shape[0]),
        "render_time_sec": float(render_time_sec),
        "renderer_metadata": renderer.last_render_metadata,
        **_depth_stats(depth),
        **edge_metrics,
        "baseline_edge_chamfer": BASELINE_EDGE_CHAMFER,
        "baseline_edge_overlap_ratio": BASELINE_EDGE_OVERLAP_RATIO,
        "edge_chamfer_delta_vs_baseline": chamfer - BASELINE_EDGE_CHAMFER,
        "edge_overlap_ratio_delta_vs_baseline": overlap - BASELINE_EDGE_OVERLAP_RATIO,
        "distortion_likely_primary": bool(
            chamfer < BASELINE_EDGE_CHAMFER and overlap > 0.25
        ),
        "camera_match_report": match_report,
    }
    _write_json(output_dir / "metrics_undistorted.json", metrics)
    print(json.dumps(_safe_jsonable(metrics), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
