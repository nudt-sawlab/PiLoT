#!/usr/bin/env python3
"""Diagnose ContextCapture camera conventions for DOM/DSM rendering."""

import argparse
import csv
import json
import os
import shutil
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_yawfix_refinement_update import _safe_jsonable
from tools.render_contextcapture_xml_domdsm_initial import (
    AXIS_TRANSFORMS,
    CONVENTIONS,
    ContextCaptureDOMDSMRenderer,
    XML_PROJECTION_ROTATION,
    _legacy_rotation_convention,
    _load_pose_file_projected,
    _match_photos,
    _parse_xml,
    _ray_rotation_description,
    _render_visuals,
)


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_XML = "data_caiwangcun/CaiWangCun.xml"
DEFAULT_QUERY_DIR = "data_caiwangcun/query/images/exif_test"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/contextcapture_xml_camera_convention_diagnosis"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _combo_name(combo: Dict[str, Any]) -> str:
    d = "dist" if combo["distortion_enabled"] else "nodist"
    return (
        f"{combo['render_ray_convention']}_{combo['axis_transform_key']}_"
        f"{combo['principal_point_mode']}_{d}_{combo['sampling_mode']}_"
        f"step{str(combo['ray_step_m']).replace('.', 'p')}_scale{str(combo['render_scale']).replace('.', 'p')}"
    )


def _score_rows(rows: List[Dict[str, Any]], min_valid_depth_ratio: float) -> List[Dict[str, Any]]:
    valid = [
        row
        for row in rows
        if float(row.get("valid_depth_ratio", 0.0)) >= min_valid_depth_ratio
        and np.isfinite(float(row.get("edge_chamfer", float("inf"))))
    ]
    source = valid if valid else rows
    return sorted(source, key=lambda r: (float(r["edge_chamfer"]), -float(r["edge_overlap_ratio"]), -float(r["valid_depth_ratio"])))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "image",
        "candidate",
        "rotation_convention",
        "legacy_rotation_convention",
        "render_ray_convention",
        "render_ray_rotation",
        "xml_projection_rotation",
        "axis_transform_key",
        "axis_transform",
        "principal_point_mode",
        "distortion_enabled",
        "sampling_mode",
        "ray_step_m",
        "render_scale",
        "edge_chamfer",
        "edge_overlap_ratio",
        "valid_depth_ratio",
        "query_edge_count",
        "render_edge_count",
        "edge_overlap_count",
        "render_time_sec",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--images", nargs="+", default=["0000.jpg"])
    parser.add_argument("--screen-scale", type=float, default=0.25)
    parser.add_argument("--topk-fullres", type=int, default=10)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=0.95)
    parser.add_argument("--checker-tile", type=int, default=128)
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
    xml_srs, photos = _parse_xml((REPO_ROOT / args.xml).resolve())
    query_dir = (REPO_ROOT / args.query_dir).resolve()
    image_paths = [query_dir / name for name in args.images]
    pose_records = _load_pose_file_projected((REPO_ROOT / args.pose_file).resolve(), xml_srs)
    matches, match_report = _match_photos(image_paths, pose_records, photos)
    _write_json(output_dir / "camera_match_report.json", {"xml_srs": xml_srs, "matches": match_report})
    renderer = ContextCaptureDOMDSMRenderer(config["render_config"], xml_srs, args.chunk_rows)
    if not image_paths:
        raise ValueError("No images requested")
    image_path = image_paths[0]
    item = matches[image_path.name]
    combos: List[Dict[str, Any]] = []
    for render_ray_convention, axis_key, pp_mode, distortion_enabled, ray_step_m, sampling_mode in product(
        CONVENTIONS,
        AXIS_TRANSFORMS.keys(),
        ["xml", "flip_y"],
        [False, True],
        [10.0, 2.0, 1.0],
        ["nearest", "bilinear"],
    ):
        combos.append(
            {
                "render_ray_convention": render_ray_convention,
                "axis_transform_key": axis_key,
                "axis_transform": AXIS_TRANSFORMS[axis_key],
                "principal_point_mode": pp_mode,
                "distortion_enabled": distortion_enabled,
                "ray_step_m": ray_step_m,
                "sampling_mode": sampling_mode,
                "render_scale": args.screen_scale,
            }
        )
    screen_rows: List[Dict[str, Any]] = []
    for idx, combo in enumerate(combos):
        row = _render_visuals(
            image_path,
            item["photo"],
            item["match"],
            renderer,
            output_dir,
            combo["render_ray_convention"],
            combo["distortion_enabled"],
            args.checker_tile,
            combo["axis_transform"],
            combo["principal_point_mode"],
            combo["sampling_mode"],
            combo["ray_step_m"],
            combo["render_scale"],
            subdir=f"screen_{idx:03d}_{_combo_name(combo)}",
        )
        row.update(combo)
        screen_rows.append(row)
    ranked_screen = _score_rows(screen_rows, max(0.01, args.min_valid_depth_ratio - 0.1))
    full_rows: List[Dict[str, Any]] = []
    for rank, screen_row in enumerate(ranked_screen[: args.topk_fullres], start=1):
        combo = {
            "render_ray_convention": screen_row["render_ray_convention"],
            "axis_transform_key": screen_row["axis_transform_key"],
            "axis_transform": screen_row["axis_transform"],
            "principal_point_mode": screen_row["principal_point_mode"],
            "distortion_enabled": screen_row["distortion_enabled"],
            "ray_step_m": screen_row["ray_step_m"],
            "sampling_mode": screen_row["sampling_mode"],
            "render_scale": 1.0,
        }
        row = _render_visuals(
            image_path,
            item["photo"],
            item["match"],
            renderer,
            output_dir,
            combo["render_ray_convention"],
            combo["distortion_enabled"],
            args.checker_tile,
            combo["axis_transform"],
            combo["principal_point_mode"],
            combo["sampling_mode"],
            combo["ray_step_m"],
            combo["render_scale"],
            subdir=f"full_top{rank:02d}_{_combo_name(combo)}",
        )
        row.update(combo)
        row["screen_rank"] = rank
        full_rows.append(row)
    ranked_full = _score_rows(full_rows, args.min_valid_depth_ratio)
    best = ranked_full[0] if ranked_full else ranked_screen[0]
    best_convention = {
        "xml_projection_rotation": XML_PROJECTION_ROTATION,
        "render_ray_convention": best["render_ray_convention"],
        "render_ray_rotation": _ray_rotation_description(best["render_ray_convention"]),
        "legacy_rotation_convention": _legacy_rotation_convention(best["render_ray_convention"]),
        "rotation_convention": _legacy_rotation_convention(best["render_ray_convention"]),
        "axis_transform_key": best["axis_transform_key"],
        "axis_transform": best["axis_transform"],
        "principal_point_mode": best["principal_point_mode"],
        "distortion_enabled": bool(best["distortion_enabled"]),
        "ray_step_m": float(best["ray_step_m"]),
        "sampling_mode": best["sampling_mode"],
        "render_scale": 1.0,
        "subdir": "xml_best_convention",
        "selected_from": "fullres_topk" if ranked_full else "screen",
        "best_metrics": {
            "edge_chamfer": best["edge_chamfer"],
            "edge_overlap_ratio": best["edge_overlap_ratio"],
            "valid_depth_ratio": best["valid_depth_ratio"],
        },
    }
    _write_csv(output_dir / "convention_grid_results.csv", screen_rows)
    _write_csv(output_dir / "convention_fullres_top_results.csv", full_rows)
    _write_json(output_dir / "best_convention.json", best_convention)
    summary = {
        "experiment": "ContextCapture XML camera convention diagnosis",
        "image": image_path.name,
        "num_screen_combinations": len(screen_rows),
        "screen_scale": args.screen_scale,
        "num_fullres_validated": len(full_rows),
        "min_valid_depth_ratio": args.min_valid_depth_ratio,
        "best_convention": best_convention,
        "top_screen": ranked_screen[:20],
        "top_fullres": ranked_full[:20],
    }
    _write_json(output_dir / "summary_metrics.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
