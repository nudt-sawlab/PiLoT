#!/usr/bin/env python3
"""Small P9.1 check for freeze pitch/roll yaw convention."""

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from tools.diagnose_yawfix_refinement_update import (
    BASE_EULER,
    _get_raster_transformers,
    _offset_between,
    _read_query_rgb,
    _render_candidate,
    _safe_jsonable,
)
from tools.run_dom_dsm_single_full import _setup_camera


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test_16x9/0000.jpg"
DEFAULT_P9_SUMMARY = "docs/experiments/dom_dsm_prepare/safe_pilot_refinement_p9/summary_metrics.json"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/p9_1_freeze_yaw_convention_check"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_yaw(yaw: float) -> float:
    return float(((yaw + 180.0) % 360.0) - 180.0)


def _with_alt(trans: List[float], alt: float) -> List[float]:
    return [float(trans[0]), float(trans[1]), float(alt)]


def _candidate(
    name: str,
    trans: List[float],
    euler: List[float],
    note: str,
) -> Dict[str, Any]:
    return {
        "name": name,
        "translation_lon_lat_alt": [float(x) for x in trans],
        "euler_pitch_roll_yaw": [float(x) for x in euler],
        "note": note,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    parser.add_argument("--p9-summary", default=DEFAULT_P9_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--checker-tile", type=int, default=32)
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists() and not args.no_clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = {
        "config_path": args.config,
        "query_image_path": args.query_image,
        "p9_summary_path": args.p9_summary,
        "output_dir": os.fspath(output_dir),
        "failure_stage": None,
        "traceback": None,
    }
    stage = "start"
    start_total = time.time()

    try:
        stage = "load_inputs"
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["default_confs"]["cam_query"]["max_size"] = args.width
        _query_resize_ratio, _raw_query_camera, render_camera_gs, _query_camera, _render_camera = _setup_camera(config)
        width, height = int(render_camera_gs[0]), int(render_camera_gs[1])
        p9 = _load_json(REPO_ROOT / args.p9_summary)
        initial = p9["methods"]["initial"]
        raw = p9["methods"]["rebuilt_cuda_raw_refined"]
        p9_bad = p9["methods"]["refined_freeze_alt_pitch_roll"]
        initial_trans = [float(x) for x in initial["translation_lon_lat_alt"]]
        initial_euler = [float(x) for x in initial["euler_pitch_roll_yaw"]]
        refined_trans = [float(x) for x in raw["translation_lon_lat_alt"]]
        refined_euler = [float(x) for x in raw["euler_pitch_roll_yaw"]]
        refined_yaw = float(refined_euler[2])
        downward_refined_yaw = _normalize_yaw(refined_yaw + 180.0)
        base_yaw = float(BASE_EULER[2])
        frozen_pr_raw_yaw = [float(BASE_EULER[0]), float(BASE_EULER[1]), refined_yaw]
        frozen_pr_downward_yaw = [float(BASE_EULER[0]), float(BASE_EULER[1]), downward_refined_yaw]
        frozen_pr_base_yaw = [float(BASE_EULER[0]), float(BASE_EULER[1]), base_yaw]
        refined_initial_alt = _with_alt(refined_trans, initial_trans[2])

        stage = "init_renderer"
        config["render_config"]["init_rot"] = initial_euler
        config["render_config"]["init_trans"] = initial_trans
        renderer = DOMDSMRenderer(config["render_config"])
        to_raster, _from_raster, raster_crs = _get_raster_transformers(config)
        query_rgb = _read_query_rgb(REPO_ROOT / args.query_image, width, height)

        stage = "build_candidates"
        candidates = [
            _candidate("initial", initial_trans, initial_euler, "Baseline initial pose."),
            _candidate("raw_refined", refined_trans, refined_euler, "Raw rebuilt CUDA refined pose from P9."),
            _candidate(
                "p9_bad_freeze_alt_pitch_roll",
                refined_initial_alt,
                frozen_pr_raw_yaw,
                "Exact P9 freeze_alt_pitch_roll reproduction: initial alt/pitch/roll plus raw refined yaw.",
            ),
            _candidate(
                "refined_trans_initial_euler",
                refined_trans,
                initial_euler,
                "User-requested refined translation with full initial Euler.",
            ),
            _candidate(
                "refined_trans_raw_yaw_frozen_pr",
                refined_trans,
                frozen_pr_raw_yaw,
                "Refined full translation with initial pitch/roll and raw refined yaw.",
            ),
            _candidate(
                "refined_trans_downward_refined_yaw",
                refined_trans,
                frozen_pr_downward_yaw,
                "Refined full translation with initial pitch/roll and raw refined yaw + 180 normalized.",
            ),
            _candidate(
                "refined_trans_base_yaw",
                refined_trans,
                frozen_pr_base_yaw,
                "Refined full translation with initial pitch/roll and base yaw 29.2.",
            ),
            _candidate(
                "refined_initial_alt_downward_refined_yaw",
                refined_initial_alt,
                frozen_pr_downward_yaw,
                "Direct candidate for a corrected P9 freeze_alt_pitch_roll rule.",
            ),
        ]

        stage = "render_candidates"
        rows = []
        for item in candidates:
            offsets = _offset_between(initial_trans, item["translation_lon_lat_alt"], to_raster)
            metrics = _render_candidate(
                item["name"],
                renderer,
                query_rgb,
                item["translation_lon_lat_alt"],
                item["euler_pitch_roll_yaw"],
                output_dir,
                args.checker_tile,
                {
                    "candidate": item["name"],
                    "note": item["note"],
                    "east_offset_m": offsets[0],
                    "north_offset_m": offsets[1],
                    "alt_offset_m": offsets[2],
                },
            )
            metrics["method"] = item["name"]
            metrics["note"] = item["note"]
            metrics["delta_vs_initial"] = {
                "edge_chamfer_delta": float(metrics["edge_chamfer"]) - float(initial["edge_chamfer"]),
                "edge_overlap_ratio_delta": float(metrics["edge_overlap_ratio"]) - float(initial["edge_overlap_ratio"]),
            }
            rows.append(metrics)

        stage = "summarize"
        by_name = {row["method"]: row for row in rows}
        bad = by_name["p9_bad_freeze_alt_pitch_roll"]
        downward = by_name["refined_initial_alt_downward_refined_yaw"]
        base = by_name["refined_trans_base_yaw"]
        raw_yaw_minus_base = _normalize_yaw(refined_yaw - base_yaw)
        downward_yaw_minus_base = _normalize_yaw(downward_refined_yaw - base_yaw)
        likely_yaw_convention_bug = (
            abs(abs(raw_yaw_minus_base) - 180.0) < 5.0
            and abs(downward_yaw_minus_base) < 5.0
            and float(downward["edge_overlap_ratio"]) > float(bad["edge_overlap_ratio"])
            and float(downward["edge_chamfer"]) < float(bad["edge_chamfer"])
        )
        recommendation = (
            "fix_p9_freeze_yaw_by_using_refined_yaw_plus_180_or_base_yaw"
            if likely_yaw_convention_bug
            else "do_not_use_raw_refined_yaw_when_freezing_pitch_roll; prefer_base_yaw_until_conversion_is_proven"
        )
        summary = {
            "config": args.config,
            "query_image": args.query_image,
            "p9_summary": args.p9_summary,
            "output_dir": os.fspath(output_dir),
            "candidates": rows,
            "rank_by_visual_chamfer": [row["method"] for row in sorted(rows, key=lambda x: float(x["edge_chamfer"]))],
            "rank_by_visual_overlap": [row["method"] for row in sorted(rows, key=lambda x: float(x["edge_overlap_ratio"]), reverse=True)],
            "yaw_analysis": {
                "base_yaw": base_yaw,
                "raw_refined_yaw": refined_yaw,
                "downward_refined_yaw_raw_plus_180_normalized": downward_refined_yaw,
                "raw_yaw_minus_base_normalized": raw_yaw_minus_base,
                "downward_yaw_minus_base_normalized": downward_yaw_minus_base,
                "likely_yaw_convention_bug": likely_yaw_convention_bug,
                "recommendation": recommendation,
            },
            "p9_bad_reproduction": {
                "expected_p9_chamfer": p9_bad["edge_chamfer"],
                "expected_p9_overlap": p9_bad["edge_overlap_ratio"],
                "measured_chamfer": bad["edge_chamfer"],
                "measured_overlap": bad["edge_overlap_ratio"],
            },
            "raster_crs": raster_crs,
            "total_time_sec": time.time() - start_total,
        }
        _write_json(output_dir / "summary_metrics.json", summary)
        _write_json(output_dir / "run_log.json", {**run_log, "summary_path": os.fspath(output_dir / "summary_metrics.json")})
        print(json.dumps(summary["yaw_analysis"], indent=2, sort_keys=True))
        return 0

    except Exception:
        run_log["failure_stage"] = stage
        run_log["traceback"] = traceback.format_exc()
        run_log["total_time_sec"] = time.time() - start_total
        _write_json(output_dir / "run_log.json", run_log)
        print(run_log["traceback"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
