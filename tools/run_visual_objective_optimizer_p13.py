#!/usr/bin/env python3
"""Run P13 visual-objective derivative-free pose optimization."""

import argparse
import copy
import csv
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.localization.localizer import RenderLocalizer
from pixloc.pixlib.datasets.view import read_image
from pixloc.utils.dom_dsm.candidate_scorer import load_query_poses_from_file
from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from pixloc.utils.dom_dsm.pose_adapter import apply_enu_offset, make_downward_euler_from_yaw
from tools.diagnose_yawfix_refinement_update import (
    _checkerboard,
    _edge_overlay,
    _get_raster_transformers,
    _make_overlay,
    _read_query_rgb,
    _safe_jsonable,
    _write_rgb,
)
from tools.run_dom_dsm_single_full import _depth_stats, _format_pose_line, _resize_query_for_refine, _setup_camera


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_DIR = "data_caiwangcun/query/images/exif_test"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/visual_objective_optimizer_p13"
EPS = 1e-9


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_pass(row: Dict[str, Any], initial: Dict[str, Any]) -> bool:
    return (
        float(row["edge_chamfer"]) <= float(initial["edge_chamfer"]) + EPS
        and float(row["edge_overlap_ratio"]) + EPS >= float(initial["edge_overlap_ratio"])
    )


def _worse(row: Dict[str, Any], initial: Dict[str, Any]) -> bool:
    return (
        float(row["edge_chamfer"]) > float(initial["edge_chamfer"]) + EPS
        or float(row["edge_overlap_ratio"]) + EPS < float(initial["edge_overlap_ratio"])
    )


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _median(values: Sequence[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def _rate(values: Sequence[bool]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _candidate_name(east: float, north: float, yaw: float) -> str:
    def s(value: float) -> str:
        prefix = "p" if value > 0 else "m" if value < 0 else "z"
        mag = f"{abs(float(value)):.3f}".rstrip("0").rstrip(".").replace(".", "p")
        return prefix + (mag or "0")
    return f"e{s(east)}_n{s(north)}_y{s(yaw)}"


def _objective(row: Dict[str, Any], overlap_weight: float) -> float:
    return float(row["edge_chamfer"]) - float(overlap_weight) * float(row["edge_overlap_ratio"])


def _render_eval(
    image_name: str,
    offset: Tuple[float, float, float],
    initial_trans: Sequence[float],
    base_yaw: float,
    renderer: DOMDSMRenderer,
    query_rgb: np.ndarray,
    output_dir: Path,
    to_raster: Any,
    from_raster: Any,
    checker_tile: int,
    overlap_weight: float,
    save_images: bool,
    cache: Dict[Tuple[float, float, float], Dict[str, Any]],
) -> Dict[str, Any]:
    east, north, yaw = [float(x) for x in offset]
    key = (round(east, 6), round(north, 6), round(yaw, 6))
    if key in cache:
        return cache[key]
    name = "initial" if abs(east) < EPS and abs(north) < EPS and abs(yaw) < EPS else _candidate_name(east, north, yaw)
    trans = apply_enu_offset(initial_trans, east, north, 0.0, to_raster, from_raster)
    euler = make_downward_euler_from_yaw(base_yaw + yaw)
    out_dir = output_dir / "candidates" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    render_rgb, depth = renderer.render(trans, euler)
    render_time = time.perf_counter() - t0
    overlay = _make_overlay(query_rgb, render_rgb)
    edge_overlay, edge_metrics = _edge_overlay(query_rgb, render_rgb)
    if save_images:
        _write_rgb(out_dir / "rendered_rgb.png", render_rgb)
        _write_rgb(out_dir / "overlay.png", overlay)
        _write_rgb(out_dir / "edge_overlay.png", edge_overlay)
        _write_rgb(out_dir / "checkerboard.png", _checkerboard(query_rgb, render_rgb, checker_tile))
    row = {
        "image": image_name,
        "candidate": name,
        "translation_lon_lat_alt": trans,
        "euler_pitch_roll_yaw": euler,
        "offset_east_m": east,
        "offset_north_m": north,
        "offset_alt_m": 0.0,
        "yaw_offset_deg": yaw,
        "render_time_sec": render_time,
        **_depth_stats(depth),
        **edge_metrics,
    }
    row["objective"] = _objective(row, overlap_weight)
    row["passes_strict_visual_gate"] = False
    row["safe_output_worse_than_initial"] = None
    _write_json(out_dir / "metrics.json", row)
    cache[key] = row
    return row


def _directions(optimize_vars: Sequence[str], step_xy: float, step_yaw: float) -> List[Tuple[float, float, float]]:
    dirs = []
    if "east" in optimize_vars:
        dirs.extend([(step_xy, 0.0, 0.0), (-step_xy, 0.0, 0.0)])
    if "north" in optimize_vars:
        dirs.extend([(0.0, step_xy, 0.0), (0.0, -step_xy, 0.0)])
    if "yaw" in optimize_vars:
        dirs.extend([(0.0, 0.0, step_yaw), (0.0, 0.0, -step_yaw)])
    return dirs


def _select_best_strict(rows: Sequence[Dict[str, Any]], initial: Dict[str, Any], overlap_weight: float) -> Dict[str, Any]:
    passing = [row for row in rows if row["candidate"] != "initial" and _safe_pass(row, initial)]
    if not passing:
        return initial
    return sorted(passing, key=lambda row: (float(row["objective"]), float(row["edge_chamfer"]), -float(row["edge_overlap_ratio"])))[0]


def _write_result_pose(path: Path, image_name: str, selected: Dict[str, Any], initial: Dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# P13 selected safe pose",
                f"image: {image_name}",
                f"method: {selected['candidate']}",
                "policy: strict_visual_gate_then_best_visual_objective",
                "lon lat alt: " + " ".join(str(float(x)) for x in selected["translation_lon_lat_alt"]),
                "euler_pitch_roll_yaw: " + " ".join(str(float(x)) for x in selected["euler_pitch_roll_yaw"]),
                f"edge_chamfer: {selected['edge_chamfer']}",
                f"edge_overlap_ratio: {selected['edge_overlap_ratio']}",
                f"initial_edge_chamfer: {initial['edge_chamfer']}",
                f"initial_edge_overlap_ratio: {initial['edge_overlap_ratio']}",
                f"safe_output_worse_than_initial: {_worse(selected, initial)}",
                _format_pose_line(image_name, selected["translation_lon_lat_alt"], selected["euler_pitch_roll_yaw"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _save_top_images(rows: Sequence[Dict[str, Any]], selected: Dict[str, Any], initial: Dict[str, Any], image_name: str, initial_trans: Sequence[float], base_yaw: float, renderer: DOMDSMRenderer, query_rgb: np.ndarray, output_dir: Path, to_raster: Any, from_raster: Any, checker_tile: int, overlap_weight: float, cache: Dict[Tuple[float, float, float], Dict[str, Any]], topk: int) -> None:
    names = {initial["candidate"], selected["candidate"]}
    names.update(row["candidate"] for row in sorted(rows, key=lambda r: (float(r["objective"]), float(r["edge_chamfer"])))[:topk])
    for row in rows:
        if row["candidate"] in names:
            _render_eval(
                image_name,
                (row["offset_east_m"], row["offset_north_m"], row["yaw_offset_deg"]),
                initial_trans,
                base_yaw,
                renderer,
                query_rgb,
                output_dir,
                to_raster,
                from_raster,
                checker_tile,
                overlap_weight,
                True,
                cache,
            )


def _optimize_one(
    image_path: Path,
    pose_entry: Dict[str, Any],
    renderer: DOMDSMRenderer,
    query_resize_ratio: Any,
    raw_query_camera: Any,
    render_camera_gs: Any,
    config: Dict[str, Any],
    to_raster: Any,
    from_raster: Any,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    image_name = image_path.name
    out_dir = (REPO_ROOT / args.output_dir / image_path.stem).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    width = int(render_camera_gs[0])
    height = int(render_camera_gs[1])
    query_rgb = _read_query_rgb(image_path, width, height)
    trans = [float(x) for x in pose_entry["translation_lon_lat_alt"]]
    base_yaw = float(pose_entry["base_yaw"])
    cache: Dict[Tuple[float, float, float], Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    initial = _render_eval(
        image_name,
        (0.0, 0.0, 0.0),
        trans,
        base_yaw,
        renderer,
        query_rgb,
        out_dir,
        to_raster,
        from_raster,
        args.checker_tile,
        args.overlap_weight,
        True,
        cache,
    )
    initial["passes_strict_visual_gate"] = True
    initial["safe_output_worse_than_initial"] = False
    rows.append(initial)
    center = (0.0, 0.0, 0.0)
    step_xy = float(args.initial_step_xy)
    step_yaw = float(args.initial_step_yaw)
    best = initial
    eval_count = 1
    reason = "max_iters"
    for iteration in range(args.max_iters):
        if eval_count >= args.max_evals:
            reason = "max_evals"
            break
        if step_xy < args.min_step_xy and step_yaw < args.min_step_yaw:
            reason = "min_step"
            break
        candidates = [center]
        for de, dn, dy in _directions(args.optimize_vars, step_xy, step_yaw):
            candidates.append((center[0] + de, center[1] + dn, center[2] + dy))
        evaluated = []
        for cand in candidates:
            before = len(cache)
            row = _render_eval(
                image_name,
                cand,
                trans,
                base_yaw,
                renderer,
                query_rgb,
                out_dir,
                to_raster,
                from_raster,
                args.checker_tile,
                args.overlap_weight,
                False,
                cache,
            )
            if len(cache) > before:
                eval_count += 1
                rows.append(row)
            row["passes_strict_visual_gate"] = _safe_pass(row, initial)
            row["safe_output_worse_than_initial"] = _worse(row, initial)
            evaluated.append(row)
            if eval_count >= args.max_evals:
                break
        safe_candidates = [row for row in evaluated if row["candidate"] != "initial" and row["passes_strict_visual_gate"]]
        improved = [row for row in safe_candidates if float(row["objective"]) < float(best["objective"]) - EPS]
        if improved:
            best = sorted(improved, key=lambda r: (float(r["objective"]), float(r["edge_chamfer"]), -float(r["edge_overlap_ratio"])))[0]
            center = (best["offset_east_m"], best["offset_north_m"], best["yaw_offset_deg"])
            moved = True
        else:
            step_xy *= float(args.step_decay)
            step_yaw *= float(args.step_decay)
            moved = False
        trace.append(
            {
                "iteration": iteration,
                "center_east_m": center[0],
                "center_north_m": center[1],
                "center_yaw_deg": center[2],
                "step_xy": step_xy,
                "step_yaw": step_yaw,
                "best_candidate": best["candidate"],
                "best_objective": best["objective"],
                "best_chamfer": best["edge_chamfer"],
                "best_overlap": best["edge_overlap_ratio"],
                "moved": moved,
                "num_evaluations": eval_count,
            }
        )
    else:
        reason = "max_iters"
    selected = _select_best_strict(rows, initial, args.overlap_weight)
    for row in rows:
        row["passes_strict_visual_gate"] = _safe_pass(row, initial)
        row["safe_output_worse_than_initial"] = _worse(row, initial)
        row["selected_by_visual_objective"] = row["candidate"] == selected["candidate"]
    _save_top_images(rows, selected, initial, image_name, trans, base_yaw, renderer, query_rgb, out_dir, to_raster, from_raster, args.checker_tile, args.overlap_weight, cache, args.save_topk_images)
    _write_result_pose(out_dir / "result_pose_safe_p13.txt", image_name, selected, initial)
    trace_path = out_dir / "optimization_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["iteration", "center_east_m", "center_north_m", "center_yaw_deg", "step_xy", "step_yaw", "best_candidate", "best_objective", "best_chamfer", "best_overlap", "moved", "num_evaluations"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trace)
    _write_json(out_dir / "optimization_trace.json", {"trace": trace})
    summary = {
        "image": image_name,
        "initial": initial,
        "selected": selected,
        "selected_is_initial": selected["candidate"] == "initial",
        "selected_offset_east_m": selected["offset_east_m"],
        "selected_offset_north_m": selected["offset_north_m"],
        "selected_yaw_offset_deg": selected["yaw_offset_deg"],
        "selected_alt_offset_m": 0.0,
        "chamfer_improvement": float(initial["edge_chamfer"]) - float(selected["edge_chamfer"]),
        "overlap_improvement": float(selected["edge_overlap_ratio"]) - float(initial["edge_overlap_ratio"]),
        "safe_output_worse_than_initial": _worse(selected, initial),
        "num_evaluations": len(cache),
        "stop_reason": reason,
        "final_step_xy": step_xy,
        "final_step_yaw": step_yaw,
        "optimization_trace": trace,
        "candidates": rows,
    }
    _write_json(out_dir / "summary_metrics.json", summary)
    return summary, rows


def _write_csvs(output_dir: Path, image_rows: List[Dict[str, Any]], candidate_rows: List[Dict[str, Any]]) -> None:
    with (output_dir / "batch_image_results.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["image", "initial_chamfer", "initial_overlap", "selected_candidate", "selected_chamfer", "selected_overlap", "selected_offset_east_m", "selected_offset_north_m", "selected_yaw_offset_deg", "chamfer_improvement", "overlap_improvement", "safe_output_worse_than_initial", "num_evaluations", "stop_reason"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(image_rows)
    with (output_dir / "batch_candidate_results.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["image", "candidate", "offset_east_m", "offset_north_m", "offset_alt_m", "yaw_offset_deg", "edge_chamfer", "edge_overlap_ratio", "objective", "passes_strict_visual_gate", "selected_by_visual_objective", "safe_output_worse_than_initial", "valid_depth_ratio", "query_edge_count", "render_edge_count"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidate_rows)


def _write_doc(output_dir: Path, summary: Dict[str, Any]) -> None:
    stats = summary["batch_statistics"]
    lines = [
        "# P13 Visual Objective Pose Optimizer",
        "",
        "## Purpose",
        "P13 replaces P12 fixed local grid search with a derivative-free optimizer driven by visual edge alignment.",
        "",
        "## Method",
        "- Not a grid search: pattern search / coordinate descent evaluates the center and coordinate neighbors.",
        "- Optimizes only east, north, and yaw offsets.",
        "- Altitude is fixed to the initial pose; pitch/roll remain [0, 180].",
        "- Objective: edge_chamfer - overlap_weight * edge_overlap_ratio.",
        "- Feature loss is not used as the optimization objective.",
        "- Final selection uses strict visual gate, with fallback to initial.",
        "",
        "## Batch Results",
        "| Image | Initial chamfer | Selected chamfer | Initial overlap | Selected overlap | Candidate | East | North | Yaw | Evals | Safe worse? |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["per_image"]:
        lines.append(
            f"| {row['image']} | {row['initial_chamfer']:.4f} | {row['selected_chamfer']:.4f} | {row['initial_overlap']:.4f} | {row['selected_overlap']:.4f} | {row['selected_candidate']} | {row['selected_offset_east_m']:.3f} | {row['selected_offset_north_m']:.3f} | {row['selected_yaw_offset_deg']:.3f} | {row['num_evaluations']} | {row['safe_output_worse_than_initial']} |"
        )
    lines += [
        "",
        "## P12 Comparison",
        f"- regression_rate: {stats['regression_rate']}",
        f"- non_initial_accept_rate: {stats['non_initial_accept_rate']}",
        f"- mean_chamfer_improvement: {stats['mean_chamfer_improvement']}",
        f"- mean_overlap_improvement: {stats['mean_overlap_improvement']}",
        f"- mean_num_evaluations: {stats['mean_num_evaluations']}",
        "- P12 fixed search used about 83 candidates/image; P13 target is fewer evaluations.",
        "",
        "## Interpretation",
        f"- does_visual_optimizer_improve_initial: {summary['interpretation']['does_visual_optimizer_improve_initial']}",
        f"- does_optimizer_reduce_evaluations_vs_p12: {summary['interpretation']['does_optimizer_reduce_evaluations_vs_p12']}",
        f"- recommended_next_step: {summary['interpretation']['recommended_next_step']}",
        "",
        "## Conclusion",
        summary["interpretation"]["conclusion"],
        "",
    ]
    (output_dir / "visual_objective_optimizer_p13_check.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--image-limit", type=int, default=5)
    parser.add_argument("--optimize-vars", nargs="+", default=["east", "north", "yaw"])
    parser.add_argument("--initial-step-xy", type=float, default=2.0)
    parser.add_argument("--initial-step-yaw", type=float, default=0.5)
    parser.add_argument("--min-step-xy", type=float, default=0.25)
    parser.add_argument("--min-step-yaw", type=float, default=0.1)
    parser.add_argument("--step-decay", type=float, default=0.5)
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--max-evals", type=int, default=40)
    parser.add_argument("--overlap-weight", type=float, default=1.0)
    parser.add_argument("--visual-gate", action="store_true")
    parser.add_argument("--save-topk-images", type=int, default=5)
    parser.add_argument("--checker-tile", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["default_confs"]["cam_query"]["max_size"] = args.width
    query_resize_ratio, raw_query_camera, render_camera_gs, _query_camera, _render_camera = _setup_camera(config)
    renderer = DOMDSMRenderer(config["render_config"])
    to_raster, from_raster, raster_crs = _get_raster_transformers(config)
    pose_map = load_query_poses_from_file(str(REPO_ROOT / args.pose_file))
    query_dir = (REPO_ROOT / args.query_dir).resolve()
    images = sorted([p for p in query_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}])[: args.image_limit]
    image_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    failed = []
    for path in images:
        key = path.name if path.name in pose_map else path.name.lower()
        if key not in pose_map:
            failed.append({"image": path.name, "error": "missing_pose"})
            continue
        try:
            summary, rows = _optimize_one(
                path,
                pose_map[key],
                renderer,
                query_resize_ratio,
                raw_query_camera,
                render_camera_gs,
                config,
                to_raster,
                from_raster,
                args,
            )
            image_rows.append(
                {
                    "image": path.name,
                    "initial_chamfer": summary["initial"]["edge_chamfer"],
                    "initial_overlap": summary["initial"]["edge_overlap_ratio"],
                    "selected_candidate": summary["selected"]["candidate"],
                    "selected_chamfer": summary["selected"]["edge_chamfer"],
                    "selected_overlap": summary["selected"]["edge_overlap_ratio"],
                    "selected_offset_east_m": summary["selected_offset_east_m"],
                    "selected_offset_north_m": summary["selected_offset_north_m"],
                    "selected_yaw_offset_deg": summary["selected_yaw_offset_deg"],
                    "chamfer_improvement": summary["chamfer_improvement"],
                    "overlap_improvement": summary["overlap_improvement"],
                    "safe_output_worse_than_initial": summary["safe_output_worse_than_initial"],
                    "num_evaluations": summary["num_evaluations"],
                    "stop_reason": summary["stop_reason"],
                }
            )
            candidate_rows.extend(rows)
        except Exception as exc:
            failed.append({"image": path.name, "error": repr(exc), "traceback": traceback.format_exc()})
    n = len(image_rows)
    chamfer_improvements = [float(r["chamfer_improvement"]) for r in image_rows]
    overlap_improvements = [float(r["overlap_improvement"]) for r in image_rows]
    evals = [int(r["num_evaluations"]) for r in image_rows]
    regression_rate = float(np.mean([bool(r["safe_output_worse_than_initial"]) for r in image_rows])) if image_rows else None
    non_initial_accept_rate = float(np.mean([r["selected_candidate"] != "initial" for r in image_rows])) if image_rows else None
    mean_chamfer = float(np.mean(chamfer_improvements)) if chamfer_improvements else None
    mean_overlap = float(np.mean(overlap_improvements)) if overlap_improvements else None
    mean_evals = float(np.mean(evals)) if evals else None
    median_evals = float(np.median(evals)) if evals else None
    p13_success = bool(
        regression_rate == 0
        and non_initial_accept_rate is not None
        and non_initial_accept_rate >= 0.6
        and mean_chamfer is not None
        and mean_chamfer > 0
        and mean_overlap is not None
        and mean_overlap > 0
        and mean_evals is not None
        and mean_evals < 83
        and all(not r["safe_output_worse_than_initial"] for r in image_rows)
    )
    if p13_success:
        conclusion = "P13 safely improves the batch with fewer evaluations than P12."
        recommended = "P14 should add multi-scale coarse-to-fine visual optimization or a differentiable edge-distance objective."
    else:
        conclusion = "P13 did not satisfy all batch success criteria under the current pattern-search settings."
        recommended = "Inspect trace files and tune step schedule or objective weighting before pipeline integration."
    summary = {
        "experiment": "P13 visual objective pose optimizer",
        "config": args.config,
        "query_dir": args.query_dir,
        "pose_file": args.pose_file,
        "output_dir": args.output_dir,
        "num_images_processed": n,
        "num_images_failed": len(failed),
        "failed_images": failed,
        "batch_statistics": {
            "regression_rate": regression_rate,
            "non_initial_accept_rate": non_initial_accept_rate,
            "mean_chamfer_improvement": mean_chamfer,
            "median_chamfer_improvement": float(np.median(chamfer_improvements)) if chamfer_improvements else None,
            "mean_overlap_improvement": mean_overlap,
            "median_overlap_improvement": float(np.median(overlap_improvements)) if overlap_improvements else None,
            "mean_num_evaluations": mean_evals,
            "median_num_evaluations": median_evals,
        },
        "per_image": image_rows,
        "interpretation": {
            "does_visual_optimizer_improve_initial": bool(mean_chamfer is not None and mean_chamfer > 0 and mean_overlap is not None and mean_overlap > 0),
            "does_optimizer_reduce_evaluations_vs_p12": bool(mean_evals is not None and mean_evals < 83),
            "recommended_next_step": recommended,
            "conclusion": conclusion,
        },
        "p13_success": p13_success,
    }
    _write_csvs(output_dir, image_rows, candidate_rows)
    _write_doc(output_dir, summary)
    _write_json(output_dir / "batch_summary_metrics.json", summary)
    print(json.dumps({"p13_success": p13_success, "num_images_processed": n, "mean_num_evaluations": mean_evals, "regression_rate": regression_rate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
