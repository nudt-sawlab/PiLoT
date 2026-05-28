#!/usr/bin/env python3
"""Run P12 batch local visual-safe candidate search for DOM/DSM queries."""

import argparse
import copy
import csv
import json
import os
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CUDA_EXT_DIR = REPO_ROOT / "DirectAbsoluteCostCuda"
if CUDA_EXT_DIR.exists() and str(CUDA_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_EXT_DIR))

from pixloc.localization.localizer import RenderLocalizer
from pixloc.pixlib.datasets.view import read_image
from pixloc.utils.dom_dsm.candidate_scorer import (
    build_p12_candidate_specs_for_image,
    deduplicate_candidates,
    load_query_poses_from_file,
)
from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from pixloc.utils.dom_dsm.feature_loss_debug import project_points_to_image, sample_feature_map
from pixloc.utils.dom_dsm.pose_adapter import compute_enu_delta_m, make_downward_euler_from_yaw
from pixloc.utils.get_depth import pad_to_multiple, zero_pad
from tools.compare_cuda_torch_feature_loss_fixed_poses import _pose_from_osg_fixed_context, _spearman_from_ranked
from tools.diagnose_yawfix_refinement_update import (
    _checkerboard,
    _edge_overlay,
    _get_raster_transformers,
    _make_overlay,
    _read_query_rgb,
    _safe_jsonable,
    _write_rgb,
)
from tools.run_dom_dsm_single_full import (
    _back_project,
    _depth_stats,
    _format_pose_line,
    _resize_query_for_refine,
    _setup_camera,
)
from tools.run_local_visual_safe_candidate_scorer_p11 import _build_alt_refine_specs, _build_yaw_refine_specs, _slug
from tools.run_structure_weighted_safe_refinement_p10 import _build_structure_weights, _save_weight_png, _weight_stats


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_DIR = "data_caiwangcun/query/images/exif_test"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/batch_local_visual_safe_candidate_search_p12"
WEIGHT_KEY_BY_MODE = {
    "uniform": "uniform_loss",
    "dom_edge": "dom_edge_weighted_loss",
    "depth_gradient": "dsm_gradient_weighted_loss",
    "combined": "combined_structure_weighted_loss",
    "low_texture_downweight": "low_texture_vegetation_water_downweighted_loss",
}
EPS = 1e-9


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rank(rows: Sequence[Dict[str, Any]], key: str, reverse: bool = False) -> List[str]:
    valid = [row for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    return [str(row["candidate"]) for row in sorted(valid, key=lambda item: float(item[key]), reverse=reverse)]


def _topk_contains(rank: Sequence[str], target: Optional[str], k: int) -> bool:
    return bool(target and target in list(rank)[:k])


def _safe_pass(row: Dict[str, Any], initial: Dict[str, Any]) -> bool:
    return (
        float(row["edge_chamfer"]) <= float(initial["edge_chamfer"]) + EPS
        and float(row["edge_overlap_ratio"]) + EPS >= float(initial["edge_overlap_ratio"])
    )


def _chamfer_pass(row: Dict[str, Any], initial: Dict[str, Any]) -> bool:
    return float(row["edge_chamfer"]) <= float(initial["edge_chamfer"]) + EPS


def _overlap_pass(row: Dict[str, Any], initial: Dict[str, Any]) -> bool:
    return float(row["edge_overlap_ratio"]) + EPS >= float(initial["edge_overlap_ratio"])


def _worse(row: Dict[str, Any], initial: Dict[str, Any]) -> bool:
    return (
        float(row["edge_chamfer"]) > float(initial["edge_chamfer"]) + EPS
        or float(row["edge_overlap_ratio"]) + EPS < float(initial["edge_overlap_ratio"])
    )


def _pose_from_spec(spec: Dict[str, Any], origin: torch.Tensor, dd: torch.Tensor, mul: Optional[float], device: str) -> Any:
    pose = _pose_from_osg_fixed_context(spec["translation_lon_lat_alt"], spec["euler_pitch_roll_yaw"], origin, dd, mul, device)
    return pose[0] if len(pose.shape) == 1 and pose.shape[0] == 1 else pose


def _strip_confidence(features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [feat[:-1] if feat.shape[0] > 1 else feat for feat in features]


def _feature_losses_all_modes(
    features_q_raw: Sequence[torch.Tensor],
    features_ref_raw: Sequence[torch.Tensor],
    scales_q: Sequence[Any],
    scales_ref: Sequence[Any],
    p3d: torch.Tensor,
    pose: Any,
    t_render: Any,
    query_camera: Any,
    render_camera: Any,
    weight_maps: Dict[str, np.ndarray],
) -> Tuple[float, Dict[str, float]]:
    fq = _strip_confidence(features_q_raw)
    fr = _strip_confidence(features_ref_raw)
    device = p3d.device
    proj_q = project_points_to_image(p3d, pose, query_camera)
    proj_r = project_points_to_image(p3d, t_render, render_camera)
    totals = {mode: [] for mode in weight_maps}
    unweighted = []
    weight_tensors = {mode: torch.as_tensor(weight, dtype=torch.float32, device=device).unsqueeze(0) for mode, weight in weight_maps.items()}
    for level in range(min(len(fq), len(fr))):
        feat_q = F.normalize(fq[level].to(device), dim=0)
        feat_r = F.normalize(fr[level].to(device), dim=0)
        q_sample, q_valid = sample_feature_map(feat_q, proj_q["points2d"], scales_q[level])
        r_sample, r_valid = sample_feature_map(feat_r, proj_r["points2d"], scales_ref[level])
        valid_base = proj_q["valid"] & proj_r["valid"] & q_valid & r_valid
        if not valid_base.any():
            continue
        residual = ((q_sample[valid_base] - r_sample[valid_base]) ** 2).sum(dim=-1)
        unweighted.append(residual.mean())
        for mode, weight_t in weight_tensors.items():
            w_sample, w_valid = sample_feature_map(weight_t, proj_r["points2d"], 1.0)
            w_all = w_sample[:, 0].clamp_min(0.0)
            valid = valid_base & w_valid & torch.isfinite(w_all) & (w_all > 0)
            if valid.any():
                residual_mode = ((q_sample[valid] - r_sample[valid]) ** 2).sum(dim=-1)
                weights = w_all[valid]
                totals[mode].append((residual_mode * weights).sum() / weights.sum().clamp_min(1e-12))
    def _mean(vals: List[torch.Tensor]) -> float:
        return float(torch.stack(vals).mean().detach().cpu().item()) if vals else float("inf")
    return _mean(unweighted), {mode: _mean(vals) for mode, vals in totals.items()}


def _render_candidate_p12(
    spec: Dict[str, Any],
    renderer: DOMDSMRenderer,
    query_rgb: np.ndarray,
    output_root: Path,
    checker_tile: int,
    save_images: bool,
) -> Dict[str, Any]:
    out_dir = output_root / "candidates" / spec["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    render_rgb, depth = renderer.render(spec["translation_lon_lat_alt"], spec["euler_pitch_roll_yaw"])
    render_time = time.perf_counter() - t0
    overlay = _make_overlay(query_rgb, render_rgb)
    edge_overlay, edge_metrics = _edge_overlay(query_rgb, render_rgb)
    if save_images:
        _write_rgb(out_dir / "rendered_rgb.png", render_rgb)
        _write_rgb(out_dir / "overlay.png", overlay)
        _write_rgb(out_dir / "edge_overlay.png", edge_overlay)
        _write_rgb(out_dir / "checkerboard.png", _checkerboard(query_rgb, render_rgb, checker_tile))
    metrics = {
        "image": spec["image"],
        "name": spec["name"],
        "candidate": spec["name"],
        "source": spec["source"],
        "stage": spec["stage"],
        "translation_lon_lat_alt": [float(x) for x in spec["translation_lon_lat_alt"]],
        "euler_pitch_roll_yaw": [float(x) for x in spec["euler_pitch_roll_yaw"]],
        "offset_east_m": float(spec["offset_east_m"]),
        "offset_north_m": float(spec["offset_north_m"]),
        "offset_alt_m": float(spec["offset_alt_m"]),
        "yaw_offset_deg": float(spec["yaw_offset_deg"]),
        "render_time_sec": render_time,
        **_depth_stats(depth),
        **edge_metrics,
    }
    _write_json(out_dir / "metrics.json", metrics)
    return metrics


def _select_visual(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    passing = [r for r in rows if r.get("passes_strict_visual_gate") and r["candidate"] != "initial"]
    return sorted(passing, key=lambda r: (float(r["edge_chamfer"]), -float(r["edge_overlap_ratio"])))[0] if passing else None


def _select_by_key(rows: Sequence[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    passing = [r for r in rows if r.get("passes_strict_visual_gate") and r["candidate"] != "initial" and r.get(key) is not None]
    valid = [r for r in passing if np.isfinite(float(r[key]))]
    return sorted(valid, key=lambda r: float(r[key]))[0] if valid else None


def _write_result_pose(path: Path, image_name: str, selected: Dict[str, Any], initial: Dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# P12 selected safe pose",
                f"image: {image_name}",
                f"method: {selected['candidate']}",
                "policy: strict_visual_gate_then_best_chamfer",
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


def _evaluate_specs(
    specs: Sequence[Dict[str, Any]],
    renderer: DOMDSMRenderer,
    query_rgb: np.ndarray,
    output_root: Path,
    checker_tile: int,
    initial_row: Optional[Dict[str, Any]],
    weight_maps: Dict[str, np.ndarray],
    features_q_raw: Sequence[torch.Tensor],
    features_ref_raw: Sequence[torch.Tensor],
    scales_q: Sequence[Any],
    scales_ref: Sequence[Any],
    p3d: torch.Tensor,
    t_render: Any,
    query_camera: Any,
    render_camera: Any,
    origin: torch.Tensor,
    dd: torch.Tensor,
    mul: Optional[float],
    device: str,
    save_images: bool,
) -> List[Dict[str, Any]]:
    rows = []
    for spec in specs:
        row = _render_candidate_p12(spec, renderer, query_rgb, output_root, checker_tile, save_images)
        pose = _pose_from_spec(spec, origin, dd, mul, device)
        unweighted, weighted = _feature_losses_all_modes(
            features_q_raw,
            features_ref_raw,
            scales_q,
            scales_ref,
            p3d,
            pose,
            t_render,
            query_camera,
            render_camera,
            weight_maps,
        )
        row["unweighted_feature_loss"] = unweighted
        row["weighted_feature_loss_by_mode"] = weighted
        row["passes_strict_visual_gate"] = True if initial_row is None and row["candidate"] == "initial" else _safe_pass(row, initial_row or row)
        row["passes_chamfer_only_gate"] = True if initial_row is None and row["candidate"] == "initial" else _chamfer_pass(row, initial_row or row)
        row["passes_overlap_only_gate"] = True if initial_row is None and row["candidate"] == "initial" else _overlap_pass(row, initial_row or row)
        row["worse_than_initial"] = False if initial_row is None and row["candidate"] == "initial" else _worse(row, initial_row or row)
        for mode, value in weighted.items():
            row[f"weighted_feature_loss_{mode}"] = value
        _write_json(output_root / "candidates" / spec["name"] / "metrics.json", row)
        rows.append(row)
    return rows


def _process_image(
    image_path: Path,
    pose_entry: Dict[str, Any],
    args: argparse.Namespace,
    config_template: Dict[str, Any],
    renderer: DOMDSMRenderer,
    localizer: RenderLocalizer,
    to_raster: Any,
    from_raster: Any,
    raster_crs: str,
    raw_query_camera: Any,
    query_resize_ratio: Any,
    render_camera_gs: Any,
    query_camera_template: Any,
    render_camera_template: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    image_name = image_path.name
    image_stem = image_path.stem
    out_dir = (REPO_ROOT / args.output_dir / image_stem).resolve()
    if out_dir.exists() and not args.skip_existing:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trans = [float(x) for x in pose_entry["translation_lon_lat_alt"]]
    base_yaw = float(pose_entry["base_yaw"])
    euler = make_downward_euler_from_yaw(base_yaw)
    width = int(render_camera_gs[0])
    height = int(render_camera_gs[1])
    query_rgb = _read_query_rgb(image_path, width, height)
    query_image = read_image(str(image_path), scale=query_resize_ratio, distortion=config_template["default_confs"]["cam_query"]["distortion"], query_camera=raw_query_camera)
    query_image_for_refine = _resize_query_for_refine(query_image, render_camera_gs)
    color, depth = renderer.render(trans, euler)
    color_for_refine = pad_to_multiple(color, 16) if config_template["default_confs"].get("padding", False) else color
    device = "cuda" if torch.cuda.is_available() else "cpu"
    origin_np = np.array(__import__("pixloc.utils.transform", fromlist=["WGS84_to_ECEF"]).WGS84_to_ECEF(trans), dtype=np.float64)
    origin = torch.tensor(origin_np, device=device, dtype=torch.float32)
    query_camera = query_camera_template.to(device)
    render_camera = render_camera_template.to(device)
    p3d, t_render, _t_init, dd = _back_project(
        depth, euler, trans, euler, trans, render_camera_gs, render_camera, origin, config_template["default_confs"]["refine"]["mul"], device, is_init=True
    )
    q_w, _ = query_camera.size
    with torch.no_grad():
        features_ref_raw, scales_ref = localizer.refiner.dense_feature_extraction(zero_pad(int(q_w.item()), color_for_refine))
        features_q_raw, scales_q = localizer.refiner.dense_feature_extraction(zero_pad(int(q_w.item()), query_image_for_refine))

    p10_weights = _build_structure_weights(color, depth)
    weight_maps = {mode: p10_weights[WEIGHT_KEY_BY_MODE[mode]]["weight"] for mode in args.weight_modes}
    weights_dir = out_dir / "weights"
    mask_quality = {}
    for mode, weight in weight_maps.items():
        mode_dir = weights_dir / mode
        _save_weight_png(mode_dir / "weight_map.png", weight)
        mask_quality[mode] = _weight_stats(weight, {"source_p10_key": WEIGHT_KEY_BY_MODE[mode]})
        _write_json(mode_dir / "weight_stats.json", mask_quality[mode])

    stage1_specs, debug = build_p12_candidate_specs_for_image(
        image_name,
        trans,
        base_yaw,
        to_raster,
        from_raster,
        coarse_east_offsets=args.coarse_east,
        coarse_north_offsets=args.coarse_north,
        yaw_refine_offsets=args.yaw_refine,
        topk_visual_for_yaw=args.topk_visual_for_yaw,
        include_known_seeds_for_debug=args.include_known_seeds_for_debug,
    )
    stage1_specs, duplicates_stage1 = deduplicate_candidates(stage1_specs)
    initial_spec = next(s for s in stage1_specs if s["name"] == "initial")
    initial_row = _evaluate_specs(
        [initial_spec], renderer, query_rgb, out_dir, args.checker_tile, None, weight_maps, features_q_raw, features_ref_raw, scales_q, scales_ref,
        p3d, t_render, query_camera, render_camera, origin, dd, config_template["default_confs"]["refine"]["mul"], device, True
    )[0]
    other_specs = [s for s in stage1_specs if s["name"] != "initial"]
    rows = [initial_row] + _evaluate_specs(
        other_specs, renderer, query_rgb, out_dir, args.checker_tile, initial_row, weight_maps, features_q_raw, features_ref_raw, scales_q, scales_ref,
        p3d, t_render, query_camera, render_camera, origin, dd, config_template["default_confs"]["refine"]["mul"], device, args.save_all_candidate_images
    )
    stage1_rows = list(rows)
    top_stage1 = sorted(stage1_rows, key=lambda r: (float(r["edge_chamfer"]), -float(r["edge_overlap_ratio"])))[: args.topk_visual_for_yaw]
    yaw_specs = _build_yaw_refine_specs(top_stage1, args.yaw_refine, base_yaw) if args.topk_visual_for_yaw > 0 else []
    for spec in yaw_specs:
        spec["image"] = image_name
    yaw_specs, duplicates_yaw = deduplicate_candidates(yaw_specs)
    rows.extend(
        _evaluate_specs(
            yaw_specs, renderer, query_rgb, out_dir, args.checker_tile, initial_row, weight_maps, features_q_raw, features_ref_raw, scales_q, scales_ref,
            p3d, t_render, query_camera, render_camera, origin, dd, config_template["default_confs"]["refine"]["mul"], device, args.save_all_candidate_images
        )
    )
    duplicates_alt: List[Dict[str, Any]] = []
    if args.enable_alt_refine:
        top_stage2 = sorted(rows, key=lambda r: (float(r["edge_chamfer"]), -float(r["edge_overlap_ratio"])))[:3]
        alt_specs = _build_alt_refine_specs(top_stage2, args.alt_refine_offsets, trans, to_raster, from_raster)
        for spec in alt_specs:
            spec["image"] = image_name
        alt_specs, duplicates_alt = deduplicate_candidates(alt_specs)
        rows.extend(
            _evaluate_specs(
                alt_specs, renderer, query_rgb, out_dir, args.checker_tile, initial_row, weight_maps, features_q_raw, features_ref_raw, scales_q, scales_ref,
                p3d, t_render, query_camera, render_camera, origin, dd, config_template["default_confs"]["refine"]["mul"], device, args.save_all_candidate_images
            )
        )

    rank_chamfer = _rank(rows, "edge_chamfer")
    rank_overlap = _rank(rows, "edge_overlap_ratio", reverse=True)
    rank_unweighted = _rank(rows, "unweighted_feature_loss")
    rank_weighted = {mode: _rank(rows, f"weighted_feature_loss_{mode}") for mode in args.weight_modes}
    for row in rows:
        row["selected_by_visual"] = False
        row["selected_by_unweighted_feature"] = False
        for mode in args.weight_modes:
            row[f"selected_by_weighted_{mode}"] = False
    selected = _select_visual(rows) or initial_row
    selected["selected_by_visual"] = True
    selected_unweighted = _select_by_key(rows, "unweighted_feature_loss")
    if selected_unweighted:
        selected_unweighted["selected_by_unweighted_feature"] = True
    selected_weighted = {}
    for mode in args.weight_modes:
        pick = _select_by_key(rows, f"weighted_feature_loss_{mode}")
        selected_weighted[mode] = pick
        if pick:
            pick[f"selected_by_weighted_{mode}"] = True

    save_names = set([initial_row["candidate"], selected["candidate"]])
    save_names.update(rank_chamfer[: args.save_topk_images])
    save_names.update(rank_unweighted[: args.save_topk_images])
    for name in save_names:
        spec_row = next((r for r in rows if r["candidate"] == name), None)
        if spec_row and not (out_dir / "candidates" / name / "rendered_rgb.png").exists():
            _render_candidate_p12(
                {
                    "image": image_name,
                    "name": name,
                    "source": spec_row["source"],
                    "stage": spec_row["stage"],
                    "translation_lon_lat_alt": spec_row["translation_lon_lat_alt"],
                    "euler_pitch_roll_yaw": spec_row["euler_pitch_roll_yaw"],
                    "offset_east_m": spec_row["offset_east_m"],
                    "offset_north_m": spec_row["offset_north_m"],
                    "offset_alt_m": spec_row["offset_alt_m"],
                    "yaw_offset_deg": spec_row["yaw_offset_deg"],
                },
                renderer,
                query_rgb,
                out_dir,
                args.checker_tile,
                True,
            )

    strict_pass = [r for r in rows if r["passes_strict_visual_gate"]]
    strict_non_initial = [r for r in strict_pass if r["candidate"] != "initial"]
    feature_top1 = rank_unweighted[0] if rank_unweighted else None
    visual_top1 = rank_chamfer[0] if rank_chamfer else None
    sp_weighted_ch = {mode: _spearman_from_ranked(rank_weighted[mode], rank_chamfer) for mode in args.weight_modes}
    sp_weighted_ov = {mode: _spearman_from_ranked(rank_weighted[mode], rank_overlap) for mode in args.weight_modes}
    best_mode = max([m for m in args.weight_modes if sp_weighted_ch[m] is not None], key=lambda m: sp_weighted_ch[m], default=None)
    summary = {
        "image": image_name,
        "initial": initial_row,
        "selected": selected,
        "selected_is_initial": selected["candidate"] == "initial",
        "selected_offset_east_m": selected["offset_east_m"],
        "selected_offset_north_m": selected["offset_north_m"],
        "selected_yaw_offset_deg": selected["yaw_offset_deg"],
        "selected_alt_offset_m": selected["offset_alt_m"],
        "chamfer_improvement": float(initial_row["edge_chamfer"]) - float(selected["edge_chamfer"]),
        "overlap_improvement": float(selected["edge_overlap_ratio"]) - float(initial_row["edge_overlap_ratio"]),
        "strict_gate_accepts_non_initial": len(strict_non_initial) > 0,
        "safe_output_worse_than_initial": _worse(selected, initial_row),
        "num_candidates_total": len(rows),
        "num_strict_pass": len(strict_pass),
        "candidates": rows,
        "rankings": {
            "rank_by_visual_chamfer": rank_chamfer,
            "rank_by_visual_overlap": rank_overlap,
            "rank_by_unweighted_feature_loss": rank_unweighted,
            "rank_by_weighted_feature_loss_by_mode": rank_weighted,
        },
        "correlation": {
            "spearman_unweighted_loss_vs_chamfer": _spearman_from_ranked(rank_unweighted, rank_chamfer),
            "spearman_unweighted_loss_vs_overlap": _spearman_from_ranked(rank_unweighted, rank_overlap),
            "spearman_weighted_loss_vs_chamfer_by_mode": sp_weighted_ch,
            "spearman_weighted_loss_vs_overlap_by_mode": sp_weighted_ov,
        },
        "feature_diagnostics": {
            "feature_top1_candidate": feature_top1,
            "feature_top1_passes_strict_gate": bool(next((r for r in rows if r["candidate"] == feature_top1), {}).get("passes_strict_visual_gate", False)),
            "feature_top1_matches_visual_top1": feature_top1 == visual_top1,
            "feature_top5_contains_visual_top1": _topk_contains(rank_unweighted, visual_top1, 5),
            "feature_top10_contains_visual_top1": _topk_contains(rank_unweighted, visual_top1, 10),
            "weighted_feature_top5_contains_visual_top1_by_mode": {
                mode: _topk_contains(rank_weighted[mode], visual_top1, 5) for mode in args.weight_modes
            },
            "selected_by_weighted_feature_by_mode": {m: selected_weighted[m]["candidate"] if selected_weighted[m] else None for m in args.weight_modes},
            "selected_by_weighted_feature_passes_gate_by_mode": {m: bool(selected_weighted[m] and selected_weighted[m]["passes_strict_visual_gate"]) for m in args.weight_modes},
            "best_weight_mode_by_rank_alignment": best_mode,
        },
        "candidate_generation": {
            "duplicate_candidates_removed": duplicates_stage1 + duplicates_yaw + duplicates_alt,
            "mask_quality": mask_quality,
            "raster_crs": raster_crs,
            "debug": debug,
        },
    }
    _write_result_pose(out_dir / "result_pose_safe_p12.txt", image_name, selected, initial_row)
    _write_json(out_dir / "summary_metrics.json", summary)
    return summary, rows


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _median(values: Sequence[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def _std(values: Sequence[float]) -> Optional[float]:
    return float(np.std(values)) if values else None


def _rate(values: Sequence[bool]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _write_csvs(output_dir: Path, candidate_rows: List[Dict[str, Any]], image_rows: List[Dict[str, Any]], modes: Sequence[str]) -> None:
    cand_fields = [
        "image", "candidate", "stage", "source", "offset_east_m", "offset_north_m", "offset_alt_m", "yaw_offset_deg",
        "edge_chamfer", "edge_overlap_ratio", "unweighted_feature_loss",
    ] + [f"weighted_feature_loss_{m}" for m in modes] + [
        "passes_strict_visual_gate", "selected_by_visual", "selected_by_unweighted_feature",
    ] + [f"selected_by_weighted_{m}" for m in modes] + ["valid_depth_ratio", "query_edge_count", "render_edge_count"]
    with (output_dir / "batch_candidate_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cand_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidate_rows)
    image_fields = [
        "image", "initial_chamfer", "initial_overlap", "selected_candidate", "selected_chamfer", "selected_overlap",
        "selected_offset_east_m", "selected_offset_north_m", "selected_offset_alt_m", "selected_yaw_offset_deg",
        "chamfer_improvement", "overlap_improvement", "strict_gate_accepts_non_initial", "safe_output_worse_than_initial",
        "num_candidates", "num_strict_pass", "feature_top1_candidate", "feature_top1_passes_strict_gate",
        "feature_top5_contains_visual_top1", "best_weight_mode_by_rank_alignment",
    ]
    with (output_dir / "batch_image_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=image_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(image_rows)


def _write_plots(output_dir: Path, image_rows: List[Dict[str, Any]]) -> None:
    plot_dir = output_dir / "offset_distribution"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        east = [r["selected_offset_east_m"] for r in image_rows]
        north = [r["selected_offset_north_m"] for r in image_rows]
        labels = [Path(r["image"]).stem for r in image_rows]
        plt.figure(figsize=(6, 5))
        plt.scatter(east, north)
        for x, y, label in zip(east, north, labels):
            plt.text(x, y, label)
        plt.xlabel("east offset m")
        plt.ylabel("north offset m")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / "selected_offsets_scatter.png")
        plt.close()
        for key, name in [("selected_offset_east_m", "selected_east_hist.png"), ("selected_offset_north_m", "selected_north_hist.png"), ("chamfer_improvement", "chamfer_improvement_hist.png"), ("overlap_improvement", "overlap_improvement_hist.png")]:
            plt.figure(figsize=(6, 4))
            plt.hist([r[key] for r in image_rows], bins=min(10, max(1, len(image_rows))))
            plt.title(key)
            plt.tight_layout()
            plt.savefig(plot_dir / name)
            plt.close()
    except Exception as exc:
        _write_json(plot_dir / "plot_error.json", {"error": repr(exc)})


def _write_doc(output_dir: Path, summary: Dict[str, Any]) -> None:
    rows = summary["per_image"]
    stats = summary["batch_statistics"]
    off = summary["offset_distribution"]
    lines = [
        "# P12 Batch Local Visual-Safe Candidate Search",
        "",
        "## Purpose",
        "P11 showed local visual-safe search improves one image. P12 applies the same fixed search policy to multiple EXIF-recovered query poses and checks generalization plus systematic offsets.",
        "",
        "## Why Batch Is Necessary",
        "Single-image search does not prove generalization. P12 keeps candidate generation fixed and does not tune the range for any one query image.",
        "",
        "## Search Policy",
        f"- coarse east offsets: {summary['candidate_generation']['coarse_east_offsets']}",
        f"- coarse north offsets: {summary['candidate_generation']['coarse_north_offsets']}",
        f"- yaw refinement offsets: {summary['candidate_generation']['yaw_refine_offsets']}",
        f"- topK visual for yaw: {summary['candidate_generation']['topk_visual_for_yaw']}",
        f"- enable alt refine: {summary['candidate_generation']['enable_alt_refine']}",
        "- final pose policy: strict_visual_gate_then_best_chamfer",
        "",
        "## Gate Policy",
        "- strict visual gate: chamfer <= initial and overlap >= initial",
        "- final selection: lowest chamfer among strict-pass candidates; fallback initial if none pass",
        "",
        "## Batch Results",
        "| Image | Initial chamfer | Selected chamfer | Initial overlap | Selected overlap | Selected candidate | East | North | Yaw | Strict non-initial? | Safe worse? |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['image']} | {row['initial_chamfer']:.4f} | {row['selected_chamfer']:.4f} | {row['initial_overlap']:.4f} | {row['selected_overlap']:.4f} | {row['selected_candidate']} | {row['selected_offset_east_m']:.3f} | {row['selected_offset_north_m']:.3f} | {row['selected_yaw_offset_deg']:.3f} | {row['strict_gate_accepts_non_initial']} | {row['safe_output_worse_than_initial']} |"
        )
    lines += [
        "",
        "## Batch Statistics",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in ["num_images_processed", "non_initial_accept_rate", "regression_rate", "mean_chamfer_improvement", "median_chamfer_improvement", "mean_overlap_improvement", "feature_top1_safe_rate", "feature_top5_contains_visual_top1_rate"]:
        lines.append(f"| {key} | {stats.get(key)} |")
    lines += [
        "",
        "## Offset Distribution",
        "| Statistic | East | North | Yaw | Alt |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| mean | {off['mean_selected_east_offset']} | {off['mean_selected_north_offset']} | {off['mean_selected_yaw_offset']} | {off['mean_selected_alt_offset']} |",
        f"| median | {off['median_selected_east_offset']} | {off['median_selected_north_offset']} | {off['median_selected_yaw_offset']} | {off['median_selected_alt_offset']} |",
        f"| std | {off['std_selected_east_offset']} | {off['std_selected_north_offset']} | {off['std_selected_yaw_offset']} | {off['std_selected_alt_offset']} |",
        f"| most common | {off['most_common_selected_offset'].get('east')} | {off['most_common_selected_offset'].get('north')} | - | - |",
        "",
        "## Feature Scorer Analysis",
        "| Scorer | Top1 safe rate | Top5 contains visual top1 rate | Mean Spearman vs chamfer | Mean Spearman vs overlap |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| unweighted | {stats['feature_top1_safe_rate']} | {stats['feature_top5_contains_visual_top1_rate']} | {stats['mean_spearman_unweighted_vs_chamfer']} | {stats['mean_spearman_unweighted_vs_overlap']} |",
    ]
    for mode in summary["candidate_generation"]["weight_modes"]:
        lines.append(
            f"| {mode} | {stats['weighted_feature_top1_safe_rate_by_mode'].get(mode)} | {stats['weighted_feature_top5_contains_visual_top1_rate_by_mode'].get(mode)} | {stats['mean_spearman_weighted_vs_chamfer_by_mode'].get(mode)} | {stats['mean_spearman_weighted_vs_overlap_by_mode'].get(mode)} |"
        )
    lines += [
        "",
        "## Interpretation",
        f"- does batch local search generalize: {summary['interpretation']['does_batch_local_search_generalize']}",
        f"- regression rate is zero: {stats['regression_rate'] == 0}",
        f"- systematic offset hypothesis: {summary['interpretation']['systematic_offset_hypothesis']}",
        f"- feature scorer generalizes: {summary['interpretation']['does_feature_scorer_generalize']}",
        f"- any structure weight helps: {summary['interpretation']['does_any_structure_weight_help_feature_ranking']}",
        f"- recommended next step: {summary['interpretation']['recommended_next_step']}",
        "",
        "## Conclusion",
        summary["interpretation"]["conclusion"],
        "",
    ]
    (output_dir / "batch_local_visual_safe_candidate_search_p12_check.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    parser.add_argument("--query-list")
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--image-limit", type=int, default=10)
    parser.add_argument("--coarse-east", type=float, nargs="+", default=[-5, -3, -1, 0, 1, 3, 5])
    parser.add_argument("--coarse-north", type=float, nargs="+", default=[-5, -3, -1, 0, 1, 3, 5])
    parser.add_argument("--yaw-refine", type=float, nargs="+", default=[-2, -1, -0.5, 0, 0.5, 1, 2])
    parser.add_argument("--topk-visual-for-yaw", type=int, default=5)
    parser.add_argument("--enable-alt-refine", action="store_true")
    parser.add_argument("--alt-refine-offsets", type=float, nargs="+", default=[-1, 0, 1])
    parser.add_argument("--weight-modes", nargs="+", default=["uniform", "dom_edge", "depth_gradient", "combined", "low_texture_downweight"])
    parser.add_argument("--visual-gate", action="store_true")
    parser.add_argument("--include-raw-refined", action="store_true")
    parser.add_argument("--include-known-seeds-for-debug", action="store_true")
    parser.add_argument("--save-all-candidate-images", action="store_true")
    parser.add_argument("--save-topk-images", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--checker-tile", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists() and not args.skip_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["default_confs"]["cam_query"]["max_size"] = args.width
    query_resize_ratio, raw_query_camera, render_camera_gs, query_camera, render_camera = _setup_camera(config)
    renderer = DOMDSMRenderer(config["render_config"])
    localizer = RenderLocalizer(copy.deepcopy(config["default_confs"]["from_render_test"]))
    to_raster, from_raster, raster_crs = _get_raster_transformers(config)
    pose_map = load_query_poses_from_file(str(REPO_ROOT / args.pose_file))
    query_dir = (REPO_ROOT / args.query_dir).resolve()
    images = sorted([p for p in query_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}])
    if args.query_list:
        wanted = {line.strip() for line in (REPO_ROOT / args.query_list).read_text(encoding="utf-8").splitlines() if line.strip()}
        images = [p for p in images if p.name in wanted or p.stem in wanted]
    if args.image_limit:
        images = images[: min(args.image_limit, 5)]
    per_image: List[Dict[str, Any]] = []
    all_candidates: List[Dict[str, Any]] = []
    failed = []
    skipped_missing_pose = []
    for path in images:
        key = path.name if path.name in pose_map else path.name.lower()
        if key not in pose_map:
            skipped_missing_pose.append(path.name)
            continue
        try:
            summary, rows = _process_image(
                path, pose_map[key], args, config, renderer, localizer, to_raster, from_raster, raster_crs,
                raw_query_camera, query_resize_ratio, render_camera_gs, query_camera, render_camera,
            )
            per_image.append(
                {
                    "image": path.name,
                    "initial_chamfer": summary["initial"]["edge_chamfer"],
                    "initial_overlap": summary["initial"]["edge_overlap_ratio"],
                    "selected_candidate": summary["selected"]["candidate"],
                    "selected_chamfer": summary["selected"]["edge_chamfer"],
                    "selected_overlap": summary["selected"]["edge_overlap_ratio"],
                    "selected_offset_east_m": summary["selected_offset_east_m"],
                    "selected_offset_north_m": summary["selected_offset_north_m"],
                    "selected_offset_alt_m": summary["selected_alt_offset_m"],
                    "selected_yaw_offset_deg": summary["selected_yaw_offset_deg"],
                    "chamfer_improvement": summary["chamfer_improvement"],
                    "overlap_improvement": summary["overlap_improvement"],
                    "strict_gate_accepts_non_initial": summary["strict_gate_accepts_non_initial"],
                    "safe_output_worse_than_initial": summary["safe_output_worse_than_initial"],
                    "num_candidates": summary["num_candidates_total"],
                    "num_strict_pass": summary["num_strict_pass"],
                    "feature_top1_candidate": summary["feature_diagnostics"]["feature_top1_candidate"],
                    "feature_top1_passes_strict_gate": summary["feature_diagnostics"]["feature_top1_passes_strict_gate"],
                    "feature_top1_matches_visual_top1": summary["feature_diagnostics"]["feature_top1_matches_visual_top1"],
                    "feature_top5_contains_visual_top1": summary["feature_diagnostics"]["feature_top5_contains_visual_top1"],
                    "feature_top10_contains_visual_top1": summary["feature_diagnostics"]["feature_top10_contains_visual_top1"],
                    "weighted_feature_top5_contains_visual_top1_by_mode": summary["feature_diagnostics"]["weighted_feature_top5_contains_visual_top1_by_mode"],
                    "best_weight_mode_by_rank_alignment": summary["feature_diagnostics"]["best_weight_mode_by_rank_alignment"],
                    "spearman_unweighted_vs_chamfer": summary["correlation"]["spearman_unweighted_loss_vs_chamfer"],
                    "spearman_unweighted_vs_overlap": summary["correlation"]["spearman_unweighted_loss_vs_overlap"],
                    "spearman_weighted_vs_chamfer_by_mode": summary["correlation"]["spearman_weighted_loss_vs_chamfer_by_mode"],
                    "spearman_weighted_vs_overlap_by_mode": summary["correlation"]["spearman_weighted_loss_vs_overlap_by_mode"],
                }
            )
            all_candidates.extend(rows)
        except Exception as exc:
            failed.append({"image": path.name, "error": repr(exc), "traceback": traceback.format_exc()})
            if args.fail_fast:
                raise
    n = len(per_image)
    bools = lambda key: [bool(r[key]) for r in per_image]
    chamfer_improvements = [float(r["chamfer_improvement"]) for r in per_image]
    overlap_improvements = [float(r["overlap_improvement"]) for r in per_image]
    east = [float(r["selected_offset_east_m"]) for r in per_image]
    north = [float(r["selected_offset_north_m"]) for r in per_image]
    yaw = [float(r["selected_yaw_offset_deg"]) for r in per_image]
    alt = [float(r["selected_offset_alt_m"]) for r in per_image]
    pair_counter = Counter((round(r["selected_offset_east_m"], 3), round(r["selected_offset_north_m"], 3)) for r in per_image)
    common_pair, common_count = pair_counter.most_common(1)[0] if pair_counter else ((None, None), 0)
    weighted_safe = {}
    weighted_top5 = {}
    mean_sp_w_ch = {}
    mean_sp_w_ov = {}
    for mode in args.weight_modes:
        selected_keys = [f"selected_by_weighted_{mode}"]
        weighted_safe[mode] = _rate([bool(r.get(selected_keys[0])) and bool(r.get("passes_strict_visual_gate")) for r in all_candidates if r.get(selected_keys[0])])
        weighted_top5[mode] = _rate([
            bool(r.get("weighted_feature_top5_contains_visual_top1_by_mode", {}).get(mode))
            for r in per_image
        ])
        mean_sp_w_ch[mode] = _mean([r["spearman_weighted_vs_chamfer_by_mode"][mode] for r in per_image if r["spearman_weighted_vs_chamfer_by_mode"].get(mode) is not None])
        mean_sp_w_ov[mode] = _mean([r["spearman_weighted_vs_overlap_by_mode"][mode] for r in per_image if r["spearman_weighted_vs_overlap_by_mode"].get(mode) is not None])
    non_initial_accept_rate = _rate([r["selected_candidate"] != "initial" for r in per_image])
    regression_rate = _rate(bools("safe_output_worse_than_initial"))
    systematic = bool(n >= 3 and common_count / max(n, 1) >= 0.5 and common_pair != (0.0, 0.0))
    p12_success = bool(
        n >= 5
        and regression_rate == 0
        and non_initial_accept_rate is not None
        and non_initial_accept_rate >= 0.3
        and (_mean(chamfer_improvements) or 0) > 0
        and any(r["selected_candidate"] != "initial" for r in per_image)
    )
    stats = {
        "num_images_processed": n,
        "non_initial_accept_rate": non_initial_accept_rate,
        "strict_gate_accept_rate": _rate(bools("strict_gate_accepts_non_initial")),
        "regression_rate": regression_rate,
        "mean_chamfer_improvement": _mean(chamfer_improvements),
        "median_chamfer_improvement": _median(chamfer_improvements),
        "mean_overlap_improvement": _mean(overlap_improvements),
        "median_overlap_improvement": _median(overlap_improvements),
        "num_chamfer_improved": sum(v > 0 for v in chamfer_improvements),
        "num_overlap_improved": sum(v > 0 for v in overlap_improvements),
        "num_both_improved": sum(c > 0 and o > 0 for c, o in zip(chamfer_improvements, overlap_improvements)),
        "feature_top1_safe_rate": _rate(bools("feature_top1_passes_strict_gate")),
        "feature_top1_matches_visual_top1_rate": _rate(bools("feature_top1_matches_visual_top1")),
        "feature_top5_contains_visual_top1_rate": _rate(bools("feature_top5_contains_visual_top1")),
        "feature_top10_contains_visual_top1_rate": _rate(bools("feature_top10_contains_visual_top1")),
        "weighted_feature_top1_safe_rate_by_mode": weighted_safe,
        "weighted_feature_top5_contains_visual_top1_rate_by_mode": weighted_top5,
        "mean_spearman_unweighted_vs_chamfer": _mean([r["spearman_unweighted_vs_chamfer"] for r in per_image if r["spearman_unweighted_vs_chamfer"] is not None]),
        "mean_spearman_unweighted_vs_overlap": _mean([r["spearman_unweighted_vs_overlap"] for r in per_image if r["spearman_unweighted_vs_overlap"] is not None]),
        "mean_spearman_weighted_vs_chamfer_by_mode": mean_sp_w_ch,
        "mean_spearman_weighted_vs_overlap_by_mode": mean_sp_w_ov,
    }
    offset_dist = {
        "selected_east_offsets": east,
        "selected_north_offsets": north,
        "selected_yaw_offsets": yaw,
        "selected_alt_offsets": alt,
        "mean_selected_east_offset": _mean(east),
        "median_selected_east_offset": _median(east),
        "std_selected_east_offset": _std(east),
        "mean_selected_north_offset": _mean(north),
        "median_selected_north_offset": _median(north),
        "std_selected_north_offset": _std(north),
        "mean_selected_yaw_offset": _mean(yaw),
        "median_selected_yaw_offset": _median(yaw),
        "std_selected_yaw_offset": _std(yaw),
        "mean_selected_alt_offset": _mean(alt),
        "median_selected_alt_offset": _median(alt),
        "std_selected_alt_offset": _std(alt),
        "most_common_selected_offset": {"east": common_pair[0], "north": common_pair[1], "count": common_count},
    }
    if p12_success and systematic:
        conclusion = "Case A: Batch search safely improves most images and selected offsets are concentrated; this supports a systematic pose/coordinate bias hypothesis."
        recommended = "P13 should estimate and validate dataset-level offset correction."
    elif p12_success:
        conclusion = "Case B: Batch search safely improves images with dispersed offsets; this supports per-image local visual-safe refinement."
        recommended = "P13 should implement coarse-to-fine candidate refinement with visual-safe selection."
    else:
        conclusion = "Case C: Batch evidence is insufficient for generalization under the current fixed policy."
        recommended = "Inspect metric stability, query/pose quality, and search range before pipeline integration."
    summary = {
        "experiment": "P12 batch local visual-safe candidate search",
        "config": args.config,
        "query_dir": args.query_dir,
        "pose_file": args.pose_file,
        "width": args.width,
        "num_images_requested": len(images),
        "num_images_processed": n,
        "num_images_failed": len(failed),
        "failed_images": failed,
        "skipped_missing_pose": skipped_missing_pose,
        "candidate_generation": {
            "coarse_east_offsets": args.coarse_east,
            "coarse_north_offsets": args.coarse_north,
            "yaw_refine_offsets": args.yaw_refine,
            "topk_visual_for_yaw": args.topk_visual_for_yaw,
            "enable_alt_refine": args.enable_alt_refine,
            "include_raw_refined": args.include_raw_refined,
            "include_known_seeds_for_debug": args.include_known_seeds_for_debug,
            "weight_modes": args.weight_modes,
        },
        "per_image": per_image,
        "batch_statistics": stats,
        "offset_distribution": offset_dist,
        "interpretation": {
            "does_batch_local_search_generalize": p12_success,
            "does_batch_show_systematic_offset": systematic,
            "systematic_offset_hypothesis": f"most common selected east/north offset {common_pair} count {common_count}/{n}",
            "does_feature_scorer_generalize": bool(stats["feature_top5_contains_visual_top1_rate"] and stats["feature_top5_contains_visual_top1_rate"] >= 0.5),
            "does_any_structure_weight_help_feature_ranking": any(
                mean_sp_w_ch.get(m) is not None and stats["mean_spearman_unweighted_vs_chamfer"] is not None and mean_sp_w_ch[m] > stats["mean_spearman_unweighted_vs_chamfer"]
                for m in args.weight_modes
                if m != "uniform"
            ),
            "recommended_next_step": recommended,
            "conclusion": conclusion,
        },
        "p12_success": p12_success,
    }
    _write_csvs(output_dir, all_candidates, per_image, args.weight_modes)
    _write_plots(output_dir, per_image)
    _write_doc(output_dir, summary)
    _write_json(output_dir / "batch_summary_metrics.json", summary)
    print(json.dumps({"p12_success": p12_success, "num_images_processed": n, "regression_rate": regression_rate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
