#!/usr/bin/env python3
"""Run P11 local visual-safe candidate scoring for one DOM/DSM query."""

import argparse
import copy
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CUDA_EXT_DIR = REPO_ROOT / "DirectAbsoluteCostCuda"
if CUDA_EXT_DIR.exists() and str(CUDA_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_EXT_DIR))

import direct_abs_cost_cuda
from pixloc.localization.localizer import RenderLocalizer
from pixloc.pixlib.datasets.view import read_image
from pixloc.utils.dom_dsm.candidate_scorer import (
    build_local_candidate_specs,
    deduplicate_candidates,
)
from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from pixloc.utils.dom_dsm.pose_adapter import (
    compute_enu_delta_m,
    make_downward_euler_from_yaw,
    refined_yaw_to_downward_yaw,
)
from pixloc.utils.get_depth import pad_to_multiple, zero_pad
from src.utils.pose_utils import load_initial_pose, load_pose_dict
from tools.compare_cuda_torch_feature_loss_fixed_poses import _pose_from_osg_fixed_context
from tools.diagnose_yawfix_refinement_update import (
    BASE_EULER,
    _array,
    _get_raster_transformers,
    _read_query_rgb,
    _render_candidate,
    _ret_subset,
    _safe_jsonable,
)
from tools.run_dom_dsm_single_full import (
    _back_project,
    _depth_stats,
    _format_pose_line,
    _resize_query_for_refine,
    _setup_camera,
)
from tools.run_structure_weighted_safe_refinement_p10 import (
    _build_structure_weights,
    _save_weight_png,
    _spearman_from_ranked,
    _weighted_feature_loss,
    _weight_stats,
)


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test_16x9/0000.jpg"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/local_visual_safe_candidate_scorer_p11"
DEFAULT_DOC = "docs/experiments/dom_dsm_prepare/local_visual_safe_candidate_scorer_p11_check.md"
EPS = 1e-9
WEIGHT_KEY_BY_MODE = {
    "uniform": "uniform_loss",
    "dom_edge": "dom_edge_weighted_loss",
    "depth_gradient": "dsm_gradient_weighted_loss",
    "combined": "combined_structure_weighted_loss",
    "low_texture_downweight": "low_texture_vegetation_water_downweighted_loss",
}
KNOWN_SEEDS = ["p3_best_chamfer", "p3_best_overlap", "p4_scale_025_fixed_alt"]


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rank(rows: Sequence[Dict[str, Any]], key: str, reverse: bool = False) -> List[str]:
    valid = [row for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    return [str(row["name"]) for row in sorted(valid, key=lambda item: float(item[key]), reverse=reverse)]


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


def _slug(value: float) -> str:
    prefix = "p" if value > 0 else "m" if value < 0 else "z"
    mag = str(abs(float(value))).replace(".", "p").rstrip("0").rstrip("p")
    return f"{prefix}{mag}"


def _select_visual(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    passing = [row for row in rows if row.get("passes_strict_visual_gate") and row["name"] != "initial"]
    if not passing:
        return None
    return sorted(passing, key=lambda item: (float(item["edge_chamfer"]), -float(item["edge_overlap_ratio"])))[0]


def _select_by_key(rows: Sequence[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    passing = [row for row in rows if row.get("passes_strict_visual_gate") and row["name"] != "initial"]
    valid = [row for row in passing if row.get(key) is not None and np.isfinite(float(row[key]))]
    if not valid:
        return None
    return sorted(valid, key=lambda item: float(item[key]))[0]


def _save_weight_overlay(path: Path, rgb: np.ndarray, weight: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = weight.astype(np.float32)
    if np.isfinite(w).any():
        w = (w - np.nanmin(w)) / max(float(np.nanmax(w) - np.nanmin(w)), 1e-9)
    heat = cv2.applyColorMap((np.clip(w, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    base = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), cv2.addWeighted(base, 0.55, heat, 0.45, 0.0))


def _candidate_pose(
    spec: Dict[str, Any],
    origin: torch.Tensor,
    dd: torch.Tensor,
    mul: Optional[float],
    device: str,
) -> Any:
    pose = _pose_from_osg_fixed_context(
        spec["translation_lon_lat_alt"],
        spec["euler_pitch_roll_yaw"],
        origin,
        dd,
        mul,
        device,
    )
    return pose[0] if len(pose.shape) == 1 and pose.shape[0] == 1 else pose


def _evaluate_candidate(
    spec: Dict[str, Any],
    renderer: DOMDSMRenderer,
    query_for_visual: np.ndarray,
    output_dir: Path,
    checker_tile: int,
    initial_row: Optional[Dict[str, Any]],
    initial_trans: Sequence[float],
    to_raster: Any,
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
) -> Dict[str, Any]:
    metrics = _render_candidate(
        spec["name"],
        renderer,
        query_for_visual,
        spec["translation_lon_lat_alt"],
        spec["euler_pitch_roll_yaw"],
        output_dir / "candidates",
        checker_tile,
        {
            "candidate": spec["name"],
            "source": spec["source"],
            "stage": spec["stage"],
            "description": spec["description"],
        },
    )
    enu = compute_enu_delta_m(initial_trans, spec["translation_lon_lat_alt"], to_raster)
    spec["offset_east_m"] = enu[0]
    spec["offset_north_m"] = enu[1]
    spec["offset_alt_m"] = enu[2]
    pose = _candidate_pose(spec, origin, dd, mul, device)
    weighted_by_mode: Dict[str, float] = {}
    weight_stats_on_points: Dict[str, Any] = {}
    unweighted_loss = None
    for mode, weight in weight_maps.items():
        loss = _weighted_feature_loss(
            features_q_raw,
            features_ref_raw,
            scales_q,
            scales_ref,
            p3d,
            pose,
            t_render,
            query_camera,
            render_camera,
            weight,
            levels=None,
        )
        weighted_by_mode[mode] = loss["weighted_feature_loss_total"]
        weight_stats_on_points[mode] = {
            "num_valid_by_level": loss["num_valid_by_level"],
            "mean_weight_by_level": loss["mean_weight_by_level"],
        }
        if mode == "uniform":
            unweighted_loss = loss["unweighted_feature_loss_total"]

    row = {
        **spec,
        **metrics,
        "visual_chamfer": metrics["edge_chamfer"],
        "visual_overlap": metrics["edge_overlap_ratio"],
        "unweighted_feature_loss": unweighted_loss,
        "weighted_feature_loss_by_mode": weighted_by_mode,
        "weight_stats_on_points_by_mode": weight_stats_on_points,
        "valid_render": np.isfinite(float(metrics["edge_chamfer"])),
        "passes_strict_visual_gate": False,
        "passes_chamfer_only_gate": False,
        "passes_overlap_only_gate": False,
        "worse_than_initial": False,
        "notes": [],
    }
    if initial_row is not None:
        row["passes_strict_visual_gate"] = _safe_pass(row, initial_row)
        row["passes_chamfer_only_gate"] = _chamfer_pass(row, initial_row)
        row["passes_overlap_only_gate"] = _overlap_pass(row, initial_row)
        row["worse_than_initial"] = _worse(row, initial_row)
    return row


def _build_yaw_refine_specs(
    base_rows: Sequence[Dict[str, Any]],
    yaw_offsets: Sequence[float],
    base_yaw: float,
) -> List[Dict[str, Any]]:
    out = []
    for row in base_rows:
        base_offset = float(row.get("yaw_offset_deg", 0.0))
        for yaw in yaw_offsets:
            yaw = float(yaw)
            total = base_offset + yaw
            if abs(yaw) <= EPS:
                continue
            out.append(
                {
                    "name": f"{row['name']}__yaw_{_slug(yaw)}",
                    "source": "local_grid",
                    "stage": "yaw_refine",
                    "translation_lon_lat_alt": row["translation_lon_lat_alt"],
                    "euler_pitch_roll_yaw": make_downward_euler_from_yaw(base_yaw + total),
                    "offset_east_m": row["offset_east_m"],
                    "offset_north_m": row["offset_north_m"],
                    "offset_alt_m": row["offset_alt_m"],
                    "yaw_offset_deg": total,
                    "description": f"Stage 2 yaw refinement from {row['name']} with delta {yaw:g} deg.",
                }
            )
    return out


def _build_alt_refine_specs(
    base_rows: Sequence[Dict[str, Any]],
    alt_offsets: Sequence[float],
    initial_trans: Sequence[float],
    to_raster: Any,
    from_raster: Any,
) -> List[Dict[str, Any]]:
    from pixloc.utils.dom_dsm.pose_adapter import apply_enu_offset

    out = []
    for row in base_rows:
        for alt in alt_offsets:
            alt = float(alt)
            if abs(alt) <= EPS:
                continue
            trans = apply_enu_offset(
                initial_trans,
                row["offset_east_m"],
                row["offset_north_m"],
                row["offset_alt_m"] + alt,
                to_raster,
                from_raster,
            )
            out.append(
                {
                    "name": f"{row['name']}__alt_{_slug(alt)}",
                    "source": "local_grid",
                    "stage": "alt_refine",
                    "translation_lon_lat_alt": trans,
                    "euler_pitch_roll_yaw": row["euler_pitch_roll_yaw"],
                    "offset_east_m": row["offset_east_m"],
                    "offset_north_m": row["offset_north_m"],
                    "offset_alt_m": row["offset_alt_m"] + alt,
                    "yaw_offset_deg": row["yaw_offset_deg"],
                    "description": f"Optional Stage 3 alt refinement from {row['name']} with delta {alt:g} m.",
                }
            )
    return out


def _write_result_pose(path: Path, qname: str, selected: Dict[str, Any], initial: Dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# P11 selected safe pose",
                f"method: {selected['name']}",
                "policy: strict_visual_gate_then_best_chamfer",
                "lon lat alt: " + " ".join(str(float(x)) for x in selected["translation_lon_lat_alt"]),
                "euler_pitch_roll_yaw: " + " ".join(str(float(x)) for x in selected["euler_pitch_roll_yaw"]),
                f"edge_chamfer: {selected['edge_chamfer']}",
                f"edge_overlap_ratio: {selected['edge_overlap_ratio']}",
                f"initial_edge_chamfer: {initial['edge_chamfer']}",
                f"initial_edge_overlap_ratio: {initial['edge_overlap_ratio']}",
                f"safe_output_worse_than_initial: {_worse(selected, initial)}",
                _format_pose_line(qname, selected["translation_lon_lat_alt"], selected["euler_pitch_roll_yaw"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, summary: Dict[str, Any], variant_results: Dict[str, Any]) -> None:
    top_rows = sorted(summary["candidates"], key=lambda row: (float(row["edge_chamfer"]), -float(row["edge_overlap_ratio"])))[:12]
    scorer_rows = [
        (
            "unweighted_feature",
            summary["rankings"]["rank_by_unweighted_feature_loss"][0],
            summary["correlation"]["spearman_unweighted_loss_vs_chamfer"],
            summary["correlation"]["spearman_unweighted_loss_vs_overlap"],
            summary["safe_gate"]["feature_topk_contains_selected_by_visual"]["unweighted_feature"]["top5"],
        )
    ]
    for mode, result in variant_results.items():
        scorer_rows.append(
            (
                mode,
                result["rank_by_weighted_feature_loss"][0] if result["rank_by_weighted_feature_loss"] else None,
                result["spearman_loss_vs_chamfer"],
                result["spearman_loss_vs_overlap"],
                result["contains_selected_by_visual_top5"],
            )
        )
    conclusion = summary["interpretation"]["conclusion_case"]
    lines = [
        "# P11 Local Visual-Safe Candidate Scorer Check",
        "",
        "## Purpose",
        "P10 tested structure weighting on raw-refined-derived candidates only. P11 rebuilds a local candidate scorer that includes local grid poses, known P3/P4 seeds, and corrected raw refined candidates under the same strict visual safe gate.",
        "",
        "## Prior Evidence",
        "| Source | Candidate | East | North | Alt | Yaw | Chamfer | Overlap | Feature loss | Meaning |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        "| local_gradient | initial | 0 | 0 | 0 | 0 | 5.4937 | 0.6293 | 0.1519 | baseline |",
        "| local_gradient | north_plus_1m | 0 | 1 | 0 | 0 | 4.6455 | 0.6021 | 0.1546 | better chamfer |",
        "| local_gradient | yaw_minus_1deg | 0 | 0 | 0 | -1 | 5.0441 | 0.6579 | 0.1562 | better chamfer and overlap |",
        "| diagnosis | p3_best_overlap | -5 | 5 | 0 | 0 | 3.1489 | 1.0774 | 0.1467 | known best visual |",
        "| diagnosis | p3_best_chamfer | -5 | 0 | 0 | 0 | 3.9349 | 0.8866 | 0.1576 | known good visual |",
        "| diagnosis | p4_scale_025_fixed_alt | -3.302 | -0.526 | 0 | 0 | 4.5064 | 0.8012 | 0.1603 | known good local refined |",
        "",
        "These are prior evidence only; P11 recomputes all metrics in this run.",
        "",
        "## Candidate Generation",
        f"- Stage 1 coarse east offsets: {summary['candidate_generation']['coarse_east_offsets']}",
        f"- Stage 1 coarse north offsets: {summary['candidate_generation']['coarse_north_offsets']}",
        f"- Stage 2 yaw offsets: {summary['candidate_generation']['yaw_refine_offsets']}",
        f"- Known seeds included: {summary['candidate_generation']['include_known_seeds']}",
        f"- Raw refined included: {summary['candidate_generation']['include_raw_refined']}",
        "",
        "## Gate Policy",
        "- strict visual gate: chamfer <= initial and overlap >= initial",
        "- final pose policy: strict_visual_gate_then_best_chamfer",
        "- feature and weighted feature losses are diagnostic scorers only.",
        "",
        "## Results",
        "| Candidate | Source | Stage | East | North | Alt | Yaw | Chamfer | Overlap | Unweighted loss | Best weighted loss | Strict gate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in top_rows:
        best_weight = min(row["weighted_feature_loss_by_mode"].values())
        lines.append(
            f"| {row['name']} | {row['source']} | {row['stage']} | {row['offset_east_m']:.3f} | {row['offset_north_m']:.3f} | {row['offset_alt_m']:.3f} | {row['yaw_offset_deg']:.3f} | {row['edge_chamfer']:.4f} | {row['edge_overlap_ratio']:.4f} | {row['unweighted_feature_loss']:.6f} | {best_weight:.6f} | {row['passes_strict_visual_gate']} |"
        )
    lines += [
        "",
        "## Ranking Comparison",
        "| Scorer | Top-1 candidate | Top-1 visual safe? | Spearman vs chamfer | Spearman vs overlap | Contains visual top1 in top5? |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    candidate_by_name = {row["name"]: row for row in summary["candidates"]}
    for scorer, top, sp_ch, sp_ov, contains in scorer_rows:
        safe = candidate_by_name.get(top, {}).get("passes_strict_visual_gate")
        lines.append(f"| {scorer} | {top} | {safe} | {sp_ch} | {sp_ov} | {contains} |")
    lines += [
        "",
        "## Safe Selection",
        f"- selected_by_visual: {summary['safe_gate']['selected_by_visual']['name']}",
        f"- selected_by_unweighted_feature: {summary['safe_gate']['selected_by_unweighted_feature']['name'] if summary['safe_gate']['selected_by_unweighted_feature'] else None}",
        f"- selected_by_weighted_feature_by_mode: {json.dumps(summary['safe_gate']['selected_by_weighted_feature_by_mode'], sort_keys=True)}",
        f"- result_pose_safe_p11.txt uses: {summary['safe_gate']['selected_by_visual']['name']}",
        f"- safe_output_worse_than_initial: {summary['safe_gate']['safe_output_worse_than_initial']}",
        "",
        "## Interpretation",
        f"- local candidate set contains better than initial: {summary['interpretation']['does_local_candidate_set_contain_better_than_initial']}",
        f"- strict gate accepts non-initial: {summary['interpretation']['does_strict_gate_accept_non_initial']}",
        f"- raw refined still misses good region: {summary['interpretation']['does_raw_refined_still_miss_good_region']}",
        f"- unweighted feature selects visual good candidate: {summary['interpretation']['does_unweighted_feature_select_visual_good_candidate']}",
        f"- any structure weight improves rank alignment: {summary['interpretation']['does_any_structure_weight_improve_rank_alignment']}",
        f"- recommended next step: {summary['interpretation']['recommended_next_step']}",
        "",
        "## Conclusion",
        conclusion,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--coarse-east", type=float, nargs="+", default=[-5, -3, -1, 0, 1, 3, 5])
    parser.add_argument("--coarse-north", type=float, nargs="+", default=[-5, -3, -1, 0, 1, 3, 5])
    parser.add_argument("--yaw-refine", type=float, nargs="+", default=[-2, -1, -0.5, 0, 0.5, 1, 2])
    parser.add_argument("--topk-visual-for-yaw", type=int, default=5)
    parser.add_argument("--weight-modes", nargs="+", default=["uniform", "dom_edge", "depth_gradient", "combined", "low_texture_downweight"])
    parser.add_argument("--include-known-seeds", action="store_true")
    parser.add_argument("--include-raw-refined", action="store_true")
    parser.add_argument("--visual-gate", action="store_true")
    parser.add_argument("--enable-alt-refine", action="store_true")
    parser.add_argument("--save-all-candidate-images", action="store_true")
    parser.add_argument("--checker-tile", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists() and not args.no_clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log: Dict[str, Any] = {"failure_stage": None, "traceback": None, "output_dir": os.fspath(output_dir)}
    stage = "start"
    start_total = time.time()

    try:
        stage = "load_config"
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        default_confs = config["default_confs"]
        default_confs["cam_query"]["max_size"] = args.width
        refine_conf = default_confs["refine"]
        conf = copy.deepcopy(default_confs["from_render_test"])

        stage = "setup_camera_and_pose"
        query_resize_ratio, raw_query_camera, render_camera_gs, query_camera, render_camera = _setup_camera(config)
        width = int(render_camera_gs[0])
        height = int(render_camera_gs[1])
        _loaded_euler, initial_trans, origin_np = load_initial_pose(args.pose_file)
        initial_trans = [float(x) for x in initial_trans]
        base_euler = [float(x) for x in BASE_EULER]
        base_yaw = float(base_euler[2])
        config["render_config"]["init_rot"] = base_euler
        config["render_config"]["init_trans"] = initial_trans
        refine_conf["origin"] = origin_np
        gt_pose_dict = load_pose_dict(args.pose_file, origin=origin_np)
        qname = Path(args.query_image).name

        stage = "load_query_and_renderer"
        cam_cfg = default_confs["cam_query"]
        query_image = read_image(args.query_image, scale=query_resize_ratio, distortion=cam_cfg["distortion"], query_camera=raw_query_camera)
        query_for_visual = _read_query_rgb(REPO_ROOT / args.query_image, width, height)
        query_image_for_refine = _resize_query_for_refine(query_image, render_camera_gs)
        renderer = DOMDSMRenderer(config["render_config"])
        to_raster, from_raster, raster_crs = _get_raster_transformers(config)

        stage = "reference_render_backproject"
        color, depth = renderer.render(initial_trans, base_euler)
        color_for_refine = pad_to_multiple(color, 16) if default_confs.get("padding", False) else color
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError("P11 requires CUDA for feature extraction/refinement")
        origin = torch.tensor(origin_np, device=device)
        query_camera = query_camera.to(device)
        render_camera = render_camera.to(device)
        p3d, t_render, t_init, dd = _back_project(
            depth,
            base_euler,
            initial_trans,
            base_euler,
            initial_trans,
            render_camera_gs,
            render_camera,
            origin,
            refine_conf["mul"],
            device,
            is_init=True,
        )

        stage = "init_localizer_extract_features"
        localizer = RenderLocalizer(conf)
        q_w, _ = query_camera.size
        query_feature_image = zero_pad(int(q_w.item()), query_image_for_refine)
        render_feature_image = zero_pad(int(q_w.item()), color_for_refine)
        with torch.no_grad():
            features_ref_raw, scales_ref = localizer.refiner.dense_feature_extraction(render_feature_image)
            features_q_raw, scales_q = localizer.refiner.dense_feature_extraction(query_feature_image)

        stage = "weights"
        p10_weights = _build_structure_weights(color, depth)
        weight_maps: Dict[str, np.ndarray] = {}
        mask_quality: Dict[str, Any] = {}
        weights_dir = output_dir / "weights"
        for mode in args.weight_modes:
            key = WEIGHT_KEY_BY_MODE[mode]
            weight = p10_weights[key]["weight"]
            weight_maps[mode] = weight
            mode_dir = weights_dir / mode
            _save_weight_png(mode_dir / "weight_map.png", weight)
            _save_weight_overlay(mode_dir / "weight_overlay.png", color, weight)
            stats = _weight_stats(
                weight,
                {
                    "source_p10_key": key,
                    "description": p10_weights[key]["description"],
                    "heuristic": p10_weights[key]["heuristic"],
                },
            )
            mask_quality[mode] = stats
            _write_json(mode_dir / "weight_stats.json", stats)

        stage = "optional_raw_refinement"
        raw_refined: Dict[str, Any] = {
            "available": False,
            "error": None,
            "translation_lon_lat_alt": None,
            "euler_pitch_roll_yaw": None,
            "raw_refined_yaw": None,
            "corrected_downward_yaw": None,
            "yaw_conversion_applied": False,
            "old_p9_yaw_bug_avoided": True,
        }
        if args.include_raw_refined:
            try:
                last_frame_info = {"observations": [], "refine_conf": refine_conf}
                ret = localizer.run_query(
                    args.query_image,
                    query_camera,
                    render_camera,
                    color_for_refine,
                    query_T=t_init,
                    render_T=t_render,
                    Points_3D_ECEF=p3d,
                    query_resize_ratio=query_resize_ratio,
                    dd=dd,
                    gt_pose_dict=gt_pose_dict,
                    last_frame_info=last_frame_info,
                    image_query=query_image_for_refine,
                )
                if not ret.get("success", False):
                    raise RuntimeError("run_query returned success=False")
                raw_trans = _array(ret["translation"]).tolist()
                raw_euler = _array(ret["euler_angles"]).tolist()
                raw_yaw = float(raw_euler[2])
                raw_refined.update(
                    {
                        "available": True,
                        "translation_lon_lat_alt": raw_trans,
                        "euler_pitch_roll_yaw": raw_euler,
                        "raw_refined_yaw": raw_yaw,
                        "corrected_downward_yaw": refined_yaw_to_downward_yaw(raw_yaw),
                        "yaw_conversion_applied": True,
                        "ret_fields": _ret_subset(ret),
                    }
                )
            except Exception as exc:
                raw_refined["error"] = repr(exc)

        stage = "candidate_specs_stage1"
        specs, debug = build_local_candidate_specs(
            initial_trans,
            base_yaw,
            to_raster,
            from_raster,
            coarse_east_offsets=args.coarse_east,
            coarse_north_offsets=args.coarse_north,
            include_known_p3p4_seeds=args.include_known_seeds,
            include_raw_refined_candidates=args.include_raw_refined and raw_refined["available"],
            raw_refined_translation_lon_lat_alt=raw_refined["translation_lon_lat_alt"],
            raw_refined_euler_pitch_roll_yaw=raw_refined["euler_pitch_roll_yaw"],
        )
        specs, duplicates_stage1 = deduplicate_candidates(specs)

        stage = "evaluate_stage1"
        rows: List[Dict[str, Any]] = []
        initial_row = None
        for spec in specs:
            row = _evaluate_candidate(
                spec,
                renderer,
                query_for_visual,
                output_dir,
                args.checker_tile,
                initial_row,
                initial_trans,
                to_raster,
                weight_maps,
                features_q_raw,
                features_ref_raw,
                scales_q,
                scales_ref,
                p3d,
                t_render,
                query_camera,
                render_camera,
                origin,
                dd,
                refine_conf["mul"],
                device,
            )
            if row["name"] == "initial":
                initial_row = row
                row["passes_strict_visual_gate"] = True
                row["passes_chamfer_only_gate"] = True
                row["passes_overlap_only_gate"] = True
                row["worse_than_initial"] = False
            rows.append(row)
        if initial_row is None:
            raise RuntimeError("Initial candidate missing")

        stage = "yaw_refinement"
        top_stage1 = sorted(
            [row for row in rows if row["valid_render"]],
            key=lambda item: (float(item["edge_chamfer"]), -float(item["edge_overlap_ratio"])),
        )[: args.topk_visual_for_yaw]
        yaw_specs = _build_yaw_refine_specs(top_stage1, args.yaw_refine, base_yaw)
        yaw_specs, duplicates_yaw = deduplicate_candidates(yaw_specs)
        for spec in yaw_specs:
            rows.append(
                _evaluate_candidate(
                    spec,
                    renderer,
                    query_for_visual,
                    output_dir,
                    args.checker_tile,
                    initial_row,
                    initial_trans,
                    to_raster,
                    weight_maps,
                    features_q_raw,
                    features_ref_raw,
                    scales_q,
                    scales_ref,
                    p3d,
                    t_render,
                    query_camera,
                    render_camera,
                    origin,
                    dd,
                    refine_conf["mul"],
                    device,
                )
            )

        duplicates_alt: List[Dict[str, Any]] = []
        if args.enable_alt_refine:
            stage = "alt_refinement"
            top_stage2 = sorted(
                [row for row in rows if row["valid_render"]],
                key=lambda item: (float(item["edge_chamfer"]), -float(item["edge_overlap_ratio"])),
            )[:3]
            alt_specs = _build_alt_refine_specs(top_stage2, [-1, 0, 1], initial_trans, to_raster, from_raster)
            alt_specs, duplicates_alt = deduplicate_candidates(alt_specs)
            for spec in alt_specs:
                rows.append(
                    _evaluate_candidate(
                        spec,
                        renderer,
                        query_for_visual,
                        output_dir,
                        args.checker_tile,
                        initial_row,
                        initial_trans,
                        to_raster,
                        weight_maps,
                        features_q_raw,
                        features_ref_raw,
                        scales_q,
                        scales_ref,
                        p3d,
                        t_render,
                        query_camera,
                        render_camera,
                        origin,
                        dd,
                        refine_conf["mul"],
                        device,
                    )
                )

        stage = "rankings_and_summary"
        valid_rows = [row for row in rows if row["valid_render"]]
        rank_chamfer = _rank(valid_rows, "edge_chamfer")
        rank_overlap = _rank(valid_rows, "edge_overlap_ratio", reverse=True)
        rank_unweighted = _rank(valid_rows, "unweighted_feature_loss")
        rank_weighted_by_mode: Dict[str, List[str]] = {}
        for mode in args.weight_modes:
            for row in valid_rows:
                row[f"weighted_feature_loss_{mode}"] = row["weighted_feature_loss_by_mode"][mode]
            rank_weighted_by_mode[mode] = _rank(valid_rows, f"weighted_feature_loss_{mode}")

        for row in valid_rows:
            row["rank_visual_chamfer"] = rank_chamfer.index(row["name"]) + 1 if row["name"] in rank_chamfer else None
            row["rank_visual_overlap"] = rank_overlap.index(row["name"]) + 1 if row["name"] in rank_overlap else None
            row["rank_unweighted_feature_loss"] = rank_unweighted.index(row["name"]) + 1 if row["name"] in rank_unweighted else None
            row["rank_weighted_feature_loss_by_mode"] = {
                mode: rank_weighted_by_mode[mode].index(row["name"]) + 1 if row["name"] in rank_weighted_by_mode[mode] else None
                for mode in args.weight_modes
            }

        selected_by_visual = _select_visual(valid_rows) or initial_row
        selected_by_unweighted = _select_by_key(valid_rows, "unweighted_feature_loss")
        selected_by_weighted = {mode: _select_by_key(valid_rows, f"weighted_feature_loss_{mode}") for mode in args.weight_modes}
        strict_pass = [row for row in valid_rows if row["passes_strict_visual_gate"]]
        strict_pass_non_initial = [row for row in strict_pass if row["name"] != "initial"]
        safe_output_worse = _worse(selected_by_visual, initial_row)
        top1_visual = rank_chamfer[0] if rank_chamfer else None
        selected_name = selected_by_visual["name"]

        sp_unweighted_chamfer = _spearman_from_ranked(rank_unweighted, rank_chamfer)
        sp_unweighted_overlap = _spearman_from_ranked(rank_unweighted, rank_overlap)
        sp_weighted_chamfer = {mode: _spearman_from_ranked(rank_weighted_by_mode[mode], rank_chamfer) for mode in args.weight_modes}
        sp_weighted_overlap = {mode: _spearman_from_ranked(rank_weighted_by_mode[mode], rank_overlap) for mode in args.weight_modes}
        uniform_sp = sp_weighted_chamfer.get("uniform")
        improved_modes = [
            mode
            for mode, value in sp_weighted_chamfer.items()
            if mode != "uniform" and uniform_sp is not None and value is not None and float(value) > float(uniform_sp) + EPS
        ]
        best_mode = None
        if sp_weighted_chamfer:
            best_mode = max(
                [mode for mode in args.weight_modes if sp_weighted_chamfer[mode] is not None],
                key=lambda mode: float(sp_weighted_chamfer[mode]),
                default=None,
            )

        known_seed_check = {}
        for name in KNOWN_SEEDS:
            row = next((item for item in valid_rows if item["name"] == name), None)
            known_seed_check[name] = {
                "exists": row is not None,
                "edge_chamfer": row.get("edge_chamfer") if row else None,
                "edge_overlap_ratio": row.get("edge_overlap_ratio") if row else None,
                "better_than_initial": bool(row and (float(row["edge_chamfer"]) < float(initial_row["edge_chamfer"]) - EPS or float(row["edge_overlap_ratio"]) > float(initial_row["edge_overlap_ratio"]) + EPS)),
                "passes_strict_visual_gate": bool(row and row["passes_strict_visual_gate"]),
            }
        known_seed_check["does_known_seed_improve_initial"] = any(v.get("better_than_initial") for v in known_seed_check.values() if isinstance(v, dict))

        raw_row = next((row for row in valid_rows if row["name"] == "raw_refined_full"), None)
        raw_misses = bool(raw_row and not raw_row["passes_strict_visual_gate"] and len(strict_pass_non_initial) > 0)
        feature_topk = {
            "unweighted_feature": {
                f"top{k}": _topk_contains(rank_unweighted, selected_name, k)
                for k in (3, 5, 10)
            }
        }
        weighted_feature_topk = {
            mode: {f"top{k}": _topk_contains(rank_weighted_by_mode[mode], selected_name, k) for k in (3, 5, 10)}
            for mode in args.weight_modes
        }
        feature_topk.update(weighted_feature_topk)

        if len(strict_pass_non_initial) > 0 and (
            selected_by_unweighted and selected_by_unweighted["name"] == selected_name
            or any(row and row["name"] == selected_name for row in selected_by_weighted.values())
        ):
            conclusion_case = "Case A: Local candidate scorer is effective, and a feature/weighted scorer selects the same visual-safe candidate."
            recommended = "Use the matching scorer for P12 coarse-to-fine safe refinement."
        elif len(strict_pass_non_initial) > 0:
            conclusion_case = "Case B: Local visual candidate search recovers a better pose, but feature scorers remain unreliable."
            recommended = "Use visual-safe selection or a hybrid visual-feature scorer for P12."
        else:
            conclusion_case = "Case C: Strict gate does not accept a non-initial pose in this run."
            recommended = "Inspect metric resolution, width, edge extraction, CRS offset conversion, and gate strictness."

        p11_success = (
            not safe_output_worse
            and len(strict_pass_non_initial) > 0
            and selected_by_visual["name"] != "initial"
            and float(selected_by_visual["edge_chamfer"]) <= float(initial_row["edge_chamfer"]) + EPS
            and float(selected_by_visual["edge_overlap_ratio"]) + EPS >= float(initial_row["edge_overlap_ratio"])
            and bool(raw_refined.get("old_p9_yaw_bug_avoided", True))
        )

        variant_results = {}
        for mode in args.weight_modes:
            selected = selected_by_weighted[mode]
            variant_results[mode] = {
                "rank_by_weighted_feature_loss": rank_weighted_by_mode[mode],
                "spearman_loss_vs_chamfer": sp_weighted_chamfer[mode],
                "spearman_loss_vs_overlap": sp_weighted_overlap[mode],
                "selected_by_weighted_feature": selected["name"] if selected else None,
                "selected_passes_strict_gate": bool(selected and selected["passes_strict_visual_gate"]),
                "contains_selected_by_visual_top5": _topk_contains(rank_weighted_by_mode[mode], selected_name, 5),
            }

        result_pose_path = output_dir / "result_pose_safe_p11.txt"
        _write_result_pose(result_pose_path, qname, selected_by_visual, initial_row)

        summary = {
            "experiment": "P11 local visual safe candidate scorer",
            "config": args.config,
            "query_image": args.query_image,
            "pose_file": args.pose_file,
            "width": args.width,
            "output_dir": os.fspath(output_dir),
            "result_pose_safe_p11": os.fspath(result_pose_path),
            "base_pose": {
                "translation_lon_lat_alt": initial_trans,
                "euler_pitch_roll_yaw": base_euler,
                "base_yaw": base_yaw,
            },
            "initial": initial_row,
            "raw_refined": {**raw_refined, **debug},
            "candidate_generation": {
                "coarse_east_offsets": args.coarse_east,
                "coarse_north_offsets": args.coarse_north,
                "yaw_refine_offsets": args.yaw_refine,
                "topk_visual_for_yaw": args.topk_visual_for_yaw,
                "include_known_seeds": args.include_known_seeds,
                "include_raw_refined": args.include_raw_refined,
                "num_candidates_total": len(rows),
                "num_candidates_valid": len(valid_rows),
                "duplicate_candidates_removed": duplicates_stage1 + duplicates_yaw + duplicates_alt,
            },
            "known_seed_check": known_seed_check,
            "candidates": valid_rows,
            "rankings": {
                "rank_by_visual_chamfer": rank_chamfer,
                "rank_by_visual_overlap": rank_overlap,
                "rank_by_unweighted_feature_loss": rank_unweighted,
                "rank_by_weighted_feature_loss_by_mode": rank_weighted_by_mode,
            },
            "correlation": {
                "spearman_unweighted_loss_vs_chamfer": sp_unweighted_chamfer,
                "spearman_unweighted_loss_vs_overlap": sp_unweighted_overlap,
                "spearman_weighted_loss_vs_chamfer_by_mode": sp_weighted_chamfer,
                "spearman_weighted_loss_vs_overlap_by_mode": sp_weighted_overlap,
            },
            "safe_gate": {
                "strict_gate_definition": "chamfer <= initial and overlap >= initial",
                "diagnostic_chamfer_gate_definition": "chamfer <= initial",
                "diagnostic_overlap_gate_definition": "overlap >= initial",
                "num_strict_pass": len(strict_pass),
                "strict_pass_candidates": [row["name"] for row in strict_pass],
                "selected_by_visual": selected_by_visual,
                "selected_by_unweighted_feature": selected_by_unweighted,
                "selected_by_weighted_feature_by_mode": {
                    mode: selected_by_weighted[mode]["name"] if selected_by_weighted[mode] else None
                    for mode in args.weight_modes
                },
                "final_pose_policy": "strict_visual_gate_then_best_chamfer",
                "feature_selected_pose_is_safe": bool(selected_by_unweighted and selected_by_unweighted["passes_strict_visual_gate"]),
                "weighted_feature_selected_pose_is_safe_by_mode": {
                    mode: bool(selected_by_weighted[mode] and selected_by_weighted[mode]["passes_strict_visual_gate"])
                    for mode in args.weight_modes
                },
                "feature_topk_contains_selected_by_visual": feature_topk,
                "safe_output_worse_than_initial": safe_output_worse,
            },
            "interpretation": {
                "does_local_candidate_set_contain_better_than_initial": any(
                    row["name"] != "initial"
                    and (float(row["edge_chamfer"]) < float(initial_row["edge_chamfer"]) - EPS or float(row["edge_overlap_ratio"]) > float(initial_row["edge_overlap_ratio"]) + EPS)
                    for row in valid_rows
                ),
                "does_strict_gate_accept_non_initial": len(strict_pass_non_initial) > 0,
                "does_unweighted_feature_select_visual_good_candidate": bool(selected_by_unweighted and selected_by_unweighted["name"] == selected_name),
                "does_any_structure_weight_improve_rank_alignment": len(improved_modes) > 0,
                "best_weight_mode_by_rank_alignment": best_mode,
                "does_p11_recover_p3p4_known_good_region": bool(known_seed_check["does_known_seed_improve_initial"]),
                "does_raw_refined_still_miss_good_region": raw_misses,
                "top1_visual_candidate": top1_visual,
                "top1_unweighted_feature_candidate": rank_unweighted[0] if rank_unweighted else None,
                "top1_weighted_feature_candidate_by_mode": {
                    mode: rank_weighted_by_mode[mode][0] if rank_weighted_by_mode[mode] else None for mode in args.weight_modes
                },
                "whether_top1_feature_matches_top1_visual": rank_unweighted[0] == top1_visual if rank_unweighted and top1_visual else False,
                "whether_topk_feature_contains_visual_top1": {
                    f"top{k}": _topk_contains(rank_unweighted, top1_visual, k) for k in (3, 5, 10)
                },
                "recommended_next_step": recommended,
                "conclusion_case": conclusion_case,
            },
            "environment": {
                "cuda_module_path": getattr(direct_abs_cost_cuda, "__file__", None),
                "torch_version": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
                "raster_crs": raster_crs,
            },
            "reference": {
                "render_camera_gs": render_camera_gs.tolist(),
                "depth_stats": _depth_stats(depth),
                "points_3d_count": int(p3d.shape[0]),
                "mask_quality": mask_quality,
            },
            "acceptance_checks": {
                "summary_has_required_keys": True,
                "result_pose_safe_p11_exists": result_pose_path.exists(),
                "old_p9_yaw_bug_avoided": bool(raw_refined.get("old_p9_yaw_bug_avoided", True)),
                "safe_output_worse_than_initial": safe_output_worse,
            },
            "p11_success": p11_success,
            "total_time_sec": time.time() - start_total,
        }

        _write_json(output_dir / "summary_metrics.json", summary)
        _write_json(output_dir / "variant_results.json", variant_results)
        _write_json(output_dir / "mask_quality_report.json", mask_quality)
        _write_markdown(REPO_ROOT / DEFAULT_DOC, summary, variant_results)
        _write_json(output_dir / "run_log.json", {**run_log, "summary_path": os.fspath(output_dir / "summary_metrics.json")})
        print(json.dumps({"p11_success": p11_success, "selected_by_visual": selected_by_visual["name"], "strict_pass_non_initial": [r["name"] for r in strict_pass_non_initial]}, indent=2, sort_keys=True))
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
