#!/usr/bin/env python3
"""Run P10 structure-aware loss diagnostics under the safe PiLoT gate."""

import argparse
import copy
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

import direct_abs_cost_cuda
from pixloc.localization.localizer import RenderLocalizer
from pixloc.pixlib.datasets.view import read_image
from pixloc.pixlib.geometry import Pose
from pixloc.pixlib.geometry.costs import DirectAbsoluteCost2
from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from pixloc.utils.dom_dsm.feature_loss_debug import project_points_to_image, sample_feature_map
from pixloc.utils.get_depth import pad_to_multiple, zero_pad
from src.utils.pose_utils import load_initial_pose, load_pose_dict
from tools.compare_cuda_torch_feature_loss_fixed_poses import (
    _cuda_and_torch_cost2_by_level,
    _pose_from_osg_fixed_context,
    _prepare_refiner_features,
    _spearman_from_ranked,
    _stack_poses,
)
from tools.diagnose_yawfix_refinement_update import (
    BASE_EULER,
    _array,
    _get_raster_transformers,
    _offset_between,
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
from tools.run_safe_pilot_refinement_p9 import (
    EPS,
    _candidate_extra,
    _metric_delta,
    _normalize_yaw,
    _select_safe_candidate,
    _visual_passes,
    _visual_worse,
)


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test_16x9/0000.jpg"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/structure_weighted_safe_refinement_p10"
VARIANTS = [
    "uniform_loss",
    "dom_edge_weighted_loss",
    "dsm_gradient_weighted_loss",
    "combined_structure_weighted_loss",
    "low_texture_vegetation_water_downweighted_loss",
]
RANK_METHODS = [
    "initial",
    "rebuilt_cuda_raw_refined",
    "refined_freeze_alt",
    "refined_freeze_alt_pitch_roll",
]


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_float_list(value: Sequence[Any]) -> List[float]:
    return [float(x) for x in value]


def _norm01(values: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if valid is None:
        sample = arr[np.isfinite(arr)]
    else:
        sample = arr[np.asarray(valid).astype(bool) & np.isfinite(arr)]
    if sample.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(sample, 2.0))
    hi = float(np.percentile(sample, 98.0))
    if hi <= lo + 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)


def _sobel_mag(gray: np.ndarray) -> np.ndarray:
    gray_f = np.asarray(gray, dtype=np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy).astype(np.float32)


def _weight_stats(weight: np.ndarray, extras: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    finite = weight[np.isfinite(weight)]
    stats = {
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
        "non_one_ratio": float(np.mean(np.abs(weight - 1.0) > 1e-6)),
        "shape": list(weight.shape),
    }
    if extras:
        stats.update(extras)
    return stats


def _save_weight_png(path: Path, weight: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vis = _norm01(weight)
    color = cv2.applyColorMap((vis * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(path), color)


def _build_structure_weights(color: np.ndarray, depth: np.ndarray) -> Dict[str, Dict[str, Any]]:
    rgb = np.asarray(color, dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    gray = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    dom_edge_norm = _norm01(_sobel_mag(gray))
    dom_edge_weight = (1.0 + 2.0 * dom_edge_norm).astype(np.float32)

    depth_f = np.asarray(depth, dtype=np.float32)
    valid_depth = np.isfinite(depth_f) & (depth_f > 0)
    if valid_depth.any():
        fill = float(np.median(depth_f[valid_depth]))
        depth_filled = np.where(valid_depth, depth_f, fill).astype(np.float32)
    else:
        depth_filled = np.zeros_like(depth_f, dtype=np.float32)
    dsm_grad_norm = _norm01(_sobel_mag(depth_filled), valid_depth)
    dsm_grad_weight = (1.0 + 2.0 * dsm_grad_norm).astype(np.float32)
    if valid_depth.any():
        dsm_grad_weight = np.where(valid_depth, dsm_grad_weight, 1.0).astype(np.float32)

    combined = np.maximum(dom_edge_weight, dsm_grad_weight).astype(np.float32)

    exg = 2.0 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    exg_norm = _norm01(exg)
    hsv = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    low_texture = dom_edge_norm < 0.15
    vegetation = (exg_norm > 0.75) & (rgb[..., 1] > rgb[..., 0]) & (rgb[..., 1] > rgb[..., 2])
    water = (sat < 0.25) & (val < 0.45) & (dom_edge_norm < 0.20)
    downweight = np.ones_like(gray, dtype=np.float32)
    downweight[low_texture] *= 0.65
    downweight[vegetation] *= 0.55
    downweight[water] *= 0.45
    downweight = np.clip(downweight, 0.25, 1.0).astype(np.float32)

    return {
        "uniform_loss": {
            "weight": np.ones_like(gray, dtype=np.float32),
            "description": "uniform feature residual weighting",
            "heuristic": False,
        },
        "dom_edge_weighted_loss": {
            "weight": dom_edge_weight,
            "description": "Sobel edge strength from initial DOM RGB, mapped to [1, 3]",
            "heuristic": False,
            "edge_strength": dom_edge_norm,
        },
        "dsm_gradient_weighted_loss": {
            "weight": dsm_grad_weight,
            "description": "Sobel gradient of initial render depth, mapped to [1, 3]",
            "heuristic": False,
            "depth_gradient": dsm_grad_norm,
        },
        "combined_structure_weighted_loss": {
            "weight": combined,
            "description": "max(dom_edge_weight, dsm_gradient_weight)",
            "heuristic": False,
        },
        "low_texture_vegetation_water_downweighted_loss": {
            "weight": downweight,
            "description": "heuristic RGB proxy that downweights low-texture, vegetation-like, and water-like areas",
            "heuristic": True,
            "proxy_fractions": {
                "low_texture": float(low_texture.mean()),
                "vegetation": float(vegetation.mean()),
                "water": float(water.mean()),
            },
        },
    }


def _strip_confidence(features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    out = []
    for feat in features:
        out.append(feat[:-1] if feat.shape[0] > 1 else feat)
    return out


def _weighted_feature_loss(
    features_query: Sequence[torch.Tensor],
    features_render: Sequence[torch.Tensor],
    scales_query: Sequence[Any],
    scales_render: Sequence[Any],
    p3d: torch.Tensor,
    t_query_w2c: Pose,
    t_render_w2c: Pose,
    query_camera: Any,
    render_camera: Any,
    weight_map: np.ndarray,
    levels: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    if levels is None:
        levels = list(range(min(len(features_query), len(features_render))))
    fq = _strip_confidence(features_query)
    fr = _strip_confidence(features_render)
    device = p3d.device
    proj_q = project_points_to_image(p3d, t_query_w2c, query_camera)
    proj_r = project_points_to_image(p3d, t_render_w2c, render_camera)
    weight_t = torch.as_tensor(weight_map, dtype=torch.float32, device=device).unsqueeze(0)
    losses = []
    unweighted_losses = []
    num_valid = []
    mean_weights = []
    for level in levels:
        feat_q = F.normalize(fq[level].to(device), dim=0)
        feat_r = F.normalize(fr[level].to(device), dim=0)
        q_sample, q_valid = sample_feature_map(feat_q, proj_q["points2d"], scales_query[level])
        r_sample, r_valid = sample_feature_map(feat_r, proj_r["points2d"], scales_render[level])
        w_sample, w_valid = sample_feature_map(weight_t, proj_r["points2d"], 1.0)
        w = w_sample[:, 0].clamp_min(0.0)
        valid = proj_q["valid"] & proj_r["valid"] & q_valid & r_valid & w_valid & torch.isfinite(w) & (w > 0)
        if valid.any():
            residual = ((q_sample[valid] - r_sample[valid]) ** 2).sum(dim=-1)
            weights = w[valid]
            losses.append((residual * weights).sum() / weights.sum().clamp_min(1e-12))
            unweighted_losses.append(residual.mean())
            num_valid.append(int(valid.sum().item()))
            mean_weights.append(float(weights.mean().detach().cpu().item()))
        else:
            losses.append(torch.tensor(float("nan"), device=device))
            unweighted_losses.append(torch.tensor(float("nan"), device=device))
            num_valid.append(0)
            mean_weights.append(None)
    finite_weighted = [loss for loss in losses if torch.isfinite(loss)]
    finite_unweighted = [loss for loss in unweighted_losses if torch.isfinite(loss)]
    loss_total = torch.stack(finite_weighted).mean() if finite_weighted else torch.tensor(float("inf"), device=device)
    unweighted_total = (
        torch.stack(finite_unweighted).mean() if finite_unweighted else torch.tensor(float("inf"), device=device)
    )
    return {
        "weighted_feature_loss_total": float(loss_total.detach().cpu().item()),
        "weighted_feature_loss_by_level": [
            float(loss.detach().cpu().item()) if torch.isfinite(loss) else None for loss in losses
        ],
        "unweighted_feature_loss_total": float(unweighted_total.detach().cpu().item()),
        "unweighted_feature_loss_by_level": [
            float(loss.detach().cpu().item()) if torch.isfinite(loss) else None for loss in unweighted_losses
        ],
        "num_valid_by_level": num_valid,
        "mean_weight_by_level": mean_weights,
    }


def _rank(rows: Sequence[Dict[str, Any]], key: str, reverse: bool = False) -> List[str]:
    valid = [row for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    return [
        str(row["method"])
        for row in sorted(valid, key=lambda item: float(item[key]), reverse=reverse)
    ]


def _apply_ranks(rows: List[Dict[str, Any]], key: str, field: str, reverse: bool = False) -> List[str]:
    ranked = _rank(rows, key, reverse=reverse)
    rank_map = {name: i + 1 for i, name in enumerate(ranked)}
    for row in rows:
        row[field] = rank_map.get(row["method"])
    return ranked


def _candidate_specs(
    initial_trans: List[float],
    initial_euler: List[float],
    refined_trans: List[float],
    refined_euler: List[float],
    downward_refined_yaw: float,
) -> List[Dict[str, Any]]:
    freeze_alt_trans = [float(refined_trans[0]), float(refined_trans[1]), float(initial_trans[2])]
    return [
        {
            "method": "initial",
            "trans": initial_trans,
            "euler": initial_euler,
            "source_method": "pose_file_yawfix_initial",
        },
        {
            "method": "rebuilt_cuda_raw_refined",
            "trans": refined_trans,
            "euler": refined_euler,
            "source_method": "rebuilt_cuda_optimizer_raw_output",
        },
        {
            "method": "refined_freeze_alt",
            "trans": freeze_alt_trans,
            "euler": list(map(float, refined_euler)),
            "source_method": "raw_refined_with_initial_alt",
        },
        {
            "method": "refined_freeze_alt_pitch_roll",
            "trans": list(freeze_alt_trans),
            "euler": [float(initial_euler[0]), float(initial_euler[1]), float(downward_refined_yaw)],
            "source_method": "raw_refined_lon_lat_downward_yaw_with_initial_alt_pitch_roll",
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--checker-tile", type=int, default=32)
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
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

    run_log: Dict[str, Any] = {
        "config_path": args.config,
        "query_image_path": args.query_image,
        "pose_file_path": args.pose_file,
        "output_dir": os.fspath(output_dir),
        "failure_stage": None,
        "traceback": None,
    }
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

        stage = "setup_camera"
        query_resize_ratio, raw_query_camera, render_camera_gs, query_camera, render_camera = _setup_camera(config)
        width = int(render_camera_gs[0])
        height = int(render_camera_gs[1])

        stage = "load_pose"
        _loaded_euler, initial_trans, origin_np = load_initial_pose(args.pose_file)
        initial_euler = list(map(float, BASE_EULER))
        initial_trans = list(map(float, initial_trans))
        config["render_config"]["init_rot"] = initial_euler
        config["render_config"]["init_trans"] = initial_trans
        refine_conf["origin"] = origin_np
        gt_pose_dict = load_pose_dict(args.pose_file, origin=origin_np)
        qname = Path(args.query_image).name

        stage = "load_query_image"
        cam_cfg = default_confs["cam_query"]
        query_image = read_image(
            args.query_image,
            scale=query_resize_ratio,
            distortion=cam_cfg["distortion"],
            query_camera=raw_query_camera,
        )
        query_for_visual = _read_query_rgb(REPO_ROOT / args.query_image, width, height)
        query_image_for_refine = _resize_query_for_refine(query_image, render_camera_gs)

        stage = "init_renderer"
        renderer = DOMDSMRenderer(config["render_config"])
        to_raster, _from_raster, raster_crs = _get_raster_transformers(config)

        stage = "render_reference"
        color, depth = renderer.render(initial_trans, initial_euler)
        color_for_refine = pad_to_multiple(color, 16) if default_confs.get("padding", False) else color

        stage = "back_project_reference"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError("P10 requires CUDA")
        origin = torch.tensor(origin_np, device=device)
        query_camera = query_camera.to(device)
        render_camera = render_camera.to(device)
        p3d, t_render, t_init, dd = _back_project(
            depth,
            initial_euler,
            initial_trans,
            initial_euler,
            initial_trans,
            render_camera_gs,
            render_camera,
            origin,
            refine_conf["mul"],
            device,
            is_init=True,
        )

        stage = "run_rebuilt_cuda_refinement"
        localizer = RenderLocalizer(conf)
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

        stage = "build_corrected_candidates"
        refined_euler = _array(ret["euler_angles"]).tolist()
        refined_trans = _array(ret["translation"]).tolist()
        raw_refined_yaw = float(refined_euler[2])
        downward_refined_yaw = _normalize_yaw(raw_refined_yaw + 180.0)
        specs = _candidate_specs(initial_trans, initial_euler, refined_trans, refined_euler, downward_refined_yaw)
        spec_by_method = {item["method"]: item for item in specs}

        stage = "render_candidates_and_gate"
        candidate_root = output_dir / "candidates"
        method_metrics: Dict[str, Dict[str, Any]] = {}
        for item in specs:
            metrics = _render_candidate(
                item["method"],
                renderer,
                query_for_visual,
                item["trans"],
                item["euler"],
                candidate_root,
                args.checker_tile,
                {
                    "method": item["method"],
                    **_candidate_extra(item["source_method"], item["trans"], initial_trans, to_raster),
                },
            )
            metrics["passes_acceptance_gate"] = (
                True if item["method"] == "initial" else _visual_passes(metrics, method_metrics["initial"])
            )
            metrics["worse_than_initial"] = (
                False if item["method"] == "initial" else _visual_worse(metrics, method_metrics["initial"])
            )
            metrics["delta_vs_initial"] = _metric_delta(metrics, metrics if item["method"] == "initial" else method_metrics["initial"])
            method_metrics[item["method"]] = metrics

        refined_candidate_metrics = [method_metrics[name] for name in RANK_METHODS if name != "initial"]
        selected_method, selected_metrics, passing_methods = _select_safe_candidate(
            method_metrics["initial"],
            refined_candidate_metrics,
        )
        safe_metrics = _render_candidate(
            "safe_refined_acceptance_gate",
            renderer,
            query_for_visual,
            selected_metrics["translation_lon_lat_alt"],
            selected_metrics["euler_pitch_roll_yaw"],
            candidate_root,
            args.checker_tile,
            {
                "method": "safe_refined_acceptance_gate",
                "selected_source_method": selected_method,
                "passing_refined_methods": passing_methods,
                **_candidate_extra(selected_method, selected_metrics["translation_lon_lat_alt"], initial_trans, to_raster),
            },
        )
        safe_metrics["passes_acceptance_gate"] = _visual_passes(safe_metrics, method_metrics["initial"])
        safe_metrics["worse_than_initial"] = _visual_worse(safe_metrics, method_metrics["initial"])
        safe_metrics["delta_vs_initial"] = _metric_delta(safe_metrics, method_metrics["initial"])
        method_metrics["safe_refined_acceptance_gate"] = safe_metrics

        stage = "extract_features"
        refiner = localizer.refiner
        opt = refiner.optimizer
        if isinstance(opt, (list, tuple)):
            opt_for_level = lambda level: opt[refiner.conf.layer_indices[level]] if refiner.conf.layer_indices else opt[level]
        else:
            opt_for_level = lambda _level: opt
        q_w, _ = query_camera.size
        query_feature_image = zero_pad(int(q_w.item()), query_image_for_refine)
        render_feature_image = zero_pad(int(q_w.item()), color_for_refine)
        with torch.no_grad():
            features_ref_raw, scales_ref = refiner.dense_feature_extraction(render_feature_image)
            features_q_raw, scales_q = refiner.dense_feature_extraction(query_feature_image)
        features_q, features_ref, weights_q, weights_ref = _prepare_refiner_features(
            refiner,
            features_q_raw,
            features_ref_raw,
        )
        if weights_q is None or weights_ref is None:
            raise RuntimeError("CUDA optimizer path expects uncertainty weights")

        stage = "build_candidate_poses"
        all_methods = RANK_METHODS + ["safe_refined_acceptance_gate"]
        pose_specs = []
        for method in all_methods:
            if method == "safe_refined_acceptance_gate":
                pose_specs.append(
                    {
                        "method": method,
                        "trans": safe_metrics["translation_lon_lat_alt"],
                        "euler": safe_metrics["euler_pitch_roll_yaw"],
                    }
                )
            else:
                pose_specs.append(spec_by_method[method])
        poses = _stack_poses(
            _pose_from_osg_fixed_context(
                item["trans"],
                item["euler"],
                origin,
                dd,
                refine_conf["mul"],
                device,
            )
            for item in pose_specs
        )
        pose_by_method = {item["method"]: poses[i] for i, item in enumerate(pose_specs)}

        stage = "cuda_loss_for_candidates"
        torch_cost = DirectAbsoluteCost2()
        cuda_by_method = {method: {"cuda_loss_by_level": {}} for method in all_methods}
        level_order = list(reversed(range(len(features_q))))
        for level in level_order:
            f_q = features_q[level]
            f_ref = features_ref[level]
            qcam_lvl = query_camera.scale(scales_q[level]).to_tensor().to(f_q)
            rcam_lvl = render_camera.scale(scales_ref[level]).to_tensor().to(f_q)
            level_result = _cuda_and_torch_cost2_by_level(
                opt_for_level(level),
                torch_cost,
                p3d,
                f_ref,
                f_q,
                poses.to(f_q),
                qcam_lvl,
                t_render.to(f_q),
                rcam_lvl,
                (weights_ref[level].to(f_q), weights_q[level].to(f_q)),
                all_methods,
            )
            for method in all_methods:
                metrics = level_result["candidate_metrics"][method]
                cuda_by_method[method]["cuda_loss_by_level"][str(level)] = metrics
                if level == level_order[-1]:
                    cuda_by_method[method]["cuda_w_loss_sum"] = metrics["cuda_w_loss"]["sum"]
                    cuda_by_method[method]["cuda_w_loss_mean"] = metrics["cuda_w_loss"]["mean"]
                    cuda_by_method[method]["cuda_cost_sum"] = metrics["cuda_cost"]["sum"]
                    cuda_by_method[method]["cuda_cost_mean"] = metrics["cuda_cost"]["mean"]

        stage = "structure_weights"
        weight_defs = _build_structure_weights(color, depth)
        mask_quality: Dict[str, Any] = {}
        for name, data in weight_defs.items():
            variant_dir = output_dir / name
            _save_weight_png(variant_dir / "weight_map.png", data["weight"])
            extras = {
                "description": data["description"],
                "heuristic": data["heuristic"],
            }
            if "proxy_fractions" in data:
                extras["proxy_fractions"] = data["proxy_fractions"]
            mask_quality[name] = _weight_stats(data["weight"], extras)
            _write_json(variant_dir / "weight_stats.json", mask_quality[name])

        stage = "variant_losses_and_ranks"
        variant_results: Dict[str, Any] = {}
        uniform_spearman: Optional[float] = None
        for variant in VARIANTS:
            weight = weight_defs[variant]["weight"]
            rows: List[Dict[str, Any]] = []
            for method in all_methods:
                base_metrics = method_metrics[method]
                loss = _weighted_feature_loss(
                    features_q_raw,
                    features_ref_raw,
                    scales_q,
                    scales_ref,
                    p3d,
                    pose_by_method[method],
                    t_render,
                    query_camera,
                    render_camera,
                    weight,
                    levels=None,
                )
                row = {
                    "method": method,
                    "source_method": base_metrics.get("source_method") or base_metrics.get("selected_source_method"),
                    "translation_lon_lat_alt": base_metrics["translation_lon_lat_alt"],
                    "euler_pitch_roll_yaw": base_metrics["euler_pitch_roll_yaw"],
                    "visual_output_paths": {
                        "rendered_rgb": os.fspath(candidate_root / method / "rendered_rgb.png"),
                        "edge_overlay": os.fspath(candidate_root / method / "edge_overlay.png"),
                        "overlay": os.fspath(candidate_root / method / "overlay.png"),
                        "checkerboard": os.fspath(candidate_root / method / "checkerboard.png"),
                        "metrics": os.fspath(candidate_root / method / "metrics.json"),
                    },
                    "visual_chamfer": base_metrics["edge_chamfer"],
                    "visual_overlap": base_metrics["edge_overlap_ratio"],
                    "passes_acceptance_gate": base_metrics["passes_acceptance_gate"],
                    "worse_than_initial": base_metrics["worse_than_initial"],
                    **loss,
                    **cuda_by_method[method],
                }
                rows.append(row)

            rank_rows = [row for row in rows if row["method"] in RANK_METHODS]
            rank_weighted = _apply_ranks(rank_rows, "weighted_feature_loss_total", "rank_weighted_feature_loss")
            rank_unweighted = _apply_ranks(rank_rows, "unweighted_feature_loss_total", "rank_unweighted_feature_loss")
            rank_chamfer = _apply_ranks(rank_rows, "visual_chamfer", "rank_visual_chamfer")
            rank_overlap = _apply_ranks(rank_rows, "visual_overlap", "rank_visual_overlap", reverse=True)
            rank_cuda_w = _apply_ranks(rank_rows, "cuda_w_loss_mean", "rank_cuda_w_loss")

            rank_fields = {row["method"]: row for row in rank_rows}
            for row in rows:
                if row["method"] in rank_fields:
                    continue
                row["rank_weighted_feature_loss"] = None
                row["rank_unweighted_feature_loss"] = None
                row["rank_visual_chamfer"] = None
                row["rank_visual_overlap"] = None
                row["rank_cuda_w_loss"] = None

            spearman_weighted = _spearman_from_ranked(rank_weighted, rank_chamfer)
            spearman_unweighted = _spearman_from_ranked(rank_unweighted, rank_chamfer)
            if variant == "uniform_loss":
                uniform_spearman = spearman_weighted

            final_safe = next(row for row in rows if row["method"] == "safe_refined_acceptance_gate")
            initial_row = next(row for row in rows if row["method"] == "initial")
            feature_down_visual_worse = []
            for row in rows:
                if row["method"] == "initial":
                    continue
                if (
                    row["weighted_feature_loss_total"] < initial_row["weighted_feature_loss_total"] - EPS
                    and row["visual_chamfer"] > initial_row["visual_chamfer"] + EPS
                ):
                    feature_down_visual_worse.append(row["method"])

            variant_result = {
                "variant": variant,
                "description": weight_defs[variant]["description"],
                "heuristic": weight_defs[variant]["heuristic"],
                "weight_map_path": os.fspath(output_dir / variant / "weight_map.png"),
                "candidate_metrics": rows,
                "rank_by_weighted_feature_loss": rank_weighted,
                "rank_by_unweighted_feature_loss": rank_unweighted,
                "rank_by_visual_chamfer": rank_chamfer,
                "rank_by_visual_overlap": rank_overlap,
                "rank_by_cuda_w_loss": rank_cuda_w,
                "variant_best_by_visual_chamfer": rank_chamfer[0] if rank_chamfer else None,
                "variant_best_by_weighted_feature_loss": rank_weighted[0] if rank_weighted else None,
                "rank_agreement_weighted_loss_vs_chamfer": spearman_weighted,
                "rank_agreement_unweighted_loss_vs_chamfer": spearman_unweighted,
                "safe_gate_selected_method": selected_method,
                "safe_output_worse_than_initial": bool(final_safe["worse_than_initial"]),
                "feature_loss_down_visual_worse_methods": feature_down_visual_worse,
                "does_safe_gate_prevent_regression": not bool(final_safe["worse_than_initial"]),
            }
            variant_results[variant] = variant_result
            _write_json(output_dir / variant / "candidate_metrics.json", variant_result)

        stage = "summarize"
        structure_improvements = {}
        for variant, result in variant_results.items():
            if variant == "uniform_loss":
                structure_improvements[variant] = False
                continue
            weighted = result["rank_agreement_weighted_loss_vs_chamfer"]
            structure_improvements[variant] = (
                uniform_spearman is not None
                and weighted is not None
                and float(weighted) > float(uniform_spearman) + EPS
            )
        safe_output = method_metrics["safe_refined_acceptance_gate"]
        safe_output_worse = _visual_worse(safe_output, method_metrics["initial"])
        corrected_yaw_delta_from_base = _normalize_yaw(downward_refined_yaw - float(initial_euler[2]))
        old_yaw_bug_triggered = (
            float(method_metrics["refined_freeze_alt_pitch_roll"]["edge_overlap_ratio"]) <= EPS
            or float(method_metrics["refined_freeze_alt_pitch_roll"]["edge_chamfer"]) > 20.0
        )
        p10_success = (
            not safe_output_worse
            and any(structure_improvements.values())
            and not old_yaw_bug_triggered
        )

        result_pose_path = output_dir / "result_pose_safe_p10.txt"
        result_pose_path.write_text(
            "\n".join(
                [
                    "# safe_refined_acceptance_gate",
                    f"# selected_source_method: {selected_method}",
                    _format_pose_line(
                        qname,
                        safe_output["translation_lon_lat_alt"],
                        safe_output["euler_pitch_roll_yaw"],
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        summary = {
            "config": args.config,
            "query_image": args.query_image,
            "pose_file": args.pose_file,
            "output_dir": os.fspath(output_dir),
            "result_pose_safe_p10_path": os.fspath(result_pose_path),
            "yaw_fix": {
                "raw_refined_yaw": raw_refined_yaw,
                "downward_refined_yaw": downward_refined_yaw,
                "base_yaw": float(initial_euler[2]),
                "downward_yaw_delta_from_base": corrected_yaw_delta_from_base,
                "yaw_conversion_applied": True,
                "rule": "normalize_yaw(raw_refined_yaw + 180)",
                "old_p9_1_yaw_bug_triggered": old_yaw_bug_triggered,
            },
            "methods": method_metrics,
            "gate": {
                "rule": "edge_chamfer <= initial_edge_chamfer + 1e-9 and edge_overlap_ratio + 1e-9 >= initial_edge_overlap_ratio",
                "passing_refined_methods": passing_methods,
                "safe_gate_selected_method": selected_method,
            },
            "variant_names": VARIANTS,
            "variant_best_by_visual_chamfer": {
                name: result["variant_best_by_visual_chamfer"] for name, result in variant_results.items()
            },
            "variant_best_by_weighted_feature_loss": {
                name: result["variant_best_by_weighted_feature_loss"] for name, result in variant_results.items()
            },
            "rank_agreement_weighted_loss_vs_chamfer": {
                name: result["rank_agreement_weighted_loss_vs_chamfer"] for name, result in variant_results.items()
            },
            "rank_agreement_unweighted_loss_vs_chamfer": {
                name: result["rank_agreement_unweighted_loss_vs_chamfer"] for name, result in variant_results.items()
            },
            "does_structure_weight_improve_rank_alignment": structure_improvements,
            "does_safe_gate_prevent_regression": not safe_output_worse,
            "p10_success": p10_success,
            "conclusions": {
                "safe_output_worse_than_initial": safe_output_worse,
                "safe_gate_selected_method": selected_method,
                "raw_refined_worse_than_initial": _visual_worse(method_metrics["rebuilt_cuda_raw_refined"], method_metrics["initial"]),
                "corrected_freeze_alt_pitch_roll_chamfer": method_metrics["refined_freeze_alt_pitch_roll"]["edge_chamfer"],
                "corrected_freeze_alt_pitch_roll_overlap": method_metrics["refined_freeze_alt_pitch_roll"]["edge_overlap_ratio"],
                "does_structure_weight_improve_rank_alignment": any(structure_improvements.values()),
                "p10_success": p10_success,
            },
            "acceptance_checks": {
                "summary_metrics_exists": True,
                "variant_count_is_5": len(variant_results) == 5,
                "all_variants_have_weight_maps": all((output_dir / name / "weight_map.png").exists() for name in VARIANTS),
                "all_variants_have_candidate_metrics": all(
                    (output_dir / name / "candidate_metrics.json").exists() for name in VARIANTS
                ),
                "safe_gate_chamfer_not_worse": float(safe_output["edge_chamfer"]) <= float(method_metrics["initial"]["edge_chamfer"]) + EPS,
                "safe_gate_overlap_not_worse": float(safe_output["edge_overlap_ratio"]) + EPS >= float(method_metrics["initial"]["edge_overlap_ratio"]),
                "result_pose_safe_p10_exists": result_pose_path.exists(),
                "yaw_conversion_applied": True,
                "old_p9_1_yaw_bug_avoided": not old_yaw_bug_triggered,
            },
            "environment": {
                "cuda_module_path": getattr(direct_abs_cost_cuda, "__file__", None),
                "cuda_has_residual": hasattr(direct_abs_cost_cuda, "residual_jacobian_batch_quat_cuda"),
                "cuda_has_step": hasattr(direct_abs_cost_cuda, "optimizer_step_cuda"),
                "torch_version": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
                "raster_crs": raster_crs,
            },
            "reference": {
                "initial_translation_lon_lat_alt": initial_trans,
                "initial_euler_pitch_roll_yaw": initial_euler,
                "render_camera_gs": render_camera_gs.tolist(),
                "initial_render_depth_stats": _depth_stats(depth),
                "points_3d_count": int(p3d.shape[0]),
                "processed_level_order": level_order,
            },
            "assumptions": {
                "structure_weighting_scope": "Python diagnostic feature loss only; CUDA kernel is unchanged",
                "safe_gate": "P9 strict visual gate is applied to every variant",
                "vegetation_water_proxy": "RGB-only heuristic, not semantic segmentation",
                "rank_agreement_methods": RANK_METHODS,
            },
            "total_time_sec": time.time() - start_total,
        }
        _write_json(output_dir / "summary_metrics.json", summary)
        _write_json(output_dir / "variant_results.json", variant_results)
        _write_json(output_dir / "mask_quality_report.json", mask_quality)
        _write_json(output_dir / "run_log.json", {**run_log, "summary_path": os.fspath(output_dir / "summary_metrics.json")})
        print(json.dumps(summary["conclusions"], indent=2, sort_keys=True))
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
