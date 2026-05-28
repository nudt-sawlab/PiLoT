#!/usr/bin/env python3
"""Compare CUDA and PyTorch feature losses for fixed DOM/DSM poses.

This P7 diagnostic intentionally evaluates losses only. It calls the CUDA
residual/Jacobian kernel directly and never calls optimizer_step_cuda.
"""

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
from pixloc.utils.dom_dsm.feature_loss_debug import (
    compute_feature_residual_loss,
    save_residual_debug_visualization,
)
from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from pixloc.utils.get_depth import (
    WGS84_to_ECEF,
    _euler_to_matrix_ecef_batch,
    pad_to_multiple,
    zero_pad,
)
from src.utils.pose_utils import load_initial_pose
from tools.diagnose_yawfix_refinement_update import (
    BASE_EULER,
    _get_raster_transformers,
    _offset_between,
    _read_query_rgb,
    _render_candidate,
    _safe_jsonable,
)
from tools.run_dom_dsm_single_full import (
    _back_project,
    _depth_stats,
    _resize_query_for_refine,
    _setup_camera,
)


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test_16x9/0000.jpg"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/cuda_torch_loss_parity_p7"
DEFAULT_P3_SUMMARY = "docs/experiments/dom_dsm_prepare/torch_feature_loss_diagnosis/summary_metrics.json"
DEFAULT_OLD_P4_SUMMARY = "docs/experiments/dom_dsm_prepare/yawfix_refinement_diagnosis_results/summary_metrics.json"
DEFAULT_REBUILT_P4_SUMMARY = "docs/experiments/dom_dsm_prepare/yawfix_refinement_diagnosis_results_rebuilt_sm61/summary_metrics.json"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float_list(value: Sequence[Any]) -> List[float]:
    return [float(x) for x in value]


def _find_candidate(summary: Dict[str, Any], name: str) -> Dict[str, Any]:
    if name in summary and isinstance(summary[name], dict):
        return summary[name]
    for key in ("candidates", "line_search_candidates"):
        for item in summary.get(key, []):
            if item.get("candidate") == name or item.get("directory") == name:
                return item
    raise KeyError(f"Candidate not found in summary: {name}")


def _build_candidates(
    initial_trans: List[float],
    initial_euler: List[float],
    p3_summary: Dict[str, Any],
    old_p4_summary: Dict[str, Any],
    rebuilt_p4_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    p3_overlap = _find_candidate(p3_summary, "p3_best_overlap")
    p3_chamfer = _find_candidate(p3_summary, "p3_best_chamfer")
    p4_scale_025 = _find_candidate(
        old_p4_summary,
        "line_search/scale_0.25_alt_fixed_initial",
    )
    rebuilt_raw = _find_candidate(rebuilt_p4_summary, "raw_refined_full")
    old_raw = _find_candidate(old_p4_summary, "raw_refined_full")

    specs = [
        ("initial", "pose_file_yawfix_initial", initial_trans, initial_euler),
        ("p3_best_overlap", DEFAULT_P3_SUMMARY, p3_overlap["translation_lon_lat_alt"], p3_overlap["euler_pitch_roll_yaw"]),
        ("p3_best_chamfer", DEFAULT_P3_SUMMARY, p3_chamfer["translation_lon_lat_alt"], p3_chamfer["euler_pitch_roll_yaw"]),
        ("p4_scale_025", "old_p4:line_search/scale_0.25_alt_fixed_initial", p4_scale_025["translation_lon_lat_alt"], p4_scale_025["euler_pitch_roll_yaw"]),
        ("rebuilt_cuda_refined", "rebuilt_p4:raw_refined_full", rebuilt_raw["translation_lon_lat_alt"], rebuilt_raw["euler_pitch_roll_yaw"]),
        ("raw_refined_old", "old_p4:raw_refined_full", old_raw["translation_lon_lat_alt"], old_raw["euler_pitch_roll_yaw"]),
    ]
    return [
        {
            "candidate": name,
            "source_label": source,
            "translation_lon_lat_alt": _as_float_list(trans),
            "euler_pitch_roll_yaw": _as_float_list(euler),
        }
        for name, source, trans, euler in specs
    ]


def _pose_from_osg_fixed_context(
    trans: Sequence[float],
    euler: Sequence[float],
    origin: torch.Tensor,
    dd: torch.Tensor,
    mul: Optional[float],
    device: str,
) -> Pose:
    """Build a one-item PixLoc w2c Pose using sample_3d_points conventions."""
    euler_t = torch.tensor([_as_float_list(euler)], device=device, dtype=torch.float32)
    trans_ecef = torch.tensor(
        WGS84_to_ECEF(_as_float_list(trans)),
        device=device,
        dtype=torch.float32,
    ).reshape(1, 3)
    query_t_c2w = _euler_to_matrix_ecef_batch(
        euler_t,
        trans_ecef,
        _as_float_list(trans),
        device=device,
    )
    query_t_c2w[:, :3, 1] *= -1
    query_t_c2w[:, :3, 2] *= -1

    origin_scaled = origin
    if mul is not None:
        query_t_c2w[:, :3, 3] *= float(mul)
        origin_scaled = origin * float(mul)

    query_t_c2w[:, :3, 3] -= origin_scaled
    t_query = Pose.from_Rt(
        query_t_c2w[:, :3, :3],
        query_t_c2w[:, :3, 3],
    ).inv()
    shifted_t = t_query.t + t_query.R @ dd
    return Pose.from_Rt(t_query.R, shifted_t).float()


def _stack_poses(poses: Iterable[Pose]) -> Pose:
    return Pose(torch.cat([pose._data for pose in poses], dim=0))


def _prepare_refiner_features(refiner: Any, features_q: Sequence[torch.Tensor], features_ref: Sequence[torch.Tensor]):
    features_q = [f.to(refiner.device) for f in features_q]
    features_ref = [f.to(refiner.device) for f in features_ref]
    weights_q, weights_ref = None, None
    if refiner.conf.compute_uncertainty:
        weights_q = [f[-1:] for f in features_q]
        features_q = [f[:-1] for f in features_q]
        weights_ref = [f[-1:] for f in features_ref]
        features_ref = [f[:-1] for f in features_ref]
    if refiner.conf.normalize_descriptors:
        features_q = [F.normalize(f, dim=0) for f in features_q]
    return features_q, features_ref, weights_q, weights_ref


def _tensor_stats_by_candidate(tensor: torch.Tensor, num_candidates: int) -> Tuple[List[Dict[str, Any]], List[int]]:
    values = tensor.detach().float()
    while values.ndim > 0 and values.shape[0] == 1 and values.shape[0] != num_candidates:
        values = values.squeeze(0)
    raw_shape = list(tensor.shape)
    if values.ndim == 0:
        values = values.reshape(1).repeat(num_candidates)
    if values.shape[0] != num_candidates:
        values = values.reshape(num_candidates, -1)
    if values.ndim == 1:
        flat = values.reshape(num_candidates, 1)
    else:
        flat = values.reshape(num_candidates, -1)

    out = []
    for row in flat:
        finite = row[torch.isfinite(row)]
        nonzero = finite[finite != 0]
        denom = finite.numel()
        nz_denom = nonzero.numel()
        out.append(
            {
                "sum": float(finite.sum().item()) if denom else None,
                "mean": float(finite.mean().item()) if denom else None,
                "nonzero_sum": float(nonzero.sum().item()) if nz_denom else 0.0,
                "nonzero_mean": float(nonzero.mean().item()) if nz_denom else None,
                "finite_count": int(denom),
                "nonzero_count": int(nz_denom),
            }
        )
    return out, raw_shape


def _cuda_and_torch_cost2_by_level(
    opt: Any,
    torch_cost: DirectAbsoluteCost2,
    p3d: torch.Tensor,
    f_ref: torch.Tensor,
    f_query: torch.Tensor,
    poses: Pose,
    qcamera_tensor: torch.Tensor,
    render_pose: Pose,
    rcamera_tensor: torch.Tensor,
    weights_ref_query: Tuple[torch.Tensor, torch.Tensor],
    candidate_names: Sequence[str],
) -> Dict[str, Any]:
    num_candidates = len(candidate_names)
    t_flat = poses.to_flat()
    t_render = render_pose.to_flat().expand(1, -1)
    qcamera = qcamera_tensor.unsqueeze(0).expand(num_candidates, -1)
    p3d_expanded = p3d.unsqueeze(0).expand(1, -1, -1)
    ref_camera = rcamera_tensor.unsqueeze(0).expand(1, -1)
    c_ref, c_query = weights_ref_query

    inputs = {
        "pose_data_q": t_flat.unsqueeze(0).contiguous().clone(),
        "f_r": f_ref.unsqueeze(0).clone(),
        "pose_data_r": t_render.unsqueeze(0).clone(),
        "cam_data_r": ref_camera.unsqueeze(0).clone(),
        "f_q": f_query.unsqueeze(0).clone(),
        "cam_data_q": qcamera.unsqueeze(0).clone(),
        "p3D": p3d_expanded.unsqueeze(0).contiguous().clone(),
        "c_ref": c_ref.unsqueeze(0).clone(),
        "c_query": c_query.unsqueeze(0).clone(),
    }

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g_cuda, h_cuda, w_loss_cuda, cost_cuda = opt.fn(
        inputs["pose_data_q"],
        inputs["f_r"],
        inputs["pose_data_r"],
        inputs["cam_data_r"],
        inputs["f_q"],
        inputs["cam_data_q"],
        inputs["p3D"],
        inputs["c_ref"],
        inputs["c_query"],
    )
    torch.cuda.synchronize()
    cuda_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    g_torch, h_torch, w_loss_torch, valid_torch, _p2d_q, cost_torch = torch_cost.residual_jacobian_batch_quat(
        inputs["pose_data_q"],
        inputs["f_r"],
        inputs["pose_data_r"],
        inputs["cam_data_r"],
        inputs["f_q"],
        inputs["cam_data_q"],
        inputs["p3D"],
        inputs["c_ref"],
        inputs["c_query"],
    )
    torch.cuda.synchronize()
    torch_time = time.perf_counter() - t0

    cuda_w_stats, cuda_w_shape = _tensor_stats_by_candidate(w_loss_cuda, num_candidates)
    cuda_c_stats, cuda_c_shape = _tensor_stats_by_candidate(cost_cuda, num_candidates)
    torch_w_stats, torch_w_shape = _tensor_stats_by_candidate(w_loss_torch, num_candidates)
    torch_c_stats, torch_c_shape = _tensor_stats_by_candidate(cost_torch, num_candidates)
    valid_stats, valid_shape = _tensor_stats_by_candidate(valid_torch, num_candidates)

    return {
        "candidate_metrics": {
            name: {
                "cuda_w_loss": cuda_w_stats[i],
                "cuda_cost": cuda_c_stats[i],
                "torch_cost2_w_loss": torch_w_stats[i],
                "torch_cost2_cost": torch_c_stats[i],
                "torch_cost2_valid": valid_stats[i],
            }
            for i, name in enumerate(candidate_names)
        },
        "raw_shapes": {
            "cuda_g": list(g_cuda.shape),
            "cuda_H": list(h_cuda.shape),
            "cuda_w_loss": cuda_w_shape,
            "cuda_cost": cuda_c_shape,
            "torch_g": list(g_torch.shape),
            "torch_H": list(h_torch.shape),
            "torch_w_loss": torch_w_shape,
            "torch_cost": torch_c_shape,
            "torch_valid": valid_shape,
        },
        "timing_sec": {
            "cuda_residual_jacobian": cuda_time,
            "torch_direct_absolute_cost2": torch_time,
        },
    }


def _rank(rows: Sequence[Dict[str, Any]], key: str, reverse: bool = False) -> List[str]:
    valid = [row for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    return [
        row["candidate"]
        for row in sorted(valid, key=lambda item: float(item[key]), reverse=reverse)
    ]


def _apply_ranks(rows: List[Dict[str, Any]], key: str, rank_field: str, reverse: bool = False) -> List[str]:
    ranked = _rank(rows, key, reverse=reverse)
    rank_map = {name: i + 1 for i, name in enumerate(ranked)}
    for row in rows:
        row[rank_field] = rank_map.get(row["candidate"])
    return ranked


def _spearman_from_ranked(a: Sequence[str], b: Sequence[str]) -> Optional[float]:
    names = [name for name in a if name in set(b)]
    n = len(names)
    if n < 2:
        return None
    ar = {name: i + 1 for i, name in enumerate(a)}
    br = {name: i + 1 for i, name in enumerate(b)}
    d2 = sum((ar[name] - br[name]) ** 2 for name in names)
    return float(1.0 - (6.0 * d2) / (n * (n * n - 1)))


def _diagnosis_branch(torch_rank: Sequence[str], cuda_w_rank: Sequence[str], cuda_c_rank: Sequence[str]) -> Tuple[str, bool, bool, Optional[float], Optional[float]]:
    cuda_w_top1 = bool(torch_rank and cuda_w_rank and torch_rank[0] == cuda_w_rank[0])
    cuda_c_top1 = bool(torch_rank and cuda_c_rank and torch_rank[0] == cuda_c_rank[0])
    spearman_w = _spearman_from_ranked(torch_rank, cuda_w_rank)
    spearman_c = _spearman_from_ranked(torch_rank, cuda_c_rank)
    parity = (
        cuda_w_top1
        and cuda_c_top1
        and spearman_w is not None
        and spearman_c is not None
        and spearman_w >= 0.8
        and spearman_c >= 0.8
    )
    if parity:
        branch = "loss parity passed: inspect Jacobian / optimizer_step_cuda / pose update sign, side, and parameter order next"
    else:
        branch = "loss parity failed: do not trust CUDA optimizer yet; inspect CUDA loss scale, coordinates, feature normalization, and robust loss"
    return branch, cuda_w_top1, cuda_c_top1, spearman_w, spearman_c


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--p3-summary", default=DEFAULT_P3_SUMMARY)
    parser.add_argument("--old-p4-summary", default=DEFAULT_OLD_P4_SUMMARY)
    parser.add_argument("--rebuilt-p4-summary", default=DEFAULT_REBUILT_P4_SUMMARY)
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
        "optimizer_step_called": False,
        "failure_stage": None,
        "traceback": None,
    }
    stage = "start"
    start_total = time.time()
    optimizer_step_called = {"value": False}

    def _forbidden_optimizer_step_cuda(*_args: Any, **_kwargs: Any) -> None:
        optimizer_step_called["value"] = True
        raise RuntimeError("optimizer_step_cuda must not be called by this fixed-pose parity check")

    try:
        stage = "guard_optimizer_step"
        try:
            direct_abs_cost_cuda.optimizer_step_cuda = _forbidden_optimizer_step_cuda
        except Exception as exc:
            run_log["optimizer_step_guard_warning"] = repr(exc)

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
        _loaded_euler, trans, origin_np = load_initial_pose(args.pose_file)
        euler = list(map(float, BASE_EULER))
        trans = list(map(float, trans))
        config["render_config"]["init_rot"] = euler
        config["render_config"]["init_trans"] = trans
        refine_conf["origin"] = origin_np

        stage = "load_candidate_summaries"
        p3_summary = _load_json(REPO_ROOT / args.p3_summary)
        old_p4_summary = _load_json(REPO_ROOT / args.old_p4_summary)
        rebuilt_p4_summary = _load_json(REPO_ROOT / args.rebuilt_p4_summary)
        candidate_specs = _build_candidates(trans, euler, p3_summary, old_p4_summary, rebuilt_p4_summary)
        candidate_names = [item["candidate"] for item in candidate_specs]

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
        color, depth = renderer.render(trans, euler)
        if default_confs.get("padding", False):
            color_for_refine = pad_to_multiple(color, 16)
        else:
            color_for_refine = color

        stage = "back_project_reference"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError("P7 CUDA parity check requires CUDA")
        origin = torch.tensor(origin_np, device=device)
        query_camera = query_camera.to(device)
        render_camera = render_camera.to(device)
        p3d, t_render, _unused_t_init, dd = _back_project(
            depth,
            euler,
            trans,
            euler,
            trans,
            render_camera_gs,
            render_camera,
            origin,
            refine_conf["mul"],
            device,
            is_init=True,
        )

        stage = "build_candidate_poses"
        candidate_poses = _stack_poses(
            _pose_from_osg_fixed_context(
                item["translation_lon_lat_alt"],
                item["euler_pitch_roll_yaw"],
                origin,
                dd,
                refine_conf["mul"],
                device,
            )
            for item in candidate_specs
        )

        stage = "init_localizer"
        localizer = RenderLocalizer(conf)
        refiner = localizer.refiner
        opt = refiner.optimizer
        if isinstance(opt, (list, tuple)):
            opt_for_level = lambda level: opt[refiner.conf.layer_indices[level]] if refiner.conf.layer_indices else opt[level]
        else:
            opt_for_level = lambda _level: opt

        stage = "extract_features"
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
            raise RuntimeError("CUDA optimizer path expects uncertainty weights in W_ref_query")

        stage = "visual_metrics"
        rows: List[Dict[str, Any]] = []
        for item in candidate_specs:
            visual = _render_candidate(
                item["candidate"],
                renderer,
                query_for_visual,
                item["translation_lon_lat_alt"],
                item["euler_pitch_roll_yaw"],
                output_dir,
                args.checker_tile,
                {
                    "source_label": item["source_label"],
                    "east_north_alt_offset_m": _offset_between(trans, item["translation_lon_lat_alt"], to_raster),
                },
            )
            offset = _offset_between(trans, item["translation_lon_lat_alt"], to_raster)
            rows.append(
                {
                    **item,
                    "output_translation_lon_lat_alt": item["translation_lon_lat_alt"],
                    "output_euler_pitch_roll_yaw": item["euler_pitch_roll_yaw"],
                    "pose_update_delta_east_north_alt_m": [0.0, 0.0, 0.0],
                    "pose_update_delta_euler_deg": [0.0, 0.0, 0.0],
                    "east_offset_m": offset[0],
                    "north_offset_m": offset[1],
                    "alt_offset_m": offset[2],
                    "visual_overlap": visual["edge_overlap_ratio"],
                    "visual_chamfer": visual["edge_chamfer"],
                    "visual_metrics": visual,
                }
            )

        stage = "torch_feature_loss"
        pose_by_name = {name: candidate_poses[i] for i, name in enumerate(candidate_names)}
        render_rgb_for_residual = color if color.shape[:2] == query_for_visual.shape[:2] else query_for_visual
        for row in rows:
            loss = compute_feature_residual_loss(
                features_q_raw,
                features_ref_raw,
                scales_q,
                scales_ref,
                p3d,
                pose_by_name[row["candidate"]],
                t_render,
                query_camera,
                render_camera,
                levels=None,
                robust="l2",
                use_confidence=False,
            )
            row["torch_feature_loss_total"] = loss["loss_total"]
            row["torch_feature_loss_by_level"] = loss["loss_by_level"]
            row["torch_feature_num_valid_by_level"] = loss["num_valid_by_level"]
            row["torch_feature_valid_ratio_by_level"] = loss["valid_ratio_by_level"]
            save_residual_debug_visualization(
                output_dir / row["candidate"] / "torch_feature_residual",
                query_for_visual,
                render_rgb_for_residual,
                loss["points_query"],
                loss["points_render"],
                loss["residual_per_point"],
                loss["valid_mask"],
            )

        stage = "cuda_residual_jacobian"
        torch_cost = DirectAbsoluteCost2()
        per_level: Dict[str, Any] = {}
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
                candidate_poses.to(f_q),
                qcam_lvl,
                t_render.to(f_q),
                rcam_lvl,
                (weights_ref[level].to(f_q), weights_q[level].to(f_q)),
                candidate_names,
            )
            per_level[str(level)] = {
                "raw_shapes": level_result["raw_shapes"],
                "timing_sec": level_result["timing_sec"],
            }
            for row in rows:
                metrics = level_result["candidate_metrics"][row["candidate"]]
                row.setdefault("cuda_loss_by_level", {})[str(level)] = metrics
                if level == level_order[-1]:
                    row["cuda_w_loss_sum"] = metrics["cuda_w_loss"]["sum"]
                    row["cuda_w_loss_mean"] = metrics["cuda_w_loss"]["mean"]
                    row["cuda_cost_sum"] = metrics["cuda_cost"]["sum"]
                    row["cuda_cost_mean"] = metrics["cuda_cost"]["mean"]
                    row["torch_cost2_w_loss_sum"] = metrics["torch_cost2_w_loss"]["sum"]
                    row["torch_cost2_w_loss_mean"] = metrics["torch_cost2_w_loss"]["mean"]
                    row["torch_cost2_cost_sum"] = metrics["torch_cost2_cost"]["sum"]
                    row["torch_cost2_cost_mean"] = metrics["torch_cost2_cost"]["mean"]

        stage = "rank_and_summarize"
        rank_torch = _apply_ranks(rows, "torch_feature_loss_total", "rank_torch_feature_loss")
        rank_cuda_w = _apply_ranks(rows, "cuda_w_loss_mean", "rank_cuda_w_loss")
        rank_cuda_c = _apply_ranks(rows, "cuda_cost_mean", "rank_cuda_cost")
        rank_overlap = _apply_ranks(rows, "visual_overlap", "rank_visual_overlap", reverse=True)
        rank_chamfer = _apply_ranks(rows, "visual_chamfer", "rank_visual_chamfer")
        branch, cuda_w_top1, cuda_c_top1, spearman_w, spearman_c = _diagnosis_branch(
            rank_torch,
            rank_cuda_w,
            rank_cuda_c,
        )

        for row in rows:
            row["cuda_w_loss_matches_torch_top1"] = cuda_w_top1
            row["cuda_cost_matches_torch_top1"] = cuda_c_top1
            row["cuda_torch_rank_agreement"] = {
                "spearman_torch_vs_cuda_w_loss": spearman_w,
                "spearman_torch_vs_cuda_cost": spearman_c,
            }
            row["diagnosis_branch"] = branch
            row["input_pose_unchanged"] = (
                row["translation_lon_lat_alt"] == row["output_translation_lon_lat_alt"]
                and row["euler_pitch_roll_yaw"] == row["output_euler_pitch_roll_yaw"]
            )

        finite_nonzero = {
            "torch_feature_loss_total": all(
                row["torch_feature_loss_total"] is not None
                and np.isfinite(float(row["torch_feature_loss_total"]))
                and float(row["torch_feature_loss_total"]) != 0.0
                for row in rows
            ),
            "cuda_w_loss_mean": all(
                row["cuda_w_loss_mean"] is not None
                and np.isfinite(float(row["cuda_w_loss_mean"]))
                and float(row["cuda_w_loss_mean"]) != 0.0
                for row in rows
            ),
            "cuda_cost_mean": all(
                row["cuda_cost_mean"] is not None
                and np.isfinite(float(row["cuda_cost_mean"]))
                and float(row["cuda_cost_mean"]) != 0.0
                for row in rows
            ),
        }

        summary = {
            "config": args.config,
            "query_image": args.query_image,
            "pose_file": args.pose_file,
            "output_dir": os.fspath(output_dir),
            "candidate_count": len(rows),
            "candidates": rows,
            "rank_by_torch_feature_loss": rank_torch,
            "rank_by_cuda_w_loss": rank_cuda_w,
            "rank_by_cuda_cost": rank_cuda_c,
            "rank_by_visual_overlap": rank_overlap,
            "rank_by_visual_chamfer": rank_chamfer,
            "agreement": {
                "cuda_w_loss_matches_torch_top1": cuda_w_top1,
                "cuda_cost_matches_torch_top1": cuda_c_top1,
                "spearman_torch_vs_cuda_w_loss": spearman_w,
                "spearman_torch_vs_cuda_cost": spearman_c,
                "diagnosis_branch": branch,
            },
            "acceptance_checks": {
                "candidate_count_is_6": len(rows) == 6,
                "losses_finite_nonzero": finite_nonzero,
                "optimizer_step_called": optimizer_step_called["value"],
                "all_input_poses_unchanged": all(row["input_pose_unchanged"] for row in rows),
            },
            "cuda": {
                "module_path": getattr(direct_abs_cost_cuda, "__file__", None),
                "has_residual": hasattr(direct_abs_cost_cuda, "residual_jacobian_batch_quat_cuda"),
                "has_step": hasattr(direct_abs_cost_cuda, "optimizer_step_cuda"),
                "optimizer_step_called": optimizer_step_called["value"],
            },
            "torch": {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
            },
            "reference": {
                "translation_lon_lat_alt": trans,
                "euler_pitch_roll_yaw": euler,
                "render_camera_gs": render_camera_gs.tolist(),
                "depth_stats": _depth_stats(depth),
                "points_3d_count": int(p3d.shape[0]),
                "feature_scales_query": _safe_jsonable(scales_q),
                "feature_scales_render": _safe_jsonable(scales_ref),
                "processed_level_order": level_order,
                "per_level_cuda_shapes": per_level,
            },
            "assumptions": {
                "p4_scale_025": "old P4 line_search/scale_0.25_alt_fixed_initial",
                "rebuilt_cuda_refined": "rebuilt P4 raw_refined_full",
                "raw_refined_old": "old P4 raw_refined_full",
                "reference_features_and_p3d": "fixed initial render",
            },
            "total_time_sec": time.time() - start_total,
        }
        _write_json(output_dir / "summary_metrics.json", summary)
        _write_json(output_dir / "run_log.json", {**run_log, "summary_path": os.fspath(output_dir / "summary_metrics.json")})
        print(json.dumps(summary["agreement"], indent=2, sort_keys=True))
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
