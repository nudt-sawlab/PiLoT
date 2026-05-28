#!/usr/bin/env python3
"""Record rebuilt CUDA optimizer trajectories for DOM/DSM PiLoT refinement."""

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
from pixloc.localization.base_refiner import build_world_c2w_batch
from pixloc.localization.localizer import RenderLocalizer
from pixloc.pixlib.datasets.view import read_image
from pixloc.pixlib.geometry import Pose
from pixloc.pixlib.geometry.costs import DirectAbsoluteCost2
from pixloc.utils.dom_dsm.feature_loss_debug import compute_feature_residual_loss
from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from pixloc.utils.get_depth import pad_to_multiple, zero_pad
from pixloc.utils.transform import pixloc_to_osg
from src.utils.pose_utils import load_initial_pose
from tools.compare_cuda_torch_feature_loss_fixed_poses import (
    _prepare_refiner_features,
    _tensor_stats_by_candidate,
)
from tools.diagnose_yawfix_refinement_update import (
    BASE_EULER,
    _edge_overlay,
    _get_raster_transformers,
    _offset_between,
    _read_query_rgb,
    _safe_jsonable,
    _write_rgb,
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
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/rebuilt_cuda_optimizer_trajectory_p8"
DEFAULT_REBUILT_P4_SUMMARY = "docs/experiments/dom_dsm_prepare/yawfix_refinement_diagnosis_results_rebuilt_sm61/summary_metrics.json"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_list(value: Any) -> List[float]:
    return [float(x) for x in np.asarray(_safe_jsonable(value), dtype=np.float64).reshape(-1)]


def _stack_pose_items(items: Sequence[Pose]) -> Pose:
    return Pose(torch.cat([item._data for item in items], dim=0))


def _ensure_batched_pose(pose: Pose) -> Pose:
    if len(pose.shape) == 0:
        return pose.unsqueeze(0)
    return pose


def _pose_to_osg_rows(
    poses: Pose,
    dd: torch.Tensor,
    scale_mul: float,
    origin: torch.Tensor,
) -> List[Dict[str, Any]]:
    poses = _ensure_batched_pose(poses)
    c2w_batch = build_world_c2w_batch(poses, dd, scale_mul, origin)
    rows = []
    for mat in c2w_batch.detach().cpu().numpy():
        euler, trans, _t_ecef, _kf = pixloc_to_osg(mat)
        rows.append(
            {
                "translation_lon_lat_alt": _as_list(trans),
                "euler_pitch_roll_yaw": _as_list(euler),
                "yaw_deg": float(np.asarray(euler).reshape(-1)[2]),
            }
        )
    return rows


def _render_visual_record(
    name: str,
    renderer: DOMDSMRenderer,
    query_rgb: np.ndarray,
    trans: Sequence[float],
    euler: Sequence[float],
    output_dir: Path,
    save_images: bool,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    render_rgb, depth = renderer.render(list(map(float, trans)), list(map(float, euler)))
    edge_overlay, edge_metrics = _edge_overlay(query_rgb, render_rgb)
    metrics: Dict[str, Any] = {
        "render_time_sec": time.perf_counter() - t0,
        **_depth_stats(depth),
        **edge_metrics,
    }
    if save_images:
        out_dir = output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_rgb(out_dir / "rendered_rgb.png", render_rgb)
        _write_rgb(out_dir / "edge_overlay.png", edge_overlay)
    return metrics


def _make_inputs(
    poses: Pose,
    p3d: torch.Tensor,
    f_ref: torch.Tensor,
    f_query: torch.Tensor,
    qcamera_tensor: torch.Tensor,
    render_pose: Pose,
    rcamera_tensor: torch.Tensor,
    c_ref: torch.Tensor,
    c_query: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    poses = _ensure_batched_pose(poses)
    num_poses = poses.shape[0]
    return {
        "pose_data_q": poses.to_flat().unsqueeze(0).contiguous().clone(),
        "f_r": f_ref.unsqueeze(0).clone(),
        "pose_data_r": render_pose.to_flat().expand(1, -1).unsqueeze(0).clone(),
        "cam_data_r": rcamera_tensor.unsqueeze(0).expand(1, -1).unsqueeze(0).clone(),
        "f_q": f_query.unsqueeze(0).clone(),
        "cam_data_q": qcamera_tensor.unsqueeze(0).expand(num_poses, -1).unsqueeze(0).clone(),
        "p3D": p3d.unsqueeze(0).expand(1, -1, -1).unsqueeze(0).contiguous().clone(),
        "c_ref": c_ref.unsqueeze(0).clone(),
        "c_query": c_query.unsqueeze(0).clone(),
    }


def _eval_cuda_and_torch_cost2(
    opt: Any,
    torch_cost: DirectAbsoluteCost2,
    poses: Pose,
    p3d: torch.Tensor,
    f_ref: torch.Tensor,
    f_query: torch.Tensor,
    qcamera_tensor: torch.Tensor,
    render_pose: Pose,
    rcamera_tensor: torch.Tensor,
    c_ref: torch.Tensor,
    c_query: torch.Tensor,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    inputs = _make_inputs(
        poses, p3d, f_ref, f_query, qcamera_tensor, render_pose, rcamera_tensor, c_ref, c_query
    )
    g, H, w_loss, cost = opt.fn(
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
    g, H, w_loss, cost = [
        x.squeeze(0) if x.shape[0] == 1 else x for x in [g, H, w_loss, cost]
    ]
    _tg, _tH, torch_w_loss, valid_torch, _p2d_q, torch_cost_raw = torch_cost.residual_jacobian_batch_quat(
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
    num_poses = poses.shape[0]
    cuda_w_stats, _ = _tensor_stats_by_candidate(w_loss, num_poses)
    cuda_c_stats, _ = _tensor_stats_by_candidate(cost, num_poses)
    torch_w_stats, _ = _tensor_stats_by_candidate(torch_w_loss, num_poses)
    torch_c_stats, _ = _tensor_stats_by_candidate(torch_cost_raw, num_poses)
    valid_stats, _ = _tensor_stats_by_candidate(valid_torch, num_poses)
    rows = []
    for i in range(num_poses):
        rows.append(
            {
                "candidate_index": i,
                "cuda_w_loss_mean": cuda_w_stats[i]["mean"],
                "cuda_w_loss_sum": cuda_w_stats[i]["sum"],
                "cuda_cost_mean": cuda_c_stats[i]["mean"],
                "cuda_cost_sum": cuda_c_stats[i]["sum"],
                "torch_cost2_w_loss_mean": torch_w_stats[i]["mean"],
                "torch_cost2_cost_mean": torch_c_stats[i]["mean"],
                "torch_cost2_valid_count": valid_stats[i]["nonzero_count"],
            }
        )
    return {
        "candidate_losses": rows,
        "best_by_cuda_w_loss_index": min(rows, key=lambda item: float(item["cuda_w_loss_mean"]))["candidate_index"],
        "best_by_cuda_cost_index": min(rows, key=lambda item: float(item["cuda_cost_mean"]))["candidate_index"],
        "g_shape": list(g.shape),
        "H_shape": list(H.shape),
        "w_loss_shape": list(w_loss.shape),
        "cost_shape": list(cost.shape),
    }, {"g": g, "H": H}


def _torch_feature_loss_for_pose(
    pose: Pose,
    features_q_raw: Sequence[torch.Tensor],
    features_ref_raw: Sequence[torch.Tensor],
    scales_q: Sequence[Any],
    scales_ref: Sequence[Any],
    p3d: torch.Tensor,
    t_render: Pose,
    query_camera: Any,
    render_camera: Any,
) -> float:
    loss = compute_feature_residual_loss(
        features_q_raw,
        features_ref_raw,
        scales_q,
        scales_ref,
        p3d,
        pose,
        t_render,
        query_camera,
        render_camera,
        levels=None,
        robust="l2",
        use_confidence=False,
    )
    return float(loss["loss_total"])


def _trajectory_record(
    trajectory_name: str,
    global_iteration: int,
    level: Optional[int],
    local_iteration: Optional[int],
    poses: Pose,
    losses: Dict[str, Any],
    pose_rows: List[Dict[str, Any]],
    initial_trans: Sequence[float],
    to_raster: Any,
    renderer: DOMDSMRenderer,
    query_rgb: np.ndarray,
    visual_root: Path,
    save_images: bool,
    features_q_raw: Sequence[torch.Tensor],
    features_ref_raw: Sequence[torch.Tensor],
    scales_q: Sequence[Any],
    scales_ref: Sequence[Any],
    p3d: torch.Tensor,
    t_render: Pose,
    query_camera: Any,
    render_camera: Any,
    delta_summary: Optional[Dict[str, Any]] = None,
    topk_event: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    poses = _ensure_batched_pose(poses)
    best_idx = int(losses["best_by_cuda_w_loss_index"])
    best_pose = pose_rows[best_idx]
    visual = _render_visual_record(
        f"{trajectory_name}/iter_{global_iteration:03d}_level_{level}_best_{best_idx}",
        renderer,
        query_rgb,
        best_pose["translation_lon_lat_alt"],
        best_pose["euler_pitch_roll_yaw"],
        visual_root,
        save_images,
    )
    feature_loss = _torch_feature_loss_for_pose(
        poses[best_idx],
        features_q_raw,
        features_ref_raw,
        scales_q,
        scales_ref,
        p3d,
        t_render,
        query_camera,
        render_camera,
    )
    best_loss = losses["candidate_losses"][best_idx]
    offsets = _offset_between(initial_trans, best_pose["translation_lon_lat_alt"], to_raster)
    return {
        "trajectory": trajectory_name,
        "global_iteration": global_iteration,
        "level": level,
        "local_iteration": local_iteration,
        "candidate_count": len(losses["candidate_losses"]),
        "best_by_cuda_w_loss_index": best_idx,
        "best_by_cuda_cost_index": int(losses["best_by_cuda_cost_index"]),
        "best_pose": {
            **best_pose,
            "east_m": offsets[0],
            "north_m": offsets[1],
            "alt_offset_m": offsets[2],
        },
        "best_metrics": {
            **best_loss,
            "torch_feature_loss": feature_loss,
            "visual_overlap": visual["edge_overlap_ratio"],
            "visual_chamfer": visual["edge_chamfer"],
        },
        "candidate_losses": losses["candidate_losses"],
        "delta_summary": delta_summary,
        "topk_event": topk_event,
    }


def _delta_summary(delta: torch.Tensor) -> Dict[str, Any]:
    delta = delta.detach().float()
    dw, dt = delta.split([3, 3], dim=-1)
    return {
        "rotation_delta_norm_mean": float(torch.linalg.norm(dw, dim=-1).mean().item()),
        "rotation_delta_norm_max": float(torch.linalg.norm(dw, dim=-1).max().item()),
        "translation_delta_norm_mean": float(torch.linalg.norm(dt, dim=-1).mean().item()),
        "translation_delta_norm_max": float(torch.linalg.norm(dt, dim=-1).max().item()),
        "delta_shape": list(delta.shape),
    }


def _run_trajectory(
    trajectory_name: str,
    initial_poses: Pose,
    opt_getter: Any,
    features_q: Sequence[torch.Tensor],
    features_ref: Sequence[torch.Tensor],
    weights_q: Sequence[torch.Tensor],
    weights_ref: Sequence[torch.Tensor],
    features_q_raw: Sequence[torch.Tensor],
    features_ref_raw: Sequence[torch.Tensor],
    scales_q: Sequence[Any],
    scales_ref: Sequence[Any],
    p3d: torch.Tensor,
    t_render: Pose,
    query_camera: Any,
    render_camera: Any,
    dd: torch.Tensor,
    scale_mul: float,
    origin: torch.Tensor,
    initial_trans: Sequence[float],
    to_raster: Any,
    renderer: DOMDSMRenderer,
    query_rgb: np.ndarray,
    output_dir: Path,
    save_images: bool,
) -> Tuple[List[Dict[str, Any]], Pose, Dict[str, Any]]:
    torch_cost = DirectAbsoluteCost2()
    trajectory: List[Dict[str, Any]] = []
    T = initial_poses
    failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)
    global_iter = 0
    topk_events: List[Dict[str, Any]] = []
    level_order = list(reversed(range(len(features_q))))

    for level in level_order:
        f_q = features_q[level]
        f_ref = features_ref[level]
        qcam_lvl = query_camera.scale(scales_q[level]).to_tensor().to(f_q)
        rcam_lvl = render_camera.scale(scales_ref[level]).to_tensor().to(f_q)
        opt = opt_getter(level)
        n_iters = {0: 4, 1: 3, 2: 2}.get(level, 2)

        for local_iter in range(n_iters + 1):
            T = T.to(f_q)
            losses, linear = _eval_cuda_and_torch_cost2(
                opt,
                torch_cost,
                T,
                p3d,
                f_ref,
                f_q,
                qcam_lvl,
                t_render.to(f_q),
                rcam_lvl,
                weights_ref[level].to(f_q),
                weights_q[level].to(f_q),
            )
            pose_rows = _pose_to_osg_rows(T, dd, scale_mul, origin)
            record = _trajectory_record(
                trajectory_name,
                global_iter,
                level,
                local_iter,
                T,
                losses,
                pose_rows,
                initial_trans,
                to_raster,
                renderer,
                query_rgb,
                output_dir,
                save_images,
                features_q_raw,
                features_ref_raw,
                scales_q,
                scales_ref,
                p3d,
                t_render,
                query_camera,
                render_camera,
            )
            trajectory.append(record)
            if local_iter == n_iters:
                break

            g = linear["g"].unsqueeze(-1)
            delta = opt.optimizer_cuda(g, linear["H"], 0.1, ~failed)
            dw, dt = delta.split([3, 3], dim=-1)
            T_delta = Pose.from_aa(dw, dt)
            T = T @ T_delta
            global_iter += 1
            trajectory[-1]["delta_summary"] = _delta_summary(delta)

        if n_iters == 2 and T.shape[0] > opt.num_filter_pose:
            final_losses = torch.tensor(
                [float(item["cuda_w_loss_mean"]) for item in trajectory[-1]["candidate_losses"]],
                device=T.device,
            )
            _, topk_indices = torch.topk(-final_losses, opt.num_filter_pose, dim=-1, largest=True, sorted=True)
            event = {
                "after_level": level,
                "before_count": int(T.shape[0]),
                "after_count": int(opt.num_filter_pose),
                "selected_indices": [int(x) for x in topk_indices.detach().cpu().tolist()],
            }
            topk_events.append(event)
            trajectory[-1]["topk_event"] = event
            T = T[topk_indices]
            failed = failed[topk_indices]

    return trajectory, T, {"topk_events": topk_events, "level_order": level_order}


def _summarize_trajectory(name: str, trajectory: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    first = trajectory[0]
    last = trajectory[-1]
    values = [item["best_metrics"] for item in trajectory]
    feature_start = float(values[0]["torch_feature_loss"])
    feature_end = float(values[-1]["torch_feature_loss"])
    cuda_start = float(values[0]["cuda_w_loss_mean"])
    cuda_end = float(values[-1]["cuda_w_loss_mean"])
    chamfer_start = float(values[0]["visual_chamfer"])
    chamfer_end = float(values[-1]["visual_chamfer"])
    overlap_start = float(values[0]["visual_overlap"])
    overlap_end = float(values[-1]["visual_overlap"])
    best_visual_idx = min(range(len(values)), key=lambda i: float(values[i]["visual_chamfer"]))
    return {
        "trajectory": name,
        "num_records": len(trajectory),
        "start": first["best_pose"],
        "end": last["best_pose"],
        "feature_loss_start": feature_start,
        "feature_loss_end": feature_end,
        "cuda_w_loss_start": cuda_start,
        "cuda_w_loss_end": cuda_end,
        "visual_chamfer_start": chamfer_start,
        "visual_chamfer_end": chamfer_end,
        "visual_overlap_start": overlap_start,
        "visual_overlap_end": overlap_end,
        "feature_loss_decreased": feature_end < feature_start,
        "cuda_w_loss_decreased": cuda_end < cuda_start,
        "visual_chamfer_improved": chamfer_end < chamfer_start,
        "visual_overlap_improved": overlap_end > overlap_start,
        "best_visual_chamfer_iteration": int(best_visual_idx),
        "best_visual_chamfer": float(values[best_visual_idx]["visual_chamfer"]),
    }


def _diagnose(primary: Dict[str, Any]) -> str:
    feature_down = bool(primary["feature_loss_decreased"])
    visual_worse = (
        float(primary["visual_chamfer_end"]) > float(primary["visual_chamfer_start"])
        or float(primary["visual_overlap_end"]) < float(primary["visual_overlap_start"])
    )
    early_best = int(primary["best_visual_chamfer_iteration"])
    has_overshoot = (
        feature_down
        and early_best > 0
        and early_best < int(primary["num_records"]) - 1
        and float(primary["visual_chamfer_end"]) > float(primary["best_visual_chamfer"])
    )
    if not feature_down:
        return "case_b_feature_not_down"
    if has_overshoot:
        return "case_c_overshoot"
    if visual_worse:
        return "case_a_feature_down_visual_worse"
    return "feature_and_visual_both_improve"


def _final_pose_comparison(
    final_pose: Pose,
    rebuilt_summary_path: Path,
    dd: torch.Tensor,
    scale_mul: float,
    origin: torch.Tensor,
    to_raster: Any,
) -> Dict[str, Any]:
    expected = _load_json(rebuilt_summary_path).get("raw_refined_full", {})
    actual = _pose_to_osg_rows(final_pose, dd, scale_mul, origin)[0]
    if not expected:
        return {"status": "missing_expected_raw_refined_full", "actual": actual}
    offset = _offset_between(
        expected["translation_lon_lat_alt"],
        actual["translation_lon_lat_alt"],
        to_raster,
    )
    euler_delta = (
        np.asarray(actual["euler_pitch_roll_yaw"], dtype=np.float64)
        - np.asarray(expected["euler_pitch_roll_yaw"], dtype=np.float64)
    ).tolist()
    mismatch = (
        max(abs(x) for x in offset) > 1.0
        or max(abs(float(x)) for x in euler_delta) > 1.0
    )
    return {
        "status": "p4_reproduction_mismatch" if mismatch else "p4_reproduction_close",
        "actual": actual,
        "expected": {
            "translation_lon_lat_alt": expected["translation_lon_lat_alt"],
            "euler_pitch_roll_yaw": expected["euler_pitch_roll_yaw"],
        },
        "delta_east_north_alt_m": offset,
        "delta_euler_deg": euler_delta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rebuilt-p4-summary", default=DEFAULT_REBUILT_P4_SUMMARY)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--checker-tile", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
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
        _loaded_euler, trans, origin_np = load_initial_pose(args.pose_file)
        euler = list(map(float, BASE_EULER))
        trans = list(map(float, trans))
        config["render_config"]["init_rot"] = euler
        config["render_config"]["init_trans"] = trans
        refine_conf["origin"] = origin_np

        stage = "load_query"
        cam_cfg = default_confs["cam_query"]
        query_image = read_image(
            args.query_image,
            scale=query_resize_ratio,
            distortion=cam_cfg["distortion"],
            query_camera=raw_query_camera,
        )
        query_for_visual = _read_query_rgb(REPO_ROOT / args.query_image, width, height)
        query_image_for_refine = _resize_query_for_refine(query_image, render_camera_gs)

        stage = "render_reference"
        renderer = DOMDSMRenderer(config["render_config"])
        to_raster, _from_raster, raster_crs = _get_raster_transformers(config)
        color, depth = renderer.render(trans, euler)
        color_for_refine = pad_to_multiple(color, 16) if default_confs.get("padding", False) else color

        stage = "back_project"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError("P8 trajectory diagnosis requires CUDA")
        origin = torch.tensor(origin_np, device=device)
        query_camera = query_camera.to(device)
        render_camera = render_camera.to(device)
        p3d, t_render, t_grid_init, dd = _back_project(
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
        t_single_init = t_grid_init[0:1]

        stage = "init_localizer_extract_features"
        localizer = RenderLocalizer(conf)
        refiner = localizer.refiner
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
            raise RuntimeError("P8 requires uncertainty weights for CUDA optimizer path")

        opt = refiner.optimizer
        if isinstance(opt, (list, tuple)):
            opt_getter = lambda level: opt[refiner.conf.layer_indices[level]] if refiner.conf.layer_indices else opt[level]
        else:
            opt_getter = lambda _level: opt

        stage = "run_single_initial"
        single_traj, single_final, single_meta = _run_trajectory(
            "single_initial",
            t_single_init,
            opt_getter,
            features_q,
            features_ref,
            weights_q,
            weights_ref,
            features_q_raw,
            features_ref_raw,
            scales_q,
            scales_ref,
            p3d,
            t_render,
            query_camera,
            render_camera,
            dd,
            refine_conf["mul"],
            origin,
            trans,
            to_raster,
            renderer,
            query_for_visual,
            output_dir,
            not args.skip_images,
        )

        stage = "run_p4_grid"
        p4_traj, p4_final, p4_meta = _run_trajectory(
            "p4_grid",
            t_grid_init,
            opt_getter,
            features_q,
            features_ref,
            weights_q,
            weights_ref,
            features_q_raw,
            features_ref_raw,
            scales_q,
            scales_ref,
            p3d,
            t_render,
            query_camera,
            render_camera,
            dd,
            refine_conf["mul"],
            origin,
            trans,
            to_raster,
            renderer,
            query_for_visual,
            output_dir,
            not args.skip_images,
        )

        stage = "summarize"
        single_summary = _summarize_trajectory("single_initial", single_traj)
        p4_summary = _summarize_trajectory("p4_grid", p4_traj)
        final_loss = torch.tensor(
            [float(item["cuda_w_loss_mean"]) for item in p4_traj[-1]["candidate_losses"]],
            device=p4_final.device,
        )
        final_best_idx = int(torch.argmin(final_loss).item())
        selected_records = [
            item
            for item in p4_traj
            if int(item["best_by_cuda_w_loss_index"]) == int(item["best_by_cuda_w_loss_index"])
        ]
        selected_trajectory = [
            {
                "global_iteration": item["global_iteration"],
                "level": item["level"],
                "local_iteration": item["local_iteration"],
                "best_by_cuda_w_loss_index": item["best_by_cuda_w_loss_index"],
                "best_pose": item["best_pose"],
                "best_metrics": item["best_metrics"],
                "delta_summary": item.get("delta_summary"),
                "topk_event": item.get("topk_event"),
            }
            for item in selected_records
        ]
        p4_reproduction = _final_pose_comparison(
            p4_final[final_best_idx],
            REPO_ROOT / args.rebuilt_p4_summary,
            dd,
            refine_conf["mul"],
            origin,
            to_raster,
        )
        diagnosis_branch = _diagnose(p4_summary)

        summary = {
            "config": args.config,
            "query_image": args.query_image,
            "pose_file": args.pose_file,
            "output_dir": os.fspath(output_dir),
            "diagnosis_branch": diagnosis_branch,
            "single_initial": {**single_summary, **single_meta},
            "p4_grid": {**p4_summary, **p4_meta},
            "p4_reproduction": p4_reproduction,
            "cuda": {
                "module_path": getattr(direct_abs_cost_cuda, "__file__", None),
                "has_residual": hasattr(direct_abs_cost_cuda, "residual_jacobian_batch_quat_cuda"),
                "has_step": hasattr(direct_abs_cost_cuda, "optimizer_step_cuda"),
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
                "raster_crs": raster_crs,
                "points_3d_count": int(p3d.shape[0]),
                "initial_grid_candidate_count": int(t_grid_init.shape[0]),
            },
            "acceptance_checks": {
                "has_single_initial_iteration_0": bool(single_traj and single_traj[0]["global_iteration"] == 0),
                "has_p4_levels_2_1_0": sorted({int(item["level"]) for item in p4_traj}) == [0, 1, 2],
                "p4_has_topk_event": bool(p4_meta["topk_events"]),
                "records_have_required_metrics": all(
                    all(k in item["best_metrics"] for k in ["cuda_w_loss_mean", "cuda_cost_mean", "torch_feature_loss", "visual_overlap", "visual_chamfer"])
                    for item in single_traj + p4_traj
                ),
            },
            "total_time_sec": time.time() - start_total,
        }
        _write_json(output_dir / "trajectory_single_initial.json", {"trajectory": single_traj})
        _write_json(output_dir / "trajectory_p4_grid.json", {"trajectory": p4_traj})
        _write_json(output_dir / "selected_candidate_trajectory.json", {"trajectory": selected_trajectory})
        _write_json(output_dir / "summary_metrics.json", summary)
        _write_json(output_dir / "run_log.json", {**run_log, "summary_path": os.fspath(output_dir / "summary_metrics.json")})
        print(json.dumps({
            "diagnosis_branch": diagnosis_branch,
            "p4_reproduction": p4_reproduction["status"],
            "single_records": len(single_traj),
            "p4_records": len(p4_traj),
        }, indent=2, sort_keys=True))
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
