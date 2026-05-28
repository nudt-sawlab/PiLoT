#!/usr/bin/env python3
"""Run DOM/DSM-aware PiLoT refinement variants for one query image."""

import argparse
import copy
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from pixloc.localization.localizer import RenderLocalizer
from pixloc.pixlib.datasets.view import read_image
from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from pixloc.utils.dom_dsm.domdsm_refine import run_domdsm_back_project
from pixloc.utils.dom_dsm.point_sampling import (
    compute_combined_structure_weight,
    compute_dom_gradient_weight,
    compute_dsm_depth_gradient_weight,
    save_sampling_debug,
)
from pixloc.utils.dom_dsm.pose_adapter import compute_enu_delta_m, get_domdsm_transformers
from pixloc.utils.get_depth import pad_to_multiple
from src.utils.pose_utils import load_initial_pose, load_pose_dict
from tools.run_dom_dsm_single_full import (
    _back_project,
    _depth_stats,
    _format_pose_line,
    _resize_query_for_refine,
    _setup_camera,
)

DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test_16x9/0000.jpg"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/domdsm_aware_refinement_results_single"
BASE_EULER = [0.0, 180.0, 29.2]


def _safe_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return _safe_jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items() if k != "weight"}
    if isinstance(value, (list, tuple)):
        return [_safe_jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    cv2.imwrite(os.fspath(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def _make_overlay(query_rgb: np.ndarray, render_rgb: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(query_rgb, 0.5, render_rgb, 0.5, 0)


def _edges(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, 120, 240) > 0


def _symmetric_chamfer(query_edges: np.ndarray, render_edges: np.ndarray) -> float:
    if not np.any(query_edges) or not np.any(render_edges):
        return float("inf")
    dist_to_render = cv2.distanceTransform((~render_edges).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_query = cv2.distanceTransform((~query_edges).astype(np.uint8), cv2.DIST_L2, 3)
    return float((dist_to_render[query_edges].mean() + dist_to_query[render_edges].mean()) / 2.0)


def _edge_overlay(query_rgb: np.ndarray, render_rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    query_edges = _edges(query_rgb)
    render_edges = _edges(render_rgb)
    kernel = np.ones((3, 3), dtype=np.uint8)
    query_dilated = cv2.dilate(query_edges.astype(np.uint8), kernel, iterations=1) > 0
    render_dilated = cv2.dilate(render_edges.astype(np.uint8), kernel, iterations=1) > 0
    overlap = (query_edges & render_dilated) | (render_edges & query_dilated)
    out = _make_overlay(query_rgb, render_rgb)
    out[render_edges] = [255, 40, 40]
    out[query_edges] = [40, 255, 40]
    out[overlap] = [255, 255, 40]
    query_count = int(query_edges.sum())
    render_count = int(render_edges.sum())
    overlap_count = int(overlap.sum())
    return out, {
        "edge_overlap_ratio": float(overlap_count / max(min(query_count, render_count), 1)),
        "edge_chamfer": _symmetric_chamfer(query_edges, render_edges),
        "query_edge_count": query_count,
        "render_edge_count": render_count,
        "edge_overlap_count": overlap_count,
    }


def _checkerboard(query_rgb: np.ndarray, render_rgb: np.ndarray, tile: int) -> np.ndarray:
    height, width = query_rgb.shape[:2]
    yy, xx = np.indices((height, width))
    mask = ((xx // tile) + (yy // tile)) % 2 == 0
    out = render_rgb.copy()
    out[mask] = query_rgb[mask]
    return out


def _array(value: Any) -> np.ndarray:
    return np.asarray(_safe_jsonable(value), dtype=np.float64).reshape(-1)


def _read_query_rgb(path: Path, width: int, height: int) -> np.ndarray:
    query_bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if query_bgr is None:
        raise FileNotFoundError(path)
    query_bgr = cv2.resize(query_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)


def _all_zero_loss(loss: Any) -> Optional[bool]:
    if loss is None:
        return None
    try:
        arr = np.asarray(_safe_jsonable(loss), dtype=float)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return bool(np.allclose(arr, 0.0))


def _weight_for_mode(mode: str, render_rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    valid = (np.isfinite(depth) & (depth > 0)).astype(np.float32)
    if mode == "uniform":
        return valid
    if mode == "dom_gradient":
        return compute_dom_gradient_weight(render_rgb) * valid
    if mode == "depth_gradient":
        return compute_dsm_depth_gradient_weight(depth)
    if mode == "combined":
        return compute_combined_structure_weight(render_rgb, depth)
    raise ValueError(f"Unknown sampling mode: {mode}")


def _render_metrics(
    name: str,
    renderer: DOMDSMRenderer,
    query_rgb: np.ndarray,
    trans: List[float],
    euler: List[float],
    out_dir: Path,
    checker_tile: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    render_rgb, depth = renderer.render(trans, euler)
    render_time = time.perf_counter() - t0
    overlay = _make_overlay(query_rgb, render_rgb)
    edge_overlay, edge_metrics = _edge_overlay(query_rgb, render_rgb)
    checkerboard = _checkerboard(query_rgb, render_rgb, checker_tile)
    _write_rgb(out_dir / "rendered_rgb.png", render_rgb)
    _write_rgb(out_dir / "overlay.png", overlay)
    _write_rgb(out_dir / "edge_overlay.png", edge_overlay)
    _write_rgb(out_dir / "checkerboard.png", checkerboard)
    metrics = {
        "method": name,
        "translation_lon_lat_alt": list(map(float, trans)),
        "euler_pitch_roll_yaw": list(map(float, euler)),
        "render_time_sec": render_time,
        **_depth_stats(depth),
        **edge_metrics,
    }
    if extra:
        metrics.update(extra)
    _write_json(out_dir / "metrics.json", metrics)
    return metrics


def _run_query(
    localizer: RenderLocalizer,
    query_image_path: str,
    query_camera,
    render_camera,
    color_for_refine: np.ndarray,
    T_init,
    T_w2c,
    p3d,
    query_resize_ratio: float,
    dd,
    gt_pose_dict: Dict[str, Any],
    refine_conf: Dict[str, Any],
    query_image_for_refine: np.ndarray,
) -> Tuple[Dict[str, Any], float]:
    last_frame_info = {"observations": [], "refine_conf": refine_conf}
    t0 = time.time()
    ret = localizer.run_query(
        query_image_path,
        query_camera,
        render_camera,
        color_for_refine,
        query_T=T_init,
        render_T=T_w2c,
        Points_3D_ECEF=p3d,
        query_resize_ratio=query_resize_ratio,
        dd=dd,
        gt_pose_dict=gt_pose_dict,
        last_frame_info=last_frame_info,
        image_query=query_image_for_refine,
    )
    return ret, time.time() - t0


def _pose_after_constraints(
    refined_trans: List[float],
    refined_euler: List[float],
    initial_trans: List[float],
    initial_euler: List[float],
    freeze_alt: bool,
    freeze_pitch_roll: bool,
) -> Tuple[List[float], List[float]]:
    eval_trans = list(map(float, refined_trans))
    eval_euler = list(map(float, refined_euler))
    if freeze_alt:
        eval_trans[2] = float(initial_trans[2])
    if freeze_pitch_roll:
        # First DOM/DSM-safe version: keep the yawfix downward euler entirely to avoid refined Euler convention ambiguity.
        eval_euler = list(map(float, initial_euler))
    return eval_trans, eval_euler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--sampling-modes", nargs="+", default=["uniform", "dom_gradient", "depth_gradient", "combined"])
    parser.add_argument("--freeze-alt", action="store_true")
    parser.add_argument("--freeze-pitch-roll", action="store_true")
    parser.add_argument("--checker-tile", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = "start"
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
        width, height = int(render_camera_gs[0]), int(render_camera_gs[1])

        stage = "load_pose"
        _pose_euler, trans, origin_np = load_initial_pose(args.pose_file)
        euler = list(BASE_EULER)
        trans = list(map(float, trans))
        config["render_config"]["init_rot"] = euler
        config["render_config"]["init_trans"] = trans
        refine_conf["origin"] = origin_np
        gt_pose_dict = load_pose_dict(args.pose_file, origin=origin_np)
        to_raster, _from_raster, raster_crs = get_domdsm_transformers(config)

        stage = "load_query"
        cam_cfg = default_confs["cam_query"]
        query_image = read_image(args.query_image, scale=query_resize_ratio, distortion=cam_cfg["distortion"], query_camera=raw_query_camera)
        query_for_visual = _read_query_rgb(REPO_ROOT / args.query_image, width, height)
        query_image_for_refine = _resize_query_for_refine(query_image, render_camera_gs)

        stage = "render_initial"
        renderer = DOMDSMRenderer(config["render_config"])
        render_rgb, depth = renderer.render(trans, euler)
        initial_metrics = _render_metrics(
            "initial", renderer, query_for_visual, trans, euler, output_dir / "initial", args.checker_tile,
            {"sampling": None, "freeze_alt": False, "freeze_pitch_roll": False, "east_offset_m": 0.0, "north_offset_m": 0.0, "alt_offset_m": 0.0},
        )

        stage = "prepare_refine"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        origin = torch.tensor(origin_np, device=device)
        query_camera = query_camera.to(device)
        render_camera = render_camera.to(device)
        color_for_refine = pad_to_multiple(render_rgb, 16) if default_confs.get("padding", False) else render_rgb
        localizer = RenderLocalizer(conf)
        base_log = {
            "config_path": args.config,
            "query_image_path": args.query_image,
            "pose_file_path": args.pose_file,
            "raster_crs": raster_crs,
            "initial_translation_lon_lat_alt": trans,
            "initial_euler_pitch_roll_yaw": euler,
            "render_camera_gs": render_camera_gs,
            "query_resize_ratio": query_resize_ratio,
            "torch": {
                "version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
            },
            "cuda_kernel_warning_observed": "runtime observed on GTX 1080; not captured inside process",
        }

        stage = "baseline_back_project"
        p3d, T_w2c, T_init, dd = _back_project(
            depth, euler, trans, euler, trans, render_camera_gs, render_camera, origin, refine_conf["mul"], device, is_init=True, num_samples=args.num_samples
        )
        ret, run_time = _run_query(localizer, args.query_image, query_camera, render_camera, color_for_refine, T_init, T_w2c, p3d, query_resize_ratio, dd, gt_pose_dict, refine_conf, query_image_for_refine)
        baseline_trans = _array(ret["translation"]).tolist()
        baseline_euler = _array(ret["euler_angles"]).tolist()
        baseline_delta = compute_enu_delta_m(trans, baseline_trans, to_raster)
        baseline_metrics = _render_metrics(
            "baseline_refined", renderer, query_for_visual, baseline_trans, baseline_euler, output_dir / "baseline_refined", args.checker_tile,
            {
                "sampling": "legacy_random",
                "run_query_success": bool(ret.get("success", False)),
                "run_query_time_sec": run_time,
                "points_3d_count": int(p3d.shape[0]),
                "east_offset_m": baseline_delta[0],
                "north_offset_m": baseline_delta[1],
                "alt_offset_m": baseline_delta[2],
                "overall_loss": _safe_jsonable(ret.get("overall_loss")),
                "overall_loss_all_zero": _all_zero_loss(ret.get("overall_loss")),
                "diff_R": _safe_jsonable(ret.get("diff_R")),
                "diff_t": _safe_jsonable(ret.get("diff_t")),
            },
        )
        _write_json(output_dir / "baseline_refined" / "run_log.json", {**base_log, "ret": ret, "metrics": baseline_metrics})

        stage = "sampling_modes"
        sampling_metrics: Dict[str, Any] = {}
        for i, mode in enumerate(args.sampling_modes):
            mode_dir = output_dir / mode
            p3d, T_w2c, T_init, dd, debug = run_domdsm_back_project(
                depth, render_rgb, euler, trans, euler, trans, render_camera_gs, render_camera, origin, refine_conf["mul"], device,
                num_samples=args.num_samples, sampling_mode=mode, is_init=True, seed=args.seed + i,
            )
            weight = debug.pop("weight")
            points2d = torch.as_tensor(debug.get("points2d", []), device=device)
            save_sampling_debug(mode_dir / "sampling_debug.png", render_rgb, depth, weight, points2d)
            ret, run_time = _run_query(localizer, args.query_image, query_camera, render_camera, color_for_refine, T_init, T_w2c, p3d, query_resize_ratio, dd, gt_pose_dict, refine_conf, query_image_for_refine)
            refined_trans = _array(ret["translation"]).tolist()
            refined_euler = _array(ret["euler_angles"]).tolist()
            eval_trans, eval_euler = _pose_after_constraints(refined_trans, refined_euler, trans, euler, args.freeze_alt, args.freeze_pitch_roll)
            raw_delta = compute_enu_delta_m(trans, refined_trans, to_raster)
            eval_delta = compute_enu_delta_m(trans, eval_trans, to_raster)
            metrics = _render_metrics(
                mode, renderer, query_for_visual, eval_trans, eval_euler, mode_dir, args.checker_tile,
                {
                    "sampling": mode,
                    "freeze_alt": bool(args.freeze_alt),
                    "freeze_pitch_roll": bool(args.freeze_pitch_roll),
                    "run_query_success": bool(ret.get("success", False)),
                    "run_query_time_sec": run_time,
                    "points_3d_count": int(p3d.shape[0]),
                    "raw_refined_translation_lon_lat_alt": refined_trans,
                    "raw_refined_euler_pitch_roll_yaw": refined_euler,
                    "raw_delta_east_north_alt_m": raw_delta,
                    "east_offset_m": eval_delta[0],
                    "north_offset_m": eval_delta[1],
                    "alt_offset_m": eval_delta[2],
                    "sampling_debug": debug,
                    "overall_loss": _safe_jsonable(ret.get("overall_loss")),
                    "overall_loss_all_zero": _all_zero_loss(ret.get("overall_loss")),
                    "diff_R": _safe_jsonable(ret.get("diff_R")),
                    "diff_t": _safe_jsonable(ret.get("diff_t")),
                },
            )
            _write_json(mode_dir / "run_log.json", {**base_log, "sampling_debug": debug, "ret": ret, "metrics": metrics})
            sampling_metrics[mode] = metrics
            print(json.dumps({"mode": mode, "overlap": metrics["edge_overlap_ratio"], "chamfer": metrics["edge_chamfer"], "delta": eval_delta}, sort_keys=True), flush=True)

        all_metrics = [initial_metrics, baseline_metrics, *sampling_metrics.values()]
        best_by_chamfer = min(all_metrics, key=lambda item: float(item["edge_chamfer"]))
        best_by_overlap = max(all_metrics, key=lambda item: float(item["edge_overlap_ratio"]))
        summary = {
            "initial": initial_metrics,
            "baseline_refined": baseline_metrics,
            "sampling_modes": sampling_metrics,
            "best_by_chamfer": best_by_chamfer,
            "best_by_overlap": best_by_overlap,
            "interpretation": {
                "structure_sampling_best_method": best_by_chamfer.get("method"),
                "structure_sampling_improves_baseline_chamfer": any(m["edge_chamfer"] < baseline_metrics["edge_chamfer"] for m in sampling_metrics.values()),
                "structure_sampling_improves_initial_chamfer": any(m["edge_chamfer"] < initial_metrics["edge_chamfer"] for m in sampling_metrics.values()),
                "freeze_alt_enabled": bool(args.freeze_alt),
                "freeze_pitch_roll_enabled": bool(args.freeze_pitch_roll),
                "baseline_loss_all_zero": baseline_metrics.get("overall_loss_all_zero"),
                "cuda_loss_trustworthy": False,
            },
            "base_log": base_log,
        }
        _write_json(output_dir / "summary_metrics.json", summary)
        print(json.dumps(_safe_jsonable(summary), indent=2, sort_keys=True))
        return 0
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        _write_json(output_dir / "failure.json", {"failure_stage": stage, "traceback": tb})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
