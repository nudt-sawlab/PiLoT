#!/usr/bin/env python3
"""Diagnose PiLoT refinement updates around the yawfix initial pose."""

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
import rasterio
import torch
import yaml
from pyproj import Transformer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CUDA_EXT_DIR = REPO_ROOT / "DirectAbsoluteCostCuda"
if CUDA_EXT_DIR.exists() and str(CUDA_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_EXT_DIR))

from pixloc.localization.localizer import RenderLocalizer
from pixloc.pixlib.datasets.view import read_image
from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from pixloc.utils.get_depth import pad_to_multiple
from src.utils.pose_utils import load_initial_pose, load_pose_dict
from tools.run_dom_dsm_single_full import (
    _back_project,
    _depth_stats,
    _format_pose_line,
    _jsonable,
    _resize_query_for_refine,
    _setup_camera,
    _write_json,
)


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test_16x9/0000.jpg"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/yawfix_refinement_diagnosis_results"
BASE_EULER = [0.0, 180.0, 29.2]


def _safe_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return _safe_jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    dist_to_render = cv2.distanceTransform(
        (~render_edges).astype(np.uint8), cv2.DIST_L2, 3
    )
    dist_to_query = cv2.distanceTransform(
        (~query_edges).astype(np.uint8), cv2.DIST_L2, 3
    )
    return float(
        (dist_to_render[query_edges].mean() + dist_to_query[render_edges].mean()) / 2.0
    )


def _edge_overlay(
    query_rgb: np.ndarray,
    render_rgb: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
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
    return np.asarray(_jsonable(value), dtype=np.float64).reshape(-1)


def _read_query_rgb(path: Path, width: int, height: int) -> np.ndarray:
    query_bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if query_bgr is None:
        raise FileNotFoundError(path)
    query_bgr = cv2.resize(query_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)


def _render_candidate(
    name: str,
    renderer: DOMDSMRenderer,
    query_rgb: np.ndarray,
    trans: List[float],
    euler: List[float],
    output_dir: Path,
    checker_tile: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out_dir = output_dir / name
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

    metrics: Dict[str, Any] = {
        "candidate": name,
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


def _get_raster_transformers(config: Dict[str, Any]) -> Tuple[Transformer, Transformer, str]:
    render_config = config["render_config"]
    dom_dsm_config = render_config.get("dom_dsm", {})
    raster_path = (
        render_config.get("dom_path")
        or render_config.get("ortho_path")
        or render_config.get("dsm_path")
        or dom_dsm_config.get("dom_path")
        or dom_dsm_config.get("dsm_path")
    )
    if raster_path is None:
        raise KeyError("render_config must contain dom/dsm raster path")
    raster_path = REPO_ROOT / raster_path if not Path(raster_path).is_absolute() else Path(raster_path)
    with rasterio.open(raster_path) as ds:
        raster_crs = ds.crs
    if raster_crs is None:
        raise ValueError(f"Raster has no CRS: {raster_path}")
    to_raster = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    from_raster = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
    return to_raster, from_raster, str(raster_crs)


def _offset_between(
    a_trans: List[float],
    b_trans: List[float],
    to_raster: Transformer,
) -> List[float]:
    ax, ay = to_raster.transform(float(a_trans[0]), float(a_trans[1]))
    bx, by = to_raster.transform(float(b_trans[0]), float(b_trans[1]))
    return [float(bx - ax), float(by - ay), float(b_trans[2] - a_trans[2])]


def _interpolate_translation(
    initial_trans: List[float],
    refined_trans: List[float],
    scale: float,
    alt_mode: str,
    to_raster: Transformer,
    from_raster: Transformer,
) -> Tuple[List[float], List[float]]:
    ix, iy = to_raster.transform(float(initial_trans[0]), float(initial_trans[1]))
    rx, ry = to_raster.transform(float(refined_trans[0]), float(refined_trans[1]))
    x = ix + scale * (rx - ix)
    y = iy + scale * (ry - iy)
    lon, lat = from_raster.transform(x, y)
    if alt_mode == "fixed_initial":
        alt = float(initial_trans[2])
    elif alt_mode == "scaled_refined":
        alt = float(initial_trans[2]) + scale * (float(refined_trans[2]) - float(initial_trans[2]))
    elif alt_mode == "full_refined":
        alt = float(refined_trans[2])
    else:
        raise ValueError(f"Unknown alt mode: {alt_mode}")
    offsets = [float(x - ix), float(y - iy), float(alt - float(initial_trans[2]))]
    return [float(lon), float(lat), float(alt)], offsets


def _ret_subset(ret: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "T_opt",
        "overall_loss",
        "fail_list",
        "T_refined",
        "diff_R",
        "diff_t",
        "euler_angles",
        "translation",
        "debug_refinement",
    ]
    return {key: _safe_jsonable(ret[key]) for key in keys if key in ret}


def _candidate_improves(candidate: Dict[str, Any], reference: Dict[str, Any]) -> bool:
    return (
        float(candidate["edge_overlap_ratio"]) > float(reference["edge_overlap_ratio"])
        and float(candidate["edge_chamfer"]) < float(reference["edge_chamfer"])
    )


def _candidate_visual_signal(candidate: Dict[str, Any], reference: Dict[str, Any]) -> bool:
    return (
        float(candidate["edge_overlap_ratio"]) > float(reference["edge_overlap_ratio"])
        or float(candidate["edge_chamfer"]) < float(reference["edge_chamfer"])
    )


def _summarize_interpretation(
    initial: Dict[str, Any],
    raw_full: Dict[str, Any],
    line_candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    scaled_candidates = [
        item
        for item in line_candidates
        if float(item.get("scale", 0.0)) in {0.25, 0.5}
    ]
    positive_scale = [item for item in line_candidates if float(item.get("scale", 0.0)) > 0.0]
    scale_one = [item for item in line_candidates if abs(float(item.get("scale", 0.0)) - 1.0) < 1e-9]
    best_line_overlap = max(line_candidates, key=lambda item: float(item["edge_overlap_ratio"]))
    best_line_chamfer = min(line_candidates, key=lambda item: float(item["edge_chamfer"]))
    best_fixed = min(
        [item for item in line_candidates if item.get("alt_mode") == "fixed_initial"],
        key=lambda item: float(item["edge_chamfer"]),
    )
    non_fixed = [item for item in line_candidates if item.get("alt_mode") != "fixed_initial"]
    best_non_fixed = min(non_fixed, key=lambda item: float(item["edge_chamfer"])) if non_fixed else None
    any_scaled_improves = any(_candidate_improves(item, initial) for item in scaled_candidates)
    any_positive_signal = any(_candidate_visual_signal(item, initial) for item in positive_scale)
    scale_one_worse_than_initial = bool(
        scale_one
        and all(not _candidate_improves(item, initial) for item in scale_one)
        and all(
            float(item["edge_overlap_ratio"]) < float(initial["edge_overlap_ratio"])
            or float(item["edge_chamfer"]) > float(initial["edge_chamfer"])
            for item in scale_one
        )
    )
    overshooting = bool(any_scaled_improves and scale_one_worse_than_initial)
    alt_harmful = bool(
        best_non_fixed is not None
        and float(best_fixed["edge_chamfer"]) < float(best_non_fixed["edge_chamfer"])
        and float(best_fixed["edge_overlap_ratio"]) >= float(best_non_fixed["edge_overlap_ratio"])
    )
    return {
        "does_raw_refined_improve_initial": _candidate_improves(raw_full, initial),
        "does_scaled_refined_improve_initial": any_scaled_improves,
        "best_scale": best_line_overlap.get("scale"),
        "best_alt_mode": best_line_overlap.get("alt_mode"),
        "best_line_by_edge_overlap_ratio": best_line_overlap,
        "best_line_by_edge_chamfer": best_line_chamfer,
        "is_update_direction_useful": any_positive_signal,
        "is_update_overshooting": overshooting,
        "is_alt_update_harmful": alt_harmful,
        "scale_1_worse_than_initial": scale_one_worse_than_initial,
        "best_fixed_initial_alt_by_chamfer": best_fixed,
        "best_non_fixed_alt_by_chamfer": best_non_fixed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--line-scales", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--alt-modes", nargs="+", default=["fixed_initial", "scaled_refined", "full_refined"])
    parser.add_argument("--checker-tile", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "refinement_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    log: Dict[str, Any] = {
        "config_path": args.config,
        "query_image_path": args.query_image,
        "pose_file_path": args.pose_file,
        "output_dir": os.fspath(output_dir),
        "renderer_width": args.width,
        "line_scales": args.line_scales,
        "alt_modes": args.alt_modes,
        "failure_stage": None,
        "traceback": None,
    }
    start_total = time.time()
    stage = "start"

    try:
        stage = "load_config"
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        default_confs = config["default_confs"]
        default_confs["cam_query"]["max_size"] = args.width
        refine_conf = default_confs["refine"]
        conf = copy.deepcopy(default_confs["from_render_test"])
        log["checkpoint_path"] = conf.get("checkpoint")

        stage = "setup_camera"
        (
            query_resize_ratio,
            raw_query_camera,
            render_camera_gs,
            query_camera,
            render_camera,
        ) = _setup_camera(config)
        width = int(render_camera_gs[0])
        height = int(render_camera_gs[1])
        log["query_resize_ratio"] = query_resize_ratio
        log["render_camera_gs"] = render_camera_gs

        stage = "load_pose"
        euler, trans, origin_np = load_initial_pose(args.pose_file)
        euler = list(map(float, BASE_EULER))
        trans = list(map(float, trans))
        config["render_config"]["init_rot"] = euler
        config["render_config"]["init_trans"] = trans
        refine_conf["origin"] = origin_np
        gt_pose_dict = load_pose_dict(args.pose_file, origin=origin_np)
        qname = Path(args.query_image).name
        log["initial_translation_lon_lat_alt"] = trans
        log["initial_euler_pitch_roll_yaw"] = euler
        log["pose_format"] = "image_name lon lat alt roll pitch yaw"

        stage = "load_query_image"
        cam_cfg = default_confs["cam_query"]
        query_image = read_image(
            args.query_image,
            scale=query_resize_ratio,
            distortion=cam_cfg["distortion"],
            query_camera=raw_query_camera,
        )
        query_for_visual = _read_query_rgb(REPO_ROOT / args.query_image, width, height)
        log["query_image_shape"] = list(query_image.shape)
        log["query_for_visual_shape"] = list(query_for_visual.shape)

        stage = "init_renderer"
        renderer = DOMDSMRenderer(config["render_config"])
        to_raster, from_raster, raster_crs = _get_raster_transformers(config)
        log["raster_crs"] = raster_crs

        stage = "render_initial"
        initial_metrics = _render_candidate(
            "initial",
            renderer,
            query_for_visual,
            trans,
            euler,
            output_dir,
            args.checker_tile,
            {
                "east_offset_m": 0.0,
                "north_offset_m": 0.0,
                "alt_offset_m": 0.0,
            },
        )

        stage = "render_initial_for_refine"
        t0 = time.time()
        color, depth = renderer.render(trans, euler)
        log["initial_render_time_sec_for_refine"] = time.time() - t0
        log["initial_render_stats_for_refine"] = _depth_stats(depth)
        query_image_for_refine = _resize_query_for_refine(query_image, render_camera_gs)

        stage = "back_project"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        }
        origin = torch.tensor(origin_np, device=device)
        query_camera = query_camera.to(device)
        render_camera = render_camera.to(device)
        t0 = time.time()
        p3d, T_w2c, T_init, dd = _back_project(
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
        log["back_project_time_sec"] = time.time() - t0
        log["points_3d_count"] = int(p3d.shape[0])

        if default_confs.get("padding", False):
            color_for_refine = pad_to_multiple(color, 16)
        else:
            color_for_refine = color
        log["refine_color_shape"] = list(color_for_refine.shape)

        stage = "init_localizer"
        t0 = time.time()
        localizer = RenderLocalizer(conf)
        log["localizer_init_time_sec"] = time.time() - t0

        stage = "run_query"
        last_frame_info = {"observations": [], "refine_conf": refine_conf}
        t0 = time.time()
        ret = localizer.run_query(
            args.query_image,
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
        log["run_query_time_sec"] = time.time() - t0
        log["run_query_success"] = bool(ret.get("success", False))
        log["ret_fields"] = _ret_subset(ret)
        if not ret.get("success", False):
            raise RuntimeError("run_query returned success=False")

        stage = "save_refinement_raw"
        refined_euler = _array(ret["euler_angles"]).tolist()
        refined_trans = _array(ret["translation"]).tolist()
        refined_delta_lon_lat_alt = (_array(refined_trans) - _array(trans)).tolist()
        refined_delta_east_north_alt_m = _offset_between(trans, refined_trans, to_raster)
        log.update(
            {
                "refined_translation_lon_lat_alt": refined_trans,
                "refined_euler_pitch_roll_yaw": refined_euler,
                "refined_delta_lon_lat_alt": refined_delta_lon_lat_alt,
                "refined_delta_east_north_alt_m": refined_delta_east_north_alt_m,
                "diff_R": _safe_jsonable(ret.get("diff_R")),
                "diff_t": _safe_jsonable(ret.get("diff_t")),
                "overall_loss": _safe_jsonable(ret.get("overall_loss")),
                "fail_list": _safe_jsonable(ret.get("fail_list")),
                "selected_candidate_index": _safe_jsonable(ret.get("selected_candidate_index")),
            }
        )
        raw_log = {
            **log,
            "initial_translation_lon_lat_alt": trans,
            "initial_euler_pitch_roll_yaw": euler,
            "raw_ret": _ret_subset(ret),
            "timing": {
                "run_query_time_sec": log["run_query_time_sec"],
                "back_project_time_sec": log["back_project_time_sec"],
                "localizer_init_time_sec": log["localizer_init_time_sec"],
            },
        }
        _write_json(raw_dir / "run_log.json", raw_log)
        (raw_dir / "result_pose.txt").write_text(
            "\n".join(
                [
                    "# initial",
                    _format_pose_line(qname, trans, euler),
                    "# raw_refined",
                    _format_pose_line(qname, refined_trans, refined_euler),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        stage = "render_raw_refined"
        raw_full_metrics = _render_candidate(
            "raw_refined_full",
            renderer,
            query_for_visual,
            refined_trans,
            refined_euler,
            output_dir,
            args.checker_tile,
            {
                "east_offset_m": refined_delta_east_north_alt_m[0],
                "north_offset_m": refined_delta_east_north_alt_m[1],
                "alt_offset_m": refined_delta_east_north_alt_m[2],
            },
        )
        raw_translation_initial_rotation_metrics = _render_candidate(
            "raw_refined_translation_initial_rotation",
            renderer,
            query_for_visual,
            refined_trans,
            euler,
            output_dir,
            args.checker_tile,
            {
                "east_offset_m": refined_delta_east_north_alt_m[0],
                "north_offset_m": refined_delta_east_north_alt_m[1],
                "alt_offset_m": refined_delta_east_north_alt_m[2],
            },
        )

        stage = "line_search"
        line_root = output_dir / "line_search"
        line_candidates = []
        for scale in args.line_scales:
            for alt_mode in args.alt_modes:
                candidate_trans, offsets = _interpolate_translation(
                    trans,
                    refined_trans,
                    float(scale),
                    alt_mode,
                    to_raster,
                    from_raster,
                )
                name = f"line_search/scale_{scale:g}_alt_{alt_mode}"
                metrics = _render_candidate(
                    name,
                    renderer,
                    query_for_visual,
                    candidate_trans,
                    euler,
                    output_dir,
                    args.checker_tile,
                    {
                        "candidate": "line_search",
                        "scale": float(scale),
                        "alt_mode": alt_mode,
                        "east_offset_m": offsets[0],
                        "north_offset_m": offsets[1],
                        "alt_offset_m": offsets[2],
                        "directory": name,
                    },
                )
                line_candidates.append(metrics)
                print(
                    json.dumps(
                        {
                            "scale": scale,
                            "alt_mode": alt_mode,
                            "east": offsets[0],
                            "north": offsets[1],
                            "alt": offsets[2],
                            "overlap": metrics["edge_overlap_ratio"],
                            "chamfer": metrics["edge_chamfer"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        stage = "write_summary"
        all_metrics = [initial_metrics, raw_full_metrics, raw_translation_initial_rotation_metrics, *line_candidates]
        summary = {
            "initial": initial_metrics,
            "raw_refined_full": raw_full_metrics,
            "raw_refined_translation_initial_rotation": raw_translation_initial_rotation_metrics,
            "line_search_candidates": line_candidates,
            "best_by_edge_overlap_ratio": max(
                all_metrics, key=lambda item: float(item["edge_overlap_ratio"])
            ),
            "best_by_edge_chamfer": min(
                all_metrics, key=lambda item: float(item["edge_chamfer"])
            ),
            "refinement_raw": raw_log,
            "interpretation": _summarize_interpretation(
                initial_metrics,
                raw_full_metrics,
                line_candidates,
            ),
            "total_time_sec": time.time() - start_total,
        }
        _write_json(output_dir / "summary_metrics.json", summary)
        print(json.dumps(_safe_jsonable(summary), indent=2, sort_keys=True))
        return 0

    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        log["failure_stage"] = stage
        log["traceback"] = tb
        log["total_time_sec"] = time.time() - start_total
        _write_json(raw_dir / "run_log.json", log)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
