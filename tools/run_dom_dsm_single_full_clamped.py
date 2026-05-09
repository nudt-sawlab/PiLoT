#!/usr/bin/env python3
"""Single-image DOM/DSM refinement with post-refinement pose clamp diagnostics."""

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
    _save_depth_png,
    _save_overlay,
    _setup_camera,
    _write_json,
)


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test_16x9/0000.jpg"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt"
DEFAULT_OUTPUT_DIR = "outputs/exif_test_16x9_yawfix_clamped"


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
    return float((dist_to_render[query_edges].mean() + dist_to_query[render_edges].mean()) / 2.0)


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


def _clamp_pose_delta(
    initial_trans: List[float],
    initial_euler: List[float],
    refined_trans: Any,
    refined_euler: Any,
    max_trans_delta: np.ndarray,
    max_euler_delta: np.ndarray,
) -> Tuple[List[float], List[float]]:
    init_t = _array(initial_trans)
    init_e = _array(initial_euler)
    refined_t = _array(refined_trans)
    refined_e = _array(refined_euler)
    clamped_t = init_t + np.clip(refined_t - init_t, -max_trans_delta, max_trans_delta)
    clamped_e = init_e + np.clip(refined_e - init_e, -max_euler_delta, max_euler_delta)
    return clamped_t.tolist(), clamped_e.tolist()


def _strategy_poses(
    initial_trans: List[float],
    initial_euler: List[float],
    refined_trans: Any,
    refined_euler: Any,
    max_trans_delta: np.ndarray,
    max_euler_delta: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    clamped_trans, clamped_euler = _clamp_pose_delta(
        initial_trans,
        initial_euler,
        refined_trans,
        refined_euler,
        max_trans_delta,
        max_euler_delta,
    )
    return {
        "keep_initial": {
            "description": "Do not accept any refinement update.",
            "translation_lon_lat_alt": list(map(float, initial_trans)),
            "euler_pitch_roll_yaw": list(map(float, initial_euler)),
        },
        "raw_refined": {
            "description": "Use the raw pose returned by localizer.run_query().",
            "translation_lon_lat_alt": _array(refined_trans).tolist(),
            "euler_pitch_roll_yaw": _array(refined_euler).tolist(),
        },
        "clamp_translation_only": {
            "description": "Accept translation update only; keep initial euler.",
            "translation_lon_lat_alt": _array(refined_trans).tolist(),
            "euler_pitch_roll_yaw": list(map(float, initial_euler)),
        },
        "clamp_rotation_only": {
            "description": "Accept euler update only; keep initial translation.",
            "translation_lon_lat_alt": list(map(float, initial_trans)),
            "euler_pitch_roll_yaw": _array(refined_euler).tolist(),
        },
        "clamp_small_delta": {
            "description": (
                "Clamp lon/lat/alt and pitch/roll/yaw deltas around the initial pose."
            ),
            "translation_lon_lat_alt": clamped_trans,
            "euler_pitch_roll_yaw": clamped_euler,
        },
    }


def _render_strategy(
    name: str,
    strategy: Dict[str, Any],
    renderer: DOMDSMRenderer,
    query_rgb: np.ndarray,
    output_dir: Path,
    checker_tile: int,
) -> Dict[str, Any]:
    strategy_dir = output_dir / name
    strategy_dir.mkdir(parents=True, exist_ok=True)
    trans = strategy["translation_lon_lat_alt"]
    euler = strategy["euler_pitch_roll_yaw"]

    t0 = time.time()
    render_rgb, depth = renderer.render(trans, euler)
    render_time = time.time() - t0

    overlay = _make_overlay(query_rgb, render_rgb)
    edge_overlay, edge_metrics = _edge_overlay(query_rgb, render_rgb)
    checkerboard = _checkerboard(query_rgb, render_rgb, checker_tile)

    _write_rgb(strategy_dir / "rendered_rgb.png", render_rgb)
    _write_rgb(strategy_dir / "overlay.png", overlay)
    _write_rgb(strategy_dir / "edge_overlay.png", edge_overlay)
    _write_rgb(strategy_dir / "checkerboard.png", checkerboard)

    metrics = {
        "strategy": name,
        "description": strategy["description"],
        "translation_lon_lat_alt": trans,
        "euler_pitch_roll_yaw": euler,
        "render_time_sec": render_time,
        **_depth_stats(depth),
        **edge_metrics,
    }
    _write_json(strategy_dir / "metrics.json", metrics)
    return metrics


def parse_delta(value: str, expected: int, name: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != expected:
        raise argparse.ArgumentTypeError(f"{name} must contain {expected} comma-separated values")
    return np.asarray(parts, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checker-tile", default=32, type=int)
    parser.add_argument("--max-trans-delta", default="0.00002,0.00002,1.0")
    parser.add_argument("--max-euler-delta", default="5.0,5.0,5.0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_trans_delta = parse_delta(args.max_trans_delta, 3, "--max-trans-delta")
    max_euler_delta = parse_delta(args.max_euler_delta, 3, "--max-euler-delta")

    run_log_path = output_dir / "run_log.json"
    result_pose_path = output_dir / "result_pose.txt"
    start_total = time.time()
    stage = "start"
    log: Dict[str, Any] = {
        "config_path": args.config,
        "query_image_path": args.query_image,
        "pose_file_path": args.pose_file,
        "output_dir": os.fspath(output_dir),
        "failure_stage": None,
        "traceback": None,
        "clamp_thresholds": {
            "max_trans_delta_lon_lat_alt": max_trans_delta.tolist(),
            "max_euler_delta_pitch_roll_yaw": max_euler_delta.tolist(),
        },
    }

    try:
        stage = "load_config"
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        default_confs = config["default_confs"]
        refine_conf = default_confs["refine"]
        conf = default_confs["from_render_test"]
        log["checkpoint_path"] = conf.get("checkpoint")

        stage = "setup_camera"
        (
            query_resize_ratio,
            raw_query_camera,
            render_camera_gs,
            query_camera,
            render_camera,
        ) = _setup_camera(config)
        log["query_resize_ratio"] = query_resize_ratio
        log["render_camera_gs"] = render_camera_gs

        stage = "load_pose"
        euler, trans, origin_np = load_initial_pose(args.pose_file)
        config["render_config"]["init_rot"] = euler
        config["render_config"]["init_trans"] = trans
        refine_conf["origin"] = origin_np
        gt_pose_dict = load_pose_dict(args.pose_file, origin=origin_np)
        qname = Path(args.query_image).name
        log["initial_pose"] = {
            "image_name": qname,
            "translation_lon_lat_alt": trans,
            "euler_pitch_roll_yaw": euler,
            "pose_format": "image_name lon lat alt roll pitch yaw",
        }

        stage = "load_query_image"
        cam_cfg = default_confs["cam_query"]
        query_image = read_image(
            args.query_image,
            scale=query_resize_ratio,
            distortion=cam_cfg["distortion"],
            query_camera=raw_query_camera,
        )
        log["query_image_shape"] = list(query_image.shape)

        stage = "init_renderer"
        renderer = DOMDSMRenderer(config["render_config"])

        stage = "render_initial"
        t0 = time.time()
        color, depth = renderer.render(trans, euler)
        log["render_time_sec"] = time.time() - t0
        log["render_stats"] = _depth_stats(depth)
        log["render_shape"] = list(color.shape)
        cv2.imwrite(
            os.fspath(output_dir / "rendered_rgb.png"),
            cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
        )
        _save_depth_png(output_dir / "rendered_depth.png", depth)
        _save_overlay(output_dir / "query_render_overlay.png", query_image, color)
        query_image_for_refine = _resize_query_for_refine(query_image, render_camera_gs)
        log["query_image_for_refine_shape"] = list(query_image_for_refine.shape)

        stage = "back_project"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
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
        if not ret.get("success", False):
            raise RuntimeError("run_query returned success=False")

        stage = "postprocess_strategies"
        refined_euler = ret["euler_angles"]
        refined_trans = ret["translation"]
        log["raw_refined_pose"] = {
            "image_name": qname,
            "translation_lon_lat_alt": refined_trans,
            "euler_pitch_roll_yaw": refined_euler,
            "pose_format": "image_name lon lat alt roll pitch yaw",
        }
        initial_output = np.array(
            [*np.asarray(trans, dtype=float), euler[1], euler[0], euler[2]],
            dtype=float,
        )
        refined_output = np.array(
            [
                *_array(refined_trans),
                _array(refined_euler)[1],
                _array(refined_euler)[0],
                _array(refined_euler)[2],
            ],
            dtype=float,
        )
        log["raw_pose_delta_lon_lat_alt_roll_pitch_yaw"] = refined_output - initial_output

        strategies = _strategy_poses(
            trans,
            euler,
            refined_trans,
            refined_euler,
            max_trans_delta,
            max_euler_delta,
        )
        query_for_visual = _resize_query_for_refine(query_image, render_camera_gs)
        strategy_metrics = []
        result_lines = ["# initial", _format_pose_line(qname, trans, euler)]
        for name, strategy in strategies.items():
            metrics = _render_strategy(
                name,
                strategy,
                renderer,
                query_for_visual,
                output_dir,
                args.checker_tile,
            )
            strategy_metrics.append(metrics)
            result_lines.extend(
                [
                    f"# {name}",
                    _format_pose_line(
                        qname,
                        strategy["translation_lon_lat_alt"],
                        strategy["euler_pitch_roll_yaw"],
                    ),
                ]
            )
        log["strategy_metrics"] = strategy_metrics
        log["best_by_edge_overlap_ratio"] = max(
            strategy_metrics,
            key=lambda item: item["edge_overlap_ratio"],
        )["strategy"]
        log["best_by_edge_chamfer"] = min(
            strategy_metrics,
            key=lambda item: item["edge_chamfer"],
        )["strategy"]
        _write_json(output_dir / "summary_metrics.json", log)
        result_pose_path.write_text("\n".join(result_lines) + "\n", encoding="utf-8")
        log["total_time_sec"] = time.time() - start_total
        _write_json(run_log_path, log)
        print(json.dumps(_jsonable(log), indent=2, sort_keys=True))
        return 0

    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        log["failure_stage"] = stage
        log["traceback"] = tb
        log["total_time_sec"] = time.time() - start_total
        result_pose_path.write_text(
            "\n".join(["# failure", f"stage: {stage}", tb]) + "\n",
            encoding="utf-8",
        )
        _write_json(run_log_path, log)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
