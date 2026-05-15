#!/usr/bin/env python3
"""Run a safe PiLoT refinement gate for one DOM/DSM query."""

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
from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from pixloc.utils.get_depth import pad_to_multiple
from src.utils.pose_utils import load_initial_pose, load_pose_dict
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


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test_16x9/0000.jpg"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/safe_pilot_refinement_p9"
EPS = 1e-9


def _normalize_yaw(yaw: float) -> float:
    return float(((float(yaw) + 180.0) % 360.0) - 180.0)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _visual_passes(candidate: Dict[str, Any], initial: Dict[str, Any]) -> bool:
    return (
        float(candidate["edge_chamfer"]) <= float(initial["edge_chamfer"]) + EPS
        and float(candidate["edge_overlap_ratio"]) + EPS >= float(initial["edge_overlap_ratio"])
    )


def _visual_worse(candidate: Dict[str, Any], initial: Dict[str, Any]) -> bool:
    return (
        float(candidate["edge_chamfer"]) > float(initial["edge_chamfer"]) + EPS
        or float(candidate["edge_overlap_ratio"]) + EPS < float(initial["edge_overlap_ratio"])
    )


def _improves_over(candidate: Dict[str, Any], reference: Dict[str, Any]) -> bool:
    return (
        float(candidate["edge_chamfer"]) < float(reference["edge_chamfer"]) - EPS
        and float(candidate["edge_overlap_ratio"]) > float(reference["edge_overlap_ratio"]) + EPS
    )


def _metric_delta(candidate: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, float]:
    return {
        "edge_chamfer_delta": float(candidate["edge_chamfer"]) - float(reference["edge_chamfer"]),
        "edge_overlap_ratio_delta": float(candidate["edge_overlap_ratio"]) - float(reference["edge_overlap_ratio"]),
    }


def _candidate_extra(
    source_method: str,
    trans: List[float],
    initial_trans: List[float],
    to_raster: Any,
) -> Dict[str, Any]:
    offsets = _offset_between(initial_trans, trans, to_raster)
    return {
        "source_method": source_method,
        "east_offset_m": offsets[0],
        "north_offset_m": offsets[1],
        "alt_offset_m": offsets[2],
    }


def _select_safe_candidate(
    initial_metrics: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any], List[str]]:
    passing = [item for item in candidates if _visual_passes(item, initial_metrics)]
    passing_names = [str(item["method"]) for item in passing]
    if not passing:
        return "initial", initial_metrics, passing_names
    selected = sorted(
        passing,
        key=lambda item: (float(item["edge_chamfer"]), -float(item["edge_overlap_ratio"])),
    )[0]
    return str(selected["method"]), selected, passing_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
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

        stage = "init_renderer"
        renderer = DOMDSMRenderer(config["render_config"])
        to_raster, _from_raster, raster_crs = _get_raster_transformers(config)

        stage = "render_initial"
        initial_metrics = _render_candidate(
            "initial",
            renderer,
            query_for_visual,
            initial_trans,
            initial_euler,
            output_dir,
            args.checker_tile,
            {
                "method": "initial",
                "source_method": "pose_file_yawfix_initial",
                "east_offset_m": 0.0,
                "north_offset_m": 0.0,
                "alt_offset_m": 0.0,
            },
        )

        stage = "render_initial_for_refine"
        t0 = time.time()
        color, depth = renderer.render(initial_trans, initial_euler)
        query_image_for_refine = _resize_query_for_refine(query_image, render_camera_gs)

        stage = "back_project"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        origin = torch.tensor(origin_np, device=device)
        query_camera = query_camera.to(device)
        render_camera = render_camera.to(device)
        p3d, T_w2c, T_init, dd = _back_project(
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
        color_for_refine = pad_to_multiple(color, 16) if default_confs.get("padding", False) else color

        stage = "run_rebuilt_cuda_refinement"
        localizer = RenderLocalizer(conf)
        last_frame_info = {"observations": [], "refine_conf": refine_conf}
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
        if not ret.get("success", False):
            raise RuntimeError("run_query returned success=False")

        stage = "build_methods"
        refined_euler = _array(ret["euler_angles"]).tolist()
        refined_trans = _array(ret["translation"]).tolist()
        downward_refined_yaw = _normalize_yaw(float(refined_euler[2]) + 180.0)
        freeze_alt_trans = [float(refined_trans[0]), float(refined_trans[1]), float(initial_trans[2])]
        freeze_alt_euler = list(map(float, refined_euler))
        freeze_alt_pitch_roll_trans = list(freeze_alt_trans)
        freeze_alt_pitch_roll_euler = [
            float(initial_euler[0]),
            float(initial_euler[1]),
            downward_refined_yaw,
        ]

        methods = [
            {
                "method": "rebuilt_cuda_raw_refined",
                "trans": refined_trans,
                "euler": refined_euler,
                "source_method": "rebuilt_cuda_optimizer_raw_output",
            },
            {
                "method": "refined_freeze_alt",
                "trans": freeze_alt_trans,
                "euler": freeze_alt_euler,
                "source_method": "raw_refined_with_initial_alt",
            },
            {
                "method": "refined_freeze_alt_pitch_roll",
                "trans": freeze_alt_pitch_roll_trans,
                "euler": freeze_alt_pitch_roll_euler,
                "source_method": "raw_refined_lon_lat_downward_yaw_with_initial_alt_pitch_roll",
            },
        ]

        stage = "render_refined_methods"
        method_metrics: Dict[str, Dict[str, Any]] = {"initial": initial_metrics}
        candidate_metrics = []
        for item in methods:
            metrics = _render_candidate(
                item["method"],
                renderer,
                query_for_visual,
                item["trans"],
                item["euler"],
                output_dir,
                args.checker_tile,
                {
                    "method": item["method"],
                    **_candidate_extra(item["source_method"], item["trans"], initial_trans, to_raster),
                },
            )
            metrics["passes_acceptance_gate"] = _visual_passes(metrics, initial_metrics)
            metrics["worse_than_initial"] = _visual_worse(metrics, initial_metrics)
            metrics["delta_vs_initial"] = _metric_delta(metrics, initial_metrics)
            method_metrics[item["method"]] = metrics
            candidate_metrics.append(metrics)

        stage = "select_safe_gate"
        selected_method, selected_metrics, passing_methods = _select_safe_candidate(
            initial_metrics,
            candidate_metrics,
        )
        selected_trans = selected_metrics["translation_lon_lat_alt"]
        selected_euler = selected_metrics["euler_pitch_roll_yaw"]
        safe_metrics = _render_candidate(
            "safe_refined_acceptance_gate",
            renderer,
            query_for_visual,
            selected_trans,
            selected_euler,
            output_dir,
            args.checker_tile,
            {
                "method": "safe_refined_acceptance_gate",
                "selected_source_method": selected_method,
                "passing_refined_methods": passing_methods,
                **_candidate_extra(selected_method, selected_trans, initial_trans, to_raster),
            },
        )
        safe_metrics["passes_acceptance_gate"] = _visual_passes(safe_metrics, initial_metrics)
        safe_metrics["worse_than_initial"] = _visual_worse(safe_metrics, initial_metrics)
        safe_metrics["delta_vs_initial"] = _metric_delta(safe_metrics, initial_metrics)
        method_metrics["safe_refined_acceptance_gate"] = safe_metrics

        stage = "write_outputs"
        raw_metrics = method_metrics["rebuilt_cuda_raw_refined"]
        freeze_alt_metrics = method_metrics["refined_freeze_alt"]
        freeze_alt_pitch_roll_metrics = method_metrics["refined_freeze_alt_pitch_roll"]
        raw_worse = _visual_worse(raw_metrics, initial_metrics)
        safe_worse = _visual_worse(safe_metrics, initial_metrics)
        safe_pilot_success = (
            not safe_worse
            and (not raw_worse or selected_method != "rebuilt_cuda_raw_refined" or _visual_passes(raw_metrics, initial_metrics))
        )

        result_pose_path = output_dir / "result_pose_safe.txt"
        result_pose_path.write_text(
            "\n".join(
                [
                    "# safe_refined_acceptance_gate",
                    f"# selected_source_method: {selected_method}",
                    _format_pose_line(qname, selected_trans, selected_euler),
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
            "result_pose_safe_path": os.fspath(result_pose_path),
            "methods": method_metrics,
            "raw_refinement": {
                "translation_lon_lat_alt": refined_trans,
                "euler_pitch_roll_yaw": refined_euler,
                "raw_refined_yaw": float(refined_euler[2]),
                "downward_refined_yaw": downward_refined_yaw,
                "yaw_conversion_applied": True,
                "ret_fields": _ret_subset(ret),
                "initial_render_time_sec_for_refine": time.time() - t0,
            },
            "gate": {
                "rule": "edge_chamfer <= initial_edge_chamfer + 1e-9 and edge_overlap_ratio + 1e-9 >= initial_edge_overlap_ratio",
                "passing_refined_methods": passing_methods,
                "safe_gate_selected_method": selected_method,
            },
            "conclusions": {
                "raw_refined_worse_than_initial": raw_worse,
                "freeze_alt_improves_raw": _improves_over(freeze_alt_metrics, raw_metrics),
                "freeze_alt_pitch_roll_improves_raw": _improves_over(freeze_alt_pitch_roll_metrics, raw_metrics),
                "freeze_alt_delta_vs_raw": _metric_delta(freeze_alt_metrics, raw_metrics),
                "freeze_alt_pitch_roll_delta_vs_raw": _metric_delta(freeze_alt_pitch_roll_metrics, raw_metrics),
                "safe_gate_selected_method": selected_method,
                "safe_output_worse_than_initial": safe_worse,
                "safe_pilot_success": safe_pilot_success,
                "avoids_feature_down_visual_worse_failure": bool(raw_worse and not safe_worse),
            },
            "acceptance_checks": {
                "summary_has_5_methods": len(method_metrics) == 5,
                "safe_gate_chamfer_not_worse": float(safe_metrics["edge_chamfer"]) <= float(initial_metrics["edge_chamfer"]) + EPS,
                "safe_gate_overlap_not_worse": float(safe_metrics["edge_overlap_ratio"]) + EPS >= float(initial_metrics["edge_overlap_ratio"]),
                "result_pose_safe_exists": result_pose_path.exists(),
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
            },
            "total_time_sec": time.time() - start_total,
        }
        _write_json(output_dir / "summary_metrics.json", summary)
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
