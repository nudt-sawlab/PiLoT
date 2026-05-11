#!/usr/bin/env python3
"""Compare fixed DOM/DSM candidate poses with PyTorch-only PiLoT feature loss."""

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.localization.localizer import RenderLocalizer
from pixloc.pixlib.datasets.view import read_image
from pixloc.pixlib.geometry import Pose
from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from pixloc.utils.dom_dsm.domdsm_refine import run_domdsm_back_project
from pixloc.utils.dom_dsm.feature_loss_debug import (
    compute_feature_residual_loss,
    extract_pilot_features,
    save_residual_debug_visualization,
)
from pixloc.utils.dom_dsm.pose_adapter import apply_enu_offset, get_domdsm_transformers
from pixloc.utils.get_depth import _euler_to_matrix_ecef_batch, pad_to_multiple
from pixloc.utils.transform import WGS84_to_ECEF
from src.utils.pose_utils import load_initial_pose
from tools.run_dom_dsm_single_full import _depth_stats, _resize_query_for_refine, _setup_camera

DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test_16x9/0000.jpg"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/torch_feature_loss_diagnosis"
BASE_EULER = [0.0, 180.0, 29.2]

CANDIDATES = [
    ("initial", [0.0, 0.0, 0.0], BASE_EULER),
    ("p4_scale_025_fixed_alt", [-3.301955, -0.526292, 0.0], BASE_EULER),
    ("p3_best_chamfer", [-5.0, 0.0, 0.0], BASE_EULER),
    ("p3_best_overlap", [-5.0, 5.0, 0.0], BASE_EULER),
    ("raw_refined_translation_fixed_alt", [-13.2078, -2.1052, 0.0], BASE_EULER),
    ("raw_refined_translation_full_alt", [-13.2078, -2.1052, 4.097], BASE_EULER),
    ("raw_refined_full", [-13.2078, -2.1052, 4.097], [180.0, 0.0, -144.8]),
]


def _jsonable(v: Any) -> Any:
    if torch.is_tensor(v): return _jsonable(v.detach().cpu().numpy())
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.floating, np.integer)): return v.item()
    if isinstance(v, dict): return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_jsonable(x) for x in v]
    try:
        json.dumps(v); return v
    except TypeError:
        return repr(v)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(os.fspath(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _read_query_rgb(path: Path, width: int, height: int) -> np.ndarray:
    bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(path)
    bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _overlay(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(q, 0.5, r, 0.5, 0)


def _edges(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, 120, 240) > 0


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if not np.any(a) or not np.any(b): return float("inf")
    da = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
    return float((db[a].mean() + da[b].mean()) / 2.0)


def _edge_overlay(q: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    qe, re = _edges(q), _edges(r)
    kernel = np.ones((3, 3), dtype=np.uint8)
    overlap = (qe & (cv2.dilate(re.astype(np.uint8), kernel, iterations=1) > 0)) | (re & (cv2.dilate(qe.astype(np.uint8), kernel, iterations=1) > 0))
    out = _overlay(q, r)
    out[re] = [255, 40, 40]
    out[qe] = [40, 255, 40]
    out[overlap] = [255, 255, 40]
    qc, rc, oc = int(qe.sum()), int(re.sum()), int(overlap.sum())
    return out, {"edge_overlap_ratio": float(oc / max(min(qc, rc), 1)), "edge_chamfer": _chamfer(qe, re), "query_edge_count": qc, "render_edge_count": rc, "edge_overlap_count": oc}


def _checker(q: np.ndarray, r: np.ndarray, tile: int = 32) -> np.ndarray:
    yy, xx = np.indices(q.shape[:2])
    mask = ((xx // tile) + (yy // tile)) % 2 == 0
    out = r.copy(); out[mask] = q[mask]
    return out


def _build_query_pose(trans: Sequence[float], euler: Sequence[float], base_trans: Sequence[float], origin: torch.Tensor, mul: float, dd: torch.Tensor, device: str) -> Pose:
    euler_t = torch.tensor([euler], device=device, dtype=torch.float32)
    ecef = torch.tensor([WGS84_to_ECEF(trans)], device=device, dtype=torch.float32)
    T_c2w = _euler_to_matrix_ecef_batch(euler_t, ecef, list(base_trans), device=device)
    T_c2w[:, :3, 1] *= -1
    T_c2w[:, :3, 2] *= -1
    if mul is not None:
        T_c2w[:, :3, 3] *= float(mul)
        origin_scaled = origin * float(mul)
    else:
        origin_scaled = origin
    T_c2w[:, :3, 3] -= origin_scaled
    T_w2c = Pose.from_Rt(T_c2w[:, :3, :3], T_c2w[:, :3, 3]).inv()
    tt = T_w2c.t + T_w2c.R @ dd
    return Pose.from_Rt(T_w2c.R, tt)[0]


def _rank(items: List[Dict[str, Any]], key: str, reverse: bool = False) -> List[str]:
    return [x["candidate"] for x in sorted(items, key=lambda m: float(m[key]), reverse=reverse)]


def _spearman(x: List[float], y: List[float]) -> float:
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        for rank, i in enumerate(order): r[i] = float(rank)
        return r
    rx, ry = np.asarray(ranks(x)), np.asarray(ranks(y))
    if len(rx) < 2 or np.std(rx) == 0 or np.std(ry) == 0: return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    p.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-points", type=int, default=500)
    p.add_argument("--sampling-mode", default="combined")
    return p.parse_args()


def run_diagnosis(args, candidates=CANDIDATES, clean_output=True) -> Dict[str, Any]:
    os.chdir(REPO_ROOT)
    out_root = (REPO_ROOT / args.output_dir).resolve()
    if clean_output and out_root.exists(): shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(args.config, "r", encoding="utf-8") as f: config = yaml.safe_load(f)
    default_confs = config["default_confs"]
    default_confs["cam_query"]["max_size"] = args.width
    conf = copy.deepcopy(default_confs["from_render_test"])
    refine_conf = default_confs["refine"]
    query_resize_ratio, raw_query_camera, render_camera_gs, query_camera, render_camera = _setup_camera(config)
    width, height = int(render_camera_gs[0]), int(render_camera_gs[1])
    _pose_euler, base_trans, origin_np = load_initial_pose(args.pose_file)
    base_trans = list(map(float, base_trans)); base_euler = list(BASE_EULER)
    refine_conf["origin"] = origin_np
    config["render_config"]["init_rot"] = base_euler
    config["render_config"]["init_trans"] = base_trans
    to_raster, from_raster, raster_crs = get_domdsm_transformers(config)
    renderer = DOMDSMRenderer(config["render_config"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    origin = torch.tensor(origin_np, device=device)
    query_camera = query_camera.to(device); render_camera = render_camera.to(device)
    query_image = read_image(args.query_image, scale=query_resize_ratio, distortion=default_confs["cam_query"]["distortion"], query_camera=raw_query_camera)
    query_refine = _resize_query_for_refine(query_image, render_camera_gs)
    query_refine = pad_to_multiple(query_refine, 16) if default_confs.get("padding", False) else query_refine
    query_visual = _read_query_rgb(REPO_ROOT / args.query_image, width, height)
    localizer = RenderLocalizer(conf)
    feat_q = extract_pilot_features(localizer, query_refine, "query")
    metrics = []
    for name, offset, euler in candidates:
        cand_dir = out_root / name; cand_dir.mkdir(parents=True, exist_ok=True)
        trans = apply_enu_offset(base_trans, offset[0], offset[1], offset[2], to_raster, from_raster)
        t0 = time.perf_counter(); render_rgb, depth = renderer.render(trans, euler); render_time = time.perf_counter() - t0
        _write_rgb(cand_dir / "rendered_rgb.png", render_rgb)
        _write_rgb(cand_dir / "overlay.png", _overlay(query_visual, render_rgb))
        eo, em = _edge_overlay(query_visual, render_rgb); _write_rgb(cand_dir / "edge_overlay.png", eo)
        _write_rgb(cand_dir / "checkerboard.png", _checker(query_visual, render_rgb))
        p3d, T_render, _unused, dd, bp_debug = run_domdsm_back_project(depth, render_rgb, euler, trans, euler, trans, render_camera_gs, render_camera, origin, refine_conf["mul"], device, num_samples=args.num_points, sampling_mode=args.sampling_mode, is_init=False, seed=0)
        T_query = _build_query_pose(trans, euler, base_trans, origin, refine_conf["mul"], dd, device)
        render_refine = pad_to_multiple(render_rgb, 16) if default_confs.get("padding", False) else render_rgb
        feat_r = extract_pilot_features(localizer, render_refine, "render")
        loss = compute_feature_residual_loss(feat_q["features"], feat_r["features"], feat_q["scales"], feat_r["scales"], p3d, T_query, T_render, query_camera, render_camera)
        save_residual_debug_visualization(cand_dir, query_visual, render_rgb, loss["points_query"], loss["points_render"], loss["residual_per_point"], loss["valid_mask"])
        m = {"candidate": name, "east_offset_m": offset[0], "north_offset_m": offset[1], "alt_offset_m": offset[2], "euler_pitch_roll_yaw": euler, "translation_lon_lat_alt": trans, "render_time_sec": render_time, **_depth_stats(depth), **em, "torch_feature_loss": loss["loss_total"], "loss_by_level": loss["loss_by_level"], "num_valid_by_level": loss["num_valid_by_level"], "valid_ratio_by_level": loss["valid_ratio_by_level"]}
        _write_json(cand_dir / "metrics.json", m)
        metrics.append(m)
        print(json.dumps({"candidate": name, "loss": m["torch_feature_loss"], "chamfer": m["edge_chamfer"], "overlap": m["edge_overlap_ratio"]}, sort_keys=True), flush=True)
    ranks_chamfer = _rank(metrics, "edge_chamfer")
    ranks_overlap = _rank(metrics, "edge_overlap_ratio", reverse=True)
    ranks_loss = _rank(metrics, "torch_feature_loss")
    loss_vals = [m["torch_feature_loss"] for m in metrics]
    chamfer_vals = [m["edge_chamfer"] for m in metrics]
    overlap_vals = [m["edge_overlap_ratio"] for m in metrics]
    by_name = {m["candidate"]: m for m in metrics}
    aligned = _spearman(loss_vals, chamfer_vals) > 0.3 and _spearman(loss_vals, [-x for x in overlap_vals]) > 0.3
    has_raw = "raw_refined_translation_full_alt" in by_name and "raw_refined_full" in by_name and "initial" in by_name
    rejects_raw = None
    if has_raw:
        rejects_raw = (
            by_name["raw_refined_translation_full_alt"]["torch_feature_loss"] > by_name["initial"]["torch_feature_loss"]
            and by_name["raw_refined_full"]["torch_feature_loss"] > by_name["initial"]["torch_feature_loss"]
        )
    summary = {
        "config": args.config,
        "query_image": args.query_image,
        "pose_file": args.pose_file,
        "raster_crs": raster_crs,
        "candidates": metrics,
        "rank_by_visual_chamfer": ranks_chamfer,
        "rank_by_visual_overlap": ranks_overlap,
        "rank_by_torch_feature_loss": ranks_loss,
        "correlation": {
            "spearman_feature_loss_vs_chamfer": _spearman(loss_vals, chamfer_vals),
            "spearman_feature_loss_vs_overlap": _spearman(loss_vals, overlap_vals),
        },
        "interpretation": {
            "does_feature_loss_prefer_small_offset": ranks_loss[0] in {"p4_scale_025_fixed_alt", "p3_best_chamfer", "p3_best_overlap"},
            "does_feature_loss_reject_raw_refined": rejects_raw,
            "is_feature_loss_aligned_with_visual_metric": bool(aligned),
            "likely_failure_mode": "cuda_optimizer_path_invalid" if aligned else "feature_domain_or_pose_metric_mismatch",
        },
    }
    _write_json(out_root / "summary_metrics.json", summary)
    return summary


def main():
    args = parse_args()
    run_diagnosis(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
