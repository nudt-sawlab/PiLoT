#!/usr/bin/env python3
"""Scan small XML pose rotation residuals with SIFT residual fields."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import yaml
from pyproj import Transformer
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from tools.diagnose_yawfix_refinement_update import _checkerboard, _edge_overlay, _safe_jsonable, _write_rgb
from tools.render_contextcapture_xml_domdsm_initial import _load_pose_file_projected, _match_photos, _parse_xml


DEFAULT_XML = "data_caiwangcun/CaiWangCun.xml"
DEFAULT_QUERY_DIR = "data_caiwangcun/query/images/exif_test"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_CONFIG = "configs/caiwangcun_domdsm.yaml"
DEFAULT_PATCH_BATCH_DIR = "docs/experiments/dom_dsm_prepare/query_dom_patch_alignment_batch"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/query_render_pose_residual_grid"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SINGLE_AXIS_DELTAS = [-0.10, -0.05, 0.0, 0.05, 0.10]
PITCH_ROLL_DELTAS = [-0.05, 0.0, 0.05]


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row.keys() if not isinstance(row.get(k), (dict, list))})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_rgb(path: Path, scale: float = 1.0) -> np.ndarray:
    bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if scale != 1.0:
        rgb = cv2.resize(rgb, (int(round(rgb.shape[1] * scale)), int(round(rgb.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return rgb


def _list_images(query_dir: Path, limit: Optional[int]) -> List[Path]:
    images = sorted(p for p in query_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return images[:limit] if limit and limit > 0 else images


def _render_path_for_baseline(batch_dir: Path, idx: int, image_path: Path) -> Path:
    return batch_dir / "images" / f"{idx:03d}_{image_path.stem}" / "gpu_pinhole_render.png"


def _delta_key(dp: float, dr: float, dy: float) -> str:
    return f"dp{dp:+.2f}_dr{dr:+.2f}_dy{dy:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _candidate_deltas() -> List[Tuple[float, float, float]]:
    vals = {(0.0, 0.0, 0.0)}
    for d in SINGLE_AXIS_DELTAS:
        vals.add((float(d), 0.0, 0.0))
        vals.add((0.0, float(d), 0.0))
        vals.add((0.0, 0.0, float(d)))
    for dp in PITCH_ROLL_DELTAS:
        for dr in PITCH_ROLL_DELTAS:
            vals.add((float(dp), float(dr), 0.0))
    return sorted(vals, key=lambda x: (abs(x[0]) + abs(x[1]) + abs(x[2]), x[0], x[1], x[2]))


def _make_gpu_config(config: Dict[str, Any], intr: Any) -> Dict[str, Any]:
    render_config = copy.deepcopy(config["render_config"])
    render_config["render_camera"] = [
        int(intr.width),
        int(intr.height),
        float(intr.cx),
        float(intr.cy),
        float(intr.fx),
        float(intr.fy),
    ]
    render_config["dom_dsm"]["render_backend"] = "gpu_mesh"
    render_config["dom_dsm"]["gpu_renderer"] = "nvdiffrast"
    render_config["dom_dsm"]["texture_v_flip"] = False
    render_config["dom_dsm"]["output_y_flip"] = True
    render_config["dom_dsm"]["debug_texture_mode"] = "none"
    render_config["dom_dsm"]["debug_every"] = 0
    return render_config


def _trans_wgs84(xml_srs: str, photo: Any) -> List[float]:
    to_wgs84 = Transformer.from_crs(xml_srs, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(photo.center_xml[0], photo.center_xml[1])
    return [float(lon), float(lat), float(photo.center_xml[2])]


def _perturbed_rotation(photo: Any, dp: float, dr: float, dy: float) -> np.ndarray:
    r_delta = Rotation.from_euler("xyz", [float(dp), float(dr), float(dy)], degrees=True).as_matrix()
    return photo.rotation.T @ r_delta


def _render_candidate(
    config: Dict[str, Any],
    xml_srs: str,
    photo: Any,
    dp: float,
    dr: float,
    dy: float,
    path: Path,
) -> Tuple[np.ndarray, Dict[str, Any], float]:
    if path.exists():
        return _read_rgb(path), {"source": os.fspath(path), "rendered_now": False}, 0.0
    renderer = DOMDSMRenderer(_make_gpu_config(config, photo.intrinsics))
    t0 = time.perf_counter()
    color, _depth = renderer.render_matrix(_trans_wgs84(xml_srs, photo), _perturbed_rotation(photo, dp, dr, dy))
    elapsed = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_rgb(path, color)
    meta = dict(renderer.last_render_metadata)
    meta.update({"source": os.fspath(path), "rendered_now": True})
    return color, meta, float(elapsed)


def _detect_sift(gray: np.ndarray) -> Tuple[Any, Any]:
    sift = cv2.SIFT_create(nfeatures=9000)
    return sift.detectAndCompute(gray, None)


def _fit_residual(src: np.ndarray, residual: np.ndarray, shape: Tuple[int, int], inv_scale: float) -> Dict[str, Any]:
    h, w = shape
    src_full = src * inv_scale
    res_full = residual * inv_scale
    center = np.asarray([w * inv_scale / 2.0, h * inv_scale / 2.0], dtype=np.float64)
    rel = src_full - center
    radius = np.linalg.norm(rel, axis=1)
    norm = np.linalg.norm(res_full, axis=1)
    dx, dy = res_full[:, 0], res_full[:, 1]
    out: Dict[str, Any] = {
        "median_dx_px": float(np.median(dx)),
        "median_dy_px": float(np.median(dy)),
        "median_residual_norm_px": float(np.median(norm)),
        "p90_residual_norm_px": float(np.percentile(norm, 90)),
        "direction_consistency": float(np.hypot(dx.mean(), dy.mean()) / max(norm.mean(), 1e-9)),
    }
    if len(res_full) >= 8:
        out["corr_dx_x"] = float(np.corrcoef(src_full[:, 0], dx)[0, 1])
        out["corr_dy_y"] = float(np.corrcoef(src_full[:, 1], dy)[0, 1])
        out["corr_norm_radius"] = float(np.corrcoef(radius, norm)[0, 1])
    else:
        out.update({"corr_dx_x": None, "corr_dy_y": None, "corr_norm_radius": None})
    return out


def _residual_metrics(query_rgb: np.ndarray, render_rgb: np.ndarray, query_features: Tuple[Any, Any], scale: float) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray], Optional[List[Any]], Any, Any]:
    q_gray = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2GRAY)
    r_gray = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2GRAY)
    kq, dq = query_features
    kr, dr = _detect_sift(r_gray)
    row: Dict[str, Any] = {"query_keypoints": len(kq), "render_keypoints": len(kr)}
    if dq is None or dr is None or len(kq) < 8 or len(kr) < 8:
        row.update({"status": "insufficient_features", "good_matches": 0, "inliers": 0})
        return row, None, None, None, kq, kr
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(dq, dr, k=2)
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    row["good_matches"] = len(good)
    if len(good) < 8:
        row.update({"status": "insufficient_matches", "inliers": 0})
        return row, None, None, good, kq, kr
    src = np.float32([kq[m.queryIdx].pt for m in good])
    dst = np.float32([kr[m.trainIdx].pt for m in good])
    _H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    inlier_mask = mask.ravel().astype(bool) if mask is not None else np.zeros(len(good), dtype=bool)
    src_i = src[inlier_mask]
    dst_i = dst[inlier_mask]
    residual = dst_i - src_i
    row["inliers"] = int(inlier_mask.sum())
    row["inlier_ratio"] = float(inlier_mask.mean())
    row["status"] = "ok" if row["inliers"] >= 8 else "too_few_inliers"
    if row["status"] == "ok":
        row.update(_fit_residual(src_i, residual, q_gray.shape[:2], 1.0 / scale))
    return row, src_i, residual, [good[i] for i, keep in enumerate(inlier_mask) if keep], kq, kr


def _edge_metrics_scaled(query_rgb: np.ndarray, render_rgb: np.ndarray) -> Dict[str, Any]:
    _edge, metrics = _edge_overlay(query_rgb, render_rgb)
    return {f"edge_{k}": v for k, v in metrics.items()}


def _score(row: Dict[str, Any]) -> Tuple[float, float, float]:
    if row.get("status") != "ok":
        return (float("inf"), float("inf"), float("inf"))
    return (
        float(row.get("median_residual_norm_px") or float("inf")),
        abs(float(row.get("corr_norm_radius") or 0.0)),
        float(row.get("p90_residual_norm_px") or float("inf")),
    )


def _draw_best(
    output_dir: Path,
    image_name: str,
    query_rgb_s: np.ndarray,
    render_rgb_s: np.ndarray,
    render_rgb_full: np.ndarray,
    matches: Optional[List[Any]],
    kq: Any,
    kr: Any,
    src: Optional[np.ndarray],
    residual: Optional[np.ndarray],
    scale: float,
) -> None:
    _write_rgb(output_dir / f"{image_name}_best_render.png", render_rgb_full)
    _write_rgb(output_dir / f"{image_name}_baseline_vs_best_checkerboard.png", _checkerboard(query_rgb_s, render_rgb_s, 32))
    if matches:
        q_gray = cv2.cvtColor(query_rgb_s, cv2.COLOR_RGB2GRAY)
        r_gray = cv2.cvtColor(render_rgb_s, cv2.COLOR_RGB2GRAY)
        vis = cv2.drawMatches(q_gray, kq, r_gray, kr, matches[:200], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(os.fspath(output_dir / f"{image_name}_best_sift_inliers.png"), vis)
    if src is not None and residual is not None and len(src) > 0:
        src_full = src / scale
        res_full = residual / scale
        plt.figure(figsize=(8, 6))
        plt.quiver(src_full[:, 0], src_full[:, 1], res_full[:, 0], res_full[:, 1], np.linalg.norm(res_full, axis=1), angles="xy", scale_units="xy", scale=1.0, cmap="viridis")
        plt.gca().invert_yaxis()
        plt.xlabel("query x")
        plt.ylabel("query y")
        plt.title(f"{image_name} best pose residuals")
        plt.colorbar(label="residual norm px")
        plt.tight_layout()
        plt.savefig(output_dir / f"{image_name}_best_residual_quiver.png", dpi=160)
        plt.close()


def _analyze_image(
    args: argparse.Namespace,
    config: Dict[str, Any],
    xml_srs: str,
    image_idx: int,
    image_path: Path,
    photo: Any,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    image_dir = output_dir / "renders" / f"{image_idx:03d}_{image_path.stem}"
    query_rgb_s = _read_rgb(image_path, args.scale)
    query_features = _detect_sift(cv2.cvtColor(query_rgb_s, cv2.COLOR_RGB2GRAY))
    rows: List[Dict[str, Any]] = []
    best_payload = None
    best_row = None
    baseline_path = (REPO_ROOT / args.patch_batch_dir / "images" / f"{image_idx:03d}_{image_path.stem}" / "gpu_pinhole_render.png").resolve()
    for dp, dr, dy in _candidate_deltas():
        if abs(dp) < 1e-12 and abs(dr) < 1e-12 and abs(dy) < 1e-12 and baseline_path.exists():
            render_full = _read_rgb(baseline_path)
            meta = {"source": os.fspath(baseline_path), "rendered_now": False, "baseline_reused": True}
            render_time = 0.0
            render_path = baseline_path
        else:
            render_path = image_dir / f"render_photo{photo.photo_id}_{_delta_key(dp, dr, dy)}.png"
            render_full, meta, render_time = _render_candidate(config, xml_srs, photo, dp, dr, dy, render_path)
        render_s = cv2.resize(render_full, (query_rgb_s.shape[1], query_rgb_s.shape[0]), interpolation=cv2.INTER_AREA)
        metrics, src, residual, matches, kq, kr = _residual_metrics(query_rgb_s, render_s, query_features, args.scale)
        row = {
            "image_name": image_path.stem,
            "query_image": os.fspath(image_path.relative_to(REPO_ROOT)),
            "xml_photo_id": photo.photo_id,
            "render_path": os.fspath(render_path.relative_to(REPO_ROOT)) if render_path.is_relative_to(REPO_ROOT) else os.fspath(render_path),
            "delta_pitch_deg": float(dp),
            "delta_roll_deg": float(dr),
            "delta_yaw_deg": float(dy),
            "rotation_perturbation": "R_perturbed = R_xml.T @ R_delta_camera_local_xyz",
            "render_time_sec": float(render_time),
            "backend_used": meta.get("backend_used"),
            "fallback_reason": meta.get("fallback_reason"),
            "rendered_now": meta.get("rendered_now"),
            "scale": float(args.scale),
            **metrics,
            **_edge_metrics_scaled(query_rgb_s, render_s),
        }
        rows.append(row)
        if row.get("status") == "ok" and (best_row is None or _score(row) < _score(best_row)):
            best_row = row
            best_payload = (query_rgb_s, render_s, render_full, matches, kq, kr, src, residual)
    if best_row is not None and best_payload is not None:
        _draw_best(output_dir, image_path.stem, *best_payload, args.scale)
    return rows


def _best_by_image(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for image in sorted({r["image_name"] for r in rows}):
        image_rows = [r for r in rows if r["image_name"] == image and r.get("status") == "ok"]
        if image_rows:
            out.append(sorted(image_rows, key=_score)[0])
    return out


def _summary(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    baseline = [r for r in rows if abs(r["delta_pitch_deg"]) < 1e-12 and abs(r["delta_roll_deg"]) < 1e-12 and abs(r["delta_yaw_deg"]) < 1e-12 and r.get("status") == "ok"]
    grouped: Dict[Tuple[float, float, float], List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            grouped.setdefault((row["delta_pitch_deg"], row["delta_roll_deg"], row["delta_yaw_deg"]), []).append(row)
    candidate_summaries = []
    for key, group in grouped.items():
        candidate_summaries.append(
            {
                "delta_pitch_deg": key[0],
                "delta_roll_deg": key[1],
                "delta_yaw_deg": key[2],
                "num_images": len(group),
                "mean_median_residual_norm_px": float(np.mean([r["median_residual_norm_px"] for r in group])),
                "mean_p90_residual_norm_px": float(np.mean([r["p90_residual_norm_px"] for r in group])),
                "mean_abs_corr_norm_radius": float(np.mean([abs(r["corr_norm_radius"]) for r in group if r.get("corr_norm_radius") is not None])),
                "mean_inlier_ratio": float(np.mean([r["inlier_ratio"] for r in group])),
            }
        )
    candidate_summaries = sorted(candidate_summaries, key=lambda r: (r["mean_median_residual_norm_px"], r["mean_abs_corr_norm_radius"]))
    best = candidate_summaries[0] if candidate_summaries else None
    baseline_median = float(np.mean([r["median_residual_norm_px"] for r in baseline])) if baseline else None
    baseline_corr = float(np.mean([abs(r["corr_norm_radius"]) for r in baseline])) if baseline else None
    best_median = best["mean_median_residual_norm_px"] if best else None
    best_corr = best["mean_abs_corr_norm_radius"] if best else None
    median_improvement = None if baseline_median in (None, 0.0) or best_median is None else float((baseline_median - best_median) / baseline_median)
    corr_improvement = None if baseline_corr is None or best_corr is None else float(baseline_corr - best_corr)
    pose_likely = bool((median_improvement is not None and median_improvement > 0.2) or (corr_improvement is not None and corr_improvement > 0.2))
    return {
        "experiment": "Query-render small pose residual grid",
        "xml": args.xml,
        "query_dir": args.query_dir,
        "config": args.config,
        "output_dir": args.output_dir,
        "scale": float(args.scale),
        "rotation_perturbation": "R_perturbed = R_xml.T @ R_delta_camera_local_xyz",
        "num_rows": len(rows),
        "num_ok_rows": sum(1 for r in rows if r.get("status") == "ok"),
        "baseline_mean_median_residual_norm_px": baseline_median,
        "baseline_mean_abs_corr_norm_radius": baseline_corr,
        "best_global_delta_pitch_deg": None if best is None else best["delta_pitch_deg"],
        "best_global_delta_roll_deg": None if best is None else best["delta_roll_deg"],
        "best_global_delta_yaw_deg": None if best is None else best["delta_yaw_deg"],
        "best_global_candidate": best,
        "median_residual_improvement_ratio": median_improvement,
        "corr_norm_radius_improvement": corr_improvement,
        "pose_residual_likely": pose_likely,
        "best_by_image": _best_by_image(rows),
        "top20_global_candidates": candidate_summaries[:20],
        "recommended_next_step": "small pose perturbation explains residuals; refine pose grid around the best delta"
        if pose_likely
        else "small pose grid did not sufficiently reduce residuals; return to DOM/DSM phase, local 3D modeling, or robust objective design",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--patch-batch-dir", default=DEFAULT_PATCH_BATCH_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(REPO_ROOT / args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    xml_srs, photos = _parse_xml((REPO_ROOT / args.xml).resolve())
    pose_records = _load_pose_file_projected((REPO_ROOT / args.pose_file).resolve(), xml_srs)
    images = _list_images((REPO_ROOT / args.query_dir).resolve(), args.limit)
    matches, _report = _match_photos(images, pose_records, photos)
    rows: List[Dict[str, Any]] = []
    for idx, image_path in enumerate(images):
        photo = matches[image_path.name]["photo"]
        rows.extend(_analyze_image(args, config, xml_srs, idx, image_path, photo, output_dir))
    summary = _summary(rows, args)
    _write_csv(output_dir / "pose_residual_grid_results.csv", rows)
    _write_json(output_dir / "pose_residual_grid_summary.json", summary)
    print(json.dumps(_safe_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
