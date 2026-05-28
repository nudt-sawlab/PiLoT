#!/usr/bin/env python3
"""Scan camera-domain variants using SIFT residual fields as the main metric."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_yawfix_refinement_update import _safe_jsonable
from tools.render_contextcapture_xml_domdsm_initial import _load_pose_file_projected, _match_photos, _parse_xml


DEFAULT_XML = "data_caiwangcun/CaiWangCun.xml"
DEFAULT_QUERY_DIR = "data_caiwangcun/query/images/exif_test"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_PATCH_BATCH_DIR = "docs/experiments/dom_dsm_prepare/query_dom_patch_alignment_batch"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/query_render_camera_domain_residual_scan"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DISTORTION_SCALES = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
FOCAL_SCALES = [0.995, 1.0, 1.005]
PP_OFFSETS = [(0.0, 0.0), (0.0, -30.0), (0.0, 30.0), (-30.0, 0.0), (30.0, 0.0)]


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row.keys() if not isinstance(row.get(k), (dict, list))})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _list_images(query_dir: Path, limit: Optional[int]) -> List[Path]:
    images = sorted(p for p in query_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return images[:limit] if limit and limit > 0 else images


def _read_rgb(path: Path, scale: float) -> np.ndarray:
    bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if scale != 1.0:
        rgb = cv2.resize(rgb, (int(round(rgb.shape[1] * scale)), int(round(rgb.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return rgb


def _render_path_for_image(batch_dir: Path, idx: int, image_path: Path) -> Path:
    return batch_dir / "images" / f"{idx:03d}_{image_path.stem}" / "gpu_pinhole_render.png"


def _ensure_render(args: argparse.Namespace, idx: int, image_path: Path, render_path: Path) -> None:
    if render_path.exists():
        return
    cmd = [
        sys.executable,
        os.fspath(REPO_ROOT / "tools/check_query_dom_patch_alignment_batch.py"),
        "--query-dir",
        args.query_dir,
        "--limit",
        str(idx + 1),
        "--output-dir",
        args.patch_batch_dir,
        "--keep-existing",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    if not render_path.exists():
        raise FileNotFoundError(f"Render image missing after generation: {render_path}")


def _camera_matrix(intr: Any, scale: float, distortion_scale: float = 1.0, focal_scale: float = 1.0, cx_offset: float = 0.0, cy_offset: float = 0.0) -> np.ndarray:
    del distortion_scale
    return np.asarray(
        [
            [float(intr.fx) * float(focal_scale) * scale, 0.0, (float(intr.cx) + float(cx_offset)) * scale],
            [0.0, float(intr.fy) * float(focal_scale) * scale, (float(intr.cy) + float(cy_offset)) * scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _dist_coeffs(intr: Any, distortion_scale: float) -> np.ndarray:
    return np.asarray(
        [intr.k1 * distortion_scale, intr.k2 * distortion_scale, intr.p1 * distortion_scale, intr.p2 * distortion_scale, intr.k3 * distortion_scale],
        dtype=np.float64,
    )


def _distort_pinhole_render_to_raw(render_rgb: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    h, w = render_rgb.shape[:2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    xd = ((u - K[0, 2]) / K[0, 0]).astype(np.float32)
    yd = ((v - K[1, 2]) / K[1, 1]).astype(np.float32)
    distorted_points = np.stack([xd, yd], axis=-1).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(distorted_points, np.eye(3, dtype=np.float64), D, P=None)
    xu = undistorted[:, 0, 0].reshape(h, w)
    yu = undistorted[:, 0, 1].reshape(h, w)
    map_x = (K[0, 0] * xu + K[0, 2]).astype(np.float32)
    map_y = (K[1, 1] * yu + K[1, 2]).astype(np.float32)
    return cv2.remap(render_rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _detect_sift(gray: np.ndarray) -> Tuple[Any, Any]:
    sift = cv2.SIFT_create(nfeatures=9000)
    return sift.detectAndCompute(gray, None)


def _fit_residual_modes(src: np.ndarray, dst: np.ndarray, residual: np.ndarray, image_shape: Tuple[int, int], inv_scale: float) -> Dict[str, Any]:
    h, w = image_shape
    src_full = src * inv_scale
    residual_full = residual * inv_scale
    center = np.asarray([w * inv_scale / 2.0, h * inv_scale / 2.0], dtype=np.float64)
    rel = src_full - center
    radius = np.linalg.norm(rel, axis=1)
    norm = np.linalg.norm(residual_full, axis=1)
    dx = residual_full[:, 0]
    dy = residual_full[:, 1]
    out: Dict[str, Any] = {
        "median_dx_px": float(np.median(dx)),
        "median_dy_px": float(np.median(dy)),
        "median_residual_norm_px": float(np.median(norm)),
        "p90_residual_norm_px": float(np.percentile(norm, 90)),
        "direction_consistency": float(np.hypot(dx.mean(), dy.mean()) / max(norm.mean(), 1e-9)),
    }
    if len(residual_full) >= 8:
        out["corr_dx_x"] = float(np.corrcoef(src_full[:, 0], dx)[0, 1])
        out["corr_dy_y"] = float(np.corrcoef(src_full[:, 1], dy)[0, 1])
        out["corr_norm_radius"] = float(np.corrcoef(radius, norm)[0, 1])
    else:
        out.update({"corr_dx_x": None, "corr_dy_y": None, "corr_norm_radius": None})
    return out


def _residual_metrics_from_features(
    query_gray: np.ndarray,
    render_gray: np.ndarray,
    query_features: Tuple[Any, Any],
    scale: float,
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray], Optional[List[Any]], Optional[Any], Optional[Any]]:
    kq, dq = query_features
    kr, dr = _detect_sift(render_gray)
    row: Dict[str, Any] = {
        "query_keypoints": len(kq),
        "render_keypoints": len(kr),
    }
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
        row.update(_fit_residual_modes(src_i, dst_i, residual, query_gray.shape[:2], 1.0 / scale))
    return row, src_i, residual, [good[i] for i, keep in enumerate(inlier_mask) if keep], kq, kr


def _candidate_images(query_rgb: np.ndarray, render_rgb: np.ndarray, intr: Any, scale: float, candidate: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    K = _camera_matrix(
        intr,
        scale,
        distortion_scale=float(candidate["distortion_scale"]),
        focal_scale=float(candidate["focal_scale"]),
        cx_offset=float(candidate["cx_offset_px"]),
        cy_offset=float(candidate["cy_offset_px"]),
    )
    D = _dist_coeffs(intr, float(candidate["distortion_scale"]))
    mode = candidate["candidate"]
    if mode == "raw_query_vs_pinhole_render":
        return query_rgb, render_rgb
    if mode in {"undistorted_query_vs_pinhole_render", "undistorted_query_with_scaled_distortion_vs_pinhole_render"}:
        return cv2.undistort(query_rgb, K, D, None, K), render_rgb
    if mode in {"raw_query_vs_distorted_render", "raw_query_vs_distorted_render_with_scaled_distortion"}:
        return query_rgb, _distort_pinhole_render_to_raw(render_rgb, K, D)
    raise ValueError(mode)


def _iter_candidates() -> Iterable[Dict[str, Any]]:
    yield {
        "candidate": "raw_query_vs_pinhole_render",
        "distortion_scale": 0.0,
        "focal_scale": 1.0,
        "cx_offset_px": 0.0,
        "cy_offset_px": 0.0,
    }
    yield {
        "candidate": "undistorted_query_vs_pinhole_render",
        "distortion_scale": 1.0,
        "focal_scale": 1.0,
        "cx_offset_px": 0.0,
        "cy_offset_px": 0.0,
    }
    yield {
        "candidate": "raw_query_vs_distorted_render",
        "distortion_scale": 1.0,
        "focal_scale": 1.0,
        "cx_offset_px": 0.0,
        "cy_offset_px": 0.0,
    }
    for distortion_scale in DISTORTION_SCALES:
        for focal_scale in FOCAL_SCALES:
            for cx_offset, cy_offset in PP_OFFSETS:
                yield {
                    "candidate": "undistorted_query_with_scaled_distortion_vs_pinhole_render",
                    "distortion_scale": float(distortion_scale),
                    "focal_scale": float(focal_scale),
                    "cx_offset_px": float(cx_offset),
                    "cy_offset_px": float(cy_offset),
                }
                yield {
                    "candidate": "raw_query_vs_distorted_render_with_scaled_distortion",
                    "distortion_scale": float(distortion_scale),
                    "focal_scale": float(focal_scale),
                    "cx_offset_px": float(cx_offset),
                    "cy_offset_px": float(cy_offset),
                }


def _score(row: Dict[str, Any]) -> Tuple[float, float, float]:
    if row.get("status") != "ok":
        return (float("inf"), float("inf"), float("inf"))
    corr = abs(float(row.get("corr_norm_radius") or 0.0))
    median = float(row.get("median_residual_norm_px") or float("inf"))
    p90 = float(row.get("p90_residual_norm_px") or float("inf"))
    return (median, corr, p90)


def _draw_visuals(output_dir: Path, image_name: str, query_gray: np.ndarray, render_gray: np.ndarray, matches: Optional[List[Any]], kq: Any, kr: Any, src: Optional[np.ndarray], residual: Optional[np.ndarray], scale: float) -> None:
    if matches:
        vis = cv2.drawMatches(query_gray, kq, render_gray, kr, matches[:200], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(os.fspath(output_dir / f"{image_name}_best_sift_inliers.png"), vis)
    if src is not None and residual is not None and len(src) > 0:
        src_full = src / scale
        residual_full = residual / scale
        plt.figure(figsize=(8, 6))
        plt.quiver(src_full[:, 0], src_full[:, 1], residual_full[:, 0], residual_full[:, 1], np.linalg.norm(residual_full, axis=1), angles="xy", scale_units="xy", scale=1.0, cmap="viridis")
        plt.gca().invert_yaxis()
        plt.xlabel("query x")
        plt.ylabel("query y")
        plt.title(f"{image_name} best camera-domain residuals")
        plt.colorbar(label="residual norm px")
        plt.tight_layout()
        plt.savefig(output_dir / f"{image_name}_best_residual_quiver.png", dpi=160)
        plt.close()


def _analyze_image(query_path: Path, render_path: Path, photo: Any, scale: float, output_dir: Path) -> List[Dict[str, Any]]:
    query_rgb = _read_rgb(query_path, scale)
    render_rgb = _read_rgb(render_path, scale)
    rows: List[Dict[str, Any]] = []
    cache_query_features: Dict[str, Tuple[Any, Any]] = {}
    best_payload = None
    best_row = None
    for candidate in _iter_candidates():
        cand_query, cand_render = _candidate_images(query_rgb, render_rgb, photo.intrinsics, scale, candidate)
        q_gray = cv2.cvtColor(cand_query, cv2.COLOR_RGB2GRAY)
        r_gray = cv2.cvtColor(cand_render, cv2.COLOR_RGB2GRAY)
        qkey = f"{candidate['candidate']}|{candidate['distortion_scale']}|{candidate['focal_scale']}|{candidate['cx_offset_px']}|{candidate['cy_offset_px']}"
        query_features = cache_query_features.get(qkey)
        if query_features is None:
            query_features = _detect_sift(q_gray)
            cache_query_features[qkey] = query_features
        metrics, src, residual, matches, kq, kr = _residual_metrics_from_features(q_gray, r_gray, query_features, scale)
        row = {
            "image_name": query_path.stem,
            "query_image": os.fspath(query_path.relative_to(REPO_ROOT)),
            "render_image": os.fspath(render_path.relative_to(REPO_ROOT)),
            "xml_photo_id": photo.photo_id,
            "scale": scale,
            **candidate,
            **metrics,
        }
        rows.append(row)
        if row.get("status") == "ok" and (best_row is None or _score(row) < _score(best_row)):
            best_row = row
            best_payload = (q_gray, r_gray, matches, kq, kr, src, residual)
    if best_payload is not None:
        _draw_visuals(output_dir, query_path.stem, *best_payload, scale)
    return rows


def _best_by_image(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for image in sorted({r["image_name"] for r in rows}):
        image_rows = [r for r in rows if r["image_name"] == image and r.get("status") == "ok"]
        if image_rows:
            out.append(sorted(image_rows, key=_score)[0])
    return out


def _summary(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    baseline_rows = [r for r in rows if r["candidate"] == "raw_query_vs_pinhole_render" and r.get("status") == "ok"]
    best_rows = _best_by_image(rows)
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row["candidate"], row["distortion_scale"], row["focal_scale"], row["cx_offset_px"], row["cy_offset_px"])
        grouped.setdefault(key, []).append(row)
    candidate_summaries = []
    for key, group in grouped.items():
        candidate_summaries.append(
            {
                "candidate": key[0],
                "distortion_scale": key[1],
                "focal_scale": key[2],
                "cx_offset_px": key[3],
                "cy_offset_px": key[4],
                "num_images": len(group),
                "mean_median_residual_norm_px": float(np.mean([float(r["median_residual_norm_px"]) for r in group])),
                "mean_p90_residual_norm_px": float(np.mean([float(r["p90_residual_norm_px"]) for r in group])),
                "mean_abs_corr_norm_radius": float(np.mean([abs(float(r["corr_norm_radius"])) for r in group if r.get("corr_norm_radius") is not None])),
                "mean_inlier_ratio": float(np.mean([float(r["inlier_ratio"]) for r in group])),
            }
        )
    candidate_summaries = sorted(candidate_summaries, key=lambda r: (r["mean_median_residual_norm_px"], r["mean_abs_corr_norm_radius"]))
    best_global = candidate_summaries[0] if candidate_summaries else None
    baseline_median = float(np.mean([float(r["median_residual_norm_px"]) for r in baseline_rows])) if baseline_rows else None
    best_median = best_global["mean_median_residual_norm_px"] if best_global else None
    baseline_corr = float(np.mean([abs(float(r["corr_norm_radius"])) for r in baseline_rows])) if baseline_rows else None
    best_corr = best_global["mean_abs_corr_norm_radius"] if best_global else None
    median_improvement_ratio = None if baseline_median in (None, 0.0) or best_median is None else float((baseline_median - best_median) / baseline_median)
    corr_improvement = None if baseline_corr is None or best_corr is None else float(baseline_corr - best_corr)
    mismatch = bool((median_improvement_ratio is not None and median_improvement_ratio > 0.2) or (corr_improvement is not None and corr_improvement > 0.2))
    return {
        "experiment": "Query-render camera domain residual scan",
        "xml": args.xml,
        "query_dir": args.query_dir,
        "patch_batch_dir": args.patch_batch_dir,
        "output_dir": args.output_dir,
        "scale": args.scale,
        "num_rows": len(rows),
        "num_ok_rows": sum(1 for r in rows if r.get("status") == "ok"),
        "num_images_with_baseline": len(baseline_rows),
        "baseline_mean_median_residual_norm_px": baseline_median,
        "baseline_mean_abs_corr_norm_radius": baseline_corr,
        "best_global_candidate": best_global,
        "best_distortion_scale": None if best_global is None else best_global["distortion_scale"],
        "best_focal_scale": None if best_global is None else best_global["focal_scale"],
        "best_cx_offset_px": None if best_global is None else best_global["cx_offset_px"],
        "best_cy_offset_px": None if best_global is None else best_global["cy_offset_px"],
        "median_residual_improvement_ratio": median_improvement_ratio,
        "corr_norm_radius_improvement": corr_improvement,
        "camera_domain_mismatch_likely": mismatch,
        "best_by_image": best_rows,
        "top20_global_candidates": candidate_summaries[:20],
        "recommended_next_step": "camera-domain parameters improve SIFT residuals; refine around the best distortion/K offsets"
        if mismatch
        else "camera-domain scan did not sufficiently reduce radial residuals; run small pose residual grid next",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--patch-batch-dir", default=DEFAULT_PATCH_BATCH_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--scale", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_srs, photos = _parse_xml((REPO_ROOT / args.xml).resolve())
    pose_records = _load_pose_file_projected((REPO_ROOT / args.pose_file).resolve(), xml_srs)
    images = _list_images((REPO_ROOT / args.query_dir).resolve(), args.limit)
    matches, _match_report = _match_photos(images, pose_records, photos)
    batch_dir = (REPO_ROOT / args.patch_batch_dir).resolve()
    rows: List[Dict[str, Any]] = []
    for idx, image_path in enumerate(images):
        render_path = _render_path_for_image(batch_dir, idx, image_path)
        _ensure_render(args, idx, image_path, render_path)
        photo = matches[image_path.name]["photo"]
        rows.extend(_analyze_image(image_path, render_path, photo, float(args.scale), output_dir))
    summary = _summary(rows, args)
    _write_csv(output_dir / "camera_domain_residual_scan_results.csv", rows)
    _write_json(output_dir / "camera_domain_residual_scan_summary.json", summary)
    print(json.dumps(_safe_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
