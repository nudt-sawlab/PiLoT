#!/usr/bin/env python3
"""Estimate query-render feature residual vector fields with SIFT + RANSAC."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_yawfix_refinement_update import _safe_jsonable


DEFAULT_QUERY_DIR = "data_caiwangcun/query/images/exif_test"
DEFAULT_PATCH_BATCH_DIR = "docs/experiments/dom_dsm_prepare/query_dom_patch_alignment_batch"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/query_render_feature_residual_field"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


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


def _read_gray(path: Path, scale: float) -> np.ndarray:
    gray = cv2.imread(os.fspath(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    if scale != 1.0:
        gray = cv2.resize(gray, (int(round(gray.shape[1] * scale)), int(round(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return gray


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
        raise FileNotFoundError(f"Render image still missing after batch generation: {render_path}")


def _fit_residual_modes(src: np.ndarray, dst: np.ndarray, residual: np.ndarray, image_shape: Tuple[int, int]) -> Dict[str, Any]:
    h, w = image_shape
    center = np.asarray([w / 2.0, h / 2.0], dtype=np.float64)
    rel = src - center
    radius = np.linalg.norm(rel, axis=1)
    norm = np.linalg.norm(residual, axis=1)
    dx = residual[:, 0]
    dy = residual[:, 1]
    out: Dict[str, Any] = {
        "median_dx_px": float(np.median(dx)),
        "median_dy_px": float(np.median(dy)),
        "median_residual_norm_px": float(np.median(norm)),
        "p90_residual_norm_px": float(np.percentile(norm, 90)),
        "direction_consistency": float(np.hypot(dx.mean(), dy.mean()) / max(norm.mean(), 1e-9)),
    }
    if len(residual) >= 8:
        out["corr_dx_x"] = float(np.corrcoef(src[:, 0], dx)[0, 1])
        out["corr_dy_y"] = float(np.corrcoef(src[:, 1], dy)[0, 1])
        out["corr_norm_radius"] = float(np.corrcoef(radius, norm)[0, 1])
        out["radial_pattern_likely"] = bool(abs(out["corr_norm_radius"]) > 0.45)
        out["rolling_or_pose_y_pattern_likely"] = bool(abs(out["corr_dy_y"]) > 0.45 or abs(out["corr_dx_x"]) > 0.45)
    else:
        out.update({"corr_dx_x": None, "corr_dy_y": None, "corr_norm_radius": None, "radial_pattern_likely": False, "rolling_or_pose_y_pattern_likely": False})
    return out


def _analyze_pair(query_path: Path, render_path: Path, scale: float, output_dir: Path, image_name: str) -> Dict[str, Any]:
    q = _read_gray(query_path, scale)
    r = _read_gray(render_path, scale)
    sift = cv2.SIFT_create(nfeatures=12000)
    kq, dq = sift.detectAndCompute(q, None)
    kr, dr = sift.detectAndCompute(r, None)
    row: Dict[str, Any] = {
        "image_name": image_name,
        "query_image": os.fspath(query_path.relative_to(REPO_ROOT)),
        "render_image": os.fspath(render_path.relative_to(REPO_ROOT)),
        "scale": scale,
        "query_keypoints": len(kq),
        "render_keypoints": len(kr),
    }
    if dq is None or dr is None or len(kq) < 8 or len(kr) < 8:
        row.update({"status": "insufficient_features", "good_matches": 0, "inliers": 0})
        return row
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(dq, dr, k=2)
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    row["good_matches"] = len(good)
    if len(good) < 8:
        row.update({"status": "insufficient_matches", "inliers": 0})
        return row
    src = np.float32([kq[m.queryIdx].pt for m in good])
    dst = np.float32([kr[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    inlier_mask = mask.ravel().astype(bool) if mask is not None else np.zeros(len(good), dtype=bool)
    src_i = src[inlier_mask] / scale
    dst_i = dst[inlier_mask] / scale
    residual = dst_i - src_i
    row["inliers"] = int(inlier_mask.sum())
    row["inlier_ratio"] = float(inlier_mask.mean())
    row["status"] = "ok" if row["inliers"] >= 8 else "too_few_inliers"
    if row["status"] == "ok":
        row.update(_fit_residual_modes(src_i, dst_i, residual, (int(q.shape[0] / scale), int(q.shape[1] / scale))))
    vis = cv2.drawMatches(
        q,
        kq,
        r,
        kr,
        [good[i] for i, keep in enumerate(inlier_mask) if keep][:200],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(os.fspath(output_dir / f"{image_name}_sift_inliers.png"), vis)
    if row["status"] == "ok":
        plt.figure(figsize=(8, 6))
        plt.quiver(src_i[:, 0], src_i[:, 1], residual[:, 0], residual[:, 1], np.linalg.norm(residual, axis=1), angles="xy", scale_units="xy", scale=1.0, cmap="viridis")
        plt.gca().invert_yaxis()
        plt.xlabel("query x")
        plt.ylabel("query y")
        plt.title(f"{image_name} feature residuals")
        plt.colorbar(label="residual norm px")
        plt.tight_layout()
        plt.savefig(output_dir / f"{image_name}_residual_quiver.png", dpi=160)
        plt.close()
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    parser.add_argument("--patch-batch-dir", default=DEFAULT_PATCH_BATCH_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--scale", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images = _list_images((REPO_ROOT / args.query_dir).resolve(), args.limit)
    batch_dir = (REPO_ROOT / args.patch_batch_dir).resolve()
    rows = []
    for idx, image_path in enumerate(images):
        render_path = _render_path_for_image(batch_dir, idx, image_path)
        _ensure_render(args, idx, image_path, render_path)
        rows.append(_analyze_pair(image_path, render_path, args.scale, output_dir, image_path.stem))
    ok = [r for r in rows if r.get("status") == "ok"]
    dir_cons = [r.get("direction_consistency") for r in ok if r.get("direction_consistency") is not None]
    summary = {
        "experiment": "Query-render feature residual field",
        "query_dir": args.query_dir,
        "patch_batch_dir": args.patch_batch_dir,
        "num_images": len(rows),
        "num_ok": len(ok),
        "mean_inlier_ratio": float(np.mean([r["inlier_ratio"] for r in ok])) if ok else None,
        "mean_direction_consistency": float(np.mean(dir_cons)) if dir_cons else None,
        "radial_pattern_image_count": sum(1 for r in ok if r.get("radial_pattern_likely")),
        "rolling_or_pose_pattern_image_count": sum(1 for r in ok if r.get("rolling_or_pose_y_pattern_likely")),
        "rows": rows,
        "conclusion": "stable residual pattern exists; inspect camera model or small pose residuals"
        if any(r.get("radial_pattern_likely") or r.get("rolling_or_pose_y_pattern_likely") for r in ok)
        else "feature residuals do not show a strong global camera-model pattern",
    }
    _write_csv(output_dir / "feature_residual_results.csv", rows)
    _write_json(output_dir / "feature_residual_summary.json", summary)
    print(json.dumps(_safe_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
