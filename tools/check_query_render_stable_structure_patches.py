#!/usr/bin/env python3
"""Summarize patch alignment separately for stable structures and unstable texture classes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_yawfix_refinement_update import _safe_jsonable


DEFAULT_BATCH_DIR = "docs/experiments/dom_dsm_prepare/query_dom_patch_alignment_batch"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/query_render_stable_structure_patches"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row.keys() if not isinstance(row.get(k), (dict, list))})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    value = row.get(key)
    if value in (None, "", "None"):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _bool(row: Dict[str, Any], key: str) -> bool:
    return str(row.get(key, "")).lower() in {"true", "1", "yes"}


def _ensure_batch(args: argparse.Namespace, batch_dir: Path) -> None:
    csv_path = batch_dir / "batch_patch_alignment_results.csv"
    if csv_path.exists():
        return
    cmd = [
        sys.executable,
        os.fspath(REPO_ROOT / "tools/check_query_dom_patch_alignment_batch.py"),
        "--limit",
        str(args.limit),
        "--output-dir",
        args.batch_dir,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _patch_path(batch_dir: Path, row: Dict[str, Any], name: str) -> Optional[Path]:
    rel = row.get("batch_patch_rel_dir")
    if not rel:
        return None
    path = batch_dir / rel / name
    return path if path.exists() else None


def _scene_class(image_rgb: np.ndarray) -> Dict[str, Any]:
    rgb = image_rgb.astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    exg = 2 * g - r - b
    vegetation_ratio = float((exg > 25).mean())
    shadow_ratio = float((v < 55).mean())
    water_ratio = float(((b > g + 8) & (b > r + 8) & (s > 35)).mean())
    bright_ratio = float((v > 95).mean())
    low_sat_ratio = float((s < 55).mean())
    if shadow_ratio > 0.35:
        cls = "shadow"
    elif vegetation_ratio > 0.35:
        cls = "vegetation"
    elif water_ratio > 0.20:
        cls = "water"
    elif bright_ratio > 0.45 and low_sat_ratio > 0.20:
        cls = "stable_structure"
    else:
        cls = "mixed_or_texture"
    return {
        "scene_class": cls,
        "vegetation_ratio": vegetation_ratio,
        "shadow_ratio": shadow_ratio,
        "water_ratio": water_ratio,
        "bright_ratio": bright_ratio,
        "low_sat_ratio": low_sat_ratio,
    }


def _mean(vals: List[Optional[float]]) -> Optional[float]:
    finite = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else None


def _pct(vals: List[Optional[float]], q: float) -> Optional[float]:
    finite = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.percentile(np.asarray(finite), q)) if finite else None


def _direction_consistency(rows: List[Dict[str, Any]]) -> Optional[float]:
    if not rows:
        return None
    dx = np.asarray([_float(r, "best_dx_px", 0.0) for r in rows], dtype=np.float64)
    dy = np.asarray([_float(r, "best_dy_px", 0.0) for r in rows], dtype=np.float64)
    norm = np.hypot(dx, dy)
    if float(norm.mean()) <= 1e-9:
        return None
    return float(np.hypot(float(dx.mean()), float(dy.mean())) / float(norm.mean()))


def _group_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    high = [r for r in rows if _bool(r, "high_confidence")]
    return {
        "count": len(rows),
        "high_confidence_count": len(high),
        "median_offset_norm_px_high": _pct([_float(r, "offset_norm_px") for r in high], 50),
        "median_dx_px_high": _pct([_float(r, "best_dx_px") for r in high], 50),
        "median_dy_px_high": _pct([_float(r, "best_dy_px") for r in high], 50),
        "direction_consistency_high": _direction_consistency(high),
        "mean_query_render_ncc_high": _mean([_float(r, "query_render_ncc") for r in high]),
        "mean_query_dom_ncc_high": _mean([_float(r, "query_dom_ncc") for r in high]),
        "mean_edge_chamfer_after_high": _mean([_float(r, "edge_chamfer_after") for r in high]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", default=DEFAULT_BATCH_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    batch_dir = (REPO_ROOT / args.batch_dir).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_batch(args, batch_dir)
    rows = _read_csv(batch_dir / "batch_patch_alignment_results.csv")
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        query_patch_path = _patch_path(batch_dir, row, "query.png")
        if query_patch_path is not None:
            bgr = cv2.imread(os.fspath(query_patch_path), cv2.IMREAD_COLOR)
            if bgr is not None:
                row.update(_scene_class(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            else:
                row["scene_class"] = "unknown"
        else:
            row["scene_class"] = "unknown"
        row["stable_structure_patch"] = bool(
            row["scene_class"] == "stable_structure"
            and row.get("patch_label") in {"stable_match", "edge_unreliable", "texture_mismatch"}
            and (_float(row, "query_gradient", 0.0) or 0.0) >= 9.0
        )
        enriched.append(row)

    by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_scene[row["scene_class"]].append(row)
    stable = [r for r in enriched if r.get("stable_structure_patch")]
    unstable = [r for r in enriched if r["scene_class"] in {"vegetation", "shadow", "water"}]
    summary = {
        "experiment": "Stable structure patch attribution",
        "batch_dir": args.batch_dir,
        "num_rows": len(enriched),
        "scene_counts": dict(Counter(r["scene_class"] for r in enriched)),
        "patch_label_counts": dict(Counter(r.get("patch_label", "unknown") for r in enriched)),
        "stable_structure": _group_stats(stable),
        "unstable_texture": _group_stats(unstable),
        "by_scene": {scene: _group_stats(scene_rows) for scene, scene_rows in sorted(by_scene.items())},
    }
    stable_ncc = summary["stable_structure"]["mean_query_render_ncc_high"]
    unstable_ncc = summary["unstable_texture"]["mean_query_render_ncc_high"]
    summary["stable_structure_better_than_unstable"] = bool(
        stable_ncc is not None and unstable_ncc is not None and stable_ncc > unstable_ncc + 0.1
    )
    summary["conclusion"] = (
        "stable structures align better than vegetation/shadow/water; query-DOM content differences contaminate global metrics"
        if summary["stable_structure_better_than_unstable"]
        else "stable-structure separation does not yet isolate a cleaner geometric signal; inspect scene labels and feature residuals"
    )
    _write_csv(output_dir / "stable_structure_patch_results.csv", enriched)
    _write_json(output_dir / "stable_structure_patch_summary.json", summary)
    print(json.dumps(_safe_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
