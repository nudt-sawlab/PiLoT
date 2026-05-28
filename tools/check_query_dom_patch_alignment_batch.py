#!/usr/bin/env python3
"""Run patch-level query/render/DOM alignment diagnostics over multiple query images."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_yawfix_refinement_update import _safe_jsonable


DEFAULT_XML = "data_caiwangcun/CaiWangCun.xml"
DEFAULT_QUERY_DIR = "data_caiwangcun/query/images/exif_test"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_CONFIG = "configs/caiwangcun_domdsm.yaml"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/query_dom_patch_alignment_batch"
SINGLE_SCRIPT = "tools/check_query_dom_patch_alignment.py"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row.keys() if not isinstance(row.get(k), (dict, list))})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else None


def _percentile(values: Iterable[Optional[float]], q: float) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.percentile(np.asarray(finite, dtype=np.float64), q)) if finite else None


def _direction_consistency(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    dx = np.asarray([_float(r, "best_dx_px", 0.0) for r in rows], dtype=np.float64)
    dy = np.asarray([_float(r, "best_dy_px", 0.0) for r in rows], dtype=np.float64)
    if dx.size == 0:
        return None
    norms = np.hypot(dx, dy)
    mean_norm = float(norms.mean())
    if mean_norm <= 1.0e-9:
        return None
    return float(np.hypot(float(dx.mean()), float(dy.mean())) / mean_norm)


def _label_patch(row: Dict[str, Any]) -> Dict[str, str]:
    q_grad = _float(row, "query_gradient", 0.0) or 0.0
    r_grad = _float(row, "render_gradient", 0.0) or 0.0
    q_edges = _float(row, "query_edge_count", 0.0) or 0.0
    r_edges = _float(row, "render_edge_count", 0.0) or 0.0
    high = _bool(row, "high_confidence")
    ncc = _float(row, "query_render_ncc", 0.0) or 0.0
    ssim = _float(row, "query_render_ssim", 0.0) or 0.0
    qd_ncc = _float(row, "query_dom_ncc")
    rd_ncc = _float(row, "render_dom_ncc")
    edge_improve = _float(row, "edge_chamfer_improvement", 0.0) or 0.0
    edge_after = _float(row, "edge_chamfer_after", float("inf")) or float("inf")
    hist_qr = _float(row, "query_render_hist_distance", 0.0) or 0.0
    hist_qd = _float(row, "query_dom_hist_distance")

    if q_grad < 9.0 or r_grad < 9.0 or q_edges < 180 or r_edges < 180:
        return {"patch_label": "low_texture", "label_reason": "gradient or edge count below diagnostic threshold"}
    if high and ncc >= 0.45 and ssim >= 0.30 and edge_after <= 15.0:
        return {"patch_label": "stable_match", "label_reason": "high confidence with strong query-render photometric and edge agreement"}
    if edge_improve > 1.0 and (ncc < 0.25 or ssim < 0.20):
        return {"patch_label": "edge_unreliable", "label_reason": "edge chamfer improves but photometric agreement is weak"}
    if qd_ncc is not None and rd_ncc is not None and qd_ncc < 0.15 and rd_ncc < 0.15:
        return {"patch_label": "texture_mismatch", "label_reason": "query-DOM and render-DOM similarities are both low"}
    if hist_qr > 0.45 and (hist_qd is None or hist_qd > 0.45):
        return {"patch_label": "occlusion_or_change", "label_reason": "large color distribution difference against render and DOM"}
    if high:
        return {"patch_label": "stable_match", "label_reason": "high-confidence local alignment, weaker auxiliary DOM evidence"}
    return {"patch_label": "texture_mismatch", "label_reason": "no stable high-confidence match and weak auxiliary evidence"}


def _list_images(query_dir: Path, limit: Optional[int]) -> List[Path]:
    images = sorted(p for p in query_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return images[:limit] if limit is not None and limit > 0 else images


def _run_single(args: argparse.Namespace, image_path: Path, image_output: Path) -> Dict[str, Any]:
    render_path = image_output / "gpu_pinhole_render.png"
    cmd = [
        sys.executable,
        os.fspath(REPO_ROOT / SINGLE_SCRIPT),
        "--xml",
        args.xml,
        "--query-image",
        os.fspath(image_path.relative_to(REPO_ROOT)),
        "--pose-file",
        args.pose_file,
        "--config",
        args.config,
        "--render-image",
        os.fspath(render_path.relative_to(REPO_ROOT)),
        "--output-dir",
        os.fspath(image_output.relative_to(REPO_ROOT)),
        "--patch-size",
        str(args.patch_size),
        "--stride",
        str(args.stride),
        "--search-radius",
        str(args.search_radius),
        "--max-patches",
        str(args.max_patches),
        "--min-gradient",
        str(args.min_gradient),
        "--min-edge-count",
        str(args.min_edge_count),
        "--ray-step-m",
        str(args.ray_step_m),
        "--ray-max-m",
        str(args.ray_max_m),
    ]
    if args.keep_existing:
        cmd.append("--keep-existing")
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    log_path = image_output / "single_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND:\n" + " ".join(cmd) + "\n\nSTDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        return {
            "status": "failed",
            "query_image": os.fspath(image_path.relative_to(REPO_ROOT)),
            "output_dir": os.fspath(image_output.relative_to(REPO_ROOT)),
            "returncode": int(result.returncode),
            "log": os.fspath(log_path.relative_to(REPO_ROOT)),
        }
    summary = _read_json(image_output / "patch_alignment_summary.json")
    summary["status"] = "ok"
    summary["query_image"] = os.fspath(image_path.relative_to(REPO_ROOT))
    summary["output_dir"] = os.fspath(image_output.relative_to(REPO_ROOT))
    summary["log"] = os.fspath(log_path.relative_to(REPO_ROOT))
    return summary


def _aggregate_rows(image_path: Path, image_output: Path, image_summary: Dict[str, Any], batch_output: Path) -> List[Dict[str, Any]]:
    rows = _read_csv(image_output / "patch_alignment_results.csv")
    rel_image_output = os.path.relpath(image_output, batch_output).replace("\\", "/")
    out: List[Dict[str, Any]] = []
    for row in rows:
        label = _label_patch(row)
        row.update(label)
        row["query_image"] = os.fspath(image_path.relative_to(REPO_ROOT))
        row["image_name"] = image_path.name
        row["image_output_dir"] = rel_image_output
        row["xml_photo_id"] = image_summary.get("xml_photo_id")
        row["per_image_direction_consistency"] = image_summary.get("direction_consistency")
        row["per_image_high_confidence_patch_count"] = image_summary.get("high_confidence_patch_count")
        if row.get("patch_rel_dir"):
            row["batch_patch_rel_dir"] = f"{rel_image_output}/{row['patch_rel_dir']}"
        out.append(row)
    return out


def _summarize_batch(image_summaries: List[Dict[str, Any]], rows: List[Dict[str, Any]], failures: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    high = [r for r in rows if _bool(r, "high_confidence")]
    labels = Counter(r.get("patch_label", "unknown") for r in rows)
    high_labels = Counter(r.get("patch_label", "unknown") for r in high)
    by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[row["image_name"]].append(row)

    per_image = []
    for summary in image_summaries:
        image_name = Path(summary["query_image"]).name
        image_rows = by_image.get(image_name, [])
        image_high = [r for r in image_rows if _bool(r, "high_confidence")]
        per_image.append(
            {
                "query_image": summary["query_image"],
                "status": summary.get("status"),
                "xml_photo_id": summary.get("xml_photo_id"),
                "candidate_patch_count": summary.get("candidate_patch_count"),
                "high_confidence_patch_count": summary.get("high_confidence_patch_count"),
                "direction_consistency": summary.get("direction_consistency"),
                "median_best_dx_px": summary.get("median_best_dx_px"),
                "median_best_dy_px": summary.get("median_best_dy_px"),
                "median_offset_norm_px": summary.get("median_offset_norm_px"),
                "stable_local_shift_likely": summary.get("stable_local_shift_likely"),
                "label_counts": dict(Counter(r.get("patch_label", "unknown") for r in image_rows)),
                "high_confidence_label_counts": dict(Counter(r.get("patch_label", "unknown") for r in image_high)),
                "output_dir": summary.get("output_dir"),
            }
        )

    low_direction_images = [
        p for p in per_image if p.get("direction_consistency") is not None and float(p["direction_consistency"]) < 0.5
    ]
    mismatch_count = labels.get("texture_mismatch", 0) + labels.get("edge_unreliable", 0) + labels.get("occlusion_or_change", 0)
    total_rows = max(len(rows), 1)
    consistent_global_shift = bool(
        len(per_image) > 0
        and len(low_direction_images) <= max(1, len(per_image) // 3)
        and (_direction_consistency(high) or 0.0) > 0.7
    )
    if consistent_global_shift:
        next_step = "multi-image high-confidence offsets are directionally stable; run small pose/principal-point/height residual grid search"
    elif mismatch_count / total_rows >= 0.45:
        next_step = "texture, change, or edge unreliability dominates; replace global edge/chamfer with robust patch filtering or semantic/structure weighting"
    elif len(low_direction_images) >= max(1, int(0.6 * len(per_image))):
        next_step = "offset directions remain dispersed across images; broaden manual patch review before changing geometry"
    else:
        next_step = "mixed evidence; inspect per-image reports and rerun with more images or stricter patch labels"

    return {
        "experiment": "Batch query DOM patch alignment",
        "query_dir": args.query_dir,
        "xml": args.xml,
        "pose_file": args.pose_file,
        "config": args.config,
        "output_dir": args.output_dir,
        "limit": args.limit,
        "num_images_requested": len(image_summaries) + len(failures),
        "num_images_ok": len(image_summaries),
        "num_images_failed": len(failures),
        "num_patch_rows": len(rows),
        "num_high_confidence_patches": len(high),
        "batch_direction_consistency_high_confidence": _direction_consistency(high),
        "median_best_dx_px_high_confidence": _percentile([_float(r, "best_dx_px") for r in high], 50),
        "median_best_dy_px_high_confidence": _percentile([_float(r, "best_dy_px") for r in high], 50),
        "median_offset_norm_px_high_confidence": _percentile([_float(r, "offset_norm_px") for r in high], 50),
        "offset_norm_p90_px_high_confidence": _percentile([_float(r, "offset_norm_px") for r in high], 90),
        "label_counts": dict(labels),
        "high_confidence_label_counts": dict(high_labels),
        "label_ratios": {k: float(v / total_rows) for k, v in labels.items()},
        "mean_query_render_ncc_high_confidence": _mean([_float(r, "query_render_ncc") for r in high]),
        "mean_query_dom_ncc_high_confidence": _mean([_float(r, "query_dom_ncc") for r in high]),
        "mean_render_dom_ncc_high_confidence": _mean([_float(r, "render_dom_ncc") for r in high]),
        "mean_edge_chamfer_before_high_confidence": _mean([_float(r, "edge_chamfer_before") for r in high]),
        "mean_edge_chamfer_after_high_confidence": _mean([_float(r, "edge_chamfer_after") for r in high]),
        "low_direction_consistency_image_count": len(low_direction_images),
        "consistent_global_shift_likely": consistent_global_shift,
        "recommended_next_step": next_step,
        "per_image": per_image,
        "failures": failures,
    }


def _write_plots(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    high = [r for r in rows if _bool(r, "high_confidence")]
    if not high:
        return
    dx = np.asarray([_float(r, "best_dx_px", 0.0) for r in high], dtype=np.float64)
    dy = np.asarray([_float(r, "best_dy_px", 0.0) for r in high], dtype=np.float64)
    norm = np.hypot(dx, dy)

    plt.figure(figsize=(6, 6))
    plt.scatter(dx, dy, c=norm, cmap="viridis")
    plt.axhline(0, color="0.7")
    plt.axvline(0, color="0.7")
    plt.xlabel("dx px")
    plt.ylabel("dy px")
    plt.title("Batch high-confidence patch offsets")
    plt.colorbar(label="offset norm px")
    plt.tight_layout()
    plt.savefig(output_dir / "batch_offset_scatter.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(norm, bins=24)
    plt.xlabel("offset norm px")
    plt.ylabel("count")
    plt.title("Batch offset norm histogram")
    plt.tight_layout()
    plt.savefig(output_dir / "batch_offset_histogram.png", dpi=160)
    plt.close()

    labels = Counter(r.get("patch_label", "unknown") for r in rows)
    plt.figure(figsize=(8, 4))
    names = list(labels.keys())
    vals = [labels[n] for n in names]
    plt.bar(names, vals)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("patch count")
    plt.title("Patch label counts")
    plt.tight_layout()
    plt.savefig(output_dir / "batch_patch_label_counts.png", dpi=160)
    plt.close()


def _write_html(path: Path, summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    parts = [
        "<html><head><meta charset='utf-8'><style>body{font-family:sans-serif} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:5px} img{max-width:160px} .bad{color:#b00020}.muted{color:#777}</style></head><body>",
        "<h1>Batch Query DOM Patch Alignment</h1>",
        "<pre>" + json.dumps(_safe_jsonable({k: v for k, v in summary.items() if k not in {"per_image", "failures"}}), indent=2, sort_keys=True) + "</pre>",
        "<h2>Per image</h2>",
        "<table><tr><th>image</th><th>photo</th><th>patches</th><th>high conf</th><th>dir consistency</th><th>median dx/dy/norm</th><th>labels</th><th>report</th></tr>",
    ]
    for item in summary.get("per_image", []):
        out = item.get("output_dir", "")
        parts.append(
            "<tr>"
            f"<td>{item.get('query_image')}</td>"
            f"<td>{item.get('xml_photo_id')}</td>"
            f"<td>{item.get('candidate_patch_count')}</td>"
            f"<td>{item.get('high_confidence_patch_count')}</td>"
            f"<td>{item.get('direction_consistency')}</td>"
            f"<td>{item.get('median_best_dx_px')}, {item.get('median_best_dy_px')}, {item.get('median_offset_norm_px')}</td>"
            f"<td>{item.get('label_counts')}</td>"
            f"<td><a href='{os.path.relpath(REPO_ROOT / out / 'patch_alignment_report.html', path.parent).replace(chr(92), '/') if out else ''}'>report</a></td>"
            "</tr>"
        )
    parts.extend(
        [
            "</table>",
            "<h2>Patch samples</h2>",
            "<table><tr><th>image</th><th>#</th><th>label</th><th>high</th><th>dx,dy</th><th>NCC</th><th>edge before/after</th><th>query</th><th>render</th><th>after</th><th>DOM</th></tr>",
        ]
    )
    for row in rows[:200]:
        patch_dir = row.get("batch_patch_rel_dir", "")
        label = row.get("patch_label", "")
        cls = " class='bad'" if label in {"texture_mismatch", "edge_unreliable", "occlusion_or_change"} else ""
        parts.append(
            f"<tr{cls}>"
            f"<td>{row.get('image_name')}</td><td>{row.get('index')}</td><td>{label}</td><td>{row.get('high_confidence')}</td>"
            f"<td>{row.get('best_dx_px')}, {row.get('best_dy_px')}</td><td>{row.get('query_render_ncc')}</td>"
            f"<td>{row.get('edge_chamfer_before')} / {row.get('edge_chamfer_after')}</td>"
            f"<td><img src='{patch_dir}/query.png'></td>"
            f"<td><img src='{patch_dir}/render.png'></td>"
            f"<td><img src='{patch_dir}/overlay_after.png'></td>"
            f"<td><img src='{patch_dir}/dom.png'></td>"
            "</tr>"
        )
    parts.append("</table></body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--search-radius", type=int, default=80)
    parser.add_argument("--max-patches", type=int, default=100)
    parser.add_argument("--min-gradient", type=float, default=9.0)
    parser.add_argument("--min-edge-count", type=int, default=180)
    parser.add_argument("--ray-step-m", type=float, default=2.0)
    parser.add_argument("--ray-max-m", type=float, default=700.0)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_dir = (REPO_ROOT / args.query_dir).resolve()
    images = _list_images(query_dir, args.limit)
    if not images:
        raise FileNotFoundError(f"No query images found in {query_dir}")

    image_summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    batch_rows: List[Dict[str, Any]] = []
    per_image_root = output_dir / "images"
    for idx, image_path in enumerate(images):
        image_output = per_image_root / f"{idx:03d}_{image_path.stem}"
        summary = _run_single(args, image_path, image_output)
        if summary.get("status") != "ok":
            failures.append(summary)
            continue
        image_summaries.append(summary)
        batch_rows.extend(_aggregate_rows(image_path, image_output, summary, output_dir))

    summary = _summarize_batch(image_summaries, batch_rows, failures, args)
    _write_csv(output_dir / "batch_patch_alignment_results.csv", batch_rows)
    _write_json(output_dir / "batch_patch_alignment_summary.json", summary)
    _write_plots(output_dir, batch_rows)
    _write_html(output_dir / "batch_patch_alignment_report.html", summary, batch_rows)
    print(json.dumps(_safe_jsonable(summary), indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
