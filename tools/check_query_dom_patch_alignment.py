#!/usr/bin/env python3
"""Patch-level query/render/DOM alignment diagnostics for one XML-matched image."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import yaml
from pyproj import Transformer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from tools.check_query_domdsm_point_consistency import _intersect_dsm, _ray_from_pixel
from tools.diagnose_yawfix_refinement_update import (
    _checkerboard,
    _edge_overlay,
    _make_overlay,
    _safe_jsonable,
    _write_rgb,
)
from tools.render_contextcapture_xml_domdsm_initial import (
    ContextCaptureDOMDSMRenderer,
    _load_pose_file_projected,
    _match_photos,
    _parse_xml,
)


DEFAULT_XML = "data_caiwangcun/CaiWangCun.xml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test/0000.jpg"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_CONFIG = "configs/caiwangcun_domdsm.yaml"
DEFAULT_RENDER_IMAGE = "docs/experiments/dom_dsm_prepare/query_image_domain_vs_render_0000/gpu_pinhole_render.png"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/query_dom_patch_alignment_0000"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _camera_matrix(intr: Any) -> np.ndarray:
    return np.asarray(
        [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


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


def _render_gpu_if_needed(config: Dict[str, Any], intr: Any, xml_srs: str, photo: Any, render_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    if render_path.exists():
        return _read_rgb(render_path), {"source": os.fspath(render_path), "rendered_now": False}
    render_config = _make_gpu_config(config, intr)
    renderer = DOMDSMRenderer(render_config)
    to_wgs84 = Transformer.from_crs(xml_srs, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(photo.center_xml[0], photo.center_xml[1])
    rgb, _depth = renderer.render_matrix([float(lon), float(lat), float(photo.center_xml[2])], photo.rotation.T)
    render_path.parent.mkdir(parents=True, exist_ok=True)
    _write_rgb(render_path, rgb)
    meta = dict(renderer.last_render_metadata)
    meta.update({"source": os.fspath(render_path), "rendered_now": True})
    return rgb, meta


def _crop_fixed(image: np.ndarray, cx: int, cy: int, size: int) -> Optional[np.ndarray]:
    half = size // 2
    h, w = image.shape[:2]
    x0, x1 = int(cx - half), int(cx - half + size)
    y0, y1 = int(cy - half), int(cy - half + size)
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return image[y0:y1, x0:x1].copy()


def _crop_search(image: np.ndarray, cx: int, cy: int, patch_size: int, search_radius: int) -> Optional[Tuple[np.ndarray, int, int]]:
    half = patch_size // 2
    h, w = image.shape[:2]
    x0, x1 = int(cx - half - search_radius), int(cx + half + search_radius)
    y0, y1 = int(cy - half - search_radius), int(cy + half + search_radius)
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return image[y0:y1, x0:x1].copy(), x0, y0


def _gray_f32(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _edges_count(image_rgb: np.ndarray) -> int:
    gray = cv2.GaussianBlur(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    return int((cv2.Canny(gray, 120, 240) > 0).sum())


def _grad_energy(image_rgb: np.ndarray) -> float:
    gray = _gray_f32(image_rgb)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def _best_template_shift(query_patch: np.ndarray, render_search: np.ndarray, search_x0: int, search_y0: int, cx: int, cy: int) -> Dict[str, Any]:
    q = _gray_f32(query_patch)
    r = _gray_f32(render_search)
    result = cv2.matchTemplate(r, q, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
    patch_h, patch_w = query_patch.shape[:2]
    best_x = int(search_x0 + max_loc[0])
    best_y = int(search_y0 + max_loc[1])
    best_cx = best_x + patch_w // 2
    best_cy = best_y + patch_h // 2
    return {
        "best_x0": best_x,
        "best_y0": best_y,
        "best_dx_px": float(best_cx - cx),
        "best_dy_px": float(best_cy - cy),
        "ncc_score": float(max_val),
    }


def _phase_shift(query_patch: np.ndarray, render_patch: np.ndarray) -> Tuple[float, float, float]:
    q = _gray_f32(query_patch)
    r = _gray_f32(render_patch)
    window = cv2.createHanningWindow((q.shape[1], q.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(q, r, window)
    return float(dx), float(dy), float(response)


def _shifted_render_patch(render_rgb: np.ndarray, cx: int, cy: int, size: int, dx: float, dy: float) -> Optional[np.ndarray]:
    return _crop_fixed(render_rgb, int(round(cx + dx)), int(round(cy + dy)), size)


def _hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    b_resized = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA) if b.shape[:2] != a.shape[:2] else b
    ah = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    bh = cv2.cvtColor(b_resized, cv2.COLOR_RGB2HSV)
    ha = cv2.calcHist([ah], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hb = cv2.calcHist([bh], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))


def _simple_ssim(a: np.ndarray, b: np.ndarray) -> float:
    b_resized = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA) if b.shape[:2] != a.shape[:2] else b
    x = _gray_f32(a)
    y = _gray_f32(b_resized)
    c1 = 6.5025
    c2 = 58.5225
    mux, muy = float(x.mean()), float(y.mean())
    vx, vy = float(x.var()), float(y.var())
    cov = float(((x - mux) * (y - muy)).mean())
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2)))


def _same_size_ncc(a: np.ndarray, b: np.ndarray) -> float:
    b_resized = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA) if b.shape[:2] != a.shape[:2] else b
    x = _gray_f32(a)
    y = _gray_f32(b_resized)
    xs = float(x.std())
    ys = float(y.std())
    if xs < 1e-6 or ys < 1e-6:
        return 0.0
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (xs * ys))


def _gradient_orientation_consistency(a: np.ndarray, b: np.ndarray) -> float:
    b_resized = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA) if b.shape[:2] != a.shape[:2] else b
    ga = _gray_f32(a)
    gb = _gray_f32(b_resized)
    ax = cv2.Sobel(ga, cv2.CV_32F, 1, 0, ksize=3)
    ay = cv2.Sobel(ga, cv2.CV_32F, 0, 1, ksize=3)
    bx = cv2.Sobel(gb, cv2.CV_32F, 1, 0, ksize=3)
    by = cv2.Sobel(gb, cv2.CV_32F, 0, 1, ksize=3)
    amag = np.sqrt(ax * ax + ay * ay)
    bmag = np.sqrt(bx * bx + by * by)
    mask = (amag > np.percentile(amag, 75)) & (bmag > np.percentile(bmag, 75))
    if int(mask.sum()) < 32:
        return 0.0
    dot = ax * bx + ay * by
    cos = dot / np.maximum(amag * bmag, 1e-6)
    return float(np.mean(np.abs(cos[mask])))


def _edge_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    _img, metrics = _edge_overlay(a, b)
    return metrics


def _dom_patch_for_pixel(
    renderer: ContextCaptureDOMDSMRenderer,
    photo: Any,
    x: int,
    y: int,
    patch_size: int,
    step_m: float,
    ray_max_m: float,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    convention = {
        "principal_point_mode": "xml",
        "distortion_enabled": False,
        "axis_transform_key": "ppp",
    }
    origin, ray, _debug = _ray_from_pixel(renderer, photo, float(x), float(y), convention, "cam_to_world_correct")
    hit = _intersect_dsm(
        renderer,
        origin,
        ray,
        step_m=step_m,
        max_m=ray_max_m,
        dsm_sampling_mode="bilinear",
        dom_sampling_mode="bilinear",
        ray_refine_iters=10,
    )
    if hit is None:
        return None, {"dom_hit": False}
    dom_patch = _crop_fixed(renderer.dom_array[:, :, :3].astype(np.uint8), int(hit["dom_col"]), int(hit["dom_row"]), patch_size)
    if dom_patch is None:
        return None, {"dom_hit": False, **hit}
    return dom_patch, {"dom_hit": True, **hit}


def _write_patch_outputs(patch_dir: Path, query: np.ndarray, render_before: np.ndarray, render_after: np.ndarray, dom: Optional[np.ndarray]) -> Dict[str, str]:
    patch_dir.mkdir(parents=True, exist_ok=True)
    edge_before, _ = _edge_overlay(query, render_before)
    edge_after, _ = _edge_overlay(query, render_after)
    outputs = {
        "query_patch_rel": "query.png",
        "render_patch_rel": "render.png",
        "overlay_before_rel": "overlay_before.png",
        "overlay_after_rel": "overlay_after.png",
        "edge_overlay_before_rel": "edge_overlay_before.png",
        "edge_overlay_after_rel": "edge_overlay_after.png",
    }
    _write_rgb(patch_dir / "query.png", query)
    _write_rgb(patch_dir / "render.png", render_before)
    _write_rgb(patch_dir / "overlay_before.png", _make_overlay(query, render_before))
    _write_rgb(patch_dir / "overlay_after.png", _make_overlay(query, render_after))
    _write_rgb(patch_dir / "edge_overlay_before.png", edge_before)
    _write_rgb(patch_dir / "edge_overlay_after.png", edge_after)
    if dom is not None:
        _write_rgb(patch_dir / "dom.png", dom)
        triplet = np.concatenate(
            [
                query,
                render_before,
                cv2.resize(dom, (query.shape[1], query.shape[0]), interpolation=cv2.INTER_AREA),
            ],
            axis=1,
        )
        _write_rgb(patch_dir / "triplet_query_render_dom.png", triplet)
        outputs["dom_patch_rel"] = "dom.png"
        outputs["triplet_rel"] = "triplet_query_render_dom.png"
    return outputs


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else None


def _percentile(values: Sequence[Optional[float]], q: float) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.percentile(np.asarray(finite, dtype=np.float64), q)) if finite else None


def _direction_consistency(rows: List[Dict[str, Any]]) -> Optional[float]:
    if not rows:
        return None
    dx = np.asarray([float(r["best_dx_px"]) for r in rows], dtype=np.float64)
    dy = np.asarray([float(r["best_dy_px"]) for r in rows], dtype=np.float64)
    norms = np.hypot(dx, dy)
    if norms.size == 0 or float(norms.mean()) <= 1e-9:
        return None
    return float(math.hypot(float(dx.mean()), float(dy.mean())) / float(norms.mean()))


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    texture_rows = [r for r in rows if r.get("accepted_texture")]
    high = [r for r in texture_rows if r.get("high_confidence")]
    dc = _direction_consistency(high)
    median_norm = _percentile([r.get("offset_norm_px") for r in high], 50)
    median_dx = _percentile([r.get("best_dx_px") for r in high], 50)
    median_dy = _percentile([r.get("best_dy_px") for r in high], 50)
    edge_before = _mean([r.get("edge_chamfer_before") for r in high])
    edge_after = _mean([r.get("edge_chamfer_after") for r in high])
    edge_improves = bool(edge_before is not None and edge_after is not None and edge_after < edge_before * 0.8)
    query_render_ncc = _mean([r.get("query_render_ncc") for r in high])
    query_dom_ncc = _mean([r.get("query_dom_ncc") for r in high if r.get("dom_hit")])
    render_dom_ncc = _mean([r.get("render_dom_ncc") for r in high if r.get("dom_hit")])
    stable = bool(len(high) >= 10 and dc is not None and dc > 0.7 and median_norm is not None and median_norm > 5.0)
    texture_phase = bool(len(high) >= 10 and not stable and (query_dom_ncc is None or query_dom_ncc < 0.25) and (render_dom_ncc is None or render_dom_ncc > query_dom_ncc + 0.1))
    edge_reliable = bool(len(high) >= 10 and edge_improves and query_render_ncc is not None and query_render_ncc >= 0.25)
    if stable:
        next_step = "high-confidence local shifts are directionally stable; inspect small pose residuals or local DSM height errors"
    elif texture_phase:
        next_step = "query and DOM/render texture mismatch dominates; avoid using global edge metrics as geometry evidence"
    elif edge_improves and not edge_reliable:
        next_step = "edge metric can be improved by local shifting but photometric agreement is weak; replace or downweight edge-only objective"
    else:
        next_step = "review patch report manually and broaden to more images before changing geometry"
    return {
        "candidate_patch_count": int(len(rows)),
        "texture_candidate_count": int(len(texture_rows)),
        "high_confidence_patch_count": int(len(high)),
        "median_best_dx_px": median_dx,
        "median_best_dy_px": median_dy,
        "median_offset_norm_px": median_norm,
        "offset_norm_p90_px": _percentile([r.get("offset_norm_px") for r in high], 90),
        "direction_consistency": dc,
        "mean_query_render_ncc": query_render_ncc,
        "mean_query_dom_ncc": query_dom_ncc,
        "mean_render_dom_ncc": render_dom_ncc,
        "mean_edge_chamfer_before": edge_before,
        "mean_edge_chamfer_after": edge_after,
        "edge_metric_reliable": edge_reliable,
        "local_texture_phase_difference_likely": texture_phase,
        "stable_local_shift_likely": stable,
        "recommended_next_step": next_step,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row.keys() if not isinstance(row.get(k), (dict, list))})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_html(path: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    parts = [
        "<html><head><meta charset='utf-8'><style>body{font-family:sans-serif} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:5px} img{max-width:180px} .low{color:#777}</style></head><body>",
        "<h1>Query DOM Patch Alignment</h1>",
        "<pre>" + json.dumps(_safe_jsonable(summary), indent=2, sort_keys=True) + "</pre>",
        "<table><tr><th>#</th><th>xy</th><th>accepted</th><th>high conf</th><th>dx,dy</th><th>NCC</th><th>phase</th><th>edge before/after</th><th>q-r/q-d/r-d NCC</th><th>query</th><th>render</th><th>DOM</th><th>after</th><th>edges after</th></tr>",
    ]
    for row in rows:
        cls = "" if row.get("high_confidence") else " class='low'"
        patch_rel = row.get("patch_rel_dir")
        dom_img = f"<img src='{patch_rel}/dom.png'>" if row.get("dom_hit") else ""
        parts.append(
            f"<tr{cls}>"
            f"<td>{row['index']}</td><td>{row['center_x']},{row['center_y']}</td>"
            f"<td>{row.get('accepted_texture')}</td><td>{row.get('high_confidence')}</td>"
            f"<td>{row.get('best_dx_px'):.1f}, {row.get('best_dy_px'):.1f}</td>"
            f"<td>{row.get('ncc_score'):.3f}</td><td>{row.get('phase_response'):.3f}</td>"
            f"<td>{row.get('edge_chamfer_before'):.2f} / {row.get('edge_chamfer_after'):.2f}</td>"
            f"<td>{row.get('query_render_ncc'):.3f} / {row.get('query_dom_ncc')} / {row.get('render_dom_ncc')}</td>"
            f"<td><img src='{patch_rel}/query.png'></td>"
            f"<td><img src='{patch_rel}/render.png'></td>"
            f"<td>{dom_img}</td>"
            f"<td><img src='{patch_rel}/overlay_after.png'></td>"
            f"<td><img src='{patch_rel}/edge_overlay_after.png'></td>"
            "</tr>"
        )
    parts.append("</table></body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_plots(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    high = [r for r in rows if r.get("high_confidence")]
    if not high:
        return
    x = np.asarray([r["center_x"] for r in high], dtype=np.float64)
    y = np.asarray([r["center_y"] for r in high], dtype=np.float64)
    dx = np.asarray([r["best_dx_px"] for r in high], dtype=np.float64)
    dy = np.asarray([r["best_dy_px"] for r in high], dtype=np.float64)
    norm = np.hypot(dx, dy)

    plt.figure(figsize=(8, 6))
    plt.quiver(x, y, dx, dy, norm, angles="xy", scale_units="xy", scale=1.0, cmap="viridis")
    plt.gca().invert_yaxis()
    plt.xlabel("image x")
    plt.ylabel("image y")
    plt.title("High-confidence patch offsets")
    plt.colorbar(label="offset norm px")
    plt.tight_layout()
    plt.savefig(output_dir / "offset_quiver.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(dx, dy, c=norm, cmap="viridis")
    plt.axhline(0, color="0.7")
    plt.axvline(0, color="0.7")
    plt.xlabel("dx px")
    plt.ylabel("dy px")
    plt.title("Patch offset scatter")
    plt.colorbar(label="offset norm px")
    plt.tight_layout()
    plt.savefig(output_dir / "offset_scatter.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(norm, bins=20)
    plt.xlabel("offset norm px")
    plt.ylabel("count")
    plt.title("Patch offset norm histogram")
    plt.tight_layout()
    plt.savefig(output_dir / "offset_histogram.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--render-image", default=DEFAULT_RENDER_IMAGE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
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
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    with open(REPO_ROOT / args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    xml_srs, photos = _parse_xml((REPO_ROOT / args.xml).resolve())
    query_path = (REPO_ROOT / args.query_image).resolve()
    pose_records = _load_pose_file_projected((REPO_ROOT / args.pose_file).resolve(), xml_srs)
    matches, match_report = _match_photos([query_path], pose_records, photos)
    photo = matches[query_path.name]["photo"]
    intr = photo.intrinsics

    query_rgb = _read_rgb(query_path)
    render_rgb, render_meta = _render_gpu_if_needed(config, intr, xml_srs, photo, (REPO_ROOT / args.render_image).resolve())
    if query_rgb.shape[:2] != render_rgb.shape[:2]:
        raise ValueError(f"query/render shape mismatch: {query_rgb.shape[:2]} vs {render_rgb.shape[:2]}")

    cc_renderer = ContextCaptureDOMDSMRenderer(copy.deepcopy(config["render_config"]), xml_srs)
    h, w = query_rgb.shape[:2]
    half = args.patch_size // 2
    margin = half + args.search_radius
    valid_render = np.any(render_rgb > 0, axis=2)
    candidates: List[Tuple[int, int, float]] = []
    for y in range(margin, h - margin, args.stride):
        for x in range(margin, w - margin, args.stride):
            q_patch = _crop_fixed(query_rgb, x, y, args.patch_size)
            r_patch = _crop_fixed(render_rgb, x, y, args.patch_size)
            if q_patch is None or r_patch is None:
                continue
            valid_ratio = float(valid_render[y - half : y + half, x - half : x + half].mean())
            if valid_ratio < 0.8:
                continue
            q_grad = _grad_energy(q_patch)
            r_grad = _grad_energy(r_patch)
            q_edges = _edges_count(q_patch)
            r_edges = _edges_count(r_patch)
            score = min(q_grad, r_grad) + 0.01 * min(q_edges, r_edges)
            if q_grad >= args.min_gradient and r_grad >= args.min_gradient and q_edges >= args.min_edge_count and r_edges >= args.min_edge_count:
                candidates.append((x, y, score))
    candidates = sorted(candidates, key=lambda t: t[2], reverse=True)[: args.max_patches]

    rows: List[Dict[str, Any]] = []
    for idx, (x, y, texture_score) in enumerate(candidates):
        q_patch = _crop_fixed(query_rgb, x, y, args.patch_size)
        r_patch = _crop_fixed(render_rgb, x, y, args.patch_size)
        search = _crop_search(render_rgb, x, y, args.patch_size, args.search_radius)
        if q_patch is None or r_patch is None or search is None:
            continue
        search_patch, search_x0, search_y0 = search
        match = _best_template_shift(q_patch, search_patch, search_x0, search_y0, x, y)
        r_after = _shifted_render_patch(render_rgb, x, y, args.patch_size, match["best_dx_px"], match["best_dy_px"])
        if r_after is None:
            continue
        phase_dx, phase_dy, phase_response = _phase_shift(q_patch, r_patch)
        edge_before = _edge_metrics(q_patch, r_patch)
        edge_after = _edge_metrics(q_patch, r_after)
        dom_patch, dom_meta = _dom_patch_for_pixel(cc_renderer, photo, x, y, args.patch_size, args.ray_step_m, args.ray_max_m)

        q_render_ncc = _same_size_ncc(q_patch, r_after)
        q_dom_ncc = _same_size_ncc(q_patch, dom_patch) if dom_patch is not None else None
        r_dom_ncc = _same_size_ncc(r_after, dom_patch) if dom_patch is not None else None
        edge_improve = float(edge_before["edge_chamfer"] - edge_after["edge_chamfer"])
        high_conf = bool(
            match["ncc_score"] >= 0.22
            and phase_response >= 0.03
            and edge_improve > 1.0
            and edge_after["edge_overlap_ratio"] >= edge_before["edge_overlap_ratio"]
        )

        patch_dir = patches_dir / f"patch_{idx:03d}"
        outputs = _write_patch_outputs(patch_dir, q_patch, r_patch, r_after, dom_patch)
        row: Dict[str, Any] = {
            "index": idx,
            "center_x": int(x),
            "center_y": int(y),
            "patch_rel_dir": os.path.relpath(patch_dir, output_dir).replace("\\", "/"),
            "texture_score": float(texture_score),
            "accepted_texture": True,
            "high_confidence": high_conf,
            "query_gradient": _grad_energy(q_patch),
            "render_gradient": _grad_energy(r_patch),
            "query_edge_count": _edges_count(q_patch),
            "render_edge_count": _edges_count(r_patch),
            "best_dx_px": float(match["best_dx_px"]),
            "best_dy_px": float(match["best_dy_px"]),
            "offset_norm_px": float(math.hypot(match["best_dx_px"], match["best_dy_px"])),
            "ncc_score": float(match["ncc_score"]),
            "phase_dx_px": phase_dx,
            "phase_dy_px": phase_dy,
            "phase_response": phase_response,
            "edge_chamfer_before": float(edge_before["edge_chamfer"]),
            "edge_chamfer_after": float(edge_after["edge_chamfer"]),
            "edge_overlap_before": float(edge_before["edge_overlap_ratio"]),
            "edge_overlap_after": float(edge_after["edge_overlap_ratio"]),
            "edge_chamfer_improvement": edge_improve,
            "query_render_ncc": q_render_ncc,
            "query_render_ssim": _simple_ssim(q_patch, r_after),
            "query_render_hist_distance": _hist_distance(q_patch, r_after),
            "query_render_gradient_orientation": _gradient_orientation_consistency(q_patch, r_after),
            "dom_hit": bool(dom_patch is not None),
            "query_dom_ncc": q_dom_ncc,
            "render_dom_ncc": r_dom_ncc,
            "query_dom_ssim": _simple_ssim(q_patch, dom_patch) if dom_patch is not None else None,
            "render_dom_ssim": _simple_ssim(r_after, dom_patch) if dom_patch is not None else None,
            "query_dom_hist_distance": _hist_distance(q_patch, dom_patch) if dom_patch is not None else None,
            "render_dom_hist_distance": _hist_distance(r_after, dom_patch) if dom_patch is not None else None,
            "query_dom_gradient_orientation": _gradient_orientation_consistency(q_patch, dom_patch) if dom_patch is not None else None,
            "render_dom_gradient_orientation": _gradient_orientation_consistency(r_after, dom_patch) if dom_patch is not None else None,
            **{k: os.path.join(os.path.relpath(patch_dir, output_dir).replace("\\", "/"), v) for k, v in outputs.items()},
            **{f"dom_{k}": v for k, v in dom_meta.items() if isinstance(v, (str, int, float, bool)) or v is None},
        }
        rows.append(row)

    summary = _summarize(rows)
    summary.update(
        {
            "experiment": "Query DOM patch alignment",
            "xml": args.xml,
            "xml_srs": xml_srs,
            "query_image": args.query_image,
            "render_image": args.render_image,
            "config": args.config,
            "output_dir": args.output_dir,
            "xml_photo_id": photo.photo_id,
            "xml_image_path": photo.image_path,
            "K_xml": _camera_matrix(intr),
            "render_meta": render_meta,
            "patch_size": int(args.patch_size),
            "stride": int(args.stride),
            "search_radius": int(args.search_radius),
            "max_patches": int(args.max_patches),
            "min_gradient": float(args.min_gradient),
            "min_edge_count": int(args.min_edge_count),
            "camera_match_report": match_report,
        }
    )
    _write_csv(output_dir / "patch_alignment_results.csv", rows)
    _write_json(output_dir / "patch_alignment_summary.json", summary)
    _write_html(output_dir / "patch_alignment_report.html", rows, summary)
    _write_plots(output_dir, rows)
    print(json.dumps(_safe_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
