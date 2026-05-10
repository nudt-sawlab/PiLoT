#!/usr/bin/env python3
"""Grid-search yawfix translation offsets around the initial pose."""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import rasterio
import yaml
from pyproj import Transformer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test_16x9/0000.jpg"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt"
DEFAULT_OUTPUT_DIR = (
    "docs/experiments/dom_dsm_prepare/yawfix_translation_grid_results"
)
REFINED_TRANSLATION = [114.43672403988498, 30.391339299808664, 395.5591768939048]


def scaled_camera(cam_cfg: Dict[str, Any], render_width: int) -> np.ndarray:
    source_w = float(cam_cfg["width"])
    source_h = float(cam_cfg["height"])
    scale = render_width / source_w
    render_height = int(round(source_h * scale))
    fx, fy, cx, cy = map(float, cam_cfg["params"])
    return np.array(
        [render_width, render_height, cx * scale, cy * scale, fx * scale, fy * scale],
        dtype=np.float64,
    )


def read_pose_file(path: Path, image_name: str) -> Tuple[List[float], List[float]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0] != image_name:
            continue
        if len(parts) != 7:
            raise ValueError(f"Invalid pose row in {path}: {line}")
        lon, lat, alt, roll, pitch, yaw = map(float, parts[1:])
        return [lon, lat, alt], [pitch, roll, yaw]
    raise KeyError(f"{image_name} not found in {path}")


def read_query_rgb(path: Path, width: int, height: int) -> np.ndarray:
    query_bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if query_bgr is None:
        raise FileNotFoundError(path)
    query_bgr = cv2.resize(query_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    cv2.imwrite(os.fspath(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def make_overlay(query_rgb: np.ndarray, render_rgb: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(query_rgb, 0.5, render_rgb, 0.5, 0)


def edges(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, 120, 240) > 0


def symmetric_chamfer(query_edges: np.ndarray, render_edges: np.ndarray) -> float:
    if not np.any(query_edges) or not np.any(render_edges):
        return float("inf")
    dist_to_render = cv2.distanceTransform(
        (~render_edges).astype(np.uint8), cv2.DIST_L2, 3
    )
    dist_to_query = cv2.distanceTransform(
        (~query_edges).astype(np.uint8), cv2.DIST_L2, 3
    )
    return float((dist_to_render[query_edges].mean() + dist_to_query[render_edges].mean()) / 2.0)


def make_edge_overlay(query_rgb: np.ndarray, render_rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    query_edges = edges(query_rgb)
    render_edges = edges(render_rgb)
    kernel = np.ones((3, 3), dtype=np.uint8)
    query_dilated = cv2.dilate(query_edges.astype(np.uint8), kernel, iterations=1) > 0
    render_dilated = cv2.dilate(render_edges.astype(np.uint8), kernel, iterations=1) > 0
    overlap = (query_edges & render_dilated) | (render_edges & query_dilated)

    out = make_overlay(query_rgb, render_rgb)
    out[render_edges] = [255, 40, 40]
    out[query_edges] = [40, 255, 40]
    out[overlap] = [255, 255, 40]

    query_count = int(query_edges.sum())
    render_count = int(render_edges.sum())
    overlap_count = int(overlap.sum())
    return out, {
        "edge_overlap_ratio": float(overlap_count / max(min(query_count, render_count), 1)),
        "edge_chamfer": symmetric_chamfer(query_edges, render_edges),
        "query_edge_count": query_count,
        "render_edge_count": render_count,
        "edge_overlap_count": overlap_count,
    }


def make_checkerboard(query_rgb: np.ndarray, render_rgb: np.ndarray, tile: int) -> np.ndarray:
    height, width = query_rgb.shape[:2]
    yy, xx = np.indices((height, width))
    mask = ((xx // tile) + (yy // tile)) % 2 == 0
    out = render_rgb.copy()
    out[mask] = query_rgb[mask]
    return out


def depth_stats(depth: np.ndarray) -> Dict[str, Any]:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return {"valid_depth_ratio": 0.0, "depth_min": None, "depth_max": None}
    return {
        "valid_depth_ratio": float(valid.mean()),
        "depth_min": float(depth[valid].min()),
        "depth_max": float(depth[valid].max()),
    }


def parse_range(values: List[float]) -> List[float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("range must be start stop step")
    start, stop, step = values
    if step <= 0:
        raise argparse.ArgumentTypeError("range step must be positive")
    count = int(round((stop - start) / step))
    return [float(start + i * step) for i in range(count + 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE, type=Path)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--east-range", nargs=3, type=float, default=[-30, 30, 5])
    parser.add_argument("--north-range", nargs=3, type=float, default=[-30, 30, 5])
    parser.add_argument("--alt-offsets", nargs="+", type=float, default=[0.0])
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--top-k", default=10, type=int)
    parser.add_argument("--checker-tile", default=32, type=int)
    parser.add_argument("--summary-every", default=5, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def transform_offset(
    trans: List[float],
    east_m: float,
    north_m: float,
    alt_m: float,
    to_raster: Transformer,
    from_raster: Transformer,
) -> List[float]:
    lon, lat, alt = map(float, trans)
    x, y = to_raster.transform(lon, lat)
    lon2, lat2 = from_raster.transform(x + east_m, y + north_m)
    return [float(lon2), float(lat2), float(alt + alt_m)]


def metric_for_pose(
    renderer: DOMDSMRenderer,
    query_rgb: np.ndarray,
    trans: List[float],
    euler: List[float],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    render_rgb, depth = renderer.render(trans, euler)
    render_time = time.perf_counter() - t0
    _edge_overlay_img, edge_metrics = make_edge_overlay(query_rgb, render_rgb)
    return {
        **extra,
        "translation_lon_lat_alt": trans,
        "euler_pitch_roll_yaw": euler,
        "render_time_sec": render_time,
        **depth_stats(depth),
        **edge_metrics,
    }


def save_visuals(
    out_dir: Path,
    renderer: DOMDSMRenderer,
    query_rgb: np.ndarray,
    trans: List[float],
    euler: List[float],
    metrics: Dict[str, Any],
    checker_tile: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    render_rgb, _depth = renderer.render(trans, euler)
    overlay = make_overlay(query_rgb, render_rgb)
    edge_overlay, _edge_metrics = make_edge_overlay(query_rgb, render_rgb)
    checkerboard = make_checkerboard(query_rgb, render_rgb, checker_tile)
    write_rgb(out_dir / "rendered_rgb.png", render_rgb)
    write_rgb(out_dir / "overlay.png", overlay)
    write_rgb(out_dir / "edge_overlay.png", edge_overlay)
    write_rgb(out_dir / "checkerboard.png", checkerboard)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def candidate_key(east_m: float, north_m: float, alt_m: float, name: str = "grid") -> str:
    return f"{name}:E{east_m:+.3f}:N{north_m:+.3f}:A{alt_m:+.3f}"


def load_partial(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    items: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        items[item["candidate_key"]] = item
    return items


def append_partial(path: Path, item: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, sort_keys=True) + "\n")


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    raster_crs: Any,
    width: int,
    height: int,
    base_trans: List[float],
    base_euler: List[float],
    east_values: List[float],
    north_values: List[float],
    alt_values: List[float],
    refined_offset: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    grid_only = [
        item for item in candidates if item["candidate"] in {"grid", "base_initial"}
    ]
    top_by_overlap = sorted(grid_only, key=lambda item: item["edge_overlap_ratio"], reverse=True)
    top_by_chamfer = sorted(grid_only, key=lambda item: item["edge_chamfer"])
    base_metrics = next(
        (item for item in candidates if item["candidate"] == "base_initial"),
        None,
    )
    refined_metrics = next(
        (
            item
            for item in candidates
            if item["candidate"] == "refined_translation_initial_rotation"
        ),
        None,
    )
    summary = {
        "config_path": os.fspath(args.config),
        "query_image_path": os.fspath(args.query_image),
        "pose_file_path": os.fspath(args.pose_file),
        "output_dir": os.fspath(args.output_dir),
        "image_size": {"width": width, "height": height},
        "raster_crs": str(raster_crs),
        "base_translation_lon_lat_alt": base_trans,
        "base_euler_pitch_roll_yaw": base_euler,
        "east_values_m": east_values,
        "north_values_m": north_values,
        "alt_offsets_m": alt_values,
        "refined_translation_offset_m": refined_offset,
        "base_initial_metrics": base_metrics,
        "refined_translation_initial_rotation_metrics": refined_metrics,
        "candidate_count": len(candidates),
        "grid_candidate_count": len(grid_only),
        "top_by_edge_overlap_ratio": top_by_overlap[: args.top_k],
        "top_by_edge_chamfer": top_by_chamfer[: args.top_k],
        "best_by_edge_overlap_ratio": top_by_overlap[0] if top_by_overlap else None,
        "best_by_edge_chamfer": top_by_chamfer[0] if top_by_chamfer else None,
        "all_candidates": candidates,
    }
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists() and not args.resume:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "partial_metrics.jsonl"
    partial = load_partial(partial_path) if args.resume else {}

    config = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    render_config = dict(config["render_config"])
    render_config["dom_dsm"] = dict(render_config["dom_dsm"])
    render_config["dom_dsm"]["debug_dir"] = None
    render_config["render_camera"] = scaled_camera(
        config["default_confs"]["cam_query"],
        args.width,
    )

    dom_path = Path(render_config["dom_dsm"]["dom_path"])
    with rasterio.open(dom_path) as dom:
        raster_crs = dom.crs
    to_raster = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    from_raster = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)

    width = int(render_config["render_camera"][0])
    height = int(render_config["render_camera"][1])
    query_rgb = read_query_rgb((REPO_ROOT / args.query_image), width, height)
    base_trans, base_euler = read_pose_file(REPO_ROOT / args.pose_file, Path(args.query_image).name)
    renderer = DOMDSMRenderer(render_config)

    refined_offset = None
    if REFINED_TRANSLATION:
        bx, by = to_raster.transform(base_trans[0], base_trans[1])
        rx, ry = to_raster.transform(REFINED_TRANSLATION[0], REFINED_TRANSLATION[1])
        refined_offset = {
            "east_offset_m": float(rx - bx),
            "north_offset_m": float(ry - by),
            "alt_offset_m": float(REFINED_TRANSLATION[2] - base_trans[2]),
            "translation_lon_lat_alt": REFINED_TRANSLATION,
        }

    candidates: List[Dict[str, Any]] = list(partial.values())
    base_key = candidate_key(0.0, 0.0, 0.0, "base_initial")
    if base_key not in partial:
        base_metrics = metric_for_pose(
            renderer,
            query_rgb,
            base_trans,
            base_euler,
            {
                "candidate": "base_initial",
                "candidate_key": base_key,
                "east_offset_m": 0.0,
                "north_offset_m": 0.0,
                "alt_offset_m": 0.0,
            },
        )
        append_partial(partial_path, base_metrics)
        candidates.append(base_metrics)

    if refined_offset:
        refined_key = candidate_key(
            refined_offset["east_offset_m"],
            refined_offset["north_offset_m"],
            refined_offset["alt_offset_m"],
            "refined_translation_initial_rotation",
        )
        if refined_key not in partial:
            refined_metrics = metric_for_pose(
                renderer,
                query_rgb,
                REFINED_TRANSLATION,
                base_euler,
                {
                    "candidate": "refined_translation_initial_rotation",
                    "candidate_key": refined_key,
                    "east_offset_m": refined_offset["east_offset_m"],
                    "north_offset_m": refined_offset["north_offset_m"],
                    "alt_offset_m": refined_offset["alt_offset_m"],
                },
            )
            append_partial(partial_path, refined_metrics)
            candidates.append(refined_metrics)

    east_values = parse_range(args.east_range)
    north_values = parse_range(args.north_range)
    alt_values = [float(v) for v in args.alt_offsets]
    total_grid = len(east_values) * len(north_values) * len(alt_values)
    done_grid = 0
    for alt_m in alt_values:
        for north_m in north_values:
            for east_m in east_values:
                done_grid += 1
                if east_m == 0.0 and north_m == 0.0 and alt_m == 0.0:
                    continue
                key = candidate_key(east_m, north_m, alt_m)
                if key in partial:
                    continue
                trans = transform_offset(base_trans, east_m, north_m, alt_m, to_raster, from_raster)
                metrics = metric_for_pose(
                    renderer,
                    query_rgb,
                    trans,
                    base_euler,
                    {
                        "candidate": "grid",
                        "candidate_key": key,
                        "east_offset_m": east_m,
                        "north_offset_m": north_m,
                        "alt_offset_m": alt_m,
                    },
                )
                candidates.append(metrics)
                append_partial(partial_path, metrics)
                print(
                    json.dumps(
                        {
                            "progress": f"{done_grid}/{total_grid}",
                            "east_offset_m": east_m,
                            "north_offset_m": north_m,
                            "alt_offset_m": alt_m,
                            "edge_overlap_ratio": metrics["edge_overlap_ratio"],
                            "edge_chamfer": metrics["edge_chamfer"],
                        },
                        sort_keys=True,
                    )
                )
                if len(candidates) % args.summary_every == 0:
                    write_summary(
                        output_dir,
                        args,
                        raster_crs,
                        width,
                        height,
                        base_trans,
                        base_euler,
                        east_values,
                        north_values,
                        alt_values,
                        refined_offset,
                        candidates,
                    )

    summary = write_summary(
        output_dir,
        args,
        raster_crs,
        width,
        height,
        base_trans,
        base_euler,
        east_values,
        north_values,
        alt_values,
        refined_offset,
        candidates,
    )

    top_by_overlap = summary["top_by_edge_overlap_ratio"]
    for rank, item in enumerate(top_by_overlap, start=1):
        name = f"top_{rank:02d}_E{item['east_offset_m']:+.0f}_N{item['north_offset_m']:+.0f}_A{item['alt_offset_m']:+.0f}"
        save_visuals(
            output_dir / name,
            renderer,
            query_rgb,
            item["translation_lon_lat_alt"],
            base_euler,
            item,
            args.checker_tile,
        )

    print(json.dumps(summary["best_by_edge_overlap_ratio"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
