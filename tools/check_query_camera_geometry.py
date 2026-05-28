#!/usr/bin/env python3
"""Check query image geometry against cam_query and render_camera_gs.

This is a read-only diagnostic script. It mirrors the camera-size arithmetic
used by the single-image DOM/DSM experiment without running localization.
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.pixlib.datasets.view import read_image


DEFAULT_CONFIG = "configs/caiwangcun_domdsm.yaml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test/0000.jpg"
ASPECT_WARNING_THRESHOLD = 0.01


def aspect(width: float, height: float) -> float:
    return float(width) / float(height)


def aspect_delta(a: float, b: float) -> float:
    return max(abs(a), abs(b)) / max(min(abs(a), abs(b)), 1e-12) - 1.0


def read_original_size(path: Path) -> Tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    return int(width), int(height)


def setup_camera_geometry(config: Dict[str, Any]) -> Tuple[float, np.ndarray, np.ndarray]:
    cam_cfg = copy.deepcopy(config["default_confs"]["cam_query"])

    query_resize_ratio = float(cam_cfg["width"]) / float(cam_cfg["max_size"])
    fx, fy, cx, cy = map(float, cam_cfg["params"])
    width = float(cam_cfg["width"])
    height = float(cam_cfg["height"])

    raw_query_camera = np.array([width, height, cx, cy, fx, fy], dtype=np.float64)
    render_camera_gs = np.array(
        [width, height, width / 2.0, height / 2.0, fx, fx],
        dtype=np.float64,
    )
    render_camera_gs = render_camera_gs / query_resize_ratio

    return query_resize_ratio, raw_query_camera, render_camera_gs


def print_aspect(name: str, width: float, height: float) -> float:
    ratio = aspect(width, height)
    print(f"{name}: width={width:g}, height={height:g}, aspect={ratio:.6f}")
    return ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE, type=Path)
    parser.add_argument(
        "--aspect-warning-threshold",
        default=ASPECT_WARNING_THRESHOLD,
        type=float,
        help="Relative aspect-ratio warning threshold. Default: 0.01.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = (REPO_ROOT / args.config).resolve()
    query_image_path = (REPO_ROOT / args.query_image).resolve()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cam_query = config["default_confs"]["cam_query"]
    query_resize_ratio, raw_query_camera, render_camera_gs = setup_camera_geometry(config)

    original_width, original_height = read_original_size(query_image_path)
    original_aspect = print_aspect("original_image", original_width, original_height)

    cam_width = float(cam_query["width"])
    cam_height = float(cam_query["height"])
    cam_aspect = print_aspect("cam_query", cam_width, cam_height)
    cam_delta = aspect_delta(original_aspect, cam_aspect)
    print(f"original_vs_cam_query_aspect_delta={cam_delta:.6%}")

    if cam_delta > args.aspect_warning_threshold:
        print(
            "WARNING: original image aspect ratio differs from cam_query aspect "
            f"ratio by {cam_delta:.2%}, above {args.aspect_warning_threshold:.2%}."
        )

    query_image = read_image(
        query_image_path,
        scale=query_resize_ratio,
        distortion=cam_query.get("distortion"),
        query_camera=raw_query_camera,
    )
    print(f"query_resize_ratio={query_resize_ratio:g}")
    print(f"query_image_shape_after_read_image={list(query_image.shape)}")
    read_height, read_width = query_image.shape[:2]
    read_aspect = print_aspect("read_image_output", read_width, read_height)

    render_width = float(render_camera_gs[0])
    render_height = float(render_camera_gs[1])
    render_aspect = print_aspect("render_camera_gs", render_width, render_height)
    print(f"render_camera_gs={render_camera_gs.tolist()}")

    read_to_render_delta = aspect_delta(read_aspect, render_aspect)
    original_to_render_delta = aspect_delta(original_aspect, render_aspect)
    print(f"read_image_vs_render_camera_gs_aspect_delta={read_to_render_delta:.6%}")
    print(f"original_vs_render_camera_gs_aspect_delta={original_to_render_delta:.6%}")

    non_uniform_stretch = read_to_render_delta > args.aspect_warning_threshold
    if non_uniform_stretch:
        print(
            "non_uniform_stretch=YES: read_image output and render_camera_gs "
            "have different aspect ratios, so resizing between them would stretch "
            "x/y by different factors."
        )
    else:
        print(
            "non_uniform_stretch=NO: read_image output and render_camera_gs "
            "aspect ratios are within the configured threshold."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
