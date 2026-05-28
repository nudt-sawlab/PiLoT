#!/usr/bin/env python3
"""Single-process DOM+DSM full PiLoT test for one query image.

This script intentionally avoids the regular entrypoint so CUDA tensors never
cross a process boundary. It renders one DOM+DSM view, back-projects the depth,
and calls RenderLocalizer in the same Python process.
"""

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

import cv2
import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.localization.localizer import RenderLocalizer
from pixloc.pixlib.datasets.view import read_image
from pixloc.pixlib.geometry import Camera
from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from pixloc.utils.get_depth import generate_render_camera, pad_to_multiple, sample_3d_points
from pixloc.utils.transform import euler_angles_to_matrix_ECEF
from src.utils.pose_utils import load_initial_pose, load_pose_dict


DEFAULT_CONFIG = "configs/caiwangcun_domdsm.yaml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test/0000.jpg"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test.txt"
DEFAULT_OUTPUT_DIR = "outputs/exif_test_single_full"


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(data), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _depth_stats(depth: np.ndarray) -> Dict[str, Optional[float]]:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    total = int(depth.size)
    if valid.size == 0:
        return {
            "valid_depth_ratio": 0.0,
            "depth_min": None,
            "depth_max": None,
        }
    return {
        "valid_depth_ratio": float(valid.size / max(total, 1)),
        "depth_min": float(valid.min()),
        "depth_max": float(valid.max()),
    }


def _save_depth_png(path: Path, depth: np.ndarray) -> None:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        depth_vis = np.zeros(depth.shape, dtype=np.uint8)
    else:
        d_min = float(valid.min())
        d_max = float(valid.max())
        scaled = (depth.astype(np.float32) - d_min) / max(d_max - d_min, 1e-6)
        scaled[depth <= 0] = 0
        depth_vis = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(os.fspath(path), depth_vis)


def _save_overlay(path: Path, query_rgb: np.ndarray, render_rgb: np.ndarray) -> None:
    if query_rgb.shape[:2] != render_rgb.shape[:2]:
        raise ValueError(
            "Query/render overlay requires equal HxW, got "
            f"query={query_rgb.shape[:2]} render={render_rgb.shape[:2]}"
        )
    overlay_rgb = cv2.addWeighted(query_rgb, 0.5, render_rgb, 0.5, 0)
    cv2.imwrite(os.fspath(path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))


def _resize_query_for_refine(
    query_rgb: np.ndarray,
    render_camera_gs: np.ndarray,
) -> np.ndarray:
    width = int(render_camera_gs[0])
    height = int(render_camera_gs[1])
    if query_rgb.shape[:2] == (height, width):
        return query_rgb
    return cv2.resize(query_rgb, (width, height), interpolation=cv2.INTER_AREA)


def _setup_camera(
    config: Dict[str, Any],
) -> Tuple[float, np.ndarray, np.ndarray, Camera, Camera]:
    default_confs = config["default_confs"]
    render_config = config["render_config"]
    cam_cfg = copy.deepcopy(default_confs["cam_query"])

    query_resize_ratio = cam_cfg["width"] / cam_cfg["max_size"]
    fx, fy, cx, cy = cam_cfg["params"]
    w, h = cam_cfg["width"], cam_cfg["height"]

    raw_query_camera = np.array([w, h, cx, cy, fx, fy])
    render_camera_gs = np.array([w, h, cx, cy, fx, fy])
    render_camera_gs = render_camera_gs / query_resize_ratio

    cam_cfg["params"] = np.array(cam_cfg["params"]) / query_resize_ratio
    cam_cfg["width"] /= query_resize_ratio
    cam_cfg["height"] /= query_resize_ratio

    query_camera = Camera.from_colmap(cam_cfg)
    render_camera = generate_render_camera(render_camera_gs).float()
    render_config["render_camera"] = render_camera_gs

    return (
        float(query_resize_ratio),
        raw_query_camera,
        render_camera_gs,
        query_camera,
        render_camera,
    )


def _back_project(
    depth_frame: np.ndarray,
    euler_angles: List[float],
    translation: List[float],
    query_euler: List[float],
    query_trans: List[float],
    render_camera_gs: np.ndarray,
    render_camera: Camera,
    origin: torch.Tensor,
    mul: float,
    device: str,
    is_init: bool = True,
    num_samples: int = 500,
    depth_min: float = 1.0,
    depth_max: float = 5000.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    depth = torch.as_tensor(depth_frame, device=device)

    T_c2w = torch.as_tensor(
        euler_angles_to_matrix_ECEF(euler_angles, translation),
        device=device,
        dtype=torch.float32,
    )

    height = int(render_camera_gs[1])
    width = int(render_camera_gs[0])

    oversample = num_samples * 4
    ys = torch.randint(0, height, size=(oversample,), device=device)
    xs = torch.randint(0, width, size=(oversample,), device=device)
    d_vals = depth[ys, xs]
    valid = (d_vals >= depth_min) & (d_vals <= depth_max)
    xs, ys = xs[valid][:num_samples], ys[valid][:num_samples]
    points2d = torch.stack((xs, ys), dim=1)

    return sample_3d_points(
        points2d,
        depth,
        T_c2w,
        render_camera,
        query_euler,
        query_trans,
        origin=origin,
        mul=mul,
        is_init_frame=is_init,
    )


def _format_pose_line(qname: str, trans: Any, euler: Any) -> str:
    trans_arr = np.asarray(_jsonable(trans), dtype=float).reshape(-1)
    euler_arr = np.asarray(_jsonable(euler), dtype=float).reshape(-1)
    return (
        f"{qname} "
        f"{' '.join(map(str, trans_arr.tolist()))} "
        f"{euler_arr[1]} {euler_arr[0]} {euler_arr[2]}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one full DOM+DSM PiLoT refinement in a single process.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Stop after initial render, depth, and query/render overlay outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log: Dict[str, Any] = {
        "config_path": args.config,
        "query_image_path": args.query_image,
        "pose_file_path": args.pose_file,
        "output_dir": os.fspath(output_dir),
        "failure_stage": None,
        "traceback": None,
    }
    run_log_path = output_dir / "run_log.json"
    result_pose_path = output_dir / "result_pose.txt"
    start_total = time.time()
    stage = "start"

    try:
        stage = "load_config"
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        default_confs = config["default_confs"]
        refine_conf = default_confs["refine"]
        conf = default_confs["from_render_test"]
        log["checkpoint_path"] = conf.get("checkpoint")

        stage = "setup_camera"
        (
            query_resize_ratio,
            raw_query_camera,
            render_camera_gs,
            query_camera,
            render_camera,
        ) = _setup_camera(config)
        log["query_resize_ratio"] = query_resize_ratio
        log["render_camera_gs"] = render_camera_gs

        stage = "load_pose"
        euler, trans, origin_np = load_initial_pose(args.pose_file)
        config["render_config"]["init_rot"] = euler
        config["render_config"]["init_trans"] = trans
        refine_conf["origin"] = origin_np
        gt_pose_dict = load_pose_dict(args.pose_file, origin=origin_np)
        qname = Path(args.query_image).name
        log["initial_pose"] = {
            "image_name": qname,
            "translation_lon_lat_alt": trans,
            "euler_pitch_roll_yaw": euler,
            "pose_format": "image_name lon lat alt roll pitch yaw",
        }

        stage = "load_query_image"
        cam_cfg = default_confs["cam_query"]
        query_image = read_image(
            args.query_image,
            scale=query_resize_ratio,
            distortion=cam_cfg["distortion"],
            query_camera=raw_query_camera,
        )
        log["query_image_shape"] = list(query_image.shape)

        stage = "init_renderer"
        renderer = DOMDSMRenderer(config["render_config"])

        stage = "render"
        t0 = time.time()
        color, depth = renderer.render(trans, euler)
        render_time = time.time() - t0
        log["render_time_sec"] = render_time
        log["render_stats"] = _depth_stats(depth)
        log["render_shape"] = list(color.shape)

        stage = "save_render_outputs"
        cv2.imwrite(
            os.fspath(output_dir / "rendered_rgb.png"),
            cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
        )
        _save_depth_png(output_dir / "rendered_depth.png", depth)
        _save_overlay(output_dir / "query_render_overlay.png", query_image, color)
        query_image_for_refine = _resize_query_for_refine(query_image, render_camera_gs)
        log["query_image_for_refine_shape"] = list(query_image_for_refine.shape)
        log["query_image_for_refine_note"] = (
            "Resized to render camera dimensions before run_query so "
            "BaseRefiner.zero_pad(512) can accept the single test image. "
            "This only affects the standalone experiment script."
        )
        if args.render_only:
            log["render_only"] = True
            log["total_time_sec"] = time.time() - start_total
            _write_json(run_log_path, log)
            return 0

        stage = "back_project"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
        }
        origin = torch.tensor(origin_np, device=device)
        query_camera = query_camera.to(device)
        render_camera = render_camera.to(device)
        t0 = time.time()
        p3d, T_w2c, T_init, dd = _back_project(
            depth,
            euler,
            trans,
            euler,
            trans,
            render_camera_gs,
            render_camera,
            origin,
            refine_conf["mul"],
            device,
            is_init=True,
        )
        log["back_project_time_sec"] = time.time() - t0
        log["points_3d_count"] = int(p3d.shape[0])

        if default_confs.get("padding", False):
            color_for_refine = pad_to_multiple(color, 16)
        else:
            color_for_refine = color
        log["refine_color_shape"] = list(color_for_refine.shape)

        stage = "init_localizer"
        t0 = time.time()
        localizer = RenderLocalizer(conf)
        log["localizer_init_time_sec"] = time.time() - t0

        stage = "run_query"
        last_frame_info = {"observations": [], "refine_conf": refine_conf}
        t0 = time.time()
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
        log["run_query_time_sec"] = time.time() - t0
        log["run_query_success"] = bool(ret.get("success", False))

        stage = "save_results"
        initial_line = _format_pose_line(qname, trans, euler)
        result_lines = [
            "# initial",
            initial_line,
        ]
        if ret.get("success", False):
            refined_euler = ret["euler_angles"]
            refined_trans = ret["translation"]
            refined_line = _format_pose_line(qname, refined_trans, refined_euler)
            initial_output = np.array(
                [*np.asarray(trans, dtype=float), euler[1], euler[0], euler[2]],
                dtype=float,
            )
            refined_output = np.array(
                [
                    *np.asarray(_jsonable(refined_trans), dtype=float).reshape(-1),
                    np.asarray(_jsonable(refined_euler), dtype=float).reshape(-1)[1],
                    np.asarray(_jsonable(refined_euler), dtype=float).reshape(-1)[0],
                    np.asarray(_jsonable(refined_euler), dtype=float).reshape(-1)[2],
                ],
                dtype=float,
            )
            delta = refined_output - initial_output
            log["refined_pose"] = {
                "image_name": qname,
                "translation_lon_lat_alt": refined_trans,
                "euler_pitch_roll_yaw": refined_euler,
                "pose_format": "image_name lon lat alt roll pitch yaw",
            }
            log["pose_delta_lon_lat_alt_roll_pitch_yaw"] = delta
            result_lines.extend(
                [
                    "# refined",
                    refined_line,
                    "# delta lon lat alt roll pitch yaw",
                    " ".join(map(str, delta.tolist())),
                ]
            )
        else:
            result_lines.extend(
                [
                    "# failure",
                    "run_query returned success=False",
                ]
            )

        result_pose_path.write_text("\n".join(result_lines) + "\n", encoding="utf-8")
        log["total_time_sec"] = time.time() - start_total
        _write_json(run_log_path, log)
        return 0 if ret.get("success", False) else 2

    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        log["failure_stage"] = stage
        log["traceback"] = tb
        log["total_time_sec"] = time.time() - start_total
        result_pose_path.write_text(
            "\n".join(
                [
                    "# failure",
                    f"stage: {stage}",
                    tb,
                ]
            ),
            encoding="utf-8",
        )
        _write_json(run_log_path, log)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
