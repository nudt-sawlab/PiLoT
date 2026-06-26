#!/usr/bin/env python3
"""Run PiLoT on N frames; stop on first hard failure, report pose/depth stats."""

from __future__ import annotations

import argparse
import copy
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pixloc.localization.localizer import RenderLocalizer
from pixloc.pixlib.datasets.view import read_image_list
from pixloc.pixlib.geometry import Camera
from pixloc.utils.citygs.citygs_render import CityGaussianRenderer
from pixloc.utils.citygs.pose_convert import euler_trans_to_colmap_c2w
from pixloc.utils.get_depth import (
    generate_render_camera,
    pad_to_multiple,
    sample_3d_points,
)
from src.utils.pose_utils import load_initial_pose_normalized, load_pose_dict_normalized

MIN_DEPTH_VALID = 1000


def load_gt_poses(gt_path: Path) -> dict:
    poses = {}
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 7:
            name = parts[0]
            poses[name] = {
                "trans": list(map(float, parts[1:4])),
                "euler": list(map(float, parts[4:7])),
            }
    return poses


def trans_error(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def euler_error_deg(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def bad_pitch_flip(pitch_in: float, pitch_out: float) -> bool:
    return abs(abs(pitch_in - pitch_out) - 180.0) < 5.0


def depth_valid_count(depth: np.ndarray) -> int:
    return int((np.isfinite(depth) & (depth > 0.1) & (depth < 200)).sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-frames", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(ROOT / "configs/demos/smbu_seq2.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dc = config["default_confs"]
    rc = config["render_config"]
    citygs = rc["citygs"]

    gt_path = Path(dc["gt_pose_path"])
    gt_poses = load_gt_poses(gt_path)
    euler, trans, _ = load_initial_pose_normalized(
        str(gt_path),
        init_pose_txt=citygs.get("init_pose_txt"),
        init_pose_frame=int(citygs.get("init_pose_frame", 0)),
    )
    origin = np.asarray(trans, dtype=np.float64)
    mul = 0.001
    dc["refine"]["origin"] = trans
    dc["refine"]["mul"] = mul
    dc["refine"]["coordinate_system"] = "normalized"

    cam_cfg = copy.deepcopy(dc["cam_query"])
    ratio = cam_cfg["width"] / cam_cfg["max_size"]
    cam_cfg["params"] = np.array(cam_cfg["params"]) / ratio
    cam_cfg["width"] /= ratio
    cam_cfg["height"] /= ratio
    query_cam = Camera.from_colmap(cam_cfg).cuda()
    render_cam = generate_render_camera(
        np.array(rc["render_camera"], dtype=float)
    ).float().cuda()
    origin_t = torch.tensor(origin, dtype=torch.float32, device="cuda")

    seq = dc["sequence_name"]
    img_dir = os.path.join(dc["dataset_path"], "images", seq)
    img_list = sorted(
        glob.glob(os.path.join(img_dir, "*.png")),
        key=lambda p: int(os.path.basename(p).split(".")[0].split("_")[0]),
    )[: args.num_frames]
    raw_cam = np.array([960, 540, 480, 270, 868, 868])
    query_list = read_image_list(
        img_list, scale=ratio, distortion=cam_cfg["distortion"], query_camera=raw_cam
    )

    gt_dict = load_pose_dict_normalized(str(gt_path), origin=origin)
    renderer = CityGaussianRenderer(rc)
    localizer = RenderLocalizer(dc["from_render_test"])

    last_euler, last_trans = euler, trans
    last_frame_info = {"observations": [], "refine_conf": dc["refine"]}

    print(f"=== PiLoT {len(img_list)}-frame verification ===\n")

    rows = []
    t_total = time.time()

    for idx, (img_path, img_tensor) in enumerate(zip(img_list, query_list)):
        qname = os.path.basename(img_path)
        t0 = time.time()

        render_euler, render_trans = (
            (euler, trans) if idx == 0 else (last_euler, last_trans)
        )
        c2w = euler_trans_to_colmap_c2w(render_trans, render_euler)
        color, depth = renderer.render_c2w(c2w)
        n_depth = depth_valid_count(depth)

        if n_depth < MIN_DEPTH_VALID:
            print(f"Frame {idx} ({qname}): [FAIL] depth valid={n_depth}")
            return 1

        valid = np.isfinite(depth) & (depth > 0.1) & (depth < 200)
        sel = torch.from_numpy(np.stack(np.where(valid), axis=1)).cuda()
        n = min(500, sel.shape[0])
        perm = torch.randperm(sel.shape[0], device="cuda")[:n]
        sel = sel[perm]
        mkpts = torch.stack((sel[:, 1].float(), sel[:, 0].float()), dim=1)
        T_c2w = torch.as_tensor(c2w, dtype=torch.float32, device="cuda")

        p3d, T_w2c, T_init, dd = sample_3d_points(
            mkpts, depth, T_c2w, render_cam,
            last_euler, last_trans,
            origin=origin_t, mul=mul,
            is_init_frame=(idx == 0),
            coordinate_system="normalized",
        )

        ret = localizer.run_query(
            img_path, query_cam, render_cam,
            pad_to_multiple(color, 16),
            query_T=T_init, render_T=T_w2c,
            Points_3D_ECEF=p3d, dd=dd,
            gt_pose_dict=gt_dict,
            last_frame_info=last_frame_info,
            query_resize_ratio=ratio,
            image_query=img_tensor,
        )

        elapsed = (time.time() - t0) * 1000
        ok = ret.get("success", True)

        if not ok:
            print(f"Frame {idx} ({qname}): [FAIL] optimization failed ({elapsed:.0f}ms)")
            return 1

        out_euler = ret["euler_angles"]
        out_trans = ret["translation"]
        if hasattr(out_euler, "tolist"):
            out_euler = out_euler.tolist()

        c2w_out = euler_trans_to_colmap_c2w(out_trans, out_euler)
        _, depth_next = renderer.render_c2w(c2w_out)
        n_next = depth_valid_count(depth_next)

        flip = bad_pitch_flip(render_euler[0], out_euler[0])
        if flip or n_next < MIN_DEPTH_VALID:
            print(f"Frame {idx} ({qname}): [FAIL] flip={flip} next_depth={n_next}")
            print(f"  in  euler={render_euler}")
            print(f"  out euler={out_euler}")
            return 1

        gt = gt_poses.get(qname, {})
        t_err = trans_error(out_trans, gt.get("trans", out_trans)) if gt else float("nan")
        e_err = euler_error_deg(out_euler, gt.get("euler", out_euler)) if gt else float("nan")

        rows.append({
            "idx": idx,
            "t_err": t_err,
            "e_err": e_err,
            "n_depth": n_depth,
            "n_next": n_next,
            "ms": elapsed,
        })

        if idx % 5 == 0 or idx == len(img_list) - 1:
            print(
                f"Frame {idx:3d} | {elapsed:6.0f}ms | depth={n_depth:6d} "
                f"next={n_next:6d} | t_err={t_err:.3f}m e_err={e_err:.2f}° | OK"
            )

        last_euler, last_trans = out_euler, out_trans

    total_s = time.time() - t_total
    t_errs = [r["t_err"] for r in rows if not np.isnan(r["t_err"])]
    e_errs = [r["e_err"] for r in rows if not np.isnan(r["e_err"])]

    print(f"\n=== Summary ({len(rows)} frames, {total_s:.1f}s) ===")
    print(f"  depth valid:  min={min(r['n_depth'] for r in rows)} "
          f"max={max(r['n_depth'] for r in rows)}")
    print(f"  next depth:   min={min(r['n_next'] for r in rows)}")
    print(f"  trans err vs GT: mean={np.mean(t_errs):.3f}m max={np.max(t_errs):.3f}m")
    print(f"  euler err vs GT: mean={np.mean(e_errs):.2f}° max={np.max(e_errs):.2f}°")
    print(f"  time/frame:   mean={np.mean([r['ms'] for r in rows]):.0f}ms")
    print("\nPASSED — coordinate chain stable, stopping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
