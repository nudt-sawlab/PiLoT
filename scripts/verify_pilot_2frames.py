#!/usr/bin/env python3
"""Run PiLoT localization on first 2 frames to verify coord + depth chain."""

from __future__ import annotations

import copy
import glob
import os
import sys
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
from pixloc.utils.citygs.pose_convert import (
    c2w_colmap_to_euler_trans,
    euler_trans_to_colmap_c2w,
)
from pixloc.utils.get_depth import (
    generate_render_camera,
    pad_to_multiple,
    sample_3d_points,
)
from src.utils.pose_utils import load_initial_pose_normalized, load_pose_dict_normalized

MAX_FRAMES = 2


def main() -> int:
    with open(ROOT / "configs/demos/smbu_seq2.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dc = config["default_confs"]
    rc = config["render_config"]
    citygs = rc["citygs"]

    gt_path = dc["gt_pose_path"]
    euler, trans, origin = load_initial_pose_normalized(
        gt_path,
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
    img_list = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:MAX_FRAMES]
    raw_cam = np.array([960, 540, 480, 270, 868, 868])
    query_list = read_image_list(
        img_list, scale=ratio, distortion=cam_cfg["distortion"], query_camera=raw_cam
    )

    gt_dict = load_pose_dict_normalized(gt_path, origin=origin)
    renderer = CityGaussianRenderer(rc)
    localizer = RenderLocalizer(dc["from_render_test"])

    last_euler, last_trans = euler, trans
    all_ok = True

    print("=== PiLoT 2-frame localization smoke test ===\n")

    for idx, (img_path, img_tensor) in enumerate(zip(img_list, query_list)):
        render_euler, render_trans = (
            (euler, trans) if idx == 0 else (last_euler, last_trans)
        )
        c2w = euler_trans_to_colmap_c2w(render_trans, render_euler)
        color, depth = renderer.render_c2w(c2w)

        valid = np.isfinite(depth) & (depth > 0.1) & (depth < 200)
        n_valid = int(valid.sum())
        print(f"Frame {idx}: depth valid={n_valid}/{depth.size}")
        if n_valid < 1000:
            print("  [FAIL] empty/invalid depth")
            return 1

        depth_t = torch.as_tensor(depth, device="cuda")
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
            last_frame_info={"observations": [], "refine_conf": dc["refine"]},
            query_resize_ratio=ratio,
            image_query=img_tensor,
        )

        if not ret.get("success", True):
            print(f"  [FAIL] optimization failed")
            all_ok = False
            continue

        out_euler = ret["euler_angles"]
        out_trans = ret["translation"]
        if hasattr(out_euler, "tolist"):
            out_euler = out_euler.tolist()

        # roundtrip: output euler/trans -> c2w -> render depth check
        c2w_out = euler_trans_to_colmap_c2w(out_trans, out_euler)
        _, depth_next = renderer.render_c2w(c2w_out)
        n_next = int((np.isfinite(depth_next) & (depth_next > 0.1) & (depth_next < 200)).sum())

        pitch_in = render_euler[0]
        pitch_out = out_euler[0]
        pitch_flip = abs(abs(pitch_in - pitch_out) - 180) < 5  # bad sign flip

        print(f"  in  euler={render_euler}")
        print(f"  out euler={out_euler}")
        print(f"  pitch delta={pitch_out - pitch_in:.2f} "
              f"[{'BAD FLIP' if pitch_flip else 'OK'}]")
        print(f"  next render depth valid={n_next} [{'OK' if n_next > 1000 else 'FAIL'}]")

        if pitch_flip or n_next < 1000:
            all_ok = False

        last_euler, last_trans = out_euler, out_trans

    print()
    if all_ok:
        print("2-frame PiLoT smoke test PASSED.")
        return 0
    print("2-frame PiLoT smoke test FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
