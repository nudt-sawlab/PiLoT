#!/usr/bin/env python3
"""End-to-end coordinate check: render -> back_project -> build_world roundtrip."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pixloc.localization.base_refiner import build_world_c2w_batch
from pixloc.pixlib.geometry import Camera, Pose
from pixloc.utils.citygs.pose_convert import (
    c2w_colmap_to_euler_trans,
    euler_trans_to_colmap_c2w,
)
from pixloc.utils.get_depth import generate_render_camera, sample_3d_points

INIT_TRANS = [-2.472183076709024, -6.551210367578361, 0.21583654951171902]
INIT_EULER = [73.70141101125306, 0.006372290178700057, 147.94607339944355]
MUL = 0.001
MAX_FRAMES = 3


def load_config():
    with open(ROOT / "configs/demos/smbu_seq2.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["render_config"]["init_rot"] = INIT_EULER
    cfg["render_config"]["init_trans"] = INIT_TRANS
    return cfg


def depth_stats(depth: np.ndarray) -> dict:
    finite = np.isfinite(depth) & (depth > 0)
    if not finite.any():
        return {"valid": 0, "total": depth.size, "min": None, "max": None}
    vals = depth[finite]
    in_range = np.sum((vals >= 0.1) & (vals <= 200.0))
    return {
        "valid": int(finite.sum()),
        "in_range": int(in_range),
        "total": depth.size,
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def check_c2w(name: str, got: np.ndarray, ref: np.ndarray, atol=0.05) -> bool:
    diff = np.abs(got - ref).max()
    ok = diff < atol
    print(f"  [{'OK' if ok else 'WARN'}] {name}: max_diff={diff:.6f}")
    return ok


def main() -> int:
    print("=== E2E coordinate verification (render + back_project) ===\n")
    cfg = load_config()
    origin = np.asarray(INIT_TRANS, dtype=np.float64)

    from pixloc.utils.citygs.citygs_render import CityGaussianRenderer

    renderer = CityGaussianRenderer(cfg["render_config"])
    render_cam = generate_render_camera(
        np.array(cfg["render_config"]["render_camera"], dtype=float)
    ).float()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    origin_t = torch.as_tensor(origin, dtype=torch.float32, device=device)

    # Load GT poses for first frames
    gt_path = Path(cfg["default_confs"]["gt_pose_path"])
    gt_lines = [l.strip() for l in gt_path.read_text().splitlines() if l.strip()]

    last_euler, last_trans = INIT_EULER, INIT_TRANS
    all_ok = True

    for idx in range(MAX_FRAMES):
        target = f"{idx}_0.png"
        line = next((l for l in gt_lines if l.split()[0] == target), None)
        if line is None:
            print(f"Frame {idx}: skip (no GT line)")
            continue

        parts = line.split()
        gt_trans = list(map(float, parts[1:4]))
        gt_euler = list(map(float, parts[4:7]))
        c2w_gt = euler_trans_to_colmap_c2w(gt_trans, gt_euler)

        # Render with current pose state (init for frame 0, last for later)
        render_euler, render_trans = (
            (INIT_EULER, INIT_TRANS) if idx == 0 else (last_euler, last_trans)
        )
        c2w_render = euler_trans_to_colmap_c2w(render_trans, render_euler)
        color, depth = renderer.render_c2w(c2w_render)
        ds = depth_stats(depth)

        print(f"Frame {idx} render pose euler={render_euler}")
        print(f"  depth: valid={ds['valid']}/{ds['total']} in[0.1,200]={ds['in_range']} "
              f"min={ds['min']} max={ds['max']}")

        if ds["in_range"] < 1000:
            print(f"  [FAIL] insufficient depth for frame {idx}")
            all_ok = False
            break

        # back_project path (same as main.py)
        T_c2w = torch.as_tensor(c2w_render, dtype=torch.float32, device=device)
        H, W = int(cfg["render_config"]["render_camera"][1]), int(
            cfg["render_config"]["render_camera"][0]
        )
        d_t = torch.as_tensor(depth, device=device)
        valid = (d_t >= 0.1) & (d_t <= 200.0) & torch.isfinite(d_t)
        sel = valid.nonzero()
        n = min(200, sel.shape[0])
        perm = torch.randperm(sel.shape[0], device=device)[:n]
        sel = sel[perm]
        mkpts = torch.stack((sel[:, 1].float(), sel[:, 0].float()), dim=1)

        _, T_render, T_query, dd = sample_3d_points(
            mkpts,
            depth,
            T_c2w,
            render_cam,
            last_euler,
            last_trans,
            origin=origin_t,
            device=device,
            mul=MUL,
            is_init_frame=(idx == 0),
            coordinate_system="normalized",
        )

        # Simulate LM output = no change (identity): w2c stays as T_render
        # build_world should recover render c2w
        c2w_rec = build_world_c2w_batch(
            T_render.unsqueeze(0),
            dd,
            MUL,
            torch.as_tensor(origin, dtype=torch.float64),
        )[0].cpu().numpy()

        ok_r = check_c2w("render roundtrip", c2w_rec, c2w_render, atol=1e-3)
        all_ok &= ok_r

        # Simulate best query = first candidate w2c
        c2w_q = build_world_c2w_batch(
            T_query[0:1],
            dd,
            MUL,
            torch.as_tensor(origin, dtype=torch.float64),
        )[0].cpu().numpy()
        euler_q, trans_q = c2w_colmap_to_euler_trans(c2w_q)
        last_euler, last_trans = euler_q, trans_q

        print(f"  query cand0 euler={euler_q}")
        print(f"  vs GT euler={gt_euler}")

        # GT comparison (pose error, not roundtrip)
        c2w_q_rebuild = euler_trans_to_colmap_c2w(trans_q, euler_q)
        check_c2w("vs GT (cand0)", c2w_q_rebuild, c2w_gt, atol=0.5)

        # Next-frame render check: would next frame have depth?
        _, depth_next = renderer.render_c2w(
            euler_trans_to_colmap_c2w(last_trans, last_euler)
        )
        ds_next = depth_stats(depth_next)
        next_ok = ds_next["in_range"] > 1000
        print(f"  next-frame depth in_range={ds_next['in_range']} "
              f"[{'OK' if next_ok else 'FAIL'}]")
        all_ok &= next_ok

    print()
    if all_ok:
        print("E2E verification PASSED.")
        return 0
    print("E2E verification FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
