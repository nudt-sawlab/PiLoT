#!/usr/bin/env python3
"""Verify PiLoT coordinate transforms: forward (world -> LM) and inverse (LM -> world).

The learned optimizer (LM) operates in a normalized space produced by
``sample_3d_points`` and recovered by ``build_world_c2w_batch``.

Original ECEF pipeline (unchanged PiLoT):
  world c2w (ECEF, no Y/Z flip)
    -> preprocess_pose_for_pixloc  (flip Y/Z columns)
    -> * mul, - origin*mul, invert -> w2c
    -> + R_w2c @ dd  (lever arm)
  LM optimizes w2c here
    -> build_world_c2w_batch: undo dd, w2c->c2w, /mul, flip Y/Z cols, +origin
    -> pixloc_to_osg  (ECEF/ENU -> WGS84 euler for next frame)

Normalized (CityGaussian COLMAP) — same LM space, different world frame:
  world c2w (COLMAP, no flip)  == euler_trans_to_colmap_c2w
    -> same preprocess / mul / origin / dd / invert
  LM optimizes w2c (identical convention)
    -> build_world_c2w_batch (same formula)
    -> c2w_colmap_to_euler_trans  (build_world output is already COLMAP c2w)
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pixloc.localization.base_refiner import build_world_c2w_batch
from pixloc.pixlib.geometry import Camera, Pose
from pixloc.utils.citygs.pose_convert import (
    c2w_colmap_to_euler_trans,
    euler_trans_to_colmap_c2w,
)
from pixloc.utils.get_depth import (
    generate_render_camera,
    preprocess_pose_for_pixloc,
    sample_3d_points,
)
from pixloc.utils.transform import euler_angles_to_matrix_ECEF

# SMBU frame-0 init (fx868)
INIT_TRANS = [-2.472183076709024, -6.551210367578361, 0.21583654951171902]
INIT_EULER = [73.70141101125306, 0.006372290178700057, 147.94607339944355]
MUL = 0.001


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def forward_query_w2c(
    c2w_world: np.ndarray,
    origin: np.ndarray,
    mul: float,
    dd: np.ndarray,
) -> Pose:
    """Replicate sample_3d_points query branch (world c2w -> LM w2c)."""
    device = _device()
    T = torch.as_tensor(c2w_world, dtype=torch.float32, device=device).clone()
    T[:3, 1] *= -1
    T[:3, 2] *= -1

    origin_t = torch.as_tensor(origin, dtype=torch.float32, device=device) * mul
    T[:3, 3] = T[:3, 3] * mul - origin_t

    query_c2w = Pose.from_Rt(T[:3, :3], T[:3, 3])
    T_query = query_c2w.inv()
    dd_t = torch.as_tensor(dd, dtype=torch.float32, device=device)
    tt = T_query.t + T_query.R @ dd_t
    return Pose.from_Rt(T_query.R, tt)


def forward_render_w2c(
    c2w_world: np.ndarray,
    origin: np.ndarray,
    mul: float,
    dd: np.ndarray,
) -> Pose:
    """Replicate sample_3d_points render-pose branch (world c2w -> LM w2c)."""
    device = _device()
    T = torch.as_tensor(c2w_world, dtype=torch.float32, device=device)
    cam = Camera.from_colmap({
        "model": "PINHOLE",
        "width": 960,
        "height": 540,
        "params": np.array([868.0, 868.0, 480.0, 270.0]),
    })
    _, render_T = preprocess_pose_for_pixloc(copy.deepcopy(cam), T)
    render_T = render_T.to(dtype=torch.float32)

    origin_t = torch.as_tensor(origin, dtype=torch.float32, device=device) * mul
    render_T[:3, 3] = render_T[:3, 3] * mul - origin_t

    render_c2w = Pose.from_Rt(render_T[:3, :3], render_T[:3, 3])
    T_render = render_c2w.inv()
    dd_t = torch.as_tensor(dd, dtype=torch.float32, device=device)
    tt = T_render.t + T_render.R @ dd_t
    return Pose.from_Rt(T_render.R, tt)


def inverse_lm_to_world_c2w(
    w2c_lm: Pose,
    origin: np.ndarray,
    mul: float,
    dd: np.ndarray,
) -> np.ndarray:
    """build_world_c2w_batch — output is already world COLMAP / ECEF c2w."""
    return build_world_c2w_batch(
        w2c_lm.unsqueeze(0),
        torch.as_tensor(dd, dtype=torch.float64),
        mul,
        torch.as_tensor(origin, dtype=torch.float64),
    )[0].cpu().numpy()


def check_matrix(name: str, got: np.ndarray, expected: np.ndarray, atol: float = 1e-4) -> bool:
    diff = np.abs(got - expected).max()
    ok = diff < atol
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name}: max_diff={diff:.6e}")
    return ok


def test_manual_roundtrip_normalized() -> bool:
    print("\n=== 1. Manual roundtrip (normalized COLMAP, mul=0.001) ===")
    origin = np.asarray(INIT_TRANS, dtype=np.float64)
    c2w_in = euler_trans_to_colmap_c2w(INIT_TRANS, INIT_EULER)
    dd = np.array([0.1, -0.2, 0.05], dtype=np.float64)

    w2c_lm = forward_render_w2c(c2w_in, origin, MUL, dd)
    c2w_out = inverse_lm_to_world_c2w(w2c_lm, origin, MUL, dd)

    ok_rot = check_matrix("rotation", c2w_out[:3, :3], c2w_in[:3, :3])
    ok_t = check_matrix("translation", c2w_out[:3, 3], c2w_in[:3, 3])
    return ok_rot and ok_t


def test_euler_roundtrip_normalized() -> bool:
    print("\n=== 2. Euler/trans roundtrip (render path) ===")
    origin = np.asarray(INIT_TRANS, dtype=np.float64)
    c2w_in = euler_trans_to_colmap_c2w(INIT_TRANS, INIT_EULER)
    dd = np.zeros(3, dtype=np.float64)

    w2c_lm = forward_render_w2c(c2w_in, origin, MUL, dd)
    c2w_out = inverse_lm_to_world_c2w(w2c_lm, origin, MUL, dd)
    euler_out, trans_out = c2w_colmap_to_euler_trans(c2w_out)
    c2w_rebuild = euler_trans_to_colmap_c2w(trans_out, euler_out)

    ok = check_matrix("euler_trans_to_colmap rebuild", c2w_rebuild, c2w_in)
    print(f"  in  euler={INIT_EULER}")
    print(f"  out euler={euler_out}")
    return ok


def test_manual_roundtrip_query_normalized() -> bool:
    print("\n=== 3. Manual roundtrip (query branch, normalized) ===")
    origin = np.asarray(INIT_TRANS, dtype=np.float64)
    c2w_in = euler_trans_to_colmap_c2w(INIT_TRANS, INIT_EULER)
    dd = np.array([0.05, 0.1, -0.03], dtype=np.float64)

    w2c_lm = forward_query_w2c(c2w_in, origin, MUL, dd)
    c2w_out = inverse_lm_to_world_c2w(w2c_lm, origin, MUL, dd)

    ok_rot = check_matrix("query rotation", c2w_out[:3, :3], c2w_in[:3, :3])
    ok_t = check_matrix("query translation", c2w_out[:3, 3], c2w_in[:3, 3])
    return ok_rot and ok_t


def test_sample3d_grid_is_perturbed() -> bool:
    """Grid candidate 0 is a perturbed pose, not necessarily the base pose."""
    print("\n=== 4. sample_3d_points grid (cand 0 is perturbed, informational) ===")
    device = _device()
    origin = torch.as_tensor(INIT_TRANS, dtype=torch.float32, device=device)
    c2w_render = euler_trans_to_colmap_c2w(INIT_TRANS, INIT_EULER)
    render_cam = generate_render_camera(
        np.array([960, 540, 480, 270, 868, 868], dtype=float)
    ).float()

    # Fake depth: one pixel at center
    depth = np.zeros((540, 960), dtype=np.float32)
    depth[270, 480] = 5.0
    mkpts = torch.tensor([[480.0, 270.0]])

    _, T_render, T_query, dd = sample_3d_points(
        mkpts,
        depth,
        torch.as_tensor(c2w_render, dtype=torch.float32),
        render_cam,
        INIT_EULER,
        INIT_TRANS,
        origin=origin,
        device=device,
        mul=MUL,
        is_init_frame=False,
        coordinate_system="normalized",
    )

    # Candidate 0 should match init query pose in LM space
    w2c_q0 = T_query[0]
    c2w_out = inverse_lm_to_world_c2w(
        w2c_q0, INIT_TRANS, MUL, dd.detach().cpu().numpy()
    )
    c2w_in = euler_trans_to_colmap_c2w(INIT_TRANS, INIT_EULER)
    diff = np.abs(c2w_out - c2w_in).max()
    print(f"  cand0 vs init max_diff={diff:.4f} (cand0 is yaw+8, not center)")
    return True


def test_ecef_manual_roundtrip() -> bool:
    """Original ECEF path — reference for LM convention."""
    print("\n=== 5. Manual roundtrip (original ECEF, mul=0.001) ===")
    trans_wgs = [114.0, 22.5, 100.0]
    euler = [10.0, 2.0, 45.0]
    c2w_in = euler_angles_to_matrix_ECEF(euler, trans_wgs)
    origin = c2w_in[:3, 3].copy()
    dd = np.zeros(3)

    w2c_lm = forward_render_w2c(c2w_in, origin, MUL, dd)
    c2w_out = inverse_lm_to_world_c2w(w2c_lm, origin, MUL, dd)
    # ECEF: build_world recovers the same world c2w as input (flip cancels)
    ok = check_matrix("ECEF c2w", c2w_out, c2w_in, atol=1e-3)
    return ok


def test_lm_scale() -> None:
    print("\n=== 6. LM input scale check (optimizer training range) ===")
    origin = np.asarray(INIT_TRANS, dtype=np.float64)
    c2w = euler_trans_to_colmap_c2w(INIT_TRANS, INIT_EULER)
    w2c = forward_render_w2c(c2w, origin, MUL, np.zeros(3))
    t_lm = w2c.t.detach().cpu().numpy()
    print(f"  LM w2c |t| with mul={MUL}: {np.linalg.norm(t_lm):.6f}")
    print(f"  (ECEF typical after mul+origin: ~0.01-0.1; should be same order)")
    w2c_bad = forward_render_w2c(c2w, origin, 1.0, np.zeros(3))
    t_bad = w2c_bad.t.detach().cpu().numpy()
    print(f"  LM w2c |t| with mul=1.0 (WRONG): {np.linalg.norm(t_bad):.6f}")


def main() -> int:
    print("PiLoT coordinate roundtrip verification")
    print("LM space = preprocess_flip -> *mul -> -origin*mul -> w2c -> +R@dd")
    print("Inverse  = build_world_c2w_batch(dd, mul, origin) -> COLMAP c2w")

    results = [
        test_manual_roundtrip_normalized(),
        test_euler_roundtrip_normalized(),
        test_manual_roundtrip_query_normalized(),
        test_ecef_manual_roundtrip(),
    ]
    test_sample3d_grid_is_perturbed()
    test_lm_scale()

    print("\n=== Summary ===")
    names = [
        "render normalized",
        "euler normalized",
        "query normalized",
        "manual ECEF",
    ]
    all_ok = True
    for name, ok in zip(names, results):
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
        all_ok &= ok

    if all_ok:
        print("\nAll roundtrip tests PASSED.")
        return 0
    print("\nSome tests FAILED — check coordinate mapping.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
