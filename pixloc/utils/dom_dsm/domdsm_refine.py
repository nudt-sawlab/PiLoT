"""DOM/DSM-aware wrappers for PiLoT refinement inputs."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from pixloc.pixlib.geometry import Camera, Pose
from pixloc.utils.get_depth import sample_3d_points
from pixloc.utils.transform import euler_angles_to_matrix_ECEF
from pixloc.utils.dom_dsm.point_sampling import (
    compute_combined_structure_weight,
    compute_dom_gradient_weight,
    compute_dsm_depth_gradient_weight,
    sample_domdsm_points,
)


def _weight_for_mode(render_rgb: np.ndarray, depth: np.ndarray, mode: str) -> np.ndarray:
    valid = (np.isfinite(depth) & (depth > 0)).astype(np.float32)
    if mode == "uniform":
        return valid
    if mode == "dom_gradient":
        return compute_dom_gradient_weight(render_rgb) * valid
    if mode == "depth_gradient":
        return compute_dsm_depth_gradient_weight(depth)
    if mode == "combined":
        return compute_combined_structure_weight(render_rgb, depth)
    raise ValueError(f"Unknown DOM/DSM sampling mode: {mode}")


def run_domdsm_back_project(
    depth: np.ndarray,
    render_rgb: np.ndarray,
    euler: List[float],
    trans: List[float],
    query_euler: List[float],
    query_trans: List[float],
    render_camera_gs: np.ndarray,
    render_camera: Camera,
    origin: torch.Tensor,
    mul: float,
    device: str,
    num_samples: int = 500,
    sampling_mode: str = "combined",
    is_init: bool = True,
    seed: Optional[int] = 0,
) -> Tuple[torch.Tensor, Pose, Pose, Optional[torch.Tensor], Dict[str, Any]]:
    """Sample DOM/DSM-structured 2D points and run the existing PixLoc back-project.

    This wrapper is intentionally opt-in. It does not change sample_3d_points()
    or the main pipeline; callers choose the sampling mode explicitly.
    """
    if render_rgb.shape[:2] != depth.shape[:2]:
        raise ValueError("render_rgb and depth must have matching HxW")
    weight = _weight_for_mode(render_rgb, depth, sampling_mode)
    points2d = sample_domdsm_points(
        render_rgb,
        depth,
        num_samples=num_samples,
        mode=sampling_mode,
        device=device,
        seed=seed,
    )
    T_c2w = torch.as_tensor(
        euler_angles_to_matrix_ECEF(euler, trans),
        device=device,
        dtype=torch.float32,
    )
    p3d, T_w2c, T_init, dd = sample_3d_points(
        points2d,
        depth,
        T_c2w,
        render_camera,
        query_euler,
        query_trans,
        origin=origin,
        device=device,
        mul=mul,
        is_init_frame=is_init,
    )
    valid = np.isfinite(depth) & (depth > 0)
    debug = {
        "sampling_mode": sampling_mode,
        "requested_num_samples": int(num_samples),
        "sampled_points2d_count": int(points2d.shape[0]),
        "points_3d_count": int(p3d.shape[0]),
        "valid_depth_ratio": float(valid.mean()) if valid.size else 0.0,
        "weight_min": float(weight.min()) if weight.size else 0.0,
        "weight_max": float(weight.max()) if weight.size else 0.0,
        "weight_mean": float(weight.mean()) if weight.size else 0.0,
        "points2d": points2d.detach().cpu().numpy().tolist(),
        "weight": weight,
    }
    return p3d, T_w2c, T_init, dd, debug
