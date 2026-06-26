"""Depth processing, 3D sampling, and camera utilities."""
import copy
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from pixloc.utils.transform import (
    WGS84_to_ECEF,
    get_rotation_enu_in_ecef,
)
from ..pixlib.geometry import Camera, Pose


# ---------------------------------------------------------------------------
# Depth interpolation
# ---------------------------------------------------------------------------

def interpolate_depth_grid(
    pos: torch.Tensor,
    depth: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bilinear interpolation of depth at sub-pixel positions.

    Args:
        pos: [N, 2] pixel coordinates (row, col).
        depth: [H, W] depth map.

    Returns:
        (depth_values, valid_positions, valid_indices).
    """
    device = depth.device
    H, W = depth.shape
    depth4 = depth.unsqueeze(0).unsqueeze(0)

    j = pos[:, 1].to(device)
    i = pos[:, 0].to(device)
    x = 2.0 * j / (W - 1) - 1.0
    y = 2.0 * i / (H - 1) - 1.0
    grid = torch.stack([x, y], dim=1).view(1, -1, 1, 2)

    sampled = F.grid_sample(
        depth4, grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(-1)

    valid = (
        (i >= 0) & (i <= H - 1)
        & (j >= 0) & (j <= W - 1)
        & (sampled > 0)
    )
    ids = valid.nonzero(as_tuple=False).view(-1)
    return sampled[ids], pos[ids].t(), ids


def read_valid_depth(
    mkpts: torch.Tensor,
    depth: np.ndarray,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read interpolated depth at keypoint locations.

    Args:
        mkpts: [N, 2] keypoint coordinates (x, y).
        depth: Depth map (numpy array or tensor).
        device: Computation device.

    Returns:
        (interpolated_depth, valid_indices).
    """
    depth_t = torch.as_tensor(depth, device=device, dtype=torch.float32)
    mkpts_f = mkpts.float().to(device)
    mkpts_rc = mkpts_f[:, [1, 0]].to(device)
    depth_interp, _, valid = interpolate_depth_grid(mkpts_rc, depth_t)
    return depth_interp, valid


# ---------------------------------------------------------------------------
# 3D back-projection
# ---------------------------------------------------------------------------

def back_project_points_3d(
    depth: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    K_inv: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """Back-project 2D points to 3D using depth, pose, and intrinsics.

    Args:
        depth: [N] depth values.
        R: [3, 3] rotation matrix (camera-to-world).
        t: [3] translation vector.
        K_inv: [3, 3] inverse intrinsic matrix.
        points: [N, 2] image coordinates.

    Returns:
        [N, 3] 3D points in world coordinates.
    """
    if points.shape[-1] != 3:
        points_2d = torch.cat(
            [points, torch.ones_like(points[:, :1])], dim=-1
        )
        points_2d = points_2d.T
    else:
        points_2d = points.T

    t_exp = t.unsqueeze(1).repeat(1, points_2d.size(-1))

    points_2d = points_2d.float()
    K_inv = K_inv.float()
    R = R.float()
    depth = depth.float()
    t_exp = t_exp.float()

    points_3d = R @ (K_inv @ (depth * points_2d)) + t_exp
    return points_3d.T


# ---------------------------------------------------------------------------
# Pose preprocessing
# ---------------------------------------------------------------------------

def preprocess_pose_for_pixloc(
    camera: Camera,
    pose: torch.Tensor,
    device: str = "cuda",
    flip_yz: bool = True,
) -> Tuple[Camera, torch.Tensor]:
    """Optionally flip Y/Z axes of pose for the OSG/ECEF PixLoc convention.

    The Y/Z column flip is only required for the OSG renderer (y-up world).
    CityGaussian renders in the native COLMAP convention (z-forward, y-down),
    which already matches PixLoc's pinhole camera, so for the normalized path
    `flip_yz` must be False — flipping there mirrors the back-projected geometry
    and inverts roll/yaw/x/y corrections once ref and query have a baseline.

    Args:
        camera: Camera intrinsics.
        pose: [4, 4] or [N, 4, 4] pose matrix.
        device: Computation device.
        flip_yz: Whether to negate the Y and Z columns.

    Returns:
        (camera, modified_pose).
    """
    pose = pose.to(device)
    if flip_yz:
        pose[..., 0:3, 1] *= -1
        pose[..., 0:3, 2] *= -1
    return camera, pose


# ---------------------------------------------------------------------------
# Camera generation
# ---------------------------------------------------------------------------

def generate_render_camera(camera_params: np.ndarray) -> Camera:
    """Build a Camera object from raw camera parameters.

    Supports parameter arrays of length 5, 6, 7, or 8.

    Args:
        camera_params: Camera parameter array.

    Returns:
        PixLoc Camera object.
    """
    n = len(camera_params)
    if n == 5:
        w, h, sw, sh, f = camera_params
        fx = w * (f / sw)
        fy = h * (f / sh)
        cx, cy = w / 2, h / 2
    elif n == 6:
        w, h, cx, cy, fx, fy = camera_params
    elif n == 7:
        w, h, cx, cy, sw, sh, f = camera_params
        fx = w * (f / sw)
        fy = h * (f / sh)
    elif n == 8:
        w, h, cx, cy, sw, sh, fx_mm, fy_mm = camera_params
        fx = w * (fx_mm / sw)
        fy = h * (fy_mm / sh)
    else:
        raise ValueError(f"Unsupported camera parameter length: {n}")

    cam_dict = {
        "model": "PINHOLE",
        "width": w,
        "height": h,
        "params": np.array([fx, fy, cx, cy]),
    }
    return Camera.from_colmap(cam_dict)


# ---------------------------------------------------------------------------
# Image padding
# ---------------------------------------------------------------------------

def zero_pad(size: int, image: np.ndarray) -> np.ndarray:
    """Zero-pad an image to a square of the given size.

    Args:
        size: Target square size.
        image: Input image [H, W, ...].

    Returns:
        Zero-padded image of shape (size, size, ...).
    """
    h, w = image.shape[:2]
    padded = np.zeros((size, size) + image.shape[2:], dtype=image.dtype)
    padded[:h, :w] = image
    return padded


def pad_to_multiple(image: np.ndarray, multiple: int = 16) -> np.ndarray:
    """Pad image dimensions to the nearest multiple.

    Args:
        image: Input image [H, W, ...].
        multiple: Target multiple for height and width.

    Returns:
        Padded image.
    """
    h, w = image.shape[:2]
    target_h = (h + (multiple - 1)) // multiple * multiple
    target_w = (w + (multiple - 1)) // multiple * multiple
    padded = np.zeros(
        (target_h, target_w, *image.shape[2:]), dtype=image.dtype
    )
    padded[:h, :w] = image
    return padded


# ---------------------------------------------------------------------------
# Rotation / translation grid generation
# ---------------------------------------------------------------------------

def _enu_to_ecef_rotation_tensor(
    lon: float,
    lat: float,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute 3x3 ENU-to-ECEF rotation matrix on GPU.

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees.
        device: Computation device.
        dtype: Tensor dtype.

    Returns:
        [3, 3] rotation matrix.
    """
    lon_t = torch.as_tensor(lon, device=device, dtype=dtype)
    lat_t = torch.as_tensor(lat, device=device, dtype=dtype)

    lon_rad = lon_t * (torch.pi / 180.0)
    lat_rad = lat_t * (torch.pi / 180.0)

    up = torch.stack([
        torch.cos(lon_rad) * torch.cos(lat_rad),
        torch.sin(lon_rad) * torch.cos(lat_rad),
        torch.sin(lat_rad),
    ])
    east = torch.stack([
        -torch.sin(lon_rad),
        torch.cos(lon_rad),
        torch.zeros_like(lon_rad),
    ])
    north = torch.cross(up, east, dim=0)
    return torch.stack([east, north, up], dim=1)


def _euler_to_rotation_batch(
    angles: torch.Tensor,
    translation: list,
    degrees: bool = True,
    device: str = "cuda",
) -> torch.Tensor:
    """Batch Euler angles to rotation matrices in ECEF.

    Args:
        angles: [N, 3] tensor of [pitch, roll, yaw].
        translation: [lon, lat, alt] for ENU-to-ECEF transform.
        degrees: Whether angles are in degrees.
        device: Computation device.

    Returns:
        [N, 3, 3] rotation matrices.
    """
    if degrees:
        angles = angles * (torch.pi / 180.0)
    p, r, y = angles.unbind(1)

    zero = torch.zeros_like(p)
    one = torch.ones_like(p)

    Rx = torch.stack([
        torch.stack([one, zero, zero], dim=1),
        torch.stack([zero, torch.cos(p), -torch.sin(p)], dim=1),
        torch.stack([zero, torch.sin(p), torch.cos(p)], dim=1),
    ], dim=1)

    Ry = torch.stack([
        torch.stack([torch.cos(r), zero, torch.sin(r)], dim=1),
        torch.stack([zero, one, zero], dim=1),
        torch.stack([-torch.sin(r), zero, torch.cos(r)], dim=1),
    ], dim=1)

    Rz = torch.stack([
        torch.stack([torch.cos(y), -torch.sin(y), zero], dim=1),
        torch.stack([torch.sin(y), torch.cos(y), zero], dim=1),
        torch.stack([zero, zero, one], dim=1),
    ], dim=1)

    lon, lat, _ = translation
    rot_enu_in_ecef = _enu_to_ecef_rotation_tensor(lon, lat).to(device)
    R_local = Rz @ Ry @ Rx
    return rot_enu_in_ecef.unsqueeze(0) @ R_local


def _euler_to_matrix_model_batch(
    euler_angles: torch.Tensor,
    translation: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """Batch Euler angles + model-space translations to 4x4 c2w matrices."""
    mats = []
    for i in range(euler_angles.shape[0]):
        rot = R.from_euler(
            "xyz", euler_angles[i].detach().cpu().numpy(), degrees=True
        ).as_matrix()
        mats.append(torch.tensor(rot, device=device, dtype=torch.float32))
    R_batch = torch.stack(mats, dim=0)
    N = R_batch.shape[0]
    T = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)
    T[:, :3, :3] = R_batch
    T[:, :3, 3] = translation
    return T


def _euler_to_matrix_ecef_batch(
    euler_angles: torch.Tensor,
    translation: torch.Tensor,
    center_pose: list,
    device: str = "cuda",
) -> torch.Tensor:
    """Batch Euler angles + translations to 4x4 ECEF pose matrices.

    Args:
        euler_angles: [N, 3] Euler angles.
        translation: [N, 3] ECEF translations.
        center_pose: [lon, lat, alt] for ENU reference.
        device: Computation device.

    Returns:
        [N, 4, 4] transformation matrices.
    """
    R_batch = _euler_to_rotation_batch(
        euler_angles, center_pose, degrees=True, device=device
    )
    N = R_batch.shape[0]
    T = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)
    T[:, :3, :3] = R_batch
    T[:, :3, 3] = translation
    return T


def generate_rotation_grid(
    base_pitch: float,
    base_roll: float,
    base_yaw: float,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate a pitch-yaw rotation grid around a base orientation.

    Args:
        base_pitch: Base pitch angle in degrees.
        base_roll: Base roll angle in degrees.
        base_yaw: Base yaw angle in degrees.
        device: Computation device.

    Returns:
        [N, 3] tensor of [pitch, roll, yaw] angles.
    """
    pitch_vals = torch.tensor(
        [9, 7, 5, 3, 1, 0, -1, -3, -5, -7, -9], device=device
    )
    yaw_vals = torch.tensor(
        [9, 7, 5, 3, 1, 0, -1, -3, -5, -7, -9], device=device
    )
    roll_vals = torch.tensor([0], device=device)

    P, Y, R_grid = torch.meshgrid(
        pitch_vals, yaw_vals, roll_vals, indexing="ij"
    )
    N = P.numel()
    pitch = P.reshape(N) + base_pitch
    roll = R_grid.reshape(N) + base_roll
    yaw = Y.reshape(N) + base_yaw
    return torch.stack((pitch, roll, yaw), dim=1)


def generate_rotation_grid_normalized(
    base_pitch: float,
    base_roll: float,
    base_yaw: float,
    device: str = "cuda",
) -> torch.Tensor:
    """Tight rotation grid for normalized-model sequential tracking.

    7 pitch x 7 yaw x 3 roll = 147 candidates. Roll is included because it has
    no other search dimension and drifts unconstrained otherwise.
    """
    dtype = torch.float32
    pitch_vals = torch.tensor(
        [0, -0.5, 0.5, -1, 1, -2, 2], device=device, dtype=dtype
    )
    yaw_vals = torch.tensor(
        [0, -0.5, 0.5, -1, 1, -2, 2], device=device, dtype=dtype
    )
    roll_vals = torch.tensor([0], device=device, dtype=dtype)

    P, Y, R_grid = torch.meshgrid(
        pitch_vals, yaw_vals, roll_vals, indexing="ij"
    )
    N = P.numel()
    pitch = P.reshape(N) + base_pitch
    roll = R_grid.reshape(N) + base_roll
    yaw = Y.reshape(N) + base_yaw
    return torch.stack((pitch, roll, yaw), dim=1)


def generate_yaw_rotation_grid(
    base_pitch: float,
    base_roll: float,
    base_yaw: float,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate a yaw-only rotation grid around a base orientation.

    Args:
        base_pitch: Base pitch angle in degrees.
        base_roll: Base roll angle in degrees.
        base_yaw: Base yaw angle in degrees.
        device: Computation device.

    Returns:
        [N, 3] tensor of [pitch, roll, yaw] angles.
    """
    pitch_vals = torch.tensor([0], device=device)
    yaw_vals = torch.tensor(
        [8, 6, 4, 2, 0, -2, -4, -6, -8], device=device
    )
    roll_vals = torch.tensor([0], device=device)

    P, Y, R_grid = torch.meshgrid(
        pitch_vals, yaw_vals, roll_vals, indexing="ij"
    )
    N = P.numel()
    pitch = P.reshape(N) + base_pitch
    roll = R_grid.reshape(N) + base_roll
    yaw = Y.reshape(N) + base_yaw
    return torch.stack((pitch, roll, yaw), dim=1)


def generate_translation_grid(
    base_trans: torch.Tensor,
    max_x: float,
    step_x: float,
    max_y: float,
    step_y: float,
    max_z: float,
    step_z: float,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate a 3D translation grid around ``base_trans`` (includes zero offset).

    Args:
        base_trans: [3] base translation.
        max_x, step_x: X-axis range and step.
        max_y, step_y: Y-axis range and step.
        max_z, step_z: Z-axis range and step.
        device: Computation device.

    Returns:
        [M, 3] tensor of translation vectors.
    """
    bx, by, bz = base_trans[0], base_trans[1], base_trans[2]

    def _create_range(max_val, step):
        if max_val == 0 or step <= 0:
            return torch.tensor([0.0], device=device)
        neg = torch.arange(-max_val, 0, step, device=device)
        zero = torch.tensor([0.0], device=device)
        pos = torch.arange(step, max_val + step, step, device=device)
        return torch.cat([neg, zero, pos])

    range_x = _create_range(max_x, step_x) + bx
    range_y = _create_range(max_y, step_y) + by
    range_z = torch.tensor([bz], device=device)

    gx, gy, gz = torch.meshgrid(range_x, range_y, range_z, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Main 3D sampling function
# ---------------------------------------------------------------------------

def sample_3d_points(
    mkpts_r: torch.Tensor,
    depth_mat: np.ndarray,
    T_c2w: torch.Tensor,
    camera: Camera,
    query_euler_angles: list,
    query_translation: list,
    origin: Optional[torch.Tensor] = None,
    device: str = "cuda",
    mul: Optional[float] = None,
    is_init_frame: bool = True,
    trust_prior: bool = False,
    coordinate_system: str = "ecef",
) -> Tuple[torch.Tensor, Pose, Pose, torch.Tensor]:
    """Sample 3D points from depth and generate initial pose candidates.

    This function:
      1. Generates rotation/translation grids for pose candidates.
      2. Pre-processes the render pose for PixLoc convention.
      3. Back-projects 2D points to 3D using depth.
      4. Applies coordinate normalization (scaling, origin subtraction, centering).

    Args:
        mkpts_r: [N, 2] 2D keypoints for back-projection.
        depth_mat: Depth map.
        T_c2w: [4, 4] render camera-to-world pose in ECEF.
        camera: Render camera intrinsics.
        query_euler_angles: [pitch, roll, yaw] of the query.
        query_translation: [lon, lat, alt] of the query.
        origin: ECEF origin for normalization.
        device: Computation device.
        mul: Optional coordinate scaling factor.
        is_init_frame: If True, generates broader search grid.
        trust_prior: If True, use the exact query pose as the only candidate.

    Returns:
        (points_3d, render_pose, candidate_poses, center_offset).
    """
    dtype = torch.float32
    T_c2w = T_c2w.to(device=device, dtype=dtype)
    if origin is not None:
        origin = origin.to(device=device, dtype=dtype)
    if torch.is_tensor(mkpts_r):
        mkpts_r = mkpts_r.to(device=device, dtype=dtype)

    normalized = coordinate_system == "normalized"

    # --- Generate rotation candidates ---
    if trust_prior:
        euler_grid = torch.tensor(
            [query_euler_angles], device=device, dtype=torch.float32
        )
        trans_grid = torch.tensor(
            [query_translation], device=device, dtype=torch.float32
        )
    elif is_init_frame:
        euler_grid = generate_yaw_rotation_grid(
            base_pitch=query_euler_angles[0],
            base_roll=query_euler_angles[1],
            base_yaw=query_euler_angles[2],
        )
        if normalized:
            trans_base = torch.tensor(
                query_translation, device=device, dtype=torch.float32
            )
            trans_grid = generate_translation_grid(
                base_trans=trans_base,
                max_x=2, step_x=1,
                max_y=2, step_y=1,
                max_z=0, step_z=1,
                device=euler_grid.device,
            )
        else:
            trans_ecef = WGS84_to_ECEF(query_translation)
            trans_grid = generate_translation_grid(
                base_trans=trans_ecef,
                max_x=10, step_x=5,
                max_y=10, step_y=5,
                max_z=0, step_z=1,
                device=euler_grid.device,
            )
    else:
        if normalized:
            # 7x7 pitch-yaw rotation seeds (roll fixed); translation is NOT
            # scattered — single base point. Total candidates = 49.
            euler_grid = generate_rotation_grid_normalized(
                base_pitch=query_euler_angles[0],
                base_roll=query_euler_angles[1],
                base_yaw=query_euler_angles[2],
            )
            trans_grid = torch.tensor(
                [query_translation], device=device, dtype=torch.float32
            )
        else:
            euler_grid = generate_rotation_grid(
                base_pitch=query_euler_angles[0],
                base_roll=query_euler_angles[1],
                base_yaw=query_euler_angles[2],
            )
            trans_base = torch.tensor(
                WGS84_to_ECEF(query_translation),
                device=device, dtype=torch.float32,
            )
            trans_grid = trans_base.reshape(1, 3)

    # --- Cartesian product of rotations x translations ---
    M = trans_grid.shape[0]
    N = euler_grid.shape[0]
    euler_expanded = euler_grid.repeat_interleave(M, dim=0)
    trans_expanded = trans_grid.repeat(N, 1)

    # COLMAP/normalized renders already match PixLoc's pinhole convention; only
    # the OSG/ECEF path needs the Y/Z axis flip (see preprocess_pose_for_pixloc).
    flip_yz = not normalized

    if normalized:
        query_T_c2w = _euler_to_matrix_model_batch(
            euler_expanded, trans_expanded, device=device
        )
    else:
        query_T_c2w = _euler_to_matrix_ecef_batch(
            euler_expanded, trans_expanded, query_translation
        )
    if flip_yz:
        query_T_c2w[:, :3, 1] *= -1
        query_T_c2w[:, :3, 2] *= -1

    # --- Preprocess render pose ---
    render_cam, render_T = preprocess_pose_for_pixloc(
        copy.deepcopy(camera), copy.deepcopy(T_c2w), flip_yz=flip_yz
    )
    cx, cy = render_cam.c
    fx, fy = render_cam.f
    render_K = torch.tensor(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device, dtype=dtype
    )
    K_inv = render_K.inverse()

    render_T = render_T.to(device=device, dtype=dtype)
    mkpts_r = mkpts_r.to(device=device, dtype=dtype)

    # --- Back-project to 3D ---
    depth_vals, valid = read_valid_depth(mkpts_r, depth=depth_mat, device=device)
    points_3d = back_project_points_3d(
        depth_vals, render_T[:3, :3], render_T[:3, 3], K_inv, mkpts_r[valid]
    )

    # --- Apply scaling ---
    if mul is not None:
        points_3d = points_3d * mul
        render_T[:3, 3] = render_T[:3, 3] * mul
        origin = origin * mul
        query_T_c2w[:, :3, 3] = query_T_c2w[:, :3, 3] * mul

    # --- Origin subtraction ---
    if origin is None:
        origin = points_3d[0]
    points_3d_local = points_3d - origin

    render_T[:3, 3] -= origin
    render_pose_c2w = Pose.from_Rt(render_T[:3, :3], render_T[:3, 3])
    T_render = render_pose_c2w.inv().float()

    query_T_c2w[:, :3, 3] -= origin
    T_query = Pose.from_Rt(
        query_T_c2w[:, :3, :3], query_T_c2w[:, :3, 3]
    ).inv()

    # --- Center normalization ---
    if points_3d_local.numel() == 0:
        dd = torch.zeros(3, device=device, dtype=dtype)
        points_3d_centered = points_3d_local
    else:
        pts_max = points_3d_local.max(dim=0)[0]
        pts_min = points_3d_local.min(dim=0)[0]
        dd = (pts_min + (pts_max - pts_min) / 2).to(dtype=dtype)
        points_3d_centered = points_3d_local - dd

        tt = T_render.t + T_render.R @ dd
        T_render = Pose.from_Rt(T_render.R, tt)

        tt = T_query.t + T_query.R @ dd
        T_query = Pose.from_Rt(T_query.R, tt)

    return points_3d_centered, T_render, T_query, dd
