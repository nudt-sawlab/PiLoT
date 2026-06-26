"""Convert PiLoT WGS84/ECEF poses to CityGaussian model coordinates."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from pixloc.utils.transform import (
    WGS84_to_ECEF,
    euler_angles_to_matrix_ECEF,
    get_rotation_enu_in_ecef,
)

# Render2Loc / Target2loc OSG axis flip: R @ diag(1, -1, -1)
_OS_D = np.diag([1.0, -1.0, -1.0])


def apply_pilot_opencv_fix(c2w: np.ndarray) -> np.ndarray:
    """Apply Render2Loc OSG axis flip (same as Target2loc pose_convention)."""
    c2w = np.asarray(c2w, dtype=np.float64)
    out = c2w.copy()
    out[:3, :3] = c2w[:3, :3] @ _OS_D
    return out


def colmap_c2w_to_render2loc(c2w: np.ndarray) -> np.ndarray:
    """COLMAP/3DGS c2w -> Render2Loc internal (OSG) c2w for back-projection."""
    return apply_pilot_opencv_fix(c2w)


def render2loc_c2w_to_colmap(c2w: np.ndarray) -> np.ndarray:
    """Render2Loc internal (OSG) c2w -> COLMAP/3DGS c2w for rendering."""
    return apply_pilot_opencv_fix(c2w)


def parse_matrix(values) -> np.ndarray:
    """Parse a 4x4 matrix from a nested list in YAML."""
    mat = np.asarray(values, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix, got shape {mat.shape}")
    return mat


def ecef_pose_from_pilot(trans: List[float], euler: List[float]) -> np.ndarray:
    """Build the PiLoT ECEF camera-to-world matrix with axis convention fix."""
    return apply_pilot_opencv_fix(euler_angles_to_matrix_ECEF(euler, trans))


def compute_ecef_to_model_from_reference(
    reference_c2w_model: np.ndarray,
    reference_euler: List[float],
    reference_trans: List[float],
) -> np.ndarray:
    """Estimate ``ecef_to_model`` from one known GPS pose and COLMAP view."""
    reference_c2w_ecef = ecef_pose_from_pilot(reference_trans, reference_euler)
    return reference_c2w_model @ np.linalg.inv(reference_c2w_ecef)


def pilot_pose_to_model_c2w(
    trans: List[float],
    euler: List[float],
    ecef_to_model: np.ndarray,
) -> np.ndarray:
    """Map a PiLoT pose to a CityGaussian camera-to-world matrix."""
    c2w_ecef = ecef_pose_from_pilot(trans, euler)
    return ecef_to_model @ c2w_ecef


def c2w_to_w2c_rt(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert c2w to CityGaussian world-to-camera ``R, T``."""
    w2c = np.linalg.inv(c2w)
    return (
        w2c[:3, :3].astype(np.float32),
        w2c[:3, 3].astype(np.float32),
    )


def resolve_ecef_to_model(
    cfg: dict,
    reference_c2w_model: Optional[np.ndarray],
) -> np.ndarray:
    """Resolve the ECEF-to-model transform from config."""
    if "ecef_to_model" in cfg and cfg["ecef_to_model"] is not None:
        return parse_matrix(cfg["ecef_to_model"])

    ref_cfg = cfg.get("reference_calibration")
    if ref_cfg and reference_c2w_model is not None:
        return compute_ecef_to_model_from_reference(
            reference_c2w_model,
            ref_cfg["euler"],
            ref_cfg["trans"],
        )

    raise ValueError(
        "CityGaussian renderer requires either 'ecef_to_model' (4x4) or "
        "'reference_calibration' with matching 'reference_image' in config."
    )


def euler_trans_to_colmap_c2w(
    trans: Union[List[float], np.ndarray],
    euler: Union[List[float], np.ndarray],
) -> np.ndarray:
    """Build a COLMAP camera-to-world matrix in normalized model space."""
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.from_euler("xyz", euler, degrees=True).as_matrix()
    c2w[:3, 3] = np.asarray(trans, dtype=np.float64).reshape(3)
    return c2w


def euler_trans_to_pilot_c2w(
    trans: List[float],
    euler: List[float],
) -> np.ndarray:
    """Model-space c2w with the PiLoT Y/Z axis convention used by back-projection."""
    return apply_pilot_opencv_fix(euler_trans_to_colmap_c2w(trans, euler))


def pilot_c2w_to_colmap_c2w(c2w: np.ndarray) -> np.ndarray:
    """Undo OSG axis flip to obtain a COLMAP c2w for 3DGS rendering."""
    return render2loc_c2w_to_colmap(c2w)


def colmap_c2w_to_pilot_c2w(c2w: np.ndarray) -> np.ndarray:
    """COLMAP c2w -> OSG back-projection convention (Target2loc compatible)."""
    return colmap_c2w_to_render2loc(c2w)


def c2w_colmap_to_euler_trans(c2w: np.ndarray) -> Tuple[List[float], List[float]]:
    """Extract xyz translation and xyz-Euler (deg) from a COLMAP c2w matrix."""
    mat = np.asarray(c2w, dtype=np.float64)
    trans = mat[:3, 3].tolist()
    euler = R.from_matrix(mat[:3, :3]).as_euler("xyz", degrees=True).tolist()
    return euler, trans


def c2w_pilot_to_euler_trans(c2w: np.ndarray) -> Tuple[List[float], List[float]]:
    """Convert a PiLoT-style c2w matrix back to (euler_xyz_deg, translation_xyz)."""
    mat = np.asarray(c2w, dtype=np.float64)
    rot = mat[:3, :3].copy()
    rot[:, 1] *= -1
    rot[:, 2] *= -1
    euler = R.from_matrix(rot).as_euler("xyz", degrees=True).tolist()
    trans = mat[:3, 3].tolist()
    return euler, trans


def pixloc_to_model(T_refined_c2w: np.ndarray) -> Tuple[List[float], List[float]]:
    """Convert refined c2w (normalized) to euler + translation for rendering."""
    euler, trans = c2w_pilot_to_euler_trans(T_refined_c2w)
    return euler, trans


def read_render2loc_pose_txt(
    pose_txt: str,
    frame: int = 0,
) -> Tuple[np.ndarray, List[float], List[float], Optional[np.ndarray]]:
    """Read Render2Loc normalized pose: name x y z euler_x euler_y euler_z [qw qx qy qz]."""
    target = f"{frame}_0.png"
    path = Path(pose_txt)
    if not path.is_file():
        raise FileNotFoundError(f"Pose file not found: {path}")

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] != target:
            continue
        if len(parts) < 7:
            raise ValueError(f"Invalid pose line: {line}")
        trans = list(map(float, parts[1:4]))
        euler = list(map(float, parts[4:7]))
        q_w2c = (
            np.asarray(list(map(float, parts[7:11])), dtype=np.float64)
            if len(parts) >= 11
            else None
        )
        c2w = euler_trans_to_colmap_c2w(trans, euler)
        return c2w, euler, trans, q_w2c

    raise ValueError(f"{target} not found in {path}")
