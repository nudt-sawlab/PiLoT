"""Convert 6-DOF pose (WGS84 + euler) to a 4x4 camera-to-world matrix
compatible with the COLMAP / 3D Gaussian Splatting coordinate system.

Adapted from osg_to_colmap.py.
"""

import math

import numpy as np
import pyproj


def wgs84_to_cgcs2000(lon: float, lat: float, alt: float) -> np.ndarray:
    """WGS84 (lon, lat, alt) -> CGCS2000 projected coordinates (x, y, alt)."""
    wgs84 = pyproj.CRS("EPSG:4326")
    cgcs2000 = pyproj.CRS("EPSG:4547")
    transformer = pyproj.Transformer.from_crs(wgs84, cgcs2000, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.array([x, y, alt], dtype=np.float64)


def euler_to_rotmat_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """ZYX euler angles (degrees) -> 3x3 rotation matrix."""
    yaw = math.radians(yaw)
    pitch = math.radians(pitch)
    roll = math.radians(roll)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def dof6_to_matrix(
    lat: float,
    lon: float,
    alt: float,
    roll_in: float,
    pitch_in: float,
    yaw_in: float,
    cgcs_offset: np.ndarray = np.array([401448, 3131258, 0], dtype=np.float64),
) -> np.ndarray:
    """Convert a 6-DOF pose to a 4x4 camera-to-world transform matrix.

    The pose originates from the PiLoT/OSG system and is converted into the
    COLMAP coordinate frame used by the 3D Gaussian Splatting model.

    Args:
        lat, lon, alt: WGS84 position.
        roll_in, pitch_in, yaw_in: Euler angles (degrees) – note that
            pitch/roll are swapped w.r.t. the raw COLMAP extraction and
            a 180° offset is applied to roll.
        cgcs_offset: Scene-specific CGCS2000 origin offset ``[x0, y0, z0]``.

    Returns:
        4x4 camera-to-world matrix (np.float64).
    """
    translation = wgs84_to_cgcs2000(lon, lat, alt) - cgcs_offset

    raw_pitch = roll_in
    raw_roll = pitch_in
    true_yaw = yaw_in
    true_pitch = raw_pitch
    true_roll = raw_roll - 180

    rotation = euler_to_rotmat_zyx(true_yaw, true_pitch, true_roll)

    c2w = np.eye(4)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = translation
    return c2w
