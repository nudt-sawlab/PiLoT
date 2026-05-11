"""Pose and CRS helpers for DOM/DSM refinement experiments."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import rasterio
from pyproj import Transformer


def _as_trans(trans: Sequence[float]) -> Tuple[float, float, float]:
    if len(trans) != 3:
        raise ValueError(f"Expected [lon, lat, alt], got {trans}")
    return float(trans[0]), float(trans[1]), float(trans[2])


def _domdsm_raster_path(config: Dict[str, Any]) -> Path:
    """Return the configured DOM/DSM raster path used for CRS lookup."""
    render_config = config.get("render_config", {})
    domdsm = render_config.get("dom_dsm", {})
    path = (
        domdsm.get("dom_path")
        or domdsm.get("dsm_path")
        or render_config.get("dom_path")
        or render_config.get("dsm_path")
        or render_config.get("ortho_path")
    )
    if not path:
        raise KeyError("No DOM/DSM raster path found in render_config")
    return Path(path)


def get_domdsm_transformers(config: Dict[str, Any]) -> Tuple[Transformer, Transformer, str]:
    """Build WGS84 <-> DOM/DSM CRS transformers from the configured raster.

    Args:
        config: PiLoT YAML config dictionary containing render_config.dom_dsm.

    Returns:
        (to_raster, from_raster, raster_crs_string), where to_raster converts
        EPSG:4326 lon/lat into DOM/DSM projected x/y meters and from_raster
        converts back to EPSG:4326.
    """
    raster_path = _domdsm_raster_path(config)
    with rasterio.open(raster_path) as ds:
        raster_crs = ds.crs
    if raster_crs is None:
        raise ValueError(f"Raster has no CRS: {raster_path}")
    to_raster = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    from_raster = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
    return to_raster, from_raster, str(raster_crs)


def wgs84_to_domxy(trans: Sequence[float], to_raster: Transformer) -> List[float]:
    """Convert [lon, lat, alt] to DOM/DSM projected [x, y, alt] meters."""
    lon, lat, alt = _as_trans(trans)
    x, y = to_raster.transform(lon, lat)
    return [float(x), float(y), alt]


def domxy_to_wgs84(xya: Sequence[float], from_raster: Transformer) -> List[float]:
    """Convert DOM/DSM projected [x, y, alt] meters to [lon, lat, alt]."""
    if len(xya) != 3:
        raise ValueError(f"Expected [x, y, alt], got {xya}")
    lon, lat = from_raster.transform(float(xya[0]), float(xya[1]))
    return [float(lon), float(lat), float(xya[2])]


def apply_enu_offset(
    trans: Sequence[float],
    east_m: float,
    north_m: float,
    alt_m: float,
    to_raster: Transformer,
    from_raster: Transformer,
) -> List[float]:
    """Apply meter offsets in the DOM/DSM projected CRS and return WGS84 pose translation."""
    x, y, alt = wgs84_to_domxy(trans, to_raster)
    return domxy_to_wgs84([x + float(east_m), y + float(north_m), alt + float(alt_m)], from_raster)


def compute_enu_delta_m(
    base_trans: Sequence[float],
    target_trans: Sequence[float],
    to_raster: Transformer,
) -> List[float]:
    """Compute target-base translation delta in DOM/DSM east/north/alt meters."""
    bx, by, balt = wgs84_to_domxy(base_trans, to_raster)
    tx, ty, talt = wgs84_to_domxy(target_trans, to_raster)
    return [float(tx - bx), float(ty - by), float(talt - balt)]


def normalize_domdsm_euler(euler: Sequence[float]) -> List[float]:
    """Return a DOMDSMRenderer euler triplet in [pitch, roll, yaw] order.

    This is intentionally a narrow, testable conversion entrypoint. The current
    yawfix convention uses [0.0, 180.0, yaw]. Equivalent refined downward-form
    handling should be added explicitly instead of inferred here.
    """
    if len(euler) != 3:
        raise ValueError(f"Expected [pitch, roll, yaw], got {euler}")
    return [float(euler[0]), float(euler[1]), float(euler[2])]


def make_domdsm_downward_euler(yaw: float) -> List[float]:
    """Create the DOMDSMRenderer downward-looking euler [pitch, roll, yaw]."""
    return [0.0, 180.0, float(yaw)]
