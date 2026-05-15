"""Candidate definitions for DOM/DSM visual-safe scoring experiments."""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pyproj import CRS, Transformer

from pixloc.utils.dom_dsm.pose_adapter import (
    apply_enu_offset,
    compute_enu_delta_m,
    make_downward_euler_from_yaw,
    refined_yaw_to_downward_yaw,
)


DEFAULT_KNOWN_SEEDS = [
    {
        "name": "p3_best_chamfer",
        "source": "known_seed",
        "stage": "known_seed",
        "east": -5.0,
        "north": 0.0,
        "alt": 0.0,
        "yaw": 0.0,
        "description": "P3 prior best chamfer seed; metrics are recomputed by P11.",
    },
    {
        "name": "p3_best_overlap",
        "source": "known_seed",
        "stage": "known_seed",
        "east": -5.0,
        "north": 5.0,
        "alt": 0.0,
        "yaw": 0.0,
        "description": "P3 prior best overlap seed; metrics are recomputed by P11.",
    },
    {
        "name": "p4_scale_025_fixed_alt",
        "source": "known_seed",
        "stage": "known_seed",
        "east": -3.301955,
        "north": -0.526292,
        "alt": 0.0,
        "yaw": 0.0,
        "description": "P4 prior scale 0.25 fixed-alt seed; metrics are recomputed by P11.",
    },
]


def offset_lonlat_by_enu(
    lon: float,
    lat: float,
    east_m: float,
    north_m: float,
    crs: str = "EPSG:32650",
    to_raster: Optional[Transformer] = None,
    from_raster: Optional[Transformer] = None,
    alt: float = 0.0,
    alt_m: float = 0.0,
) -> List[float]:
    """Offset lon/lat by projected east/north meters and return [lon, lat, alt]."""
    if to_raster is None or from_raster is None:
        raster_crs = CRS.from_user_input(crs)
        to_raster = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        from_raster = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
    return apply_enu_offset([lon, lat, alt], east_m, north_m, alt_m, to_raster, from_raster)


def _spec(
    name: str,
    source: str,
    stage: str,
    translation: Sequence[float],
    euler: Sequence[float],
    east: float,
    north: float,
    alt: float,
    yaw: float,
    description: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = {
        "name": name,
        "source": source,
        "stage": stage,
        "translation_lon_lat_alt": [float(x) for x in translation],
        "euler_pitch_roll_yaw": [float(x) for x in euler],
        "offset_east_m": float(east),
        "offset_north_m": float(north),
        "offset_alt_m": float(alt),
        "yaw_offset_deg": float(yaw),
        "description": description,
    }
    if extra:
        out.update(extra)
    return out


def _offset_name(prefix: str, value: float, unit: str) -> str:
    if value == 0:
        return f"{prefix}_0{unit}"
    sign = "plus" if value > 0 else "minus"
    mag = str(abs(float(value))).replace(".", "p").rstrip("0").rstrip("p")
    return f"{prefix}_{sign}_{mag}{unit}"


def build_local_candidate_specs(
    initial_translation_lon_lat_alt: Sequence[float],
    base_yaw: float,
    to_raster: Transformer,
    from_raster: Transformer,
    coarse_east_offsets: Sequence[float] = (-5, -3, -1, 0, 1, 3, 5),
    coarse_north_offsets: Sequence[float] = (-5, -3, -1, 0, 1, 3, 5),
    include_known_p3p4_seeds: bool = True,
    include_raw_refined_candidates: bool = True,
    raw_refined_translation_lon_lat_alt: Optional[Sequence[float]] = None,
    raw_refined_euler_pitch_roll_yaw: Optional[Sequence[float]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build initial, local, known-seed, and raw-refined candidate specs."""
    base_trans = [float(x) for x in initial_translation_lon_lat_alt]
    base_euler = make_downward_euler_from_yaw(base_yaw)
    candidates: List[Dict[str, Any]] = []

    candidates.append(
        _spec(
            "initial",
            "initial",
            "initial",
            base_trans,
            base_euler,
            0.0,
            0.0,
            0.0,
            0.0,
            "Yawfix initial pose.",
        )
    )

    one_steps = [
        ("east_minus_1m", -1.0, 0.0, 0.0, 0.0),
        ("east_plus_1m", 1.0, 0.0, 0.0, 0.0),
        ("north_minus_1m", 0.0, -1.0, 0.0, 0.0),
        ("north_plus_1m", 0.0, 1.0, 0.0, 0.0),
        ("yaw_minus_1deg", 0.0, 0.0, 0.0, -1.0),
        ("yaw_plus_1deg", 0.0, 0.0, 0.0, 1.0),
        ("alt_minus_1m", 0.0, 0.0, -1.0, 0.0),
        ("alt_plus_1m", 0.0, 0.0, 1.0, 0.0),
    ]
    for name, east, north, alt, yaw in one_steps:
        trans = apply_enu_offset(base_trans, east, north, alt, to_raster, from_raster)
        euler = make_downward_euler_from_yaw(float(base_yaw) + yaw)
        candidates.append(
            _spec(name, "local_grid", "one_step", trans, euler, east, north, alt, yaw, "One-step local diagnostic candidate.")
        )

    if include_known_p3p4_seeds:
        for seed in DEFAULT_KNOWN_SEEDS:
            trans = apply_enu_offset(base_trans, seed["east"], seed["north"], seed["alt"], to_raster, from_raster)
            euler = make_downward_euler_from_yaw(float(base_yaw) + seed["yaw"])
            candidates.append(
                _spec(
                    seed["name"],
                    seed["source"],
                    seed["stage"],
                    trans,
                    euler,
                    seed["east"],
                    seed["north"],
                    seed["alt"],
                    seed["yaw"],
                    seed["description"],
                )
            )

    for east in coarse_east_offsets:
        for north in coarse_north_offsets:
            trans = apply_enu_offset(base_trans, east, north, 0.0, to_raster, from_raster)
            name = f"coarse_e{float(east):+g}_n{float(north):+g}".replace("+", "p").replace("-", "m").replace(".", "p")
            candidates.append(
                _spec(
                    name,
                    "local_grid",
                    "coarse_translation",
                    trans,
                    base_euler,
                    float(east),
                    float(north),
                    0.0,
                    0.0,
                    "Stage 1 coarse translation grid candidate.",
                )
            )

    debug: Dict[str, Any] = {"bad_raw_refined_yaw_for_debug_only": None}
    if include_raw_refined_candidates and raw_refined_translation_lon_lat_alt is not None and raw_refined_euler_pitch_roll_yaw is not None:
        raw_trans = [float(x) for x in raw_refined_translation_lon_lat_alt]
        raw_euler = [float(x) for x in raw_refined_euler_pitch_roll_yaw]
        raw_delta = compute_enu_delta_m(base_trans, raw_trans, to_raster)
        raw_yaw = float(raw_euler[2])
        downward_yaw = refined_yaw_to_downward_yaw(raw_yaw)
        downward_yaw_offset = float(downward_yaw) - float(base_yaw)
        debug["bad_raw_refined_yaw_for_debug_only"] = make_downward_euler_from_yaw(raw_yaw)
        candidates.append(
            _spec(
                "raw_refined_full",
                "raw_refined",
                "raw_refined",
                raw_trans,
                raw_euler,
                raw_delta[0],
                raw_delta[1],
                raw_delta[2],
                downward_yaw_offset,
                "Raw rebuilt CUDA refined output; included for diagnosis only, never directly trusted.",
                {"raw_refined_yaw": raw_yaw, "corrected_downward_yaw": downward_yaw},
            )
        )
        freeze_trans = [raw_trans[0], raw_trans[1], base_trans[2]]
        candidates.append(
            _spec(
                "refined_freeze_alt_corrected_yaw",
                "corrected_refined",
                "raw_refined",
                freeze_trans,
                make_downward_euler_from_yaw(downward_yaw),
                raw_delta[0],
                raw_delta[1],
                0.0,
                downward_yaw_offset,
                "Raw refined lon/lat with initial alt and corrected downward yaw.",
                {"raw_refined_yaw": raw_yaw, "corrected_downward_yaw": downward_yaw},
            )
        )
        candidates.append(
            _spec(
                "refined_freeze_alt_base_yaw",
                "corrected_refined",
                "raw_refined",
                freeze_trans,
                make_downward_euler_from_yaw(base_yaw),
                raw_delta[0],
                raw_delta[1],
                0.0,
                0.0,
                "Raw refined lon/lat with initial alt and base yaw.",
                {"raw_refined_yaw": raw_yaw, "corrected_downward_yaw": downward_yaw},
            )
        )
    return candidates, debug


def deduplicate_candidates(
    candidates: Iterable[Dict[str, Any]],
    pose_tol: float = 1e-6,
    offset_tol: float = 1e-4,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deduplicate by name and near-identical offsets/yaw, preserving order."""
    seen_names = set()
    seen_pose_keys = set()
    deduped: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    for item in candidates:
        name = str(item["name"])
        pose_key = (
            round(float(item.get("offset_east_m", 0.0)) / offset_tol),
            round(float(item.get("offset_north_m", 0.0)) / offset_tol),
            round(float(item.get("offset_alt_m", 0.0)) / offset_tol),
            round(float(item.get("yaw_offset_deg", 0.0)) / pose_tol),
        )
        if name in seen_names or pose_key in seen_pose_keys:
            removed.append({"name": name, "reason": "duplicate_name_or_pose", "candidate": item})
            continue
        seen_names.add(name)
        seen_pose_keys.add(pose_key)
        deduped.append(item)
    return deduped, removed
