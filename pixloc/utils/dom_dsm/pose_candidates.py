"""DOM/DSM-local candidate pose generation helpers."""

from typing import Dict, List, Sequence, Tuple

from pyproj import Transformer

from pixloc.utils.dom_dsm.pose_adapter import apply_enu_offset


def generate_domdsm_candidate_poses(
    base_trans: Sequence[float],
    base_euler: Sequence[float],
    to_raster: Transformer,
    from_raster: Transformer,
    east_offsets=(-10, -5, 0, 5, 10),
    north_offsets=(-10, -5, 0, 5, 10),
    alt_offsets=(0,),
    yaw_offsets=(-8, -4, 0, 4, 8),
    freeze_pitch_roll: bool = True,
) -> Tuple[List[List[float]], List[List[float]], List[Dict[str, float]]]:
    """Generate interpretable local DOM/DSM candidate translations/eulers.

    East/north offsets are applied in projected DOM/DSM meters, not by lon/lat
    scaling. The returned poses remain WGS84 + [pitch, roll, yaw] so they can
    be inspected before any PixLoc-specific Pose construction.
    """
    if len(base_trans) != 3:
        raise ValueError("base_trans must be [lon, lat, alt]")
    if len(base_euler) != 3:
        raise ValueError("base_euler must be [pitch, roll, yaw]")
    candidate_trans_list: List[List[float]] = []
    candidate_euler_list: List[List[float]] = []
    candidate_offsets: List[Dict[str, float]] = []
    for east in east_offsets:
        for north in north_offsets:
            for alt in alt_offsets:
                trans = apply_enu_offset(base_trans, east, north, alt, to_raster, from_raster)
                for yaw_delta in yaw_offsets:
                    if freeze_pitch_roll:
                        euler = [float(base_euler[0]), float(base_euler[1]), float(base_euler[2]) + float(yaw_delta)]
                    else:
                        euler = [float(base_euler[0]), float(base_euler[1]), float(base_euler[2]) + float(yaw_delta)]
                    candidate_trans_list.append(trans)
                    candidate_euler_list.append(euler)
                    candidate_offsets.append(
                        {
                            "east_m": float(east),
                            "north_m": float(north),
                            "alt_m": float(alt),
                            "yaw_deg": float(yaw_delta),
                        }
                    )
    return candidate_trans_list, candidate_euler_list, candidate_offsets


def build_domdsm_query_pose_batch(*args, **kwargs):
    """Placeholder for converting DOM/DSM candidate lists to a PixLoc Pose batch.

    The candidate-list generation above is ready for debugging. The Pose batch
    conversion should be wired after validating axis signs against
    sample_3d_points(), preprocess_pose_for_pixloc(), and pixloc_to_osg.
    """
    raise NotImplementedError(
        "DOM/DSM PixLoc Pose batch conversion is intentionally left as a TODO."
    )
