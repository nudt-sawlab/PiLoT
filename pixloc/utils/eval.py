"""Evaluation utilities for pose and target location accuracy."""
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from pixloc.utils.transform import WGS84_to_ECEF, get_rotation_enu_in_ecef

logger = logging.getLogger(__name__)


def _euler_to_rotation_ecef(
    euler_angles: list,
    trans: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Euler angles and WGS84 translation to ECEF rotation and position.

    Args:
        euler_angles: [pitch, roll, yaw] in degrees.
        trans: [lon, lat, alt] in WGS84.

    Returns:
        (R_c2w, t_c2w) in ECEF coordinates.
    """
    lon, lat, _ = trans
    rot_pose_in_enu = R.from_euler(
        "xyz", euler_angles, degrees=True
    ).as_matrix()
    rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_enu_in_ecef, rot_pose_in_enu)
    t_c2w = WGS84_to_ECEF(trans)
    return R_c2w, np.array(t_c2w)


def _euler_to_rotation_model(euler_angles: list) -> np.ndarray:
    """Convert xyz Euler angles (degrees) to a rotation matrix in model space."""
    return R.from_euler("xyz", euler_angles, degrees=True).as_matrix()


def evaluate_pose_normalized(
    results_path: str,
    gt_path: str,
    only_localized: bool = False,
) -> Optional[str]:
    """Evaluate pose accuracy in normalized model coordinates (xyz + xyz euler)."""
    predictions: Dict[str, tuple] = {}
    test_names = []
    with open(results_path, "r", encoding="utf-8") as f:
        for data in f.read().rstrip().split("\n"):
            if not data.strip():
                continue
            tokens = data.split()
            name = tokens[0].split("/")[-1]
            t = np.array(tokens[1:4], dtype=float)
            e = list(map(float, tokens[4:7]))
            R_c2w = _euler_to_rotation_model(e)
            predictions[name] = (R_c2w, t, e)
            test_names.append(name)

    gts: Dict[str, tuple] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for data in f.read().rstrip().split("\n"):
            if not data.strip():
                continue
            tokens = data.split()
            name = tokens[0].split("/")[-1]
            t = np.array(tokens[1:4], dtype=float)
            e = list(map(float, tokens[4:7]))
            R_c2w = _euler_to_rotation_model(e)
            gts[name] = (R_c2w, t, e)

    errors_t = []
    errors_R = []
    errors_yaw = []
    for name in test_names:
        if name not in predictions:
            if only_localized:
                continue
            e_t = np.inf
            e_R = 180.0
            e_yaw = 180.0
        else:
            R_gt, t_gt, euler_gt = gts[name]
            R_pred, t_pred, euler_pred = predictions[name]
            e_t = np.linalg.norm(t_gt - t_pred)
            cos = np.clip((np.trace(R_gt.T @ R_pred) - 1) / 2, -1.0, 1.0)
            e_R = np.rad2deg(np.abs(np.arccos(cos)))
            delta = (euler_pred[-1] - euler_gt[-1] + 180) % 360 - 180
            e_yaw = abs(delta)

        errors_t.append(e_t)
        errors_R.append(e_R)
        errors_yaw.append(e_yaw)

    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)
    errors_yaw = np.array(errors_yaw)

    med_t = np.median(errors_t)
    std_t = np.std(errors_t)
    med_R = np.median(errors_R)
    std_R = np.std(errors_R)
    med_yaw = np.median(errors_yaw)
    std_yaw = np.std(errors_yaw)

    logger.info(
        "Normalized pose eval | trans med=%.3f std=%.3f | rot med=%.2f std=%.2f | yaw med=%.2f",
        med_t, std_t, med_R, std_R, med_yaw,
    )
    return (
        f"trans_med={med_t:.4f} trans_std={std_t:.4f} "
        f"rot_med={med_R:.2f} rot_std={std_R:.2f} "
        f"yaw_med={med_yaw:.2f} yaw_std={std_yaw:.2f}"
    )


def evaluate_pose(
    results_path: str,
    gt_path: str,
    only_localized: bool = False,
) -> Optional[str]:
    """Evaluate pose accuracy (translation and rotation) against ground truth.

    Args:
        results_path: Path to predicted poses file.
        gt_path: Path to ground-truth poses file.
        only_localized: If True, skip unlocalized images.

    Returns:
        Formatted evaluation string, or None on failure.
    """
    predictions: Dict[str, tuple] = {}
    test_names = []
    with open(results_path, "r") as f:
        for data in f.read().rstrip().split("\n"):
            tokens = data.split()
            name = tokens[0].split("/")[-1]
            t, e = np.split(np.array(tokens[1:], dtype=float), [3])
            e = [e[1], e[0], e[2]]
            R_c2w, t_c2w = _euler_to_rotation_ecef(e, t)
            predictions[name] = (R_c2w, t_c2w, e)
            test_names.append(name)

    gts: Dict[str, tuple] = {}
    with open(gt_path, "r") as f:
        for data in f.read().rstrip().split("\n"):
            tokens = data.split()
            name = tokens[0].split("/")[-1]
            t, e = np.split(np.array(tokens[1:], dtype=float), [3])
            e = [e[1], e[0], e[2]]
            R_c2w, t_c2w = _euler_to_rotation_ecef(e, t)
            gts[name] = (R_c2w, t_c2w, e)

    errors_t = []
    errors_R = []
    errors_yaw = []
    for name in test_names:
        if name not in predictions:
            if only_localized:
                continue
            e_t = np.inf
            e_R = 180.0
            e_yaw = 180.0
        else:
            R_gt, t_gt, euler_gt = gts[name]
            R_pred, t_pred, euler_pred = predictions[name]
            e_t = np.linalg.norm(-t_gt + t_pred, axis=0)
            cos = np.clip(
                (np.trace(np.dot(R_gt.T, R_pred)) - 1) / 2, -1.0, 1.0
            )
            e_R = np.rad2deg(np.abs(np.arccos(cos)))
            delta = (euler_pred[-1] - euler_gt[-1] + 180) % 360 - 180
            e_yaw = abs(delta)

        errors_t.append(e_t)
        errors_R.append(e_R)
        errors_yaw.append(e_yaw)

    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)
    errors_yaw = np.array(errors_yaw)

    med_t = np.median(errors_t)
    std_t = np.std(errors_t)
    med_R = np.median(errors_R)
    std_R = np.std(errors_R)
    med_yaw = np.median(errors_yaw)
    std_yaw = np.std(errors_yaw)

    # ------------------- UAV SELF-LOCALIZATION SUMMARY -------------------
    width = 54
    line = "+" + "-" * (width - 2) + "+"
    # Ensure all internal separators use the same logic
    # This sep is for 3-column rows
    sep_3col = "+" + "-" * 22 + "+" + "-" * 14 + "+" + "-" * 14 + "+"
    # This sep is for 2-column rows
    sep_2col = "+" + "-" * 35 + "+" + "-" * 16 + "+"

    # --- UAV SELF-LOCALIZATION (PiLoT) ---
    line = "+" + "-" * (width - 2) + "+"
    # Simplified internal separator without middle '+' for perfect alignment
    inner_sep = "|" + "-" * (width - 2) + "|"

    # --- UAV SELF-LOCALIZATION (PiLoT) ---
    out = "\n" + line
    out += f"\n|{'UAV SELF-LOCALIZATION (PiLoT)'.center(width - 2)}|"
    out += "\n" + line
    out += f"\n| {'Metric':<19} | {'Median':^13} | {'Std Dev':^13} |"
    out += "\n" + line
    out += f"\n| {'Trans. Error (m)':<19} | {med_t:^13.3f} | {std_t:^13.3f} |"
    out += f"\n| {'Rot. Error (deg)':<19} | {med_R:^13.3f} | {std_R:^13.3f} |"
    out += f"\n| {'Yaw Error (deg)':<19} | {med_yaw:^13.3f} | {std_yaw:^13.3f} |"
    out += "\n" + line
    out += f"\n|{'RECALL STATISTICS'.center(width - 2)}|"
    out += "\n" + line
    out += f"\n| {'Threshold':<33} | {'Success Rate':^14} |"
    out += "\n" + line
    for th_t in [1, 3, 5]:
        ratio = np.mean(errors_t < th_t)
        out += f"\n| {f'Translation < {th_t}m ({th_t*100}cm)':<33} | {ratio*100:>12.2f}% |"
    for th_R in [1.0, 3.0, 5.0]:
        ratio = np.mean(errors_R < th_R)
        out += f"\n| {f'Rotation < {th_R:.1f} deg':<33} | {ratio*100:>12.2f}% |"
    out += "\n" + line
    logger.info(out)
    return out


def evaluate_target(
    results_path: str,
    gt_path: str,
    only_localized: bool = False,
) -> Optional[Dict[str, float]]:
    """Evaluate target location accuracy in ECEF coordinates.

    Args:
        results_path: Path to predicted target locations file.
        gt_path: Path to ground-truth RTK target locations file.
        only_localized: If True, skip unlocalized images.

    Returns:
        Dictionary with error statistics, or None on failure.
    """
    if not os.path.exists(results_path):
        logger.warning("Prediction file not found: %s", results_path)
        return None
    if not os.path.exists(gt_path):
        logger.warning("Ground truth file not found: %s", gt_path)
        return None

    predictions: Dict[str, np.ndarray] = {}
    test_names = []
    total_num = 0
    test_num = 0

    with open(results_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 4:
                continue
            name = tokens[0].split("/")[-1]
            try:
                t = np.array(tokens[1:4], dtype=float)
                t_ecef = WGS84_to_ECEF(t)
                predictions[name] = t_ecef
                test_names.append(name)
                test_num += 1
            except Exception as e:
                logger.error("Failed to parse prediction for %s: %s", name, e)
                continue

    gts: Dict[str, np.ndarray] = {}
    with open(gt_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 4:
                continue
            name = tokens[0].split("/")[-1]
            try:
                t = np.array(tokens[1:4], dtype=float)
                t_ecef = WGS84_to_ECEF(t)
                gts[name] = t_ecef
                total_num += 1
            except Exception as e:
                logger.error("Failed to parse ground truth for %s: %s", name, e)
                continue

    errors_t = []
    for name in test_names:
        gt_name = name.split("_")[0] + "_0.png"
        if name not in predictions or gt_name not in gts:
            if only_localized:
                continue
            errors_t.append(np.inf)
        else:
            t_gt = np.array(gts[gt_name])
            t_pred = np.array(predictions[name])
            e_t = np.linalg.norm(t_pred - t_gt)
            errors_t.append(e_t)

    if len(errors_t) == 0:
        return None

    errors_t = np.array(errors_t)
    finite_errors = errors_t[np.isfinite(errors_t)]

    stats = {
        "MedianError": np.median(finite_errors),
        "StdError": np.std(finite_errors),
        "Recall@1m": np.mean(errors_t < 1),
        "Recall@3m": np.mean(errors_t < 3),
        "Recall@5m": np.mean(errors_t < 5),
        "Completeness": test_num / total_num if total_num > 0 else 0,
    }

    # ------------------- Target Location Section -------------------
    w = 54
    line_full  = "+" + "-"*52 + "+"
    line_inner = "| " + "-"*50 + " |"
    target_out = "\n" + line_full
    target_out += f"\n|{'TARGET INDICATOR EVALUATION'.center(w-2)}|"
    target_out += "\n" + line_full
    target_out += f"\n| {'Metric':<33} | {'Value':^14} |"
    target_out += f"\n|{'-'*35}+{'-'*16}|"
    target_out += f"\n| {'Median Error (m)':<33} | {np.median(finite_errors):^14.4f} |"
    target_out += f"\n| {'Std Deviation (m)':<33} | {np.std(finite_errors):^14.4f} |"
    target_out += f"\n| {'Completeness':<33} | {stats['Completeness']*100:>12.2f}% |"
    target_out += "\n" + line_full
    target_out += f"\n|{'TARGET RECALL STATISTICS'.center(w-2)}|"
    target_out += "\n" + line_full
    target_out += f"\n| {'Threshold':<33} | {'Success Rate':^14} |"
    target_out += f"\n|{'-'*35}+{'-'*16}|"
    target_out += f"\n| {'Recall @ 1m':<33} | {stats['Recall@1m']*100:>12.2f}% |"
    target_out += f"\n| {'Recall @ 3m':<33} | {stats['Recall@3m']*100:>12.2f}% |"
    target_out += f"\n| {'Recall @ 5m':<33} | {stats['Recall@5m']*100:>12.2f}% |"
    target_out += "\n" + line_full
    logger.info(target_out)
    return stats
