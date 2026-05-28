#!/usr/bin/env python3
"""Check ContextCapture XML tie point reprojection errors."""

import argparse
import csv
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_yawfix_refinement_update import _safe_jsonable
from tools.render_contextcapture_xml_domdsm_initial import (
    RENDER_RAY_ROTATION,
    XML_PROJECTION_ROTATION,
    _load_pose_file_projected,
    _match_photos,
    _parse_xml,
)


DEFAULT_XML = "data_caiwangcun/CaiWangCun.xml"
DEFAULT_QUERY_DIR = "data_caiwangcun/query/images/exif_test"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/contextcapture_xml_tiepoint_reprojection_check"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float_text(parent: ET.Element, path: str) -> float:
    text = parent.findtext(path)
    if text is None:
        raise KeyError(path)
    return float(text)


def _parse_tiepoints(xml_path: Path) -> List[Dict[str, Any]]:
    root = ET.parse(xml_path).getroot()
    out: List[Dict[str, Any]] = []
    for idx, tp in enumerate(root.iter("TiePoint")):
        pos = tp.find("Position")
        if pos is None:
            continue
        measurements = []
        for m in tp.findall("Measurement"):
            pid = m.findtext("PhotoId")
            if pid is None:
                continue
            measurements.append({"photo_id": str(pid), "x": _float_text(m, "x"), "y": _float_text(m, "y")})
        out.append(
            {
                "tiepoint_index": idx,
                "position": [_float_text(pos, "x"), _float_text(pos, "y"), _float_text(pos, "z")],
                "measurements": measurements,
            }
        )
    return out


def _distort(x: np.ndarray, y: np.ndarray, intr: Any) -> Tuple[np.ndarray, np.ndarray]:
    k1, k2, k3, p1, p2 = intr.k1, intr.k2, intr.k3, intr.p1, intr.p2
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x * radial + dx, y * radial + dy


PROJECTION_CONVENTIONS = [
    {
        "name": "world_to_camera_correct",
        "legacy_convention": "R_xml",
        "xml_projection_rotation": XML_PROJECTION_ROTATION,
        "axis": [1.0, 1.0, 1.0],
        "is_primary_acceptance": True,
    },
    {
        "name": "camera_to_world_wrong_for_projection",
        "legacy_convention": "R_xml_transpose",
        "xml_projection_rotation": "R_camera_to_world = R_xml.T",
        "axis": [1.0, 1.0, 1.0],
        "is_primary_acceptance": False,
    },
]


def _project(point: Sequence[float], photo: Any, convention: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    p = np.asarray(point, dtype=np.float64)
    c = np.asarray(photo.center_xml, dtype=np.float64)
    delta = p - c
    if convention["name"] == "world_to_camera_correct":
        pc = photo.rotation @ delta
    elif convention["name"] == "camera_to_world_wrong_for_projection":
        pc = photo.rotation.T @ delta
    else:
        raise ValueError(convention["name"])
    pc = pc * np.asarray(convention["axis"], dtype=np.float64)
    if abs(pc[2]) < 1e-9:
        return None
    x = pc[0] / pc[2]
    y = pc[1] / pc[2]
    xd, yd = _distort(np.asarray([x]), np.asarray([y]), photo.intrinsics)
    u = float(photo.intrinsics.fx * xd[0] + photo.intrinsics.cx)
    v = float(photo.intrinsics.fy * yd[0] + photo.intrinsics.cy)
    return u, v, float(pc[2])


def _stats(errors: List[float]) -> Dict[str, Any]:
    if not errors:
        return {
            "valid_measurement_count": 0,
            "mean_reprojection_error_px": None,
            "median_reprojection_error_px": None,
            "p90_reprojection_error_px": None,
            "max_reprojection_error_px": None,
        }
    arr = np.asarray(errors, dtype=np.float64)
    return {
        "valid_measurement_count": int(arr.size),
        "mean_reprojection_error_px": float(arr.mean()),
        "median_reprojection_error_px": float(np.median(arr)),
        "p90_reprojection_error_px": float(np.percentile(arr, 90)),
        "max_reprojection_error_px": float(arr.max()),
    }


def _interpret(median: Optional[float]) -> Dict[str, Any]:
    lt2 = bool(median is not None and median < 2.0)
    gt10 = bool(median is not None and median > 10.0)
    if lt2:
        text = "XML camera chain is basically correct for tie point reprojection."
    elif gt10:
        text = "XML convention, distortion, or coordinate parsing is still wrong."
    else:
        text = "XML camera chain is plausible but not sub-pixel; inspect convention and distortion details."
    return {"median_error_lt_2px": lt2, "median_error_gt_10px": gt10, "camera_chain_interpretation": text}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-limit", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = (REPO_ROOT / args.xml).resolve()
    xml_srs, photos = _parse_xml(xml_path)
    photo_by_id = {p.photo_id: p for p in photos}
    tiepoints = _parse_tiepoints(xml_path)
    query_dir = (REPO_ROOT / args.query_dir).resolve()
    image_paths = sorted([p for p in query_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}])
    pose_records = _load_pose_file_projected((REPO_ROOT / args.pose_file).resolve(), xml_srs)
    _matches, match_report = _match_photos(image_paths, pose_records, photos)
    matched_ids = {m["matched_photo_id"]: m for m in match_report if m.get("status") == "ok"}
    all_summaries = []
    samples = []
    for conv in PROJECTION_CONVENTIONS:
        per_photo_errors: Dict[str, List[float]] = defaultdict(list)
        total_errors: List[float] = []
        for tp in tiepoints:
            point = tp["position"]
            for meas in tp["measurements"]:
                photo = photo_by_id.get(meas["photo_id"])
                if photo is None:
                    continue
                proj = _project(point, photo, conv)
                if proj is None:
                    continue
                u, v, depth = proj
                if depth <= 0 or not np.isfinite(u) or not np.isfinite(v):
                    continue
                err = math.hypot(u - meas["x"], v - meas["y"])
                per_photo_errors[photo.photo_id].append(err)
                total_errors.append(err)
                if len(samples) < args.sample_limit:
                    samples.append(
                        {
                            "convention": conv["name"],
                            "legacy_convention": conv["legacy_convention"],
                            "xml_projection_rotation": conv["xml_projection_rotation"],
                            "is_primary_acceptance": conv["is_primary_acceptance"],
                            "photo_id": photo.photo_id,
                            "tiepoint_position_x": point[0],
                            "tiepoint_position_y": point[1],
                            "tiepoint_position_z": point[2],
                            "measurement_x": meas["x"],
                            "measurement_y": meas["y"],
                            "projected_x": u,
                            "projected_y": v,
                            "depth": depth,
                            "error_px": err,
                        }
                    )
        rows = []
        for pid, photo in photo_by_id.items():
            row = {
                "convention": conv["name"],
                "legacy_convention": conv["legacy_convention"],
                "xml_projection_rotation": conv["xml_projection_rotation"],
                "render_ray_rotation_note": f"Not used for tie point reprojection; render rays use {RENDER_RAY_ROTATION}.",
                "is_primary_acceptance": conv["is_primary_acceptance"],
                "photo_id": pid,
                "xml_image_path": photo.image_path,
                **_stats(per_photo_errors.get(pid, [])),
            }
            row.update(_interpret(row["median_reprojection_error_px"]))
            rows.append(row)
        rows.sort(key=lambda r: int(r["photo_id"]) if str(r["photo_id"]).isdigit() else str(r["photo_id"]))
        with (output_dir / f"reprojection_per_photo_{conv['name']}.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows[0].keys()) if rows else []
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "convention": conv["name"],
            "legacy_convention": conv["legacy_convention"],
            "xml_projection_rotation": conv["xml_projection_rotation"],
            "render_ray_rotation_note": f"Not used for tie point reprojection; render rays use {RENDER_RAY_ROTATION}.",
            "is_primary_acceptance": conv["is_primary_acceptance"],
            **_stats(total_errors),
        }
        summary.update(_interpret(summary["median_reprojection_error_px"]))
        all_summaries.append({"summary": summary, "per_photo": rows})
    primary = next(x for x in all_summaries if x["summary"]["is_primary_acceptance"])
    best = primary
    matched_summary = []
    best_rows_by_id = {row["photo_id"]: row for row in best["per_photo"]}
    for m in match_report:
        pid = m.get("matched_photo_id")
        if pid in best_rows_by_id:
            matched_summary.append({**m, **best_rows_by_id[pid]})
    with (output_dir / "reprojection_per_photo.csv").open("w", newline="", encoding="utf-8") as f:
        rows = best["per_photo"]
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "reprojection_measurements_sample.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0].keys()) if samples else [])
        writer.writeheader()
        writer.writerows(samples)
    _write_json(
        output_dir / "matched_query_reprojection_summary.json",
        {
            "primary_projection_convention": best["summary"]["convention"],
            "xml_projection_rotation": XML_PROJECTION_ROTATION,
            "matched_queries": matched_summary,
        },
    )
    _write_json(
        output_dir / "reprojection_summary.json",
        {
            "xml": args.xml,
            "xml_srs": xml_srs,
            "xml_projection_rotation": XML_PROJECTION_ROTATION,
            "render_ray_rotation_note": f"Not used for tie point reprojection; render rays use {RENDER_RAY_ROTATION}.",
            "conventions": [x["summary"] for x in all_summaries],
            "best": best["summary"],
            "num_tiepoints": len(tiepoints),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
