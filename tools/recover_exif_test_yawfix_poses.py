#!/usr/bin/env python3
"""Recover yawfix DOM/DSM poses from DJI EXIF/XMP metadata."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import ExifTags, Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.utils.dom_dsm.pose_adapter import normalize_angle_deg
from tools.diagnose_yawfix_refinement_update import _safe_jsonable


DEFAULT_QUERY_DIR = "data_caiwangcun/query/images/exif_test"
DEFAULT_OUTPUT_POSE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/exif_test_yawfix_pose_recovery"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rat(value: Any) -> float:
    return float(value.numerator) / float(value.denominator) if hasattr(value, "numerator") else float(value)


def _dms_to_deg(value: Any, ref: str) -> float:
    deg = _rat(value[0]) + _rat(value[1]) / 60.0 + _rat(value[2]) / 3600.0
    return -deg if ref in {"S", "W"} else deg


def _gps_ifd(exif: Any) -> Dict[str, Any]:
    gps_tag = {v: k for k, v in ExifTags.TAGS.items()}["GPSInfo"]
    gps = exif.get_ifd(gps_tag)
    return {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}


def _xmp_value(path: Path, key: str) -> Optional[float]:
    text = path.read_bytes().decode("utf-8", "ignore")
    match = re.search(r"drone-dji:" + re.escape(key) + r"=\"([^\"]+)\"", text)
    return float(match.group(1)) if match else None


def _recover_one(path: Path) -> Tuple[Optional[str], Dict[str, Any]]:
    record: Dict[str, Any] = {"image": path.name, "path": path.as_posix(), "missing_fields": []}
    try:
        exif = Image.open(path).getexif()
        gps = _gps_ifd(exif)
        record["raw_gps"] = {str(k): str(v) for k, v in gps.items()}
        required = ["GPSLatitude", "GPSLatitudeRef", "GPSLongitude", "GPSLongitudeRef", "GPSAltitude"]
        for key in required:
            if key not in gps:
                record["missing_fields"].append(key)
        gimbal_yaw = _xmp_value(path, "GimbalYawDegree")
        record["raw_gimbal_yaw_degree"] = gimbal_yaw
        if gimbal_yaw is None:
            record["missing_fields"].append("drone-dji:GimbalYawDegree")
        if record["missing_fields"]:
            record["status"] = "skipped_missing_fields"
            return None, record
        lon = _dms_to_deg(gps["GPSLongitude"], gps["GPSLongitudeRef"])
        lat = _dms_to_deg(gps["GPSLatitude"], gps["GPSLatitudeRef"])
        alt = _rat(gps["GPSAltitude"])
        if gps.get("GPSAltitudeRef") in {1, b"\x01"}:
            alt = -alt
        yaw = normalize_angle_deg(-float(gimbal_yaw))
        roll = 180.0
        pitch = 0.0
        line = f"{path.name} {lon:.10f} {lat:.10f} {alt:.3f} {roll:.1f} {pitch:.1f} {yaw:.6f}"
        record.update(
            {
                "status": "ok",
                "translation_lon_lat_alt": [lon, lat, alt],
                "roll_pitch_yaw_file_order": [roll, pitch, yaw],
                "computed_yaw": yaw,
                "pose_line": line,
            }
        )
        return line, record
    except Exception as exc:
        record["status"] = "error"
        record["error"] = repr(exc)
        return None, record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    parser.add_argument("--output-pose-file", default=DEFAULT_OUTPUT_POSE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query_dir = (REPO_ROOT / args.query_dir).resolve()
    output_pose = (REPO_ROOT / args.output_pose_file).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    images = sorted([p for p in query_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}])
    lines: List[str] = []
    records: List[Dict[str, Any]] = []
    for path in images:
        line, record = _recover_one(path)
        records.append(record)
        if line:
            lines.append(line)
    output_pose.parent.mkdir(parents=True, exist_ok=True)
    output_pose.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    summary = {
        "query_dir": args.query_dir,
        "output_pose_file": args.output_pose_file,
        "num_images": len(images),
        "num_recovered": len(lines),
        "num_skipped": len(images) - len(lines),
        "records": records,
        "yaw_rule": "yaw = normalize_angle_deg(-drone-dji:GimbalYawDegree)",
        "downward_pose_file_convention": {"roll": 180.0, "pitch": 0.0},
    }
    _write_json(output_dir / "exif_pose_recovery_summary.json", summary)
    print(json.dumps({"num_images": len(images), "num_recovered": len(lines), "output_pose_file": str(output_pose)}, indent=2))
    return 0 if lines else 1


if __name__ == "__main__":
    raise SystemExit(main())
