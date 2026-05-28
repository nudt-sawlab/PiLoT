#!/usr/bin/env python3
"""Check whether query images match their ContextCapture XML photo records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ExifTags


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_yawfix_refinement_update import _safe_jsonable
from tools.render_contextcapture_xml_domdsm_initial import _load_pose_file_projected, _match_photos, _parse_xml


DEFAULT_XML = "data_caiwangcun/CaiWangCun.xml"
DEFAULT_QUERY_DIR = "data_caiwangcun/query/images/exif_test"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/query_xml_image_identity_check"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row.keys() if not isinstance(row.get(k), (dict, list))})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path, limit_mb: Optional[int] = None) -> str:
    h = hashlib.sha256()
    remaining = None if limit_mb is None else limit_mb * 1024 * 1024
    with path.open("rb") as f:
        while True:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            if size <= 0:
                break
            chunk = f.read(size)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def _phash(path: Path) -> Optional[str]:
    bgr = cv2.imread(os.fspath(path), cv2.IMREAD_GRAYSCALE)
    if bgr is None:
        return None
    small = cv2.resize(bgr, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:8, :8]
    med = np.median(low[1:, 1:])
    bits = (low > med).astype(np.uint8).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _read_exif(path: Path) -> Dict[str, Any]:
    try:
        image = Image.open(path)
        exif_raw = image.getexif()
    except Exception as exc:
        return {"error": repr(exc)}
    tag_names = {v: k for k, v in ExifTags.TAGS.items()}
    wanted = {
        "Make": tag_names.get("Make"),
        "Model": tag_names.get("Model"),
        "DateTime": tag_names.get("DateTime"),
        "DateTimeOriginal": tag_names.get("DateTimeOriginal"),
        "FocalLength": tag_names.get("FocalLength"),
        "FocalLengthIn35mmFilm": tag_names.get("FocalLengthIn35mmFilm"),
        "BodySerialNumber": tag_names.get("BodySerialNumber"),
        "LensSerialNumber": tag_names.get("LensSerialNumber"),
    }
    out: Dict[str, Any] = {}
    for name, tag in wanted.items():
        if tag is not None and tag in exif_raw:
            value = exif_raw.get(tag)
            try:
                if hasattr(value, "numerator") and hasattr(value, "denominator"):
                    value = float(value)
            except Exception:
                pass
            out[name] = str(value)
    return out


def _source_path_candidates(xml_image_path: str) -> List[Path]:
    raw = xml_image_path.replace("\\", "/")
    candidates = [Path(raw)]
    if raw.startswith("//"):
        parts = raw.strip("/").split("/")
        if len(parts) >= 3:
            drive = parts[1].rstrip(":")
            rest = parts[2:]
            candidates.append(Path(f"{drive.upper()}:/") / Path(*rest))
            candidates.append(Path("/mnt") / drive.lower() / Path(*rest))
    return candidates


def _accessible_source(xml_image_path: str) -> Optional[Path]:
    for cand in _source_path_candidates(xml_image_path):
        if cand.exists() and cand.is_file():
            return cand
    return None


def _list_images(query_dir: Path, limit: Optional[int]) -> List[Path]:
    images = sorted(p for p in query_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return images[:limit] if limit and limit > 0 else images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_srs, photos = _parse_xml((REPO_ROOT / args.xml).resolve())
    pose_records = _load_pose_file_projected((REPO_ROOT / args.pose_file).resolve(), xml_srs)
    images = _list_images((REPO_ROOT / args.query_dir).resolve(), args.limit)
    matches, match_report = _match_photos(images, pose_records, photos)
    report_by_image = {item.get("image"): item for item in match_report}

    rows: List[Dict[str, Any]] = []
    for image_path in images:
        match = matches.get(image_path.name)
        query_bgr = cv2.imread(os.fspath(image_path), cv2.IMREAD_COLOR)
        h, w = query_bgr.shape[:2] if query_bgr is not None else (None, None)
        row: Dict[str, Any] = {
            "query_image": os.fspath(image_path.relative_to(REPO_ROOT)),
            "query_name": image_path.name,
            "query_width": w,
            "query_height": h,
            "query_size_bytes": image_path.stat().st_size,
            "query_sha256": _sha256(image_path),
            "query_phash": _phash(image_path),
            "query_exif": _read_exif(image_path),
        }
        if not match:
            row.update({"status": "no_xml_match", "identity_consistent": False})
            rows.append(row)
            continue
        photo = match["photo"]
        report = report_by_image.get(image_path.name, {})
        src = _accessible_source(photo.image_path)
        row.update(
            {
                "status": "ok",
                "matched_photo_id": photo.photo_id,
                "xml_image_path": photo.image_path,
                "xml_basename": Path(photo.image_path.replace("\\", "/")).name,
                "xml_source_accessible": src is not None,
                "xml_intr_width": photo.intrinsics.width,
                "xml_intr_height": photo.intrinsics.height,
                "xml_fx": photo.intrinsics.fx,
                "xml_fy": photo.intrinsics.fy,
                "xml_cx": photo.intrinsics.cx,
                "xml_cy": photo.intrinsics.cy,
                "match_distance_xy_m": report.get("match_distance_xy_m"),
                "match_distance_z_m": report.get("match_distance_z_m"),
                "dimension_matches_xml": bool(w == photo.intrinsics.width and h == photo.intrinsics.height),
                "pose_center_close": bool(report.get("match_distance_xy_m", 1e9) < 1.0 and abs(report.get("match_distance_z_m", 1e9)) < 2.0),
            }
        )
        if src is not None:
            row.update(
                {
                    "xml_source_local_path": os.fspath(src),
                    "xml_source_size_bytes": src.stat().st_size,
                    "xml_source_sha256": _sha256(src),
                    "xml_source_phash": _phash(src),
                    "xml_source_exif": _read_exif(src),
                }
            )
            row["hash_matches_xml_source"] = row["query_sha256"] == row["xml_source_sha256"]
            row["phash_matches_xml_source"] = row["query_phash"] == row["xml_source_phash"]
        else:
            row["hash_matches_xml_source"] = None
            row["phash_matches_xml_source"] = None
        row["basename_matches_xml"] = image_path.name.lower() == row["xml_basename"].lower()
        row["identity_consistent"] = bool(
            row["dimension_matches_xml"]
            and row["pose_center_close"]
            and (row["hash_matches_xml_source"] is True or row["xml_source_accessible"] is False)
        )
        rows.append(row)

    summary = {
        "experiment": "Query XML image identity check",
        "xml": args.xml,
        "query_dir": args.query_dir,
        "pose_file": args.pose_file,
        "xml_srs": xml_srs,
        "num_images": len(rows),
        "num_identity_consistent": sum(1 for r in rows if r.get("identity_consistent")),
        "num_xml_source_accessible": sum(1 for r in rows if r.get("xml_source_accessible")),
        "num_dimension_matches": sum(1 for r in rows if r.get("dimension_matches_xml")),
        "num_pose_center_close": sum(1 for r in rows if r.get("pose_center_close")),
        "rows": rows,
        "camera_match_report": match_report,
        "conclusion": "local XML source images are inaccessible; identity confidence is based on dimensions and EXIF/pose-to-XML center matching"
        if not any(r.get("xml_source_accessible") for r in rows)
        else "identity check includes direct local XML source comparison where accessible",
    }
    _write_csv(output_dir / "query_xml_image_identity_results.csv", rows)
    _write_json(output_dir / "query_xml_image_identity_summary.json", summary)
    print(json.dumps(_safe_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
