#!/usr/bin/env python3
"""Generate PiLoT pose files from UAV image metadata.

The output format is:
    image_name lon lat alt roll pitch yaw

GPS is read from EXIF when available. Attitude can come from DJI-style XMP
fields, a CSV sidecar, or explicit defaults.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from PIL import ExifTags, Image


GPS_TAGS = {v: k for k, v in ExifTags.GPSTAGS.items()}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _ratio_to_float(value) -> float:
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return float(value.numerator) / float(value.denominator)
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def _dms_to_decimal(values, ref: str) -> float:
    deg, minute, sec = (_ratio_to_float(v) for v in values)
    decimal = deg + minute / 60.0 + sec / 3600.0
    if ref in {"S", "W"}:
        decimal = -decimal
    return decimal


def read_exif_gps(path: Path) -> Tuple[float, float, float]:
    with Image.open(path) as img:
        exif = img.getexif()
        gps_raw = exif.get(34853)
        if not gps_raw:
            raise ValueError("missing EXIF GPSInfo")

    gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
    lat = _dms_to_decimal(gps["GPSLatitude"], gps["GPSLatitudeRef"])
    lon = _dms_to_decimal(gps["GPSLongitude"], gps["GPSLongitudeRef"])

    alt = _ratio_to_float(gps.get("GPSAltitude", 0.0))
    if gps.get("GPSAltitudeRef", 0) == 1:
        alt = -alt
    return lon, lat, alt


def _extract_xmp_number(text: str, names: Iterable[str]) -> Optional[float]:
    for name in names:
        patterns = [
            rf'{re.escape(name)}="([-+0-9.]+)"',
            rf"<[^>]*{re.escape(name)}[^>]*>([-+0-9.]+)</",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
    return None


def read_xmp_attitude(path: Path) -> Dict[str, float]:
    try:
        text = path.read_bytes().decode("utf-8", errors="ignore")
    except OSError:
        return {}

    fields = {
        "yaw": (
            "drone-dji:GimbalYawDegree",
            "drone-dji:FlightYawDegree",
            "GimbalYawDegree",
            "FlightYawDegree",
        ),
        "pitch": (
            "drone-dji:GimbalPitchDegree",
            "drone-dji:FlightPitchDegree",
            "GimbalPitchDegree",
            "FlightPitchDegree",
        ),
        "roll": (
            "drone-dji:GimbalRollDegree",
            "drone-dji:FlightRollDegree",
            "GimbalRollDegree",
            "FlightRollDegree",
        ),
        "alt": (
            "drone-dji:AbsoluteAltitude",
            "drone-dji:RelativeAltitude",
            "AbsoluteAltitude",
            "RelativeAltitude",
        ),
    }

    found: Dict[str, float] = {}
    for key, names in fields.items():
        value = _extract_xmp_number(text, names)
        if value is not None:
            found[key] = value
    return found


def load_csv_metadata(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if path is None:
        return {}

    metadata: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return metadata
        name_field = "image_name" if "image_name" in reader.fieldnames else reader.fieldnames[0]
        for row in reader:
            name = Path(row[name_field]).name
            values: Dict[str, float] = {}
            for key in ("lon", "lat", "alt", "roll", "pitch", "yaw"):
                raw = row.get(key)
                if raw not in (None, ""):
                    values[key] = float(raw)
            metadata[name] = values
    return metadata


def iter_images(image_dir: Path) -> Iterable[Path]:
    return sorted(
        (p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name,
    )


def build_pose_lines(
    image_dir: Path,
    csv_metadata: Dict[str, Dict[str, float]],
    default_roll: float,
    default_pitch: float,
    default_yaw: float,
) -> Iterable[str]:
    for image_path in iter_images(image_dir):
        csv_row = csv_metadata.get(image_path.name, {})
        xmp = read_xmp_attitude(image_path)

        if {"lon", "lat", "alt"}.issubset(csv_row):
            lon, lat, alt = csv_row["lon"], csv_row["lat"], csv_row["alt"]
        else:
            lon, lat, alt = read_exif_gps(image_path)
            alt = csv_row.get("alt", xmp.get("alt", alt))

        roll = csv_row.get("roll", xmp.get("roll", default_roll))
        pitch = csv_row.get("pitch", xmp.get("pitch", default_pitch))
        yaw = csv_row.get("yaw", xmp.get("yaw", default_yaw))

        yield (
            f"{image_path.name} {lon:.10f} {lat:.10f} {alt:.4f} "
            f"{roll:.6f} {pitch:.6f} {yaw:.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Optional CSV with image_name plus lon/lat/alt/roll/pitch/yaw columns.",
    )
    parser.add_argument("--default-roll", type=float, default=0.0)
    parser.add_argument("--default-pitch", type=float, default=0.0)
    parser.add_argument("--default-yaw", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image_dir.is_dir():
        raise NotADirectoryError(args.image_dir)

    csv_metadata = load_csv_metadata(args.metadata_csv)
    lines = list(
        build_pose_lines(
            args.image_dir,
            csv_metadata,
            args.default_roll,
            args.default_pitch,
            args.default_yaw,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"Wrote {len(lines)} poses to {args.output}")


if __name__ == "__main__":
    main()
