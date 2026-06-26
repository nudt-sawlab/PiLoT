#!/usr/bin/env python3
"""Build training-ready Mapscape data from RAW.

Read:  /mnt/data2/PublicDatasets/Mapscape/RAW
Write: /mnt/data1/UserData/liuxy24/Mapscape/crop1200@1200/Train-test

Steps per sequence:
  1. resize RGB/depth to 512x512  (crop_mapscape_raw)
  2. generate refer_info.json + Points3D  (training_generation)

Usage:
  python tools/preprocess_mapscape_from_raw.py --seq England_seq1@200@30_50 --clean
  python tools/preprocess_mapscape_from_raw.py --seq England_seq1@200@30_50 --max-frames 30 --ref-offset 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
RAW_ROOT = Path("/mnt/data2/PublicDatasets/Mapscape/RAW")
DST_ROOT = Path("/mnt/data1/UserData/liuxy24/Mapscape/crop1200@1200/Train-test")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--dst-root", type=Path, default=DST_ROOT)
    parser.add_argument("--seq", type=str, required=True)
    parser.add_argument("--resize-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--ref-offset", type=int, default=200)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--min-valid-threshold", type=int, default=800)
    args = parser.parse_args()

    crop_script = _REPO / "tools" / "crop_mapscape_raw.py"
    gen_script = _REPO / "dataset" / "training_generation.py"

    crop_cmd = [
        sys.executable, str(crop_script),
        "--raw-root", str(args.raw_root),
        "--dst-root", str(args.dst_root),
        "--seq", args.seq,
        "--resize-size", str(args.resize_size[0]), str(args.resize_size[1]),
    ]
    if args.max_frames is not None:
        crop_cmd += ["--max-frames", str(args.max_frames)]
    if args.clean:
        crop_cmd.append("--clean")

    print("==> Step 1/2: resize RAW images")
    print(" ".join(crop_cmd))
    if subprocess.run(crop_cmd, cwd=_REPO).returncode != 0:
        return 1

    gen_cmd = [
        sys.executable, str(gen_script),
        "--root", str(args.dst_root),
        "--names", args.seq,
        "--ref-offset", str(args.ref_offset),
        "--min-valid-threshold", str(args.min_valid_threshold),
    ]
    if args.max_queries is not None:
        gen_cmd += ["--max-queries", str(args.max_queries)]

    print("==> Step 2/2: generate refer_info.json + Points3D")
    print(" ".join(gen_cmd))
    if subprocess.run(gen_cmd, cwd=_REPO).returncode != 0:
        return 1

    print(f"Done. Output: {args.dst_root / args.seq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
