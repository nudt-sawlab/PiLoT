#!/usr/bin/env python3
"""Prepare a centered 16:9 crop for the single Caiwangcun query image."""

import argparse
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = "data_caiwangcun/query/images/exif_test/0000.jpg"
DEFAULT_OUTPUT = "data_caiwangcun/query/images/exif_test_16x9/0000.jpg"
EXPECTED_SIZE = (5280, 3956)
TARGET_SIZE = (5280, 2970)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = (REPO_ROOT / args.input).resolve()
    output_path = (REPO_ROOT / args.output).resolve()

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(input_path)

    height, width = image.shape[:2]
    expected_width, expected_height = EXPECTED_SIZE
    if (width, height) != EXPECTED_SIZE:
        raise ValueError(
            f"Expected {expected_width}x{expected_height}, got {width}x{height}: "
            f"{input_path}"
        )

    target_width, target_height = TARGET_SIZE
    x0 = (width - target_width) // 2
    y0 = (height - target_height) // 2
    crop = image[y0 : y0 + target_height, x0 : x0 + target_width]
    if crop.shape[:2] != (target_height, target_width):
        raise RuntimeError(f"Unexpected crop shape: {crop.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise IOError(f"Failed to write {output_path}")

    print(f"input={input_path}")
    print(f"input_size={width}x{height}")
    print(f"crop_box=left:{x0}, top:{y0}, right:{x0 + target_width}, bottom:{y0 + target_height}")
    print(f"output={output_path}")
    print(f"output_size={target_width}x{target_height}")
    print(f"output_aspect={target_width / target_height:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
