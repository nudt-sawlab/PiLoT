#!/usr/bin/env python3
"""Check DOM/DSM pose CRS offset roundtrip and euler convention entrypoints."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.utils.dom_dsm.pose_adapter import (
    apply_enu_offset,
    compute_enu_delta_m,
    get_domdsm_transformers,
    make_domdsm_downward_euler,
    normalize_domdsm_euler,
)
from src.utils.pose_utils import load_initial_pose

DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt"
DEFAULT_OUTPUT = "docs/experiments/dom_dsm_prepare/torch_feature_loss_diagnosis/pose_roundtrip.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    _euler, trans, _origin = load_initial_pose(args.pose_file)
    trans = list(map(float, trans))
    to_raster, from_raster, raster_crs = get_domdsm_transformers(config)
    tests = [
        [0.0, 0.0, 0.0],
        [-5.0, 0.0, 0.0],
        [-5.0, 5.0, 0.0],
        [-13.2078, -2.1052, 0.0],
        [-13.2078, -2.1052, 4.097],
    ]
    checks = []
    for offset in tests:
        cand = apply_enu_offset(trans, offset[0], offset[1], offset[2], to_raster, from_raster)
        delta = compute_enu_delta_m(trans, cand, to_raster)
        err = [abs(delta[i] - offset[i]) for i in range(3)]
        passed = bool(err[0] < 1e-3 and err[1] < 1e-3 and err[2] < 1e-6)
        checks.append({"offset": offset, "candidate_translation_lon_lat_alt": cand, "roundtrip_delta": delta, "abs_error": err, "passed": passed})
    result: Dict[str, Any] = {
        "config": args.config,
        "pose_file": args.pose_file,
        "base_translation_lon_lat_alt": trans,
        "raster_crs": raster_crs,
        "roundtrip_checks": checks,
        "all_roundtrip_passed": all(item["passed"] for item in checks),
        "euler_convention_check": {
            "make_domdsm_downward_euler_29_2": make_domdsm_downward_euler(29.2),
            "normalize_domdsm_euler_base": normalize_domdsm_euler([0.0, 180.0, 29.2]),
            "refined_euler_record_only": [180.0, 0.0, -144.8],
            "note": "Refined Euler convention remains separate; feature-loss diagnosis fixes initial downward euler except raw_refined_full.",
        },
    }
    out = REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_roundtrip_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
