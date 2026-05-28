#!/usr/bin/env python3
"""Finite-difference PyTorch feature-loss diagnosis around DOM/DSM yawfix initial."""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from diagnose_domdsm_torch_feature_loss import BASE_EULER, run_diagnosis

DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test_16x9/0000.jpg"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/torch_feature_loss_local_gradient"

PERTURBATIONS = [
    ("initial", [0.0, 0.0, 0.0], BASE_EULER),
    ("east_minus_1m", [-1.0, 0.0, 0.0], BASE_EULER),
    ("east_plus_1m", [1.0, 0.0, 0.0], BASE_EULER),
    ("north_minus_1m", [0.0, -1.0, 0.0], BASE_EULER),
    ("north_plus_1m", [0.0, 1.0, 0.0], BASE_EULER),
    ("yaw_minus_1deg", [0.0, 0.0, 0.0], [0.0, 180.0, 28.2]),
    ("yaw_plus_1deg", [0.0, 0.0, 0.0], [0.0, 180.0, 30.2]),
    ("alt_minus_1m", [0.0, 0.0, -1.0], BASE_EULER),
    ("alt_plus_1m", [0.0, 0.0, 1.0], BASE_EULER),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    p.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-points", type=int, default=500)
    p.add_argument("--sampling-mode", default="combined")
    return p.parse_args()


def _direction(initial: float, minus: float, plus: float, eps: float = 1e-4) -> str:
    if minus < initial - eps and minus < plus - eps:
        return "negative"
    if plus < initial - eps and plus < minus - eps:
        return "positive"
    return "flat"


def main() -> int:
    args = parse_args()
    summary = run_diagnosis(args, candidates=PERTURBATIONS, clean_output=True)
    by = {m["candidate"]: m for m in summary["candidates"]}
    init = by["initial"]["torch_feature_loss"]
    result: Dict[str, Any] = {
        "initial_loss": init,
        "east_minus_loss": by["east_minus_1m"]["torch_feature_loss"],
        "east_plus_loss": by["east_plus_1m"]["torch_feature_loss"],
        "north_minus_loss": by["north_minus_1m"]["torch_feature_loss"],
        "north_plus_loss": by["north_plus_1m"]["torch_feature_loss"],
        "yaw_minus_loss": by["yaw_minus_1deg"]["torch_feature_loss"],
        "yaw_plus_loss": by["yaw_plus_1deg"]["torch_feature_loss"],
        "alt_minus_loss": by["alt_minus_1m"]["torch_feature_loss"],
        "alt_plus_loss": by["alt_plus_1m"]["torch_feature_loss"],
        "estimated_local_direction": {
            "east": _direction(init, by["east_minus_1m"]["torch_feature_loss"], by["east_plus_1m"]["torch_feature_loss"]),
            "north": _direction(init, by["north_minus_1m"]["torch_feature_loss"], by["north_plus_1m"]["torch_feature_loss"]),
            "yaw": _direction(init, by["yaw_minus_1deg"]["torch_feature_loss"], by["yaw_plus_1deg"]["torch_feature_loss"]),
            "alt": _direction(init, by["alt_minus_1m"]["torch_feature_loss"], by["alt_plus_1m"]["torch_feature_loss"]),
        },
        "candidates": summary["candidates"],
    }
    out = REPO_ROOT / args.output_dir / "local_gradient_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
