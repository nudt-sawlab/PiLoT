#!/usr/bin/env python3
"""Run a small DOM/DSM-aware refinement batch and aggregate stability metrics."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_IMAGE_DIR = "data_caiwangcun/query/images/exif_test_16x9"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/domdsm_aware_refinement_results_batch"


def _read_pose_names(path: Path) -> List[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and not parts[0].startswith("#"):
            names.append(parts[0])
    return names


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _delta_stats(items: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    vals = [float(item[key]) for item in items if key in item]
    if not vals:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {"mean": sum(vals) / len(vals), "median": median(vals), "min": min(vals), "max": max(vals)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--sampling-mode", default="combined")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--freeze-alt", action="store_true")
    parser.add_argument("--freeze-pitch-roll", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_names = _read_pose_names(REPO_ROOT / args.pose_file)[: args.max_images]
    failures = []
    results = []
    for name in pose_names:
        image_path = REPO_ROOT / args.image_dir / name
        if not image_path.exists():
            failures.append({"image": name, "error": "image not found"})
            continue
        image_out = output_dir / Path(name).stem
        cmd = [
            sys.executable,
            "tools/run_domdsm_aware_refinement_single.py",
            "--config", args.config,
            "--query-image", os.fspath(image_path.relative_to(REPO_ROOT)),
            "--pose-file", args.pose_file,
            "--output-dir", os.fspath(image_out.relative_to(REPO_ROOT)),
            "--width", str(args.width),
            "--sampling-modes", args.sampling_mode,
        ]
        if args.freeze_alt:
            cmd.append("--freeze-alt")
        if args.freeze_pitch_roll:
            cmd.append("--freeze-pitch-roll")
        proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (image_out / "batch_command_output.log").write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            failures.append({"image": name, "error": f"returncode {proc.returncode}"})
            continue
        summary_path = image_out / "summary_metrics.json"
        if not summary_path.exists():
            failures.append({"image": name, "error": "summary_metrics.json missing"})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        initial = summary["initial"]
        refined = summary["sampling_modes"][args.sampling_mode]
        raw_delta = refined.get("raw_delta_east_north_alt_m", [None, None, None])
        results.append(
            {
                "image": name,
                "initial_chamfer": initial["edge_chamfer"],
                "refined_chamfer": refined["edge_chamfer"],
                "initial_overlap": initial["edge_overlap_ratio"],
                "refined_overlap": refined["edge_overlap_ratio"],
                "chamfer_delta": refined["edge_chamfer"] - initial["edge_chamfer"],
                "overlap_delta": refined["edge_overlap_ratio"] - initial["edge_overlap_ratio"],
                "translation_update_m": float((raw_delta[0] ** 2 + raw_delta[1] ** 2 + raw_delta[2] ** 2) ** 0.5) if None not in raw_delta else None,
                "alt_drift_m": raw_delta[2],
            }
        )
    n = len(results)
    chamfer_deltas = [r["chamfer_delta"] for r in results]
    overlap_deltas = [r["overlap_delta"] for r in results]
    update_vals = [r["translation_update_m"] for r in results if r["translation_update_m"] is not None]
    batch_summary = {
        "num_images": n,
        "pose_file_image_count": len(pose_names),
        "batch_note": "pose file has only one image; batch evaluation not yet available" if len(pose_names) <= 1 else None,
        "initial_mean_chamfer": sum(r["initial_chamfer"] for r in results) / n if n else None,
        "refined_mean_chamfer": sum(r["refined_chamfer"] for r in results) / n if n else None,
        "improve_rate": sum(1 for r in results if r["chamfer_delta"] < 0) / n if n else None,
        "median_chamfer_delta": median(chamfer_deltas) if chamfer_deltas else None,
        "median_overlap_delta": median(overlap_deltas) if overlap_deltas else None,
        "mean_translation_update_m": sum(update_vals) / len(update_vals) if update_vals else None,
        "alt_drift_stats": _delta_stats(results, "alt_drift_m"),
        "failures": failures,
        "per_image": results,
    }
    _write_json(output_dir / "batch_summary.json", batch_summary)
    print(json.dumps(batch_summary, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
