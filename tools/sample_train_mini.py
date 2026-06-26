#!/usr/bin/env python3
"""Mapscape dataset tools: generate training data from RAW, or sample a mini subset.

Modes
-----
generate (default)
  Read RAW on data2, write processed training data to Train-test on UserData:
    Step 1: resize images 512x512  (crop_mapscape_raw)
    Step 2: refer_info.json + Points3D  (training_generation)

sample
  Copy a random subset from an already-processed Train-test into Train-mini.

Examples
--------
  # RAW -> Train-test (one sequence)
  python tools/sample_train_mini.py --seq England_seq1@200@30_50 --clean

  # RAW -> Train-test (quick test, 30 frames)
  python tools/sample_train_mini.py --seq England_seq1@200@30_50 --max-frames 30 --ref-offset 5

  # Train-test -> Train-mini
  python tools/sample_train_mini.py sample --num-samples 1000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = Path(__file__).resolve().parent
for p in (_ROOT, _TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

DEFAULT_RAW_ROOT = Path("/mnt/data2/PublicDatasets/Mapscape/RAW")
DEFAULT_PROCESSED_ROOT = Path(
    "/mnt/data1/UserData/liuxy24/Mapscape/crop1200@1200/Train-test"
)
DEFAULT_MINI_ROOT = Path(
    "/mnt/data1/UserData/liuxy24/Mapscape/crop1200@1200/Train-mini"
)


def collect_pairs(src_root: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seq_dirs = sorted(p.name for p in src_root.iterdir() if p.is_dir())
    iterator = seq_dirs
    if tqdm is not None:
        iterator = tqdm(seq_dirs, desc="Scanning sequences")

    for seq in iterator:
        seq_dir = src_root / seq
        ref_path = seq_dir / "refer_info.json"
        pts_dir = seq_dir / "Points3D"
        if not ref_path.is_file() or not pts_dir.is_dir():
            continue

        with open(ref_path, "r", encoding="utf-8") as f:
            refer_info = json.load(f)

        pts_stems = {p.stem for p in pts_dir.iterdir() if p.suffix == ".npy"}
        for key in refer_info:
            if Path(key).stem in pts_stems:
                pairs.append((seq, key))
    return pairs


def pair_file_paths(src_root: Path, seq: str, key: str, entry: dict) -> list[Path]:
    seq_dir = src_root / seq
    stem = Path(key).stem
    return [
        seq_dir / "Points3D" / f"{stem}.npy",
        src_root / entry["img_path"],
        src_root / entry["img_depth"],
        src_root / entry["ref_info"]["ref_rgb"],
        src_root / entry["ref_info"]["ref_depth"],
    ]


def link_or_copy(src: Path, dst: Path, use_hardlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if use_hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def copy_seq_meta(src_seq: Path, dst_seq: Path, use_hardlink: bool) -> None:
    for name in ("camera.txt", f"{src_seq.name}.txt"):
        src = src_seq / name
        if src.is_file():
            link_or_copy(src, dst_seq / name, use_hardlink)


def _load_training_generation():
    import importlib.util

    path = _ROOT / "dataset" / "training_generation.py"
    spec = importlib.util.spec_from_file_location("training_generation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_generate(args: argparse.Namespace) -> int:
    from crop_mapscape_raw import process_sequence as resize_sequence

    tg = _load_training_generation()
    generate_refer_info = tg.generate_refer_info
    load_camera = tg.load_camera
    load_poses = tg.load_poses

    raw_root = args.raw_root.resolve()
    dst_root = args.dst.resolve()

    if not (raw_root / "images").is_dir() or not (raw_root / "poses").is_dir():
        print(f"RAW layout not found under {raw_root} (need images/ and poses/)",
              file=sys.stderr)
        return 1

    dst_root.parent.mkdir(parents=True, exist_ok=True)
    probe = dst_root.parent / ".write_probe_generate"
    try:
        probe.touch()
        probe.unlink()
    except OSError as exc:
        print(f"Cannot write to {dst_root.parent}: {exc}", file=sys.stderr)
        return 1

    if args.seq:
        seq_list = args.seq
    else:
        seq_list = sorted(
            p.name for p in (raw_root / "images").iterdir() if p.is_dir()
        )

    resize_size = tuple(args.resize_size)
    for seq in seq_list:
        print(f"\n=== [{seq}] resize RAW -> {dst_root} ===")
        resize_sequence(
            raw_root, dst_root, seq,
            resize_size, args.max_frames, args.clean,
        )

        seq_dir = dst_root / seq
        pose_path = seq_dir / f"{seq}.txt"
        camera_path = seq_dir / "camera.txt"
        query_dirs = [f for f in os.listdir(seq_dir) if "query@" in f]
        if not query_dirs:
            print(f"skip refer_info for {seq}: no query@ folder", file=sys.stderr)
            continue

        print(f"\n=== [{seq}] generate refer_info + Points3D ===")
        pose_dict = load_poses(str(pose_path))
        camera_dict = load_camera(str(camera_path))
        generate_refer_info(
            str(dst_root),
            str(seq_dir / "Points3D"),
            str(seq_dir / "ref"),
            str(seq_dir / query_dirs[0]),
            seq,
            pose_dict,
            camera_dict,
            ref_offset=args.ref_offset,
            max_queries=args.max_queries,
            min_valid_threshold=args.min_valid_threshold,
        )

    print(f"\nAll done. Processed data: {dst_root}")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    src_root = args.src.resolve()
    dst_root = args.dst.resolve()
    use_hardlink = not args.no_hardlink

    if not src_root.is_dir():
        print(f"Source not found: {src_root}", file=sys.stderr)
        return 1

    if not args.dry_run:
        dst_parent = dst_root.parent
        probe = dst_parent / ".write_probe_sample_train_mini"
        try:
            dst_parent.mkdir(parents=True, exist_ok=True)
            probe.touch()
            probe.unlink()
        except OSError as exc:
            print(f"Cannot write to {dst_parent}: {exc}", file=sys.stderr)
            return 1

    print(f"Collecting pairs from {src_root} ...")
    pairs = collect_pairs(src_root)
    total = len(pairs)
    print(f"Found {total:,} valid pairs.")

    if total == 0:
        print("No pairs found. Run preprocessing first (default command).", file=sys.stderr)
        return 1

    if args.num_samples > total:
        print(
            f"Requested {args.num_samples:,} samples but only {total:,} available; "
            f"using all pairs.",
            file=sys.stderr,
        )
        sampled = pairs
    else:
        rng = random.Random(args.seed)
        sampled = rng.sample(pairs, args.num_samples)

    print(f"Sampled {len(sampled):,} pairs (seed={args.seed}).")

    by_seq: dict[str, list[str]] = defaultdict(list)
    for seq, key in sampled:
        by_seq[seq].append(key)
    print(f"Covering {len(by_seq)} sequences.")

    if args.dry_run:
        est_bytes = 0
        missing = 0
        unique_files: set[Path] = set()
        refer_cache: dict[str, dict] = {}
        pair_iter = sampled
        if tqdm is not None:
            pair_iter = tqdm(sampled, desc="Estimating size")
        for seq, key in pair_iter:
            if seq not in refer_cache:
                with open(src_root / seq / "refer_info.json", "r", encoding="utf-8") as f:
                    refer_cache[seq] = json.load(f)
            entry = refer_cache[seq][key]
            for p in pair_file_paths(src_root, seq, key, entry):
                if p.is_file():
                    unique_files.add(p)
                else:
                    missing += 1
        for p in unique_files:
            est_bytes += p.stat().st_size
        print(f"Unique files to link/copy: {len(unique_files):,}")
        print(f"Missing files: {missing}")
        print(f"Estimated disk usage (unique files): {est_bytes / 1e9:.2f} GB")
        return 0

    dst_root.mkdir(parents=True, exist_ok=True)
    copied_files: set[Path] = set()
    missing_files: list[str] = []
    t0 = time.time()

    seq_iter = sorted(by_seq.items())
    if tqdm is not None:
        seq_iter = tqdm(seq_iter, desc="Sequences")

    refer_cache: dict[str, dict] = {}
    for seq, keys in seq_iter:
        src_seq = src_root / seq
        dst_seq = dst_root / seq
        copy_seq_meta(src_seq, dst_seq, use_hardlink)

        if seq not in refer_cache:
            with open(src_seq / "refer_info.json", "r", encoding="utf-8") as f:
                refer_cache[seq] = json.load(f)
        refer_info = refer_cache[seq]

        subset = {k: refer_info[k] for k in keys}
        dst_seq.mkdir(parents=True, exist_ok=True)
        with open(dst_seq / "refer_info.json", "w", encoding="utf-8") as f:
            json.dump(subset, f)

        key_iter = keys
        if tqdm is not None and len(by_seq) == 1:
            key_iter = tqdm(keys, desc=f"Pairs in {seq}", leave=False)

        for key in key_iter:
            entry = refer_info[key]
            for src_file in pair_file_paths(src_root, seq, key, entry):
                if not src_file.is_file():
                    missing_files.append(str(src_file))
                    continue
                rel = src_file.relative_to(src_root)
                dst_file = dst_root / rel
                if dst_file in copied_files:
                    continue
                link_or_copy(src_file, dst_file, use_hardlink)
                copied_files.add(dst_file)

    manifest = {
        "src": str(src_root),
        "dst": str(dst_root),
        "seed": args.seed,
        "num_samples": len(sampled),
        "num_sequences": len(by_seq),
        "unique_files": len(copied_files),
        "missing_files": len(missing_files),
        "use_hardlink": use_hardlink,
        "pairs_per_sequence": {seq: len(keys) for seq, keys in sorted(by_seq.items())},
    }
    with open(dst_root / "sample_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    print(f"Done in {elapsed / 60:.1f} min.")
    print(f"Destination: {dst_root}")
    print(f"Unique files linked/copied: {len(copied_files):,}")
    if missing_files:
        print(f"Warning: {len(missing_files)} missing files.")
    du_size = sum(p.stat().st_size for p in copied_files if p.exists())
    print(f"Approx unique payload: {du_size / 1e9:.2f} GB")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "sample":
        return _main_sample(argv[1:])
    return _main_generate(argv)


def _main_generate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Read Mapscape RAW and write processed data to Train-test.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=f"Mapscape RAW root (default: {DEFAULT_RAW_ROOT})",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help=f"Output directory (default: {DEFAULT_PROCESSED_ROOT})",
    )
    parser.add_argument(
        "--seq",
        nargs="+",
        default=None,
        help="Sequence name(s); default: all under RAW/images/",
    )
    parser.add_argument("--resize-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Only process first N frames (quick test)")
    parser.add_argument("--clean", action="store_true",
                        help="Remove existing output seq dir before resize")
    parser.add_argument("--ref-offset", type=int, default=200)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--min-valid-threshold", type=int, default=800)
    args = parser.parse_args(argv)
    return cmd_generate(args)


def _main_sample(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Sample a subset from Train-test into Train-mini.",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help=f"Processed Train-test (default: {DEFAULT_PROCESSED_ROOT})",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_MINI_ROOT,
        help=f"Output Train-mini (default: {DEFAULT_MINI_ROOT})",
    )
    parser.add_argument("--num-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-hardlink", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return cmd_sample(args)


if __name__ == "__main__":
    raise SystemExit(main())
