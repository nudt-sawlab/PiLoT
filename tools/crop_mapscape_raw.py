#!/usr/bin/env python3
"""Resize Mapscape RAW images to 512x512 (one output frame per source frame)."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

RAW_CAMERA = np.array([1600.0, 1200.0, 1931.7, 1931.7, 800.0, 600.0])


def _save_lines(path: Path, rows: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for name in sorted(rows.keys(), key=lambda x: int(re.search(r"(\d+)", x).group(1))):
            f.write(f"{name} {' '.join(map(str, rows[name]))}\n")


def _scale_camera(camera: np.ndarray, src_size: tuple[int, int],
                  dst_size: tuple[int, int]) -> list[float]:
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    w, h, fx, fy, cx, cy = camera
    return [
        float(dst_w), float(dst_h),
        float(fx * dst_w / src_w), float(fy * dst_h / src_h),
        float(cx * dst_w / src_w), float(cy * dst_h / src_h),
    ]


def _load_poses(pose_file: Path) -> dict[str, list[float]]:
    pose_dict: dict[str, list[float]] = {}
    with open(pose_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                pose_dict[parts[0]] = list(map(float, parts[1:]))
    return pose_dict


def _build_pose_and_camera(pose_dict: dict[str, list[float]], camera: list[float],
                           frame_ids: set[int] | None) -> tuple[dict, dict]:
    new_pose_dict: dict[str, list[float]] = {}
    new_camera_dict: dict[str, list[float]] = {}
    for pose_name in sorted(pose_dict.keys(), key=lambda x: int(re.search(r"(\d+)", x).group(1))):
        frame_id = int(re.search(r"(\d+)", pose_name).group(1))
        if frame_ids is not None and frame_id not in frame_ids:
            continue
        name = f"{pose_name.split('.')[0]}_0.png"
        new_pose_dict[name] = pose_dict[pose_name]
        new_camera_dict[name] = camera
    return new_pose_dict, new_camera_dict


def _process_image_folder(src_dir: Path, dst_dir: Path, resize_size: tuple[int, int],
                          max_frames: int | None) -> set[int]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    rgb_files = sorted(
        [f for f in os.listdir(src_dir) if f.endswith("_0.png")],
        key=lambda x: int(re.search(r"(\d+)", x).group(1)),
    )
    if max_frames is not None:
        rgb_files = rgb_files[:max_frames]

    processed_ids: set[int] = set()
    for rgb_name in rgb_files:
        frame_id = int(rgb_name.split("_")[0])
        depth_name = rgb_name.replace("_0.png", "_1.png")
        rgb_path = src_dir / rgb_name
        depth_path = src_dir / depth_name
        rgb_out = dst_dir / rgb_name
        depth_out = dst_dir / depth_name
        if rgb_out.is_file() and depth_out.is_file():
            processed_ids.add(frame_id)
            continue
        if not depth_path.is_file():
            print(f"skip missing depth: {depth_path}")
            continue
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            print(f"skip unreadable: {rgb_name}")
            continue
        depth = cv2.flip(depth, 0)
        rgb_resized = cv2.resize(rgb, resize_size, interpolation=cv2.INTER_AREA)
        depth_resized = cv2.resize(depth, resize_size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(rgb_out), rgb_resized)
        cv2.imwrite(str(depth_out), cv2.flip(depth_resized, 0))
        processed_ids.add(frame_id)
        print(f"resized {rgb_name} -> {resize_size[0]}x{resize_size[1]}")
    return processed_ids


def process_sequence(raw_root: Path, dst_root: Path, seq: str,
                     resize_size: tuple[int, int], max_frames: int | None,
                     clean: bool = False) -> None:
    seq_src = raw_root / "images" / seq
    pose_src = raw_root / "poses" / f"{seq}.txt"
    if not seq_src.is_dir():
        raise FileNotFoundError(f"missing: {seq_src}")
    if not pose_src.is_file():
        raise FileNotFoundError(f"missing: {pose_src}")

    seq_dst = dst_root / seq
    if clean and seq_dst.exists():
        shutil.rmtree(seq_dst)
    seq_dst.mkdir(parents=True, exist_ok=True)

    pose_dict = _load_poses(pose_src)
    processed_ids: set[int] = set()
    for sub in sorted(p for p in seq_src.iterdir() if p.is_dir()):
        processed_ids |= _process_image_folder(sub, seq_dst / sub.name, resize_size, max_frames)
    if not processed_ids:
        raise RuntimeError(f"no frames processed for {seq}")

    scaled_camera = _scale_camera(RAW_CAMERA, (1600, 1200), resize_size)
    frame_filter = processed_ids if max_frames is not None else None
    new_pose, new_cam = _build_pose_and_camera(pose_dict, scaled_camera, frame_filter)
    _save_lines(seq_dst / f"{seq}.txt", new_pose)
    _save_lines(seq_dst / "camera.txt", new_cam)
    print(f"done {seq}: {len(processed_ids)} frames -> {seq_dst}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/data2/PublicDatasets/Mapscape/RAW"))
    parser.add_argument("--dst-root", type=Path,
                        default=Path("/mnt/data1/UserData/liuxy24/Mapscape/crop1200@1200/Train-test"))
    parser.add_argument("--seq", type=str, required=True)
    parser.add_argument("--resize-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    process_sequence(args.raw_root, args.dst_root, args.seq,
                     tuple(args.resize_size), args.max_frames, args.clean)


if __name__ == "__main__":
    main()
