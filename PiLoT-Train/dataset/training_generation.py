import os
import sys
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pixlib.geometry.wrappers import Camera
from utils.get_depth import (
    get_3D_samples,
    get_points2D_ECEF_projection,
    sample_points_with_valid_depth,
)
from utils.transform import get_matrix_list_batch


def generate_refer_info(root, Points3D_path, reference_path, query_path, seq,
                        pose_dict, camera_dict, vis_save_path=None,
                        ref_offset=200, max_queries=None, min_valid_threshold=800):
    refer_info = {}
    os.makedirs(Points3D_path, exist_ok=True)
    origin_flag = 1
    origin_save = None
    success_num = 0
    query_items = list(pose_dict.items())
    if max_queries is not None:
        query_items = query_items[:max_queries]
    frame_ids = sorted({int(n.split('_')[0]) for n in pose_dict})

    for pose_name, pose in tqdm(query_items, desc=f"generate {seq}"):
        attempt = 0
        success_find = False
        query_rgb_file = pose_name
        query_rgb_path = os.path.join(query_path, query_rgb_file)
        query_depth_path = os.path.join(query_path, query_rgb_file.replace('_0.png', '_1.png'))
        query_depth_image = cv2.imread(query_depth_path, cv2.IMREAD_UNCHANGED)
        query_rgb_image = cv2.imread(query_rgb_path)
        if query_rgb_image is None or query_depth_image is None:
            continue
        if query_depth_image.mean() < 1 or query_depth_image.mean() > 65000:
            continue

        query_T_np = pose['T_c2w']
        if origin_flag == 1:
            origin_save = query_T_np[:3, 3]
            origin_flag = 0
        query_T = query_T_np.tolist()
        query_pose = pose['euler_angles'] + pose['translation']
        query_index = int(pose_name.split('_')[0])
        name_suffix = query_rgb_file.split('_', 1)[1]
        if query_rgb_file not in camera_dict:
            continue
        K_query = camera_dict[query_rgb_file]
        cam_query = {'model': 'PINHOLE', 'width': K_query[0], 'height': K_query[1],
                     'params': K_query[2:6]}

        ref_rgb_file = ref_rgb_image = None
        ref_T = K_ref = None
        final_indices = points2d_query = points2d_ref_valid = Points_3D_ECEF_origin = None

        while attempt < 10 and not success_find:
            attempt += 1
            offsets = [ref_offset, -ref_offset, ref_offset // 2, -ref_offset // 2]
            offset = offsets[(attempt - 1) % len(offsets)]
            if attempt > len(offsets):
                offset = random.randint(-min(50, query_index), min(50, frame_ids[-1] - query_index))
            ref_name = f"{query_index + offset}_{name_suffix}"
            if ref_name not in pose_dict:
                continue
            ref_pose = pose_dict[ref_name]
            ref_T = ref_pose['T_c2w'].tolist()
            ref_rgb_file = ref_name
            ref_rgb_path = os.path.join(reference_path, ref_rgb_file)
            ref_depth_path = os.path.join(reference_path, ref_rgb_file.replace('_0.png', '_1.png'))
            if not os.path.exists(ref_rgb_path):
                continue
            ref_depth_image = cv2.imread(ref_depth_path, cv2.IMREAD_UNCHANGED)
            ref_rgb_image = cv2.imread(ref_rgb_path)
            if ref_depth_image is None or ref_rgb_image is None:
                continue
            if ref_depth_image.mean() < 1 or not np.any(ref_depth_image < 65000):
                continue
            if ref_rgb_file not in camera_dict:
                continue
            K_ref = camera_dict[ref_rgb_file]
            cam_ref = {'model': 'PINHOLE', 'width': K_ref[0], 'height': K_ref[1],
                       'params': K_ref[2:6]}
            qcamera = Camera.from_colmap(cam_query)
            rcamera = Camera.from_colmap(cam_ref)

            points2d_ref = sample_points_with_valid_depth(ref_depth_image, num_points=10000, max_depth=2000)
            points2d_ref_valid, point3D_from_ref, _, _ = get_3D_samples(
                points2d_ref, ref_depth_image, ref_T, rcamera)
            points2d_query, _, Points_3D_ECEF_origin, query_depth_proj = get_points2D_ECEF_projection(
                np.array(query_T), qcamera, point3D_from_ref, points2d_ref_valid,
                use_valid=False, num_samples=20000)

            h, w = query_depth_image.shape[:2]
            valid = ((points2d_query[:, 0] >= 0) & (points2d_query[:, 0] < w) &
                     (points2d_query[:, 1] >= 0) & (points2d_query[:, 1] < h))
            true_indices = np.where(valid)[0]
            false_indices = np.where(~valid)[0]
            pts_q, point3D_from_query, _, query_valid_indices = get_3D_samples(
                points2d_query[true_indices], query_depth_image, query_T, qcamera)
            points2d_ref_rej, _, _, ref_depth_proj = get_points2D_ECEF_projection(
                np.array(ref_T), rcamera, point3D_from_query, pts_q, use_valid=False)
            points2d_ref_sample = points2d_ref_valid[true_indices][query_valid_indices]
            if points2d_ref_sample.shape != points2d_ref_rej.shape:
                continue

            coords = np.floor(points2d_query[true_indices][query_valid_indices]).astype(int)
            qd = query_depth_image[coords[:, 1], coords[:, 0]]
            qdp = query_depth_proj[true_indices][query_valid_indices]
            rc = np.floor(points2d_ref_sample.cpu().numpy()).astype(int)
            rd = ref_depth_image[rc[:, 1], rc[:, 0]]
            eps = 1e-6
            depth_ok = ((np.abs(qd - qdp) / np.maximum(qd, eps) <= 0.001) &
                        (np.abs(rd - ref_depth_proj) / np.maximum(rd, eps) <= 0.001))
            pix_ok = np.linalg.norm(points2d_ref_sample.cpu().numpy() - points2d_ref_rej, axis=1) <= 1
            valid_indices = np.intersect1d(np.where(pix_ok)[0], np.where(depth_ok)[0])
            true_valid = true_indices[valid_indices]
            if len(true_valid) < min_valid_threshold:
                continue
            if len(false_indices) > len(true_valid) * 0.3:
                false_indices = false_indices[:int(len(true_valid) * 0.3)]
            final_indices = np.array(false_indices.tolist() + true_valid.tolist())
            if len(false_indices) / len(final_indices) >= 0.3:
                continue
            success_find = True

        if not success_find:
            continue

        refer_info[pose_name] = {
            "img_pose": query_T, "img_pose_6": query_pose,
            "img_path": os.path.relpath(query_rgb_path, root),
            "img_intrisic": K_query,
            "img_depth": os.path.relpath(query_depth_path, root),
            "origin": origin_save.tolist(),
            "ref_info": {
                "ref_name": ref_rgb_file,
                "ref_rgb": os.path.relpath(os.path.join(reference_path, ref_rgb_file), root),
                "ref_depth": os.path.relpath(
                    os.path.join(reference_path, ref_rgb_file.replace('_0.png', '_1.png')), root),
                "ref_poses": ref_T, "ref_intrinsics": K_ref,
            },
        }
        step = len(final_indices) / min_valid_threshold
        idx = [final_indices[int(i * step)] for i in range(min_valid_threshold)]
        np.save(os.path.join(Points3D_path, pose_name.split('.')[0] + '.npy'),
                Points_3D_ECEF_origin[idx])
        success_num += 1

    out = os.path.join(root, seq, 'refer_info.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(refer_info, f, indent=4)
    print(f"saved {success_num} pairs -> {out}")


def load_poses(pose_file):
    tl, el, nl = [], [], []
    with open(pose_file, 'r', encoding='utf-8') as f:
        for line in f:
            p = line.strip().split()
            if p:
                lon, lat, alt, roll, pitch, yaw = map(float, p[1:])
                tl.append([lon, lat, alt])
                el.append([pitch, roll, yaw])
                nl.append(p[0])
    ta, ea = np.array(tl), np.array(el)
    T = get_matrix_list_batch(ta, ea)
    return {nl[i]: {'euler_angles': ea[i].tolist(), 'translation': ta[i].tolist(), 'T_c2w': T[i]}
            for i in range(len(nl))}


def load_camera(camera_file):
    d = {}
    with open(camera_file, 'r', encoding='utf-8') as f:
        for line in f:
            p = line.strip().split()
            if p:
                d[p[0]] = list(map(float, p[1:]))
    return d


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', nargs='+', default=['England_seq1@200@30_50'])
    ap.add_argument('--root', default='/mnt/data1/UserData/liuxy24/Mapscape/crop1200@1200/Train-test')
    ap.add_argument('--ref-offset', type=int, default=200)
    ap.add_argument('--max-queries', type=int, default=None)
    ap.add_argument('--min-valid-threshold', type=int, default=800)
    args = ap.parse_args()
    for seq in os.listdir(args.root):
        if seq not in args.names and seq.split('.')[0] not in args.names:
            continue
        sd = os.path.join(args.root, seq)
        if not os.path.isdir(sd):
            continue
        qdirs = [f for f in os.listdir(sd) if 'query@' in f]
        if not qdirs:
            continue
        generate_refer_info(
            args.root, os.path.join(sd, 'Points3D'), os.path.join(sd, 'ref'),
            os.path.join(sd, qdirs[0]), seq,
            load_poses(os.path.join(sd, seq + '.txt')),
            load_camera(os.path.join(sd, 'camera.txt')),
            ref_offset=args.ref_offset, max_queries=args.max_queries,
            min_valid_threshold=args.min_valid_threshold)
