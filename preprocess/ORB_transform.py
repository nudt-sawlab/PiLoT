import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pyproj import CRS, Transformer
import yaml
from transform import WGS84_to_ECEF, ECEF_to_WGS84, wgs84tocgcs2000_batch, cgcs2000towgs84
def get_utm_epsg_from_lonlat(lon, lat):
    """
    根据经纬度 (lon, lat) 计算其对应的 UTM 分带 EPSG 号。
    - 北半球：EPSG:326XX
    - 南半球：EPSG:327XX

    :param lon: 经度 (float)，通常范围 [-180, 180]
    :param lat: 纬度 (float)，通常范围 [-90, 90]
    :return: 整型 EPSG，若无法计算则返回 None
    """
    if lon < -180 or lon > 180 or lat < -90 or lat > 90:
        return None
    zone = int(math.floor((lon + 180) / 6)) + 1
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone

def wgs84_to_utm(lon, lat, alt, epsg):
    """
    将 WGS84 坐标 (lon, lat, alt) 转换到对应 UTM 坐标系 (x, y, z)。
    这里 alt 保持不变（不做大地高与正高转换）。
    """
    crs_wgs84 = CRS.from_epsg(4326)
    crs_utm   = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return [x, y, alt]

def utm_to_wgs84(x, y, alt, epsg):
    """
    将 UTM 坐标 (x, y, z) 转换回 WGS84 坐标 (lon, lat, alt)。
    """
    crs_utm   = CRS.from_epsg(epsg)
    crs_wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_utm, crs_wgs84, always_xy=True)
    lon, lat = transformer.transform(x, y)
    return [lon, lat, alt]
def load_pose_dict(file_path):
    """加载轨迹为dict: {frame_name: [x, y, z]}"""
    pose_dict = {}
    pose_dict_euler = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                fname = parts[0]
                xyz = list(map(float, parts[1:4]))
                if fname in pose_dict.keys(): 
                    continue
                pose_dict[fname] = xyz
                pose_dict_euler.append(list(map(float, parts[4:])))
    return pose_dict, pose_dict_euler

def extract_matched_points(ref_dict, orb_dict):
    """提取共同帧的参考坐标和 ORB 坐标"""
    matched_frames = sorted(set(ref_dict.keys()) & set(orb_dict.keys()))
    ref_pts = np.array([ref_dict[f] for f in matched_frames])
    orb_pts = np.array([orb_dict[f] for f in matched_frames])
    return ref_pts, orb_pts, matched_frames

def umeyama_alignment(src, dst):
    """从 src 到 dst 的相似变换：scale * R @ src + t = dst"""
    assert src.shape == dst.shape
    mu_src = src.mean(0)
    mu_dst = dst.mean(0)

    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    cov = dst_centered.T @ src_centered / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)

    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    var_src = (src_centered ** 2).sum() / src.shape[0]
    scale = np.trace(np.diag(D)) / var_src
    t = mu_dst - scale * R @ mu_src

    return scale, R, t

def transform_points(points, scale, R, t):
    return scale * (R @ points.T).T + t

ref_dict, ref_euler_dict = load_pose_dict("/mnt/sda/MapScape/query/estimation/result_images/GT/USA_seq5@8@cloudy@300-100@200.txt")
orb_dict, orb_euler_dict = load_pose_dict("/mnt/sda/MapScape/query/estimation/result_images/ORB@per30/USA_seq5@8@cloudy@300-100@200_1.txt")

ref_pts, orb_pts, matched_frames = extract_matched_points(ref_dict, orb_dict)
scale, R, t = umeyama_alignment(orb_pts, ref_pts)


# 提取两者都有的帧
matched_frames = sorted(set(ref_dict.keys()) & set(orb_dict.keys()))
gt_pts = np.array([ref_dict[f] for f in matched_frames])
orb_pts = np.array([orb_dict[f] for f in matched_frames])

# 变换所有 ORB 轨迹（包括未参与匹配的）
# 经纬度 → CGCS2000
gt_pts_cgcs = wgs84tocgcs2000_batch(gt_pts, 4547)
orb_pts_cgcs = wgs84tocgcs2000_batch(orb_pts, 4547)
# epsg = get_utm_epsg_from_lonlat(ABS_INIT[0], ABS_INIT[1])
# abs_utm_init = wgs84_to_utm(ABS_INIT[0], ABS_INIT[1], HEIGHT_INIT, epsg)

# 计算对齐变换
scale, R, t = umeyama_alignment(orb_pts_cgcs, gt_pts_cgcs)

# 对 ORB 所有点变换
all_orb_names =  sorted(orb_dict.keys(), key=lambda x: int(x.split('_')[0]))

all_orb_pts = np.array([orb_dict[f] for f in all_orb_names])

all_orb_pts_cgcs = wgs84tocgcs2000_batch(all_orb_pts, 4547)



aligned_xyz = transform_points(all_orb_pts_cgcs, scale, R, t)



# 保存为对齐后的 Render2ORB_aligned.txt
with open("/mnt/sda/MapScape/query/estimation/result_images/ORB@per30/USA_seq5@8@cloudy@300-100@200.txt", 'w') as f:
    for name, xyz, orb_euler in zip(all_orb_names, aligned_xyz, orb_euler_dict):
        xyz = cgcs2000towgs84([xyz], 0)
        f.write(f"{name} {xyz[0]} {xyz[1]} {xyz[2]} {orb_euler[0]} {orb_euler[1]} {orb_euler[2]}\n")
