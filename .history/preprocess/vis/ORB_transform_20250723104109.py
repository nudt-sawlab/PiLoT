import numpy as np
from transform import WGS84_to_ECEF, ECEF_to_WGS84, wgs84tocgcs2000
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
                pose_dict[fname] = WGS84_to_ECEF(xyz)
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
gt_pts_cgcs = wgs84tocgcs2000(gt_pts, 4547)
orb_pts_cgcs = wgs84tocgcs2000(orb_pts, 4547)

# 计算对齐变换
scale, R, t = umeyama_alignment(orb_pts_cgcs, gt_pts_cgcs)

# 对 ORB 所有点变换
all_orb_names =  sorted(orb_dict.keys(), key=lambda x: int(x.split('_')[0]))

all_orb_pts = np.array([orb_dict[f] for f in all_orb_names])

all_orb_pts_cgcs = wgs84tocgcs2000(all_orb_pts, 4547)



orb_aligned = transform_points(all_orb_pts_cgcs, scale, R, t)

all_orb_xyz = np.array([all_orb_dict[k] for k in all_orb_names])

aligned_xyz = transform_points(all_orb_xyz, scale, R, t)

# 保存为对齐后的 Render2ORB_aligned.txt
with open("/mnt/sda/MapScape/query/estimation/result_images/ORB@per30/USA_seq5@8@cloudy@300-100@200.txt", 'w') as f:
    for name, xyz, orb_euler in zip(all_orb_names, aligned_xyz, orb_euler_dict):
        xyz = ECEF_to_WGS84(xyz)
        f.write(f"{name} {xyz[0]} {xyz[1]} {xyz[2]} {orb_euler[0]} {orb_euler[1]} {orb_euler[2]}\n")
