import numpy as np
import cv2
import pyproj
# ---------- 反投影：像素+深度 -> 世界点（你当前世界系/ECEF 都行） ----------
def backproject_depth_to_world(depth, T_c2w, cam, step=4):
    """
    depth: HxW, meters
    T_c2w: 4x4 float64
    cam: [w,h,fx,fy,cx,cy]
    返回: (N,3) 世界点, 以及采样栅格(xy, valid掩码)用于回写mask
    """
    w, h, fx, fy, cx, cy = cam
    H, W = depth.shape[:2]
    K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]], dtype=np.float64)
    Kinv = np.linalg.inv(K)

    xs = np.arange(0, W, step)
    ys = np.arange(0, H, step)
    xx, yy = np.meshgrid(xs, ys)  # 小栅格
    d = depth[yy, xx].astype(np.float64)
    valid = (d > 0) & np.isfinite(d)
    if not np.any(valid):
        return np.empty((0,3), dtype=np.float64), (xx, yy, valid)

    u = xx[valid].ravel().astype(np.float64)
    v = yy[valid].ravel().astype(np.float64)
    z = d[valid].ravel()
    pix = np.stack([u, v, np.ones_like(u)], axis=0)  # 3xN
    rays = Kinv @ pix                                 # 3xN
    Xc = rays * z                                     # 3xN

    Rcw = T_c2w[:3,:3].astype(np.float64)
    tcw = T_c2w[:3,3:4].astype(np.float64)
    Xw = (Rcw @ Xc) + tcw                             # 3xN
    return Xw.T, (xx, yy, valid)

# ---------- 局部系：原点=两相机中心中点，轴=参考帧相机轴 ----------
def world_to_local(points_world, ref_T, qry_T):
    if points_world.size == 0:
        return points_world
    c1 = ref_T[:3,3]; c2 = qry_T[:3,3]
    o = 0.5*(c1 + c2)                  # 局部原点
    R1 = ref_T[:3,:3]                  # 局部轴（参考帧）
    d = points_world - o[None,:]
    Xl = (R1.T @ d.T).T                # 到局部
    return Xl, R1, o

# ---------- 体素索引 & 线性编码（向量化，快） ----------
def voxel_indices(points_local, voxel_size):
    if points_local.size == 0:
        return np.empty((0,3), dtype=np.int64)
    vs = float(voxel_size)
    ijk = np.floor(points_local / vs).astype(np.int64)  # (N,3)
    return ijk

def linearize_ijk(ijk_a, ijk_b):
    """
    统一两组体素索引的线性编码（共享最小/跨度），便于 np.intersect1d
    返回: codes_a, codes_b, mins, spans
    """
    if ijk_a.size == 0 and ijk_b.size == 0:
        return np.empty((0,), np.int64), np.empty((0,), np.int64), None, None
    if ijk_a.size == 0:
        mins = ijk_b.min(axis=0)
        spans = ijk_b.max(axis=0) - mins + 1
    elif ijk_b.size == 0:
        mins = ijk_a.min(axis=0)
        spans = ijk_a.max(axis=0) - mins + 1
    else:
        mins = np.minimum(ijk_a.min(axis=0), ijk_b.min(axis=0))
        maxs = np.maximum(ijk_a.max(axis=0), ijk_b.max(axis=0))
        spans = (maxs - mins + 1)
    spans = np.maximum(spans, 1)
    # 防止乘法溢出：用 int64，且场景尺度/voxel_size 合理即可
    ia = (ijk_a - mins[None,:]) if ijk_a.size>0 else np.empty_like(ijk_a)
    ib = (ijk_b - mins[None,:]) if ijk_b.size>0 else np.empty_like(ijk_b)
    # 线性化: (i*Sy + j)*Sz + k
    Sy, Sz = spans[1], spans[2]
    codes_a = (ia[:,0]*Sy + ia[:,1])*Sz + ia[:,2] if ia.size>0 else np.empty((0,), np.int64)
    codes_b = (ib[:,0]*Sy + ib[:,1])*Sz + ib[:,2] if ib.size>0 else np.empty((0,), np.int64)
    return codes_a.astype(np.int64), codes_b.astype(np.int64), mins, spans

def fast_iou_codes(codes_a, codes_b):
    if codes_a.size == 0 and codes_b.size == 0:
        return 1.0, np.empty((0,), np.int64)
    if codes_a.size == 0 or codes_b.size == 0:
        return 0.0, np.empty((0,), np.int64)
    ua = np.unique(codes_a)
    ub = np.unique(codes_b)
    inter = np.intersect1d(ua, ub, assume_unique=False)
    union = np.union1d(ua, ub)
    iou = float(len(inter)) / float(len(union)) if len(union)>0 else 0.0
    return iou, inter

# ---------- 可视化：交集体素 -> 像素 mask ----------
def overlap_mask_on_rgb(rgb, depth, T_c2w, cam, R1, o, inter_codes, mins, spans,
                        voxel_size=1.0, step=2, color=(0,255,0)):
    H, W = rgb.shape[:2]
    w, h, fx, fy, cx, cy = cam
    assert (W, H) == (w, h), "RGB 尺寸需与相机参数一致"

    pts_w, (xx, yy, valid) = backproject_depth_to_world(depth, T_c2w, cam, step=step)
    if pts_w.size == 0 or inter_codes.size == 0:
        return np.zeros((H,W), np.uint8), rgb.copy()

    # 到局部
    d = pts_w - o[None,:]
    Xl = (R1.T @ d.T).T
    # 当前像素的体素编码
    vs = float(voxel_size)
    ijk = np.floor(Xl / vs).astype(np.int64)
    ia = ijk - mins[None,:]
    Sy, Sz = spans[1], spans[2]
    codes = (ia[:,0]*Sy + ia[:,1])*Sz + ia[:,2]

    # 命中交集体素？
    inter_sorted = np.sort(inter_codes)
    hits = np.in1d(codes, inter_sorted, assume_unique=False)

    # 小栅格 -> 全图
    mask_small = np.zeros_like(xx, dtype=np.uint8)
    mask_small.ravel()[valid.ravel()] = hits.astype(np.uint8)
    mask = cv2.resize(mask_small*255, (W, H), interpolation=cv2.INTER_NEAREST)

    overlay = rgb.copy()
    overlay[mask>0] = (0.6*overlay[mask>0] + 0.4*np.array(color, np.uint8)).astype(np.uint8)
    vis = cv2.addWeighted(rgb, 0.7, overlay, 0.3, 0)
    edges = cv2.Canny(mask, 50, 150)
    vis[edges>0] = (0,255,255)
    return mask, vis

# ---------- 主流程：3D 体素 IoU + 可视化 ----------
def visualize_overlap_3d_voxel(
    ref_rgb, ref_depth, ref_T, rcamera,
    qry_rgb, qry_depth, qry_T, qcamera,
    voxel_size=1.0, step_fov=4, step_mask=2
):
    # 1) 反投影两帧点云（同一世界系）
    P1_w, _ = backproject_depth_to_world(ref_depth, ref_T, rcamera, step=step_fov)
    P2_w, _ = backproject_depth_to_world(qry_depth, qry_T, qcamera, step=step_fov)

    # 2) 转局部 & 体素化
    P1_l, R1, o = world_to_local(P1_w, ref_T, qry_T)
    P2_l, _, _  = world_to_local(P2_w, ref_T, qry_T)
    ijk1 = voxel_indices(P1_l, voxel_size)
    ijk2 = voxel_indices(P2_l, voxel_size)

    # 3) 统一编码并算 IoU
    codes1, codes2, mins, spans = linearize_ijk(ijk1, ijk2)
    iou, inter_codes = fast_iou_codes(codes1, codes2)

    # 4) 可视化交集体素在两张图上的投影
    mask_ref, vis_ref = overlap_mask_on_rgb(ref_rgb, ref_depth, ref_T, rcamera,
                                            R1, o, inter_codes, mins, spans,
                                            voxel_size, step=step_mask, color=(0,255,0))
    mask_qry, vis_qry = overlap_mask_on_rgb(qry_rgb, qry_depth, qry_T, qcamera,
                                            R1, o, inter_codes, mins, spans,
                                            voxel_size, step=step_mask, color=(255,0,0))
    return iou, (mask_ref, vis_ref), (mask_qry, vis_qry)

def ECEF_to_WGS84(pos):
    x, y, z = pos
    trans = pyproj.Transformer.from_crs(
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        "EPSG:4326",
        always_xy=True,
    )
    lon, lat, h = trans.transform(x, y, z, radians=False)
    return np.array([lon, lat, h], dtype=np.float64)
def WGS84_to_ECEF(pos):
    lon, lat, h = pos
    trans = pyproj.Transformer.from_crs(
        "EPSG:4326",
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        always_xy=True,
    )
    x, y, z = trans.transform(lon, lat, h, radians=False)
    return np.array([x, y, z], dtype=np.float64)
qcamera = [960, 540, 1158.8, 1158.8, 480.0, 270.0]
rcamera = [960, 540, 1158.8, 1158.8, 480.0, 270.0]
reference_rgb_path = "/mnt/sda/MapScape/query/depth/USA_seq5@8@cloudy@300-100@200/1_0.png"
reference_depth_path = "/mnt/sda/MapScape/query/depth/USA_seq5@8@cloudy@300-100@200/1_1.png"
query_rgb_path = "/mnt/sda/MapScape/query/depth/USA_seq5@8@cloudy@300-100@200/10_0.png"
query_depth_path = "/mnt/sda/MapScape/query/depth/USA_seq5@8@cloudy@300-100@200/10_1.png"

ref_depth_image = cv2.imread(reference_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
qry_depth_image = cv2.imread(query_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
# 例：构造 4x4 位姿（这里用你的 query_T / ref_T）
pose_txt = "/media/ubuntu/PS2000/poses/USA_seq5@8@cloudy@300-100@200.txt"
vis_save_path = "/mnt/sda/ycb/"
rgb_image = cv2.imread(query_rgb_path)
ref_image = cv2.imread(reference_rgb_path)
pose_dict = load_poses(pose_txt)
# query_T/ref_T 若为 list，请转 np.array 并确保为 float64

ref_pose_name = reference_depth_path.split('/')[-1].split('_')[0] +'_0.png'
query_pose_name = query_depth_path.split('/')[-1].split('_')[0] +'_0.png'
ref_pose = pose_dict[ref_pose_name]
query_pose = pose_dict[query_pose_name]

# get query pose
lon, lat, alt, roll, pitch, yaw = map(float, query_pose)
euler_angles = [pitch, roll, yaw]
translation = [lon, lat, alt]
rot_pose_in_ned = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
t_c2w = WGS84_to_ECEF(translation)
query_T = np.eye(4)

query_T[:3, :3] = R_c2w
query_T[:3, 3] = t_c2w
query_T[:3, 1] = -query_T[:3, 1]  # Y轴取反，投影后二维原点在左上角
query_T[:3, 2] = -query_T[:3, 2]  # Z轴取反


lon, lat, alt, roll, pitch, yaw = map(float, ref_pose)
euler_angles_ref = [pitch, roll, yaw]
translation_ref = [lon, lat, alt]
lon, lat, _ = translation_ref
rot_pose_in_ned = R.from_euler('xyz', euler_angles_ref, degrees=True).as_matrix()  # ZXY 东北天  
rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
t_c2w = WGS84_to_ECEF(translation_ref)
ref_T = np.eye(4)
ref_T[:3, :3] = R_c2w
ref_T[:3, 3] = t_c2w
ref_T[:3, 1] = -ref_T[:3, 1]  # Y轴取反，投影后二维原点在左上角
ref_T[:3, 2] = -ref_T[:3, 2]  # Z轴取反
# query_T/ref_T 若为 list，请转 np.array 并确保为 float64


ref_pose_name = reference_depth_path.split('/')[-1].split('_')[0] +'_0.png'
query_pose_name = query_depth_path.split('/')[-1].split('_')[0] +'_0.png'
ref_pose = pose_dict[ref_pose_name]
query_pose = pose_dict[query_pose_name]
iou3d, (mask_ref, vis_ref), (mask_qry, vis_qry) = visualize_overlap_3d_voxel(
    ref_image, ref_depth_image, ref_T, rcamera,
    rgb_image, qry_depth_image, query_T, qcamera,
    voxel_size=1.0,   # 体素边长(米)，0.5~2.0 视你的深度精度
    step_fov=4,       # 体素IoU的点云采样步长(越小越准/慢)
    step_mask=2       # 可视化时的像素采样步长
)
print("3D Voxel IoU =", iou3d)
cv2.imwrite("ref_overlap.png", vis_ref)
cv2.imwrite("qry_overlap.png", vis_qry)
