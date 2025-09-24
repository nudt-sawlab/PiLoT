import numpy as np
import cv2
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
# ==== 可选：若有 shapely，精确几何运算 ====
try:
    from shapely.geometry import Polygon, Point
    _HAS_SHAPELY = True
except Exception:
    from matplotlib.path import Path
    _HAS_SHAPELY = False
import pyproj
# ---------- 基础几何：ECEF<->ENU（纯 numpy，稳定） ----------
def _ecef_origin_from_llh(lon0, lat0, h0):
    # WGS84 椭球
    a = 6378137.0
    f = 1/298.257223563
    e2 = f*(2-f)
    lon0r = np.deg2rad(lon0); lat0r = np.deg2rad(lat0)
    sin_lat, cos_lat = np.sin(lat0r), np.cos(lat0r)
    sin_lon, cos_lon = np.sin(lon0r), np.cos(lon0r)
    N = a / np.sqrt(1 - e2*sin_lat*sin_lat)
    x0 = (N + h0) * cos_lat * cos_lon
    y0 = (N + h0) * cos_lat * sin_lon
    z0 = (N*(1 - e2) + h0) * sin_lat
    return np.array([x0, y0, z0], dtype=np.float64), (sin_lat, cos_lat, sin_lon, cos_lon)
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

def ecef_to_enu_xy(points_ecef, lon0, lat0, h0):
    if points_ecef.size == 0:
        return np.empty((0,2), dtype=np.float64)
    points_ecef = np.asarray(points_ecef, dtype=np.float64)
    mask = np.all(np.isfinite(points_ecef), axis=1)
    points_ecef = points_ecef[mask]
    p0, (sphi, cphi, slam, clam) = _ecef_origin_from_llh(lon0, lat0, h0)
    # ECEF->ENU 旋转
    R = np.array([
        [-slam,           clam,           0.0],
        [-sphi*clam,     -sphi*slam,      cphi],
        [ cphi*clam,      cphi*slam,      sphi]
    ], dtype=np.float64)
    d = points_ecef - p0[None, :]
    enu = d @ R.T
    return enu[:, :2]  # Nx2

# ---------- 相机反投影到 ECEF ----------
def backproject_depth_to_ecef(depth, T_c2w, cam, step=8):
    """
    depth: HxW (meters); cam=[w,h,fx,fy,cx,cy]; T_c2w: 4x4 ECEF
    返回稀疏 ECEF 点云 (N,3)
    """
    w, h, fx, fy, cx, cy = cam
    H, W = depth.shape[:2]
    K = np.array([[fx, 0, cx],[0, fy, cy],[0,0,1]], dtype=np.float64)
    Kinv = np.linalg.inv(K)
    xs = np.arange(0, W, step)
    ys = np.arange(0, H, step)
    xx, yy = np.meshgrid(xs, ys)
    d = depth[yy, xx].astype(np.float64)
    valid = (d > 0) & np.isfinite(d)
    if not np.any(valid):
        return np.empty((0,3), dtype=np.float64)
    u = xx[valid].ravel().astype(np.float64)
    v = yy[valid].ravel().astype(np.float64)
    d = d[valid].ravel()
    pix = np.stack([u, v, np.ones_like(u)], axis=0)   # 3xN
    rays = Kinv @ pix                                 # 3xN
    Xc = rays * d                                     # 3xN
    Rcw = T_c2w[:3,:3].astype(np.float64)
    tcw = T_c2w[:3,3:4].astype(np.float64)
    Xw = (Rcw @ Xc) + tcw
    return Xw.T  # Nx3

# ---------- 凸包 & 交集 ----------
def convex_hull_xy(xy):
    if xy.shape[0] < 3:
        return None
    try:
        hull = ConvexHull(xy)
        return xy[hull.vertices]
    except Exception:
        return None
def get_rotation_enu_in_ecef(lon, lat):
    """
    @param: lon, lat Longitude and latitude in degree
    @return: 3x3 rotation matrix of heading-pith-roll ENU in ECEF coordinate system
    Reference: https://apps.dtic.mil/dtic/tr/fulltext/u2/a484864.pdf, Section 4.3, 4.1
    Reference: https://www.fossen.biz/wiley/ed2/Ch2.pdf, p29
    """
    # 将角度转换为弧度
    latitude_rad = np.radians(lat)
    longitude_rad = np.radians(lon)
    
    # 计算向上的向量（Up Vector）
    up = np.array([
        np.cos(longitude_rad) * np.cos(latitude_rad),
        np.sin(longitude_rad) * np.cos(latitude_rad),
        np.sin(latitude_rad)
    ])
    
    # 计算向东的向量（East Vector）
    east = np.array([
        -np.sin(longitude_rad),
        np.cos(longitude_rad),
        0
    ])
    
    # 计算向北的向量（North Vector），即up向量和east向量的外积（叉积）
    north = np.cross(up, east)
    
    # 构建局部到世界坐标系的转换矩阵
    local_to_world = np.zeros((3, 3))
    local_to_world[:, 0] = east  # 东向分量
    local_to_world[:, 1] = north  # 北向分量
    local_to_world[:, 2] = up  # 上向分量
    return local_to_world
def polygon_intersection_xy(polyA, polyB):
    if polyA is None or polyB is None:
        return None
    if _HAS_SHAPELY:
        PA = Polygon(polyA); PB = Polygon(polyB)
        if not PA.is_valid or not PB.is_valid or PA.area <= 0 or PB.area <= 0:
            return None
        inter = PA.intersection(PB)
        if inter.is_empty:
            return None
        # 统一为 Nx2 顶点序列（取外边界）
        try:
            coords = np.array(inter.exterior.coords, dtype=np.float64)
            return coords[:, :2]
        except Exception:
            return None
    else:
        # 简易近似：将两凸包合并采样后，保留落在对方多边形内的点，再取凸包
        from matplotlib.path import Path
        pathA, pathB = Path(polyA), Path(polyB)
        pts = np.vstack([polyA, polyB])
        mask = pathA.contains_points(pts) | pathB.contains_points(pts)
        pts_in = pts[mask]
        if pts_in.shape[0] < 3:
            return None
        return convex_hull_xy(pts_in)

# ---------- 在每帧图像上着色重叠区域 ----------
def rasterize_overlap_mask(rgb, depth, T_c2w, cam, lon0, lat0, h0, poly_inter_xy, step=4):
    """
    返回与 rgb 同尺寸的 uint8 二值 mask（重叠区域=255），以及着色后的可视化图。
    """
    H, W = rgb.shape[:2]
    w, h, fx, fy, cx, cy = cam
    assert W == w and H == h, "RGB尺寸需与相机参数一致"

    if poly_inter_xy is None:
        mask = np.zeros((H, W), dtype=np.uint8)
        vis  = rgb.copy()
        return mask, vis

    # 稀疏采样像素 -> ENU XY -> 点在多边形内？
    xs = np.arange(0, W, step)
    ys = np.arange(0, H, step)
    xx, yy = np.meshgrid(xs, ys)
    ss = np.stack([xx.ravel(), yy.ravel()], axis=1)  # [N,2]

    # 反投影到 ECEF
    K = np.array([[fx, 0, cx],[0, fy, cy],[0,0,1]], dtype=np.float64)
    Kinv = np.linalg.inv(K)
    d = depth[yy, xx].astype(np.float64).ravel()
    valid = (d > 0) & np.isfinite(d)
    if not np.any(valid):
        mask = np.zeros((H, W), dtype=np.uint8)
        return mask, rgb.copy()
    u = ss[valid, 0].astype(np.float64)
    v = ss[valid, 1].astype(np.float64)
    pix = np.stack([u, v, np.ones_like(u)], axis=0)     # 3xN
    rays = Kinv @ pix
    Xc = rays * d[valid]                                # 3xN
    Rcw = T_c2w[:3,:3].astype(np.float64)
    tcw = T_c2w[:3,3:4].astype(np.float64)
    Xw = (Rcw @ Xc) + tcw                                # 3xN
    Xw = Xw.T                                           # Nx3

    # ECEF -> ENU xy
    XY = ecef_to_enu_xy(Xw, lon0, lat0, h0)             # Nx2

    # 点-多边形测试
    if _HAS_SHAPELY:
        P = Polygon(poly_inter_xy)
        inside = np.array([P.contains(Point(x,y)) for x,y in XY], dtype=bool)
    else:
        from matplotlib.path import Path
        inside = Path(poly_inter_xy).contains_points(XY)

    # 生成掩膜（把稀疏栅格还原到图像）
    mask_small = np.zeros_like(xx, dtype=np.uint8)
    mask_small.ravel()[valid] = inside.astype(np.uint8)
    mask = cv2.resize(mask_small*255, (W, H), interpolation=cv2.INTER_NEAREST)

    # 可视化叠加
    color = np.array([0, 255, 0], dtype=np.uint8)  # 绿色
    overlay = rgb.copy()
    overlay[mask > 0] = (0.6*overlay[mask>0] + 0.4*color).astype(np.uint8)
    vis = cv2.addWeighted(rgb, 0.7, overlay, 0.3, 0)

    # 画交集多边形（可选，投到 ENU 后无法直接画到像素，这里仅在 mask 边缘勾勒）
    edges = cv2.Canny(mask, 50, 150)
    vis[edges>0] = (0, 255, 255)  # 黄色描边
    return mask, vis

# ---------- 主流程：计算交集并可视化 ----------
def visualize_geo_overlap_on_images(
    ref_rgb, ref_depth, ref_T, rcamera,
    qry_rgb, qry_depth, qry_T, qcamera,
    step_fov=8, step_mask=4
):
    # 1) 用深度反投影 -> ENU XY 凸包
    # ENU 原点：两相机中心中点
    c1 = np.array(ref_T, dtype=np.float64)[:3,3]
    c2 = np.array(qry_T, dtype=np.float64)[:3,3]
    c_mid_ecef = 0.5*(c1+c2)

    # 将中点 ECEF 粗转回经纬高（小范围内 h0 用两相机高的均值也可）
    # 这里简单地反推：用上面的原点求法需要 lon0/lat0/h0，
    # 你已有 ECEF->LLH 的函数可替换；这里给个稳定近似：
    # ——建议直接传入 (lon0,lat0,h0) 以保证一致性。
    # 若已有函数：from your_module import ECEF_to_WGS84
    lon0, lat0, h0 = ECEF_to_WGS84(c_mid_ecef)

    # 采样反投影
    P1_ecef = backproject_depth_to_ecef(ref_depth, np.array(ref_T, dtype=np.float64), rcamera, step=step_fov)
    P2_ecef = backproject_depth_to_ecef(qry_depth, np.array(qry_T, dtype=np.float64), qcamera, step=step_fov)
    P1_xy = ecef_to_enu_xy(P1_ecef, lon0, lat0, h0)
    P2_xy = ecef_to_enu_xy(P2_ecef, lon0, lat0, h0)

    poly1 = convex_hull_xy(P1_xy)
    poly2 = convex_hull_xy(P2_xy)
    poly_inter = polygon_intersection_xy(poly1, poly2)

    # 2) 在两张图上各自渲染重叠区域
    mask_ref, vis_ref = rasterize_overlap_mask(ref_rgb, ref_depth, np.array(ref_T, dtype=np.float64),
                                               rcamera, lon0, lat0, h0, poly_inter, step=step_mask)
    mask_qry, vis_qry = rasterize_overlap_mask(qry_rgb, qry_depth, np.array(qry_T, dtype=np.float64),
                                               qcamera, lon0, lat0, h0, poly_inter, step=step_mask)
    return (mask_ref, vis_ref), (mask_qry, vis_qry), poly_inter
def load_poses(pose_file):
    """Load poses from the pose file."""
    pose_dict = {}
    with open(pose_file, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                pose_dict[parts[0]] = parts[1: ]  # Add the first element to the name list
    return pose_dict
# ===== 使用示例 =====
# (确保 ref_rgb.shape[:2] == (h,w) 与 rcamera 的 h,w 一致；深度单位米；T 为 float64)
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
(mask_ref, vis_ref), (mask_qry, vis_qry), poly_xy = visualize_geo_overlap_on_images(
    ref_image, ref_depth_image, ref_T, rcamera,
    rgb_image, qry_depth_image, query_T, qcamera,
    step_fov=8, step_mask=4
)
cv2.imwrite("ref_overlap.png", vis_ref)
cv2.imwrite("qry_overlap.png", vis_qry)
