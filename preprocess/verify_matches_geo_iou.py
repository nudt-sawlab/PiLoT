import numpy as np
import cv2
import pyproj
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False
import pyproj
import copy
from get_depth  import read_valid_depth
from transform import visualize_matches
from scipy.spatial.transform import Rotation as R
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
# ---------------------------- 工具函数 ----------------------------
def WGS84_to_ECEF(pos):
    lon, lat, h = pos
    trans = pyproj.Transformer.from_crs(
        "EPSG:4326",
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        always_xy=True,
    )
    x, y, z = trans.transform(lon, lat, h, radians=False)
    return np.array([x, y, z], dtype=np.float64)

def ECEF_to_WGS84(pos):
    x, y, z = pos
    trans = pyproj.Transformer.from_crs(
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        "EPSG:4326",
        always_xy=True,
    )
    lon, lat, h = trans.transform(x, y, z, radians=False)
    return np.array([lon, lat, h], dtype=np.float64)

def _ecef_to_enu_numpy(points_xyz, lon0, lat0, h0):
    # WGS84
    a = 6378137.0
    f = 1/298.257223563
    e2 = f*(2-f)

    lon0r = np.deg2rad(lon0)
    lat0r = np.deg2rad(lat0)
    sin_lat, cos_lat = np.sin(lat0r), np.cos(lat0r)
    sin_lon, cos_lon = np.sin(lon0r), np.cos(lon0r)

    N = a / np.sqrt(1 - e2 * sin_lat*sin_lat)
    x0 = (N + h0) * cos_lat * cos_lon
    y0 = (N + h0) * cos_lat * sin_lon
    z0 = (N*(1 - e2) + h0) * sin_lat
    p0 = np.array([x0, y0, z0], dtype=np.float64)

    R_ecef2enu = np.array([
        [-sin_lon,           cos_lon,            0.0],
        [-sin_lat*cos_lon,  -sin_lat*sin_lon,    cos_lat],
        [ cos_lat*cos_lon,   cos_lat*sin_lon,    sin_lat],
    ], dtype=np.float64)

    d = points_xyz - p0[None, :]
    enu = d @ R_ecef2enu.T
    return enu  # (N,3)

def points_ecef_to_enu_xy(points_ecef, lon0, lat0, h0):
    if points_ecef.size == 0:
        return np.empty((0,2), dtype=np.float64)
    p = np.asarray(points_ecef, dtype=np.float64)
    # 过滤非法值，避免 inf 传播
    mask = np.all(np.isfinite(p), axis=1)
    if not np.any(mask):
        return np.empty((0,2), dtype=np.float64)
    p = p[mask]
    enu = _ecef_to_enu_numpy(p, lon0, lat0, h0)
    return enu[:, :2]

def build_enu_to_ecef(lon0, lat0, h0):
    """
    ENU(e,n,u) -> ECEF(XYZ) ，上面 pipeline 的逆。
    """
    pipe = (
        f"+proj=pipeline "
        f"+step +inv +proj=topocentric +ellps=WGS84 "
        f"+lon_0={lon0} +lat_0={lat0} +h_0={h0} "
        f"+step +inv +proj=cart +ellps=WGS84"
    )
    return pyproj.Transformer.from_pipeline(pipe)

def backproject_depth_to_ecef(depth, T_c2w, cam, step=8):
    """
    depth: HxW (米，和T/K一致)
    T_c2w: 4x4, ECEF
    cam: [w, h, fx, fy, cx, cy]
    返回：Nx3 ECEF 点云（稀疏采样）
    """
    w, h, fx, fy, cx, cy = cam
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    Kinv = np.linalg.inv(K)

    H, W = depth.shape[:2]
    # 采样像素网格
    xs = np.arange(0, W, step)
    ys = np.arange(0, H, step)
    xx, yy = np.meshgrid(xs, ys)     # shape [Ny, Nx]
    d = depth[yy, xx].astype(np.float64)  # 对应深度

    # 过滤无效深度
    valid = (d > 0) & np.isfinite(d)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)

    u = xx[valid].ravel().astype(np.float64)
    v = yy[valid].ravel().astype(np.float64)
    d = d[valid].ravel()

    # 像素 -> 归一化相机坐标
    pts_pix = np.stack([u, v, np.ones_like(u)], axis=0)  # 3xN
    rays = Kinv @ pts_pix                                 # 3xN
    Xc = rays * d                                         # 3xN

    # 相机到世界（ECEF）
    Rcw = T_c2w[:3, :3]
    tcw = T_c2w[:3, 3:4]
    Xw = (Rcw @ Xc) + tcw                                 # 3xN
    return Xw.T  # Nx3


def convex_hull_polygon_xy(xy):
    if xy.shape[0] < 3:
        return None, 0.0
    try:
        hull = ConvexHull(xy)
        poly = xy[hull.vertices]
        area = hull.area  # 对2D点，ConvexHull.area 即多边形周长；area 对 2D 是周长，volume 是面积
        # 注意: scipy ConvexHull 在2D下 .volume 是面积，.area 是周长
        area = hull.volume
        return poly, area
    except Exception:
        return None, 0.0

def polygon_iou(polyA_xy, polyB_xy):
    """
    polyA_xy, polyB_xy: Nx2 顶点序列（凸包）
    返回 IoU
    """
    if polyA_xy is None or polyB_xy is None:
        return 0.0
    if _HAS_SHAPELY:
        PA = Polygon(polyA_xy)
        PB = Polygon(polyB_xy)
        if not PA.is_valid or not PB.is_valid or PA.area <= 0 or PB.area <= 0:
            return 0.0
        inter = PA.intersection(PB).area
        union = unary_union([PA, PB]).area
        return float(inter / union) if union > 0 else 0.0
    else:
        # 无 shapely：退化为栅格近似（低分辨率网格）
        all_xy = np.vstack([polyA_xy, polyB_xy])
        xmin, ymin = all_xy.min(axis=0)
        xmax, ymax = all_xy.max(axis=0)
        S = 256  # 栅格尺寸，可调
        gx = np.linspace(xmin, xmax, S)
        gy = np.linspace(ymin, ymax, S)
        Gx, Gy = np.meshgrid(gx, gy)
        grid = np.stack([Gx.ravel(), Gy.ravel()], axis=1)

        def point_in_poly(pts, poly):
            # 射线法（简单实现）；这里默认凸包，速度可接受
            from matplotlib.path import Path
            return Path(poly).contains_points(pts)

        inA = point_in_poly(grid, polyA_xy)
        inB = point_in_poly(grid, polyB_xy)
        inter = np.logical_and(inA, inB).sum()
        union = np.logical_or(inA, inB).sum()
        return float(inter / union) if union > 0 else 0.0

def compute_geo_iou(T1_c2w, cam1, depth1,
                    T2_c2w, cam2, depth2,
                    step=8):
    """
    计算两帧的 Geo-IoU：
    1) 深度反投影到 ECEF 点云
    2) 以两相机中点为 ENU 原点，投到 ENU 平面
    3) 取凸包为 FoV 投影，求 IoU
    """
    # 反投影
    P1_ecef = backproject_depth_to_ecef(depth1, T1_c2w, cam1, step=step)
    P2_ecef = backproject_depth_to_ecef(depth2, T2_c2w, cam2, step=step)

    if P1_ecef.shape[0] < 3 or P2_ecef.shape[0] < 3:
        return 0.0

    # ENU 原点：两相机中心中点
    c1 = T1_c2w[:3, 3]
    c2 = T2_c2w[:3, 3]
    c_mid_ecef = 0.5 * (c1 + c2)
    lon0, lat0, h0 = ECEF_to_WGS84(c_mid_ecef)

    P1_xy = points_ecef_to_enu_xy(P1_ecef, lon0, lat0, h0)
    P2_xy = points_ecef_to_enu_xy(P2_ecef, lon0, lat0, h0)

    # 凸包多边形
    poly1_xy, _ = convex_hull_polygon_xy(P1_xy)
    poly2_xy, _ = convex_hull_polygon_xy(P2_xy)

    # IoU
    return polygon_iou(poly1_xy, poly2_xy)
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
# ---------------------------- 示例主流程 ----------------------------
if __name__ == "__main__":
    # 示例：你已有的相机/位姿/深度读法，这里仅演示 IoU 计算的调用
    # 假设已构造好 query/ref 的 4x4 T, 相机参数，和深度图（单位米，shape HxW）

    # 例：相机参数
    qcamera = [960, 540, 1158.8, 1158.8, 480.0, 270.0]
    rcamera = [960, 540, 1158.8, 1158.8, 480.0, 270.0]

    # 例：读取深度（你的代码里 ref_depth_image = cv2.flip(...,0); 按需一致处理）

    reference_depth_path = "/mnt/sda/MapScape/query/depth/USA_seq5@8@cloudy@300-100@200/1_1.png"
    query_depth_path = "/mnt/sda/MapScape/query/depth/USA_seq5@8@cloudy@300-100@200/10_1.png"
    
    ref_depth_image = cv2.imread(reference_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
    qry_depth_image = cv2.imread(query_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
    # 例：构造 4x4 位姿（这里用你的 query_T / ref_T）
    pose_txt = "/media/ubuntu/PS2000/poses/USA_seq5@8@cloudy@300-100@200.txt"
    vis_save_path = "/mnt/sda/ycb/"

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


    geo_iou = compute_geo_iou(ref_T, rcamera, ref_depth_image,
                              query_T, qcamera, qry_depth_image,
                              step=8)
    print("Geo-IoU =", geo_iou)
