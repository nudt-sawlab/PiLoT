import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import simplekml
from tqdm import tqdm

###############################################################################
# 1) (lon,lat,alt) <-> ENU（近似）转换函数（仅用于小范围可视化）
###############################################################################
def lla_to_enu(lon_deg, lat_deg, alt_m, lon0_deg, lat0_deg, alt0_m):
    R_EARTH_LAT = 111320.0  # 每度纬度约 111.32 km
    avg_lat_rad = math.radians(lat0_deg)
    east_m  = (lon_deg - lon0_deg) * R_EARTH_LAT * math.cos(avg_lat_rad)
    north_m = (lat_deg - lat0_deg) * R_EARTH_LAT
    up_m    = alt_m - alt0_m
    return np.array([east_m, north_m, up_m], dtype=float)

def enu_to_lla(x_m, y_m, z_m, lon0_deg, lat0_deg, alt0_m):
    R_EARTH_LAT = 111320.0
    avg_lat_rad = math.radians(lat0_deg)
    d_lon = x_m / (R_EARTH_LAT * math.cos(avg_lat_rad))
    d_lat = y_m / R_EARTH_LAT
    lon_deg = lon0_deg + d_lon
    lat_deg = lat0_deg + d_lat
    alt_m   = alt0_m + z_m
    return (lon_deg, lat_deg, alt_m)

###############################################################################
# 2) 计算相机/机头 在 ENU 下的方向向量
#    （Yaw=0 => 朝北, 逆时针, Pitch=0 => 正向下, Roll 绕前向轴）
###############################################################################
def camera_direction_enu(yaw_deg, pitch_deg, roll_deg=0.0):
    """
    - Yaw=0 => 朝北(ENU的+Y), 逆时针增大(90=>向西)
    - Pitch=0 => 向下, 90=>水平
    - Roll=绕机头“前向”轴

    返回在 ENU (X东,Y北,Z上) 坐标系下的单位朝向向量
    """
    # pitch=0向下 => 这里先做 90 - pitch, 让 pitch=0 => +90° => 指向 -Z
    pitch_eff = 90.0 - pitch_deg
    # yaw=0 => 北 => ENU+Y，而 ZYX欧拉时 yaw=0通常=>+X => 做 yaw_eff= yaw - 90
    yaw_eff   = yaw_deg - 90.0

    # 先绕 z(yaw_eff), 再绕 y(pitch_eff), 再绕 x(roll_deg)
    rot = R.from_euler('ZYX', [yaw_eff, pitch_eff, roll_deg], degrees=True)
    v_body = np.array([1, 0, 0], dtype=float)  # 机体默认+X为前方
    v_enu  = rot.apply(v_body)
    v_enu  = v_enu / np.linalg.norm(v_enu)
    return v_enu

###############################################################################
# 3) 生成 “长方形底面 + 无人机位置顶点” 的四棱锥(FOV)
###############################################################################
def draw_fov_rect_pyramid(
    kml,
    lon0, lat0, alt0,
    yaw_deg, pitch_deg, roll_deg=0.0,
    dist_forward=100.0,
    width=40.0,
    height=20.0,
    name_suffix=None
):
    """
    在已有的 KML 对象 kml 上，绘制一个“长方形底面 + 顶点”的四棱锥。
    - 无人机顶点: (lon0, lat0, alt0)
    - 姿态: yaw, pitch, roll(你定义)
    - 距离: dist_forward (底面中心离顶点多少米)
    - 底面尺寸: width, height(米)
    - name_suffix: 可给形状名称后面加个字符串，便于区分多个棱锥

    不保存文件；外部可统一 kml.save(...)。
    """
    # 1) 在 ENU 里, apex= (0,0,0) (因为直接把 (lon0,lat0,alt0) 当原点)
    apex_enu = np.array([0.0, 0.0, 0.0], dtype=float)

    # 2) 相机朝向
    dir_enu = camera_direction_enu(yaw_deg, pitch_deg, roll_deg)

    # 3) 底面中心 + 四角(ENU)
    center_enu = apex_enu + dist_forward * dir_enu

    # 与 dir_enu 正交的两个向量 (u, v)
    not_collinear = np.array([0,0,1], dtype=float)
    if abs(dir_enu.dot(not_collinear)) > 0.999:
        not_collinear = np.array([1,0,0], dtype=float)
    u = np.cross(dir_enu, not_collinear); u /= np.linalg.norm(u)
    v = np.cross(dir_enu, u); v /= np.linalg.norm(v)

    half_w = width/2.0
    half_h = height/2.0

    p1_enu = center_enu + half_w*u + half_h*v
    p2_enu = center_enu + half_w*u - half_h*v
    p3_enu = center_enu - half_w*u - half_h*v
    p4_enu = center_enu - half_w*u + half_h*v
    base_enu = [p1_enu, p2_enu, p3_enu, p4_enu, p1_enu]  # 闭合

    # 4) 转回 (lon,lat,alt)
    def enu2lla(pt):
        return enu_to_lla(pt[0], pt[1], pt[2], lon0, lat0, alt0)
    base_lla = [enu2lla(pt) for pt in base_enu]

    # =========== 开始往 kml 对象里画 ===========
    suffix = f"_{name_suffix}" if name_suffix else ""

    # A) 顶点 placemark
    # apex_pnt = kml.newpoint(name="Apex"+suffix, coords=[(lon0, lat0, alt0)])
    # apex_pnt.altitudemode = simplekml.AltitudeMode.absolute

    # B) 底面 polygon
    pol_base = kml.newpolygon(name="FOV Base"+suffix)
    pol_base.outerboundaryis = base_lla
    pol_base.altitudemode = simplekml.AltitudeMode.absolute
    pol_base.extrude = 0
    # 样式(例: 半透明青色)
    pol_base.style.linestyle.color = "ff27b9fd"  # 白线
    pol_base.style.polystyle.color = "88B469FF"  # A=66, B=ff, G=ff, R=00 => 半透明青

    # C) 四个三角面
    for i in range(4):
        pA = base_lla[i]
        pB = base_lla[i+1]
        coords_tri = [(lon0, lat0, alt0), pA, pB, (lon0, lat0, alt0)]
        pol_side = kml.newpolygon(name=f"FOV Side{i+1}{suffix}")
        pol_side.outerboundaryis = coords_tri
        pol_side.altitudemode = simplekml.AltitudeMode.absolute
        pol_side.extrude = 0
        # 样式(例: 半透明红)
        pol_side.style.linestyle.color = "ff27b9fd"  # A=ff,B=00,G=00,R=ff => 不透明红
        pol_side.style.linestyle.width = 3
        pol_side.style.polystyle.fill = 0
        # pol_side.style.polystyle.color = "ffffffff"  # A=66,B=ff,G=00,R=00 => 半透明蓝(若想红=>"660000ff")
def batch_fov_visualization(txt_path, output_kml="batch_fov_pyramids.kml"):
    """
    从 txt_path 文件读取若干行：
      - 每行: filename lon lat alt roll pitch yaw
    批量在同一个 KML 中画出多个“长方形四棱锥”。
    """
    kml = simplekml.Kml()

    with open(txt_path, 'r') as f0:
        for idx, line in enumerate(f0):
            if idx % 2 == 0:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                parts = line.split()
                if len(parts) < 7:
                    print(f"Line {idx} doesn't have enough columns, skip.")
                    continue

                filename = parts[0]        # 示例：图像名
                lon = float(parts[1])
                lat = float(parts[2])
                alt = float(parts[3])
                roll  = float(parts[4])
                pitch = float(parts[5])
                yaw   = float(parts[6])

                # 可自行调整：dist_forward, width, height
                # 也可根据具体行数据做不同设置
                dist_forward = 4.0
                width = 8.0
                height= 8.0

                # 调用前面的函数，把一个四棱锥画到 kml 里
                draw_fov_rect_pyramid(
                    kml,
                    lon, lat, alt,
                    yaw, pitch, roll,
                    dist_forward=dist_forward,
                    width=width,
                    height=height,
                    name_suffix=filename  # 用filename区分
                )

    # 最后统一保存
    kml.save(output_kml)
    print(f"[INFO] Batch KML saved to {output_kml}")

import simplekml
import numpy as np
import math

# ------- 颜色映射：误差 -> RGB -------
def error_to_rgb(err, vmin=0.0, vmax=5.0):
    """
    把误差err映射到绿色(低)→黄色(中)→红色(高)的连续梯度。
    vmin/vmax 是颜色条范围，超出会裁剪。
    """
    if np.isnan(err): err = vmax
    x = (err - vmin) / max(1e-6, (vmax - vmin))
    x = float(np.clip(x, 0.0, 1.0))
    if x <= 0.5:
        # 0.0~0.5: 绿(0,255,0) -> 黄(255,255,0)
        t = x / 0.5
        r = int(0   + t * (255 - 0))
        g = 255
        b = 0
    else:
        # 0.5~1.0: 黄(255,255,0) -> 红(255,0,0)
        t = (x - 0.5) / 0.5
        r = 255
        g = int(255 - t * 255)
        b = 0
    return (r, g, b)

# ------- 画单个四棱锥到 KML（线+前端矩形面） -------
def draw_fov_rect_pyramid_error(
    kml,
    lon, lat, alt,
    yaw_deg, pitch_deg, roll_deg,
    dist_forward=4.0, width=8.0, height=8.0,
    name_suffix="",
    line_rgb=(0,255,255), line_alpha=200,
    face_rgb=(255,255,0), face_alpha=120,
    line_width=2.0
):
    """
    简化版：以 (lon,lat,alt) 为原点，近似用 ENU 局部坐标系构建前端矩形四角，再回写经纬高。
    yaw/pitch/roll：度。yaw沿Z，pitch沿Y，roll沿X（右手系）。
    """
    # 角度转弧度
    yaw   = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll  = math.radians(roll_deg)

    # 构造旋转 Rz(yaw)*Ry(pitch)*Rx(roll)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = [[cy,-sy,0],[sy,cy,0],[0,0,1]]
    Ry = [[cp,0,sp],[0,1,0],[-sp,0,cp]]
    Rx = [[1,0,0],[0,cr,-sr],[0,sr,cr]]
    def mm(A,B):
        return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    def mv(R,v):
        return [R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2],
                R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2],
                R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]]
    R = mm(mm(Rz,Ry),Rx)

    # ENU 下的前端矩形四角（相机前 dist_forward，宽width，高height）
    c  = mv(R, [dist_forward, 0, 0])
    hw = mv(R, [0,  width/2.0, 0])
    hh = mv(R, [0, 0, height/2.0])

    cam = (lon, lat, alt)
    # 近似把 ENU 偏移当作经纬高的小增量（小范围足够；更严格可用 pyproj/Transforms）
    meter_per_deg_lat = 111_320.0
    meter_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    def enu_to_llh(o):
        dx, dy, dz = o
        LON = lon + dx / meter_per_deg_lon
        LAT = lat + dy / meter_per_deg_lat
        ALT = alt + dz
        return (LON, LAT, ALT)

    p1 = enu_to_llh([c[0]+hw[0]+hh[0], c[1]+hw[1]+hh[1], c[2]+hw[2]+hh[2]])
    p2 = enu_to_llh([c[0]-hw[0]+hh[0], c[1]-hw[1]+hh[1], c[2]-hw[2]+hh[2]])
    p3 = enu_to_llh([c[0]-hw[0]-hh[0], c[1]-hw[1]-hh[1], c[2]-hw[2]-hh[2]])
    p4 = enu_to_llh([c[0]+hw[0]-hh[0], c[1]+hw[1]-hh[1], c[2]+hw[2]-hh[2]])

    # simplekml 颜色：aabbggrr；用 Color.rgb 可避免手写
    line_color = simplekml.Color.rgb(line_rgb[0], line_rgb[1], line_rgb[2], line_alpha)
    face_color = simplekml.Color.rgb(face_rgb[0], face_rgb[1], face_rgb[2], face_alpha)

    # 相机->四角连线
    for dst in [p1, p2, p3, p4]:
        ls = kml.newlinestring(name=f"edge_{name_suffix}")
        ls.coords = [cam, dst]
        ls.altitudemode = simplekml.AltitudeMode.absolute
        ls.style.linestyle.color = line_color
        ls.style.linestyle.width = line_width

    # 前端矩形边框
    rect = kml.newlinestring(name=f"rect_{name_suffix}")
    rect.coords = [p1, p2, p3, p4, p1]
    rect.altitudemode = simplekml.AltitudeMode.absolute
    rect.style.linestyle.color = line_color
    rect.style.linestyle.width = line_width

    # 前端矩形填充（可选：着色更明显）
    poly = kml.newpolygon(name=f"face_{name_suffix}")
    poly.outerboundaryis = [p1, p2, p3, p4, p1]
    poly.altitudemode = simplekml.AltitudeMode.absolute
    poly.style.polystyle.color = face_color
    poly.style.outline = 0  # 只填充由上面的 rect 负责边框
    return

# ------- 批量读取轨迹 + 误差，并着色输出 KML -------
def batch_fov_visualization_with_error(
    txt_path,
    error_txt,
    output_kml="batch_fov_pyramids_colored.kml",
    dist_forward=4.0, width=4.0, height=4.0,
    cm_vmin=0.0, cm_vmax=5.0,      # 颜色条范围（m）
    face_alpha=90, line_alpha=200,
    step=5,                         # 隔一帧取一帧
    skip_tail=50                    # 最后50帧不可视化
):
    import simplekml, numpy as np, math

    # 读误差
    errors = np.loadtxt(error_txt, dtype=float)
    if np.ndim(errors) == 0:
        errors = np.array([float(errors)])

    # 读所有行，先做清洗
    with open(txt_path, 'r') as f0:
        raw_lines = [ln.strip() for ln in f0 if ln.strip()]

    # 计算有效可视化范围：从 0 到 len-1-skip_tail
    N = len(raw_lines)
    end = max(0, N - skip_tail)

    kml = simplekml.Kml()

    def error_to_rgb(err, vmin=0.0, vmax=5.0):
        if np.isnan(err): err = vmax
        x = float(np.clip((err - vmin) / max(1e-6, (vmax - vmin)), 0.0, 1.0))
        if x <= 0.5:
            t = x / 0.5
            r, g, b = int(0 + t * 255), 255, 0      # 绿->黄
        else:
            t = (x - 0.5) / 0.5
            r, g, b = 255, int(255 - t * 255), 0    # 黄->红
        return (r, g, b)

    def draw_fov_rect_pyramid(
        kml, lon, lat, alt, yaw_deg, pitch_deg, roll_deg,
        dist_forward=4.0, width=8.0, height=8.0,
        name_suffix="", line_rgb=(0,255,255), line_alpha=200,
        face_rgb=(255,255,0), face_alpha=120, line_width=2.0
    ):
        # 角度->弧度
        cy = math.cos(math.radians(yaw_deg));    sy = math.sin(math.radians(yaw_deg))
        cp = math.cos(math.radians(pitch_deg));  sp = math.sin(math.radians(pitch_deg))
        cr = math.cos(math.radians(roll_deg));   sr = math.sin(math.radians(roll_deg))
        Rz = [[cy,-sy,0],[sy,cy,0],[0,0,1]]
        Ry = [[cp,0,sp],[0,1,0],[-sp,0,cp]]
        Rx = [[1,0,0],[0,cr,-sr],[0,sr,cr]]
        def mm(A,B):
            return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        def mv(R,v):
            return [R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2],
                    R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2],
                    R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]]
        R = mm(mm(Rz,Ry),Rx)
        c  = mv(R, [dist_forward, 0, 0])
        hw = mv(R, [0,  width/2.0, 0])
        hh = mv(R, [0, 0, height/2.0])

        cam = (lon, lat, alt)
        meter_per_deg_lat = 111_320.0
        meter_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
        def enu_to_llh(o):
            dx, dy, dz = o
            LON = lon + dx / meter_per_deg_lon
            LAT = lat + dy / meter_per_deg_lat
            ALT = alt + dz
            return (LON, LAT, ALT)

        p1 = enu_to_llh([c[0]+hw[0]+hh[0], c[1]+hw[1]+hh[1], c[2]+hw[2]+hh[2]])
        p2 = enu_to_llh([c[0]-hw[0]+hh[0], c[1]-hw[1]+hh[1], c[2]-hw[2]+hh[2]])
        p3 = enu_to_llh([c[0]-hw[0]-hh[0], c[1]-hw[1]-hh[1], c[2]-hw[2]-hh[2]])
        p4 = enu_to_llh([c[0]+hw[0]-hh[0], c[1]+hw[1]-hh[1], c[2]+hw[2]-hh[2]])

        line_color = simplekml.Color.rgb(line_rgb[0], line_rgb[1], line_rgb[2], line_alpha)
        face_color = simplekml.Color.rgb(face_rgb[0], face_rgb[1], face_rgb[2], face_alpha)

        for dst in [p1, p2, p3, p4]:
            ls = kml.newlinestring(name=f"edge_{name_suffix}")
            ls.coords = [cam, dst]
            ls.altitudemode = simplekml.AltitudeMode.absolute
            ls.style.linestyle.color = line_color
            ls.style.linestyle.width = line_width

        rect = kml.newlinestring(name=f"rect_{name_suffix}")
        rect.coords = [p1, p2, p3, p4, p1]
        rect.altitudemode = simplekml.AltitudeMode.absolute
        rect.style.linestyle.color = line_color
        rect.style.linestyle.width = line_width

        poly = kml.newpolygon(name=f"face_{name_suffix}")
        poly.outerboundaryis = [p1, p2, p3, p4, p1]
        poly.altitudemode = simplekml.AltitudeMode.absolute
        poly.style.polystyle.color = face_color
        poly.style.outline = 0

    # 主循环：隔一帧取一帧，并跳过最后 skip_tail 帧
    for idx in range(0, end, step):
        parts = raw_lines[idx].split()
        if len(parts) < 7:
            print(f"Line {idx} doesn't have enough columns, skip.")
            continue

        filename = parts[0]
        lon = float(parts[1]); lat = float(parts[2]); alt = float(parts[3])
        roll = float(parts[4]); pitch = float(parts[5]); yaw = float(parts[6]) + 90

        # 误差取同帧索引，若越界用最后一个值兜底
        err = float(errors[idx]) if idx < len(errors) else float(errors[-1])
        if err > 4:
            continue
        r, g, b = error_to_rgb(err, vmin=cm_vmin, vmax=cm_vmax)

        draw_fov_rect_pyramid(
            kml,
            lon, lat, alt,
            yaw, pitch, roll,
            dist_forward=dist_forward, width=width, height=height,
            name_suffix=f"{filename} | err={err:.2f}m",
            line_rgb=(r, g, b), line_alpha=line_alpha,
            face_rgb=(r, g, b), face_alpha=face_alpha,
            line_width=2.0
        )

    kml.save(output_kml)
    print(f"[INFO] Colored KML saved to {output_kml}  (step={step}, skipped tail={skip_tail})")
   
def batch_fov_visualization_with_error_tour(
    txt_path,
    error_txt,
    output_kml="batch_fov_30fps.kml",
    dist_forward=4.0, width=4.0, height=4.0,
    cm_vmin=0.0, cm_vmax=5.0,      # 颜色条范围（m）
    face_alpha=90, line_alpha=200,
    step=10,                         # 默认为逐帧；可改成>1降采样
    skip_tail=0,                    # 是否跳过尾帧
    fps=30,                         # 播放帧率
    base_time="2025-01-01T00:00:00Z",  # 时间轴起点（ISO8601 UTC）
    flyto_alt=120.0,                # Tour 跟随视角的高度（相对地面）
    flyto_tilt_from_pitch=True      # 用 pitch 推导 tilt（常见相机定义：前视时 tilt~90-pitch）
):
    """
    读取 txt 位姿与误差，输出带有 TimeSpan 的视锥体，并创建 gx:Tour 以 ~30fps 播放。
    在 Google Earth 中：打开KML -> 双击 Tour 播放；或用时间滑块查看逐帧。
    """
    import simplekml, numpy as np, math
    from datetime import datetime, timedelta, timezone

    # -------------- 工具：时间格式化 --------------
    def parse_base_time(s):
        # "YYYY-mm-ddTHH:MM:SSZ"
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

    def to_when(t):
        # 格式化回 KML 可识别的字符串
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")

    # -------------- 读误差 --------------
    errors = np.loadtxt(error_txt, dtype=float)
    if np.ndim(errors) == 0:
        errors = np.array([float(errors)])

    # -------------- 读位姿 --------------
    with open(txt_path, 'r') as f0:
        raw_lines = [ln.strip() for ln in f0 if ln.strip()]

    N_all = len(raw_lines)
    end = max(0, N_all - skip_tail)

    # -------------- KML 初始化 --------------
    kml = simplekml.Kml()
    doc_folder = kml.newfolder(name="FOV_30fps")

    # 颜色映射：绿->黄->红（vmin~vmax）
    def error_to_rgb(err, vmin=0.0, vmax=5.0):
        if np.isnan(err):
            err = vmax
        x = float(np.clip((err - vmin) / max(1e-6, (vmax - vmin)), 0.0, 1.0))
        if x <= 0.5:
            t = x / 0.5
            r, g, b = int(0 + t * 255), 255, 0      # 绿->黄
        else:
            t = (x - 0.5) / 0.5
            r, g, b = 255, int(255 - t * 255), 0    # 黄->红
        return (r, g, b)

    # -------------- 画单帧视锥体（带 TimeSpan） --------------
    def draw_fov_rect_pyramid(
        parent_folder, lon, lat, alt, yaw_deg, pitch_deg, roll_deg,
        dist_forward=4.0, width=8.0, height=8.0,
        name_suffix="", line_rgb=(0,255,255), line_alpha=200,
        face_rgb=(255,255,0), face_alpha=120, line_width=2.0,
        begin_when=None, end_when=None
    ):
        cy = math.cos(math.radians(yaw_deg));    sy = math.sin(math.radians(yaw_deg))
        cp = math.cos(math.radians(pitch_deg));  sp = math.sin(math.radians(pitch_deg))
        cr = math.cos(math.radians(roll_deg));   sr = math.sin(math.radians(roll_deg))
        Rz = [[cy,-sy,0],[sy,cy,0],[0,0,1]]
        Ry = [[cp,0,sp],[0,1,0],[-sp,0,cp]]
        Rx = [[1,0,0],[0,cr,-sr],[0,sr,cr]]

        def mm(A,B):
            return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        def mv(R,v):
            return [R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2],
                    R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2],
                    R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]]
        R = mm(mm(Rz,Ry),Rx)
        c  = mv(R, [dist_forward, 0, 0])
        hw = mv(R, [0,  width/2.0, 0])
        hh = mv(R, [0, 0, height/2.0])

        cam = (lon, lat, alt)
        meter_per_deg_lat = 111_320.0
        meter_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
        def enu_to_llh(o):
            dx, dy, dz = o
            LON = lon + dx / meter_per_deg_lon
            LAT = lat + dy / meter_per_deg_lat
            ALT = alt + dz
            return (LON, LAT, ALT)

        p1 = enu_to_llh([c[0]+hw[0]+hh[0], c[1]+hw[1]+hh[1], c[2]+hw[2]+hh[2]])
        p2 = enu_to_llh([c[0]-hw[0]+hh[0], c[1]-hw[1]+hh[1], c[2]-hw[2]+hh[2]])
        p3 = enu_to_llh([c[0]-hw[0]-hh[0], c[1]-hw[1]-hh[1], c[2]-hw[2]-hh[2]])
        p4 = enu_to_llh([c[0]+hw[0]-hh[0], c[1]+hw[1]-hh[1], c[2]+hw[2]-hh[2]])

        line_color = simplekml.Color.rgb(line_rgb[0], line_rgb[1], line_rgb[2], line_alpha)
        face_color = simplekml.Color.rgb(face_rgb[0], face_rgb[1], face_rgb[2], face_alpha)

        # 放到一个子文件夹，便于管理
        fd = parent_folder.newfolder(name=f"frame_{name_suffix}")

        # 四条棱线
        for j, dst in enumerate([p1, p2, p3, p4]):
            ls = fd.newlinestring(name=f"edge_{name_suffix}_{j}")
            ls.coords = [cam, dst]
            ls.altitudemode = simplekml.AltitudeMode.absolute
            ls.style.linestyle.color = line_color
            ls.style.linestyle.width = line_width
            if begin_when and end_when:
                ls.timespan.begin = begin_when
                ls.timespan.end   = end_when

        # 前方矩形边
        rect = fd.newlinestring(name=f"rect_{name_suffix}")
        rect.coords = [p1, p2, p3, p4, p1]
        rect.altitudemode = simplekml.AltitudeMode.absolute
        rect.style.linestyle.color = line_color
        rect.style.linestyle.width = line_width
        if begin_when and end_when:
            rect.timespan.begin = begin_when
            rect.timespan.end   = end_when

        # 面
        poly = fd.newpolygon(name=f"face_{name_suffix}")
        poly.outerboundaryis = [p1, p2, p3, p4, p1]
        poly.altitudemode = simplekml.AltitudeMode.absolute
        poly.style.polystyle.color = face_color
        poly.style.outline = 0
        if begin_when and end_when:
            poly.timespan.begin = begin_when
            poly.timespan.end   = end_when

    # -------------- 创建 gx:Tour（按帧推进） --------------
    # 用 Tour 的 FlyTo + Wait 模拟 30fps，并让视角跟随相机
    tour = kml.newgxtour(name="Play_30fps")
    playlist = tour.newgxplaylist()

    # 解析起始时间
    t0 = parse_base_time(base_time)
    frame_dt = timedelta(seconds=1.0 / max(1, fps))

    # 预先 FlyTo 到第一帧位置
    def add_flyto(playlist, lon, lat, alt, yaw, pitch, roll, duration=0.0):
        flyto = playlist.newgxflyto(gxduration=duration)
        # Google Earth Camera: heading(0北/顺时针), tilt[0-90], roll[-180,180]
        if flyto_tilt_from_pitch:
            # 常见视觉习惯：pitch=0看地平线，pitch>0向下；tilt ~ 90 - pitch
            tilt = float(max(0.0, min(90.0, 90.0 - pitch)))
        else:
            tilt = 75.0
        flyto.camera = simplekml.Camera(
            longitude=lon, latitude=lat, altitude=alt + flyto_alt,
            heading=yaw, tilt=tilt, roll=roll,
            altitudemode=simplekml.AltitudeMode.relativetoground
        )

    # -------------- 主循环：按时间顺序逐帧 --------------
    n_frames = 0
    for idx in range(0, end, step):
        parts = raw_lines[idx].split()
        if len(parts) < 7:
            print(f"Line {idx} not enough columns, skip.")
            continue

        filename = parts[0]
        lon = float(parts[1]); lat = float(parts[2]); alt = float(parts[3])
        roll = float(parts[4]); pitch = float(parts[5]); yaw = float(parts[6]) + 90  # 你的原偏移

        # 误差：越界用最后一个值
        err = float(errors[idx]) if idx < len(errors) else float(errors[-1])

        # 按误差筛选（可选）
        # if err > cm_vmax:
        #     continue

        r, g, b = error_to_rgb(err, vmin=cm_vmin, vmax=cm_vmax)

        # 帧的时间段
        t_begin = t0 + n_frames * frame_dt
        t_end   = t_begin + frame_dt
        begin_when = to_when(t_begin)
        end_when   = to_when(t_end)

        # 画这一帧的视锥体，绑定 timespan
        draw_fov_rect_pyramid(
            doc_folder,
            lon, lat, alt,
            yaw, pitch, roll,
            dist_forward=dist_forward, width=width, height=height,
            name_suffix=f"{idx:06d}|{filename}|err={err:.2f}m",
            line_rgb=(r, g, b), line_alpha=line_alpha,
            face_rgb=(r, g, b), face_alpha=face_alpha,
            line_width=2.0,
            begin_when=begin_when, end_when=end_when
        )

        # Tour：把视角飞到该帧（瞬时飞到）+ 等待 1/fps 秒
        add_flyto(playlist, lon, lat, alt, yaw, pitch, roll, duration=0.0)
        playlist.newgxwait(gxduration=1.0 / max(1, fps))

        n_frames += 1

    # -------------- 保存 --------------
    kml.save(output_kml)
    print(f"[INFO] KML saved to {output_kml}  | frames={n_frames}, fps={fps}, base_time={base_time}")
def batch_traj_with_axes_tour_gx(
    txt_path,
    error_txt=None,
    output_kml="traj_axes_30fps.kml",
    step=1,                      # 建议逐帧，配合 gxduration 平滑
    skip_tail=0,
    fps=30,
    base_time="2025-01-01T00:00:00Z",
    flyto_alt=120.0,
    flyto_tilt_from_pitch=True,
    axis_len_m=15.0,
    axis_width=3.0,
):
    import simplekml, numpy as np, math
    from datetime import datetime, timedelta, timezone

    def parse_base_time(s):
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

    def to_when(t):
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(txt_path, "r") as f0:
        raw_lines = [ln.strip() for ln in f0 if ln.strip()]
    N_all = len(raw_lines)
    end   = max(0, N_all - skip_tail)

    kml = simplekml.Kml()
    folder = kml.newfolder(name="Trajectory_And_Axes")

    # —— 轨迹 Track（整段）——
    track = folder.newgxtrack(name="Trajectory")
    track.altitudemode = simplekml.AltitudeMode.absolute
    track.style.linestyle.width = 3
    track.style.linestyle.color = simplekml.Color.rgb(255, 255, 255, 255)

    # —— 将 when/coord 先收集，最后一次性灌入，避免 newgxcoord 单点调用的类型问题 —— 
    whens, coords = [], []

    AX_R = simplekml.Color.rgb(255, 0,   0,   255)
    AX_G = simplekml.Color.rgb(0,   255, 0,   255)
    AX_B = simplekml.Color.rgb(0,   128, 255, 255)

    def enu_to_llh(lon, lat, alt, dx, dy, dz):
        meter_per_deg_lat = 111_320.0
        meter_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
        LON = lon + dx / meter_per_deg_lon
        LAT = lat + dy / meter_per_deg_lat
        ALT = alt + dz
        return (LON, LAT, ALT)

    def rot_zyx(yaw_deg, pitch_deg, roll_deg):
        cy = math.cos(math.radians(yaw_deg));    sy = math.sin(math.radians(yaw_deg))
        cp = math.cos(math.radians(pitch_deg));  sp = math.sin(math.radians(pitch_deg))
        cr = math.cos(math.radians(roll_deg));   sr = math.sin(math.radians(roll_deg))
        Rz = [[cy,-sy,0],[sy,cy,0],[0,0,1]]
        Ry = [[cp,0,sp],[0,1,0],[-sp,0,cp]]
        Rx = [[1,0,0],[0,cr,-sr],[0,sr,cr]]
        def mm(A,B):
            return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        return mm(mm(Rz,Ry),Rx)

    def mv(R,v):
        return [R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2],
                R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2],
                R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]]

    # —— Tour（连续插值相机）——
    tour = kml.newgxtour(name="Play_30fps")
    playlist = tour.newgxplaylist()

    t0 = parse_base_time(base_time)
    frame_dt = timedelta(seconds=1.0 / max(1, fps))

    # 为“累计轨迹”准备最终结束时间（每段线 end=最后一帧时间，从而不断累积）
    # 先遍历一次，确定总帧数
    frame_indices = [idx for idx in range(0, end, step) if len(raw_lines[idx].split()) >= 7]
    
    total_frames = len(frame_indices)
    final_end_time = to_when(t0 + (total_frames) * frame_dt)

    prev_llh = None
    n_frames = 0

    for idx in frame_indices:
        if idx > 900: continue
        parts = raw_lines[idx].split()
        _, lon_s, lat_s, alt_s, roll_s, pitch_s, yaw_s = parts[:7]
        lon = float(lon_s); lat = float(lat_s); alt = float(alt_s)
        roll = float(roll_s); pitch = float(pitch_s); yaw = float(yaw_s) + 90.0  # 你的偏移

        t_begin = t0 + n_frames * frame_dt
        when = to_when(t_begin)

        # === 收集 Track 的时刻与坐标（一次性写入） ===
        whens.append(when)
        coords.append((lon, lat, alt))

        # === 坐标轴（三条短线，仅当前帧显示） ===
        R = rot_zyx(yaw, pitch, roll)
        x_axis = mv(R, [axis_len_m, 0, 0])
        y_axis = mv(R, [0, axis_len_m, 0])
        z_axis = mv(R, [0, 0, axis_len_m])

        X_end = enu_to_llh(lon, lat, alt, x_axis[0], y_axis[1], z_axis[2])  # 注意这里别写错 y/z
        Y_end = enu_to_llh(lon, lat, alt, y_axis[0], y_axis[1], y_axis[2])
        Z_end = enu_to_llh(lon, lat, alt, z_axis[0], z_axis[1], z_axis[2])

        begin_when = when
        end_when   = to_when(t_begin + frame_dt)

        def add_axis_line(name, color, end_llh):
            ls = folder.newlinestring(name=name)
            ls.coords = [(lon, lat, alt), end_llh]
            ls.altitudemode = simplekml.AltitudeMode.absolute
            ls.style.linestyle.color = color
            ls.style.linestyle.width = axis_width
            ls.timespan.begin = begin_when
            ls.timespan.end   = end_when

        add_axis_line(f"axis_X_{idx}", AX_R, X_end)
        add_axis_line(f"axis_Y_{idx}", AX_G, Y_end)
        add_axis_line(f"axis_Z_{idx}", AX_B, Z_end)

        # === “累积轨迹”：当前段（prev->curr）出现后一直保留 ===
        if prev_llh is not None:
            seg = folder.newlinestring(name=f"trail_{idx-1}_{idx}")
            seg.coords = [prev_llh, (lon, lat, alt)]
            seg.altitudemode = simplekml.AltitudeMode.absolute
            seg.style.linestyle.width = 3
            seg.style.linestyle.color = simplekml.Color.rgb(255, 255, 255, 255)
            seg.timespan.begin = when          # 该段在当前时刻出现
            seg.timespan.end   = final_end_time  # 一直保留到动画结束

        prev_llh = (lon, lat, alt)

        # === Tour：连续插值（更丝滑）===
        # 关键改动：用 gxduration=1/fps 做平滑飞行；不再 newgxwait()
        flyto = playlist.newgxflyto(gxduration=1.0 / max(1, fps))
        if flyto_tilt_from_pitch:
            tilt = float(max(0.0, min(90.0, 90.0 - pitch)))
        else:
            tilt = 75.0
        flyto.camera = simplekml.Camera(
            longitude=lon, latitude=lat, altitude=alt + flyto_alt,
            heading=yaw, tilt=tilt, roll=roll,
            altitudemode=simplekml.AltitudeMode.relativetoground
        )

        n_frames += 1

    # 将 Track 数据一次性写入（避免 float has no len）
    track.newwhen(whens)
    track.newgxcoord(coords)

    kml.save(output_kml)
    print(f"[INFO] KML saved to {output_kml} | frames={n_frames}, fps={fps}")

def batch_traj_with_axes_tour(
    txt_path,
    error_txt=None,                 # 可选：误差，不用了也行
    output_kml="traj_axes_30fps.kml",
    step=1,
    skip_tail=0,
    fps=30,
    base_time="2025-01-01T00:00:00Z",
    flyto_alt=120.0,
    flyto_tilt_from_pitch=True,
    axis_len_m=15.0,                # 轴长度（米）
    axis_width=3.0                  # 轴线宽
):
    """
    读取位姿，输出：一条随时间播放的轨迹（gx:Track）+ 每帧头部的姿态坐标轴（3 条短线）。
    打开KML -> 双击 Tour 播放；或用时间轴查看。
    """
    import simplekml, numpy as np, math
    from datetime import datetime, timedelta, timezone

    # ---------- 工具：时间 ----------
    def parse_base_time(s):
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

    def to_when(t):
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")

    # ---------- 读数据 ----------
    with open(txt_path, "r") as f0:
        raw_lines = [ln.strip() for ln in f0 if ln.strip()]
    N_all = len(raw_lines)
    end   = max(0, N_all - skip_tail)
    end   = 900

    # 可选误差（目前不参与绘制）
    if error_txt is not None:
        try:
            errors = np.loadtxt(error_txt, dtype=float)
            if np.ndim(errors) == 0:
                errors = np.array([float(errors)])
        except Exception:
            errors = None
    else:
        errors = None

    # ---------- KML ----------
    kml = simplekml.Kml()
    folder = kml.newfolder(name="Trajectory_And_Axes")

    # 轨迹（gx:Track）：随时间播放
    track = folder.newgxtrack(name="Trajectory")
    track.altitudemode = simplekml.AltitudeMode.absolute
    track.style.linestyle.width = 3
    track.style.linestyle.color = simplekml.Color.rgb(255, 255, 255, 255)  # 白色轨迹

    # 坐标轴颜色
    AX_R = simplekml.Color.rgb(255, 0,   0,   255)  # X/前 红
    AX_G = simplekml.Color.rgb(0,   255, 0,   255)  # Y/右 绿
    AX_B = simplekml.Color.rgb(0,   128, 255, 255)  # Z/上 蓝（稍亮）

    # 地理→米的近似比例（局部ENU）
    def enu_to_llh(lon, lat, alt, dx, dy, dz):
        meter_per_deg_lat = 111_320.0
        meter_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
        LON = lon + dx / meter_per_deg_lon
        LAT = lat + dy / meter_per_deg_lat
        ALT = alt + dz
        return (LON, LAT, ALT)

    # 旋转矩阵（与原代码一致：Rz(yaw) * Ry(pitch) * Rx(roll)）
    def rot_zyx(yaw_deg, pitch_deg, roll_deg):
        cy = math.cos(math.radians(yaw_deg));    sy = math.sin(math.radians(yaw_deg))
        cp = math.cos(math.radians(pitch_deg));  sp = math.sin(math.radians(pitch_deg))
        cr = math.cos(math.radians(roll_deg));   sr = math.sin(math.radians(roll_deg))
        Rz = [[cy,-sy,0],[sy,cy,0],[0,0,1]]
        Ry = [[cp,0,sp],[0,1,0],[-sp,0,cp]]
        Rx = [[1,0,0],[0,cr,-sr],[0,sr,cr]]
        def mm(A,B):
            return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        return mm(mm(Rz,Ry),Rx)

    def mv(R,v):
        return [R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2],
                R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2],
                R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]]

    # ---------- Tour：跟随播放 ----------
    tour = kml.newgxtour(name="Play_30fps")
    playlist = tour.newgxplaylist()

    t0 = parse_base_time(base_time)
    frame_dt = timedelta(seconds=1.0 / max(1, fps))

    # 遍历帧，填充 Track 和 坐标轴
    n_frames = 0
    for idx in range(0, end, step):
        parts = raw_lines[idx].split()
        if len(parts) < 7:
            continue
        _, lon_s, lat_s, alt_s, roll_s, pitch_s, yaw_s = parts[:7]
        lon = float(lon_s); lat = float(lat_s); alt = float(alt_s)
        roll = float(roll_s); pitch = float(pitch_s); yaw = float(yaw_s) + 90.0  # 你的偏移

        # 时间戳
        t_begin = t0 + n_frames * frame_dt
        when = to_when(t_begin)

        # === 1) 轨迹点（gx:Track） ===
        track.newwhen([when])
        track.newgxcoord([(lon, lat, alt)])

        # === 2) 头部坐标轴（3条短线，仅当前帧可见） ===
        # 机体系：X前、Y右、Z上
        R = rot_zyx(yaw, pitch, roll)
        x_axis = mv(R, [axis_len_m, 0, 0])
        y_axis = mv(R, [0, axis_len_m, 0])
        z_axis = mv(R, [0, 0, axis_len_m])

        # 端点（经纬高）
        X_end = enu_to_llh(lon, lat, alt, x_axis[0], x_axis[1], x_axis[2])
        Y_end = enu_to_llh(lon, lat, alt, y_axis[0], y_axis[1], y_axis[2])
        Z_end = enu_to_llh(lon, lat, alt, z_axis[0], z_axis[1], z_axis[2])

        begin_when = when
        end_when   = to_when(t_begin + frame_dt)

        def add_axis_line(name, color, end_llh):
            ls = folder.newlinestring(name=name)
            ls.coords = [(lon, lat, alt), end_llh]
            ls.altitudemode = simplekml.AltitudeMode.absolute
            ls.style.linestyle.color = color
            ls.style.linestyle.width = axis_width
            ls.timespan.begin = begin_when
            ls.timespan.end   = end_when

        add_axis_line(f"axis_X_{idx}", AX_R, X_end)
        add_axis_line(f"axis_Y_{idx}", AX_G, Y_end)
        add_axis_line(f"axis_Z_{idx}", AX_B, Z_end)

        # === 3) Tour 跟随（相机跟着头部走） ===
        flyto = playlist.newgxflyto(gxduration=0.0)
        if flyto_tilt_from_pitch:
            tilt = float(max(0.0, min(90.0, 90.0 - pitch)))
        else:
            tilt = 75.0
        flyto.camera = simplekml.Camera(
            longitude=lon, latitude=lat, altitude=alt + flyto_alt,
            heading=yaw, tilt=tilt, roll=roll,
            altitudemode=simplekml.AltitudeMode.relativetoground
        )
        playlist.newgxwait(gxduration=1.0 / max(1, fps))

        n_frames += 1

    # 轨迹可可选加一个静态样式说明
    track.stylemap.normalstyle.linestyle.color = simplekml.Color.rgb(255, 255, 255, 255)
    track.stylemap.highlightstyle.linestyle.color = simplekml.Color.rgb(255, 255, 255, 255)

    kml.save(output_kml)
    print(f"[INFO] KML saved to {output_kml} | frames={n_frames}, fps={fps}")

def batch_traj_two_with_axes_tour(
    txt_est,
    txt_gt,
    output_kml="traj_est_vs_gt_axes_30fps.kml",
    step=1,
    skip_tail=0,
    fps=30,
    base_time="2025-01-01T00:00:00Z",
    follow="est",                 # "est" or "gt"
    axis_len_m=15.0,
    axis_width=3.0,
    tilt_from_pitch=True,
    yaw_add_90=True
):
    import simplekml, numpy as np, math
    from datetime import datetime, timedelta, timezone

    # ---------- small utils ----------
    def parse_base_time(s):
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    def to_when(t): return t.strftime("%Y-%m-%dT%H:%M:%SZ")
    def load_lines(path):
        with open(path, "r") as f: return [ln.strip() for ln in f if ln.strip()]
    def parse_pose(parts):
        _, lon_s, lat_s, alt_s, roll_s, pitch_s, yaw_s = parts[:7]
        return float(lon_s), float(lat_s), float(alt_s), float(roll_s), float(pitch_s), float(yaw_s)

    def rot_zyx(yaw_deg, pitch_deg, roll_deg):
        cy, sy = math.cos(math.radians(yaw_deg)),   math.sin(math.radians(yaw_deg))
        cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
        cr, sr = math.cos(math.radians(roll_deg)),  math.sin(math.radians(roll_deg))
        Rz = [[cy,-sy,0],[sy,cy,0],[0,0,1]]
        Ry = [[cp,0,sp],[0,1,0],[-sp,0,cp]]
        Rx = [[1,0,0],[0,cr,-sr],[0,sr,cr]]
        def mm(A,B): return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        return mm(mm(Rz,Ry),Rx)
    def mv(R,v):
        return [R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2],
                R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2],
                R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]]
    def enu_to_llh(lon, lat, alt, dx, dy, dz):
        meter_per_deg_lat = 111_320.0
        meter_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
        return (lon + dx/meter_per_deg_lon, lat + dy/meter_per_deg_lat, alt + dz)

    # ---------- load ----------
    lines_est = load_lines(txt_est)
    lines_gt  = load_lines(txt_gt)
    if not lines_est or not lines_gt: raise RuntimeError("估计或GT文件为空")

    end_est = max(0, len(lines_est) - skip_tail)
    end_gt  = max(0, len(lines_gt)  - skip_tail)
    idxs_est = [i for i in range(0, end_est, step) if len(lines_est[i].split()) >= 7]
    idxs_gt  = [i for i in range(0, end_gt,  step) if len(lines_gt[i].split())  >= 7]
    n_frames = min(len(idxs_est), len(idxs_gt))
    idxs_est, idxs_gt = idxs_est[:n_frames], idxs_gt[:n_frames]

    # ---------- kml ----------
    kml = simplekml.Kml()
    root = kml.newfolder(name="EST_vs_GT_Trajectory")

    folder_est = root.newfolder(name="EST")
    folder_gt  = root.newfolder(name="GT")

    track_est = folder_est.newgxtrack(name="EST_Track")
    track_gt  = folder_gt.newgxtrack(name="GT_Track")
    for trk in (track_est, track_gt):
        trk.altitudemode = simplekml.AltitudeMode.absolute

    # === 样式：隐藏pushpin + 标签颜色与线条一致 ===
    # 轨迹颜色
    YELLOW = simplekml.Color.rgb(255, 255,   0, 255)  # EST
    BLUE   = simplekml.Color.rgb(  0,   0, 255, 255)  # GT

    def make_track_stylemap(color, width=4, label_scale=1.0):
        st_n = simplekml.Style()
        st_n.linestyle.color = color
        st_n.linestyle.width = width
        st_n.iconstyle.scale = 0        # ← 隐藏默认pushpin
        st_n.labelstyle.color = color   # ← 标签字样同色
        st_n.labelstyle.scale = label_scale

        st_h = simplekml.Style()
        st_h.linestyle.color = color
        st_h.linestyle.width = width + 1
        st_h.iconstyle.scale = 0
        st_h.labelstyle.color = color
        st_h.labelstyle.scale = label_scale

        sm = simplekml.StyleMap()
        sm.normalstyle = st_n
        sm.highlightstyle = st_h
        return sm

    track_est.stylemap = make_track_stylemap(YELLOW, 4, 1.0)
    track_gt.stylemap  = make_track_stylemap(BLUE,   4, 1.0)

    # 轴颜色（保持RGB）—— GT 用半透明以便区分
    AX_R  = simplekml.Color.rgb(255,   0,   0, 255)
    AX_G  = simplekml.Color.rgb(  0, 255,   0, 255)
    AX_B  = simplekml.Color.rgb(  0, 128, 255, 255)
    AX_Rt = simplekml.Color.rgb(255,   0,   0, 180)
    AX_Gt = simplekml.Color.rgb(  0, 255,   0, 180)
    AX_Bt = simplekml.Color.rgb(  0, 128, 255, 180)

    # Tour（连续插值）
    tour = kml.newgxtour(name="Play_30fps")
    playlist = tour.newgxplaylist()

    t0 = parse_base_time(base_time)
    dt = timedelta(seconds=1.0 / max(1, fps))
    final_end_time = to_when(t0 + n_frames * dt)

    whens_est, coords_est = [], []
    whens_gt,  coords_gt  = [], []
    prev_est = None
    prev_gt  = None

    for f in tqdm(range(n_frames)):
        i_e, i_g = idxs_est[f], idxs_gt[f]
        lon_e, lat_e, alt_e, roll_e, pitch_e, yaw_e = parse_pose(lines_est[i_e].split())
        lon_g, lat_g, alt_g, roll_g, pitch_g, yaw_g = parse_pose(lines_gt[i_g].split())
        if yaw_add_90: yaw_e += 90.0; yaw_g += 90.0

        t    = t0 + f * dt
        when = to_when(t); when_next = to_when(t + dt)

        # Track 数据先收集
        whens_est.append(when); coords_est.append((lon_e, lat_e, alt_e))
        whens_gt.append(when);  coords_gt.append((lon_g, lat_g, alt_g))

        # === 头部坐标轴（仅当前帧显示；不设置 name，避免标签/图标） ===
        def add_axes(folder, lon, lat, alt, yaw, pitch, roll, cR, cG, cB):
            R = rot_zyx(yaw, pitch, roll)
            x, y, z = mv(R, [axis_len_m,0,0]), mv(R, [0,axis_len_m,0]), mv(R, [0,0,axis_len_m])
            X = enu_to_llh(lon, lat, alt, x[0], x[1], x[2])
            Y = enu_to_llh(lon, lat, alt, y[0], y[1], y[2])
            Z = enu_to_llh(lon, lat, alt, z[0], z[1], z[2])
            for end, col in ((X,cR),(Y,cG),(Z,cB)):
                ls = folder.newlinestring()       # ← 不给 name
                ls.coords = [(lon,lat,alt), end]
                ls.altitudemode = simplekml.AltitudeMode.absolute
                ls.style.linestyle.color = col
                ls.style.linestyle.width = axis_width
                ls.timespan.begin = when
                ls.timespan.end   = when_next

        add_axes(folder_est, lon_e, lat_e, alt_e, yaw_e, pitch_e, roll_e, AX_R, AX_G, AX_B)
        add_axes(folder_gt,  lon_g, lat_g, alt_g, yaw_g, pitch_g, roll_g, AX_Rt, AX_Gt, AX_Bt)

        # === 累积尾迹（EST=黄、GT=蓝；出现后保留到末尾；不设置 name） ===
        def add_seg(folder, p0, p1, color):
            seg = folder.newlinestring()
            seg.coords = [p0, p1]
            seg.altitudemode = simplekml.AltitudeMode.absolute
            seg.style.linestyle.color = color
            seg.style.linestyle.width = 3
            seg.timespan.begin = when
            seg.timespan.end   = final_end_time

        if prev_est is not None: add_seg(folder_est, prev_est, (lon_e,lat_e,alt_e), YELLOW)
        if prev_gt  is not None: add_seg(folder_gt,  prev_gt,  (lon_g,lat_g,alt_g), BLUE)

        prev_est = (lon_e,lat_e,alt_e)
        prev_gt  = (lon_g,lat_g,alt_g)

        # === Tour：更丝滑（连续插值；不再 Wait） ===
        cam_lon, cam_lat, cam_alt, cam_yaw, cam_pitch, cam_roll = (
            (lon_e, lat_e, alt_e, yaw_e, pitch_e, roll_e)
            if follow == "est" else
            (lon_g, lat_g, alt_g, yaw_g, pitch_g, roll_g)
        )
        flyto = playlist.newgxflyto(gxduration=1.0 / max(1, fps))
        tilt  = float(max(0.0, min(90.0, 90.0 - cam_pitch))) if tilt_from_pitch else 75.0
        flyto.camera = simplekml.Camera(
            longitude=cam_lon, latitude=cam_lat, altitude=cam_alt + 120.0,
            heading=cam_yaw, tilt=tilt, roll=cam_roll,
            altitudemode=simplekml.AltitudeMode.relativetoground
        )

    # 一次性写入 Track（注意：要传列表）
    track_est.newwhen(whens_est); track_est.newgxcoord(coords_est)
    track_gt.newwhen(whens_gt);   track_gt.newgxcoord(coords_gt)

    kml.save(output_kml)
    print(f"[INFO] KML saved to {output_kml} | frames={n_frames}, fps={fps}, follow={follow}")



def extract_track_to_kml(txt_path, output_kml, color, name="Track", step=10):
    kml = simplekml.Kml()
    ls = kml.newlinestring(name=name)
    ls.altitudemode = simplekml.AltitudeMode.absolute
    ls.style.linestyle.color = color
    ls.style.linestyle.width = 4

    with open(txt_path, "r") as f:
        for i, line in enumerate(f):
            if i % step != 0: continue  # ← 只取每 step 个点
            parts = line.strip().split()
            if len(parts) < 7: continue
            _, lon, lat, alt, *_ = parts
            ls.coords.addcoordinates([(float(lon), float(lat), float(alt))])

    kml.save(output_kml)
    print(f"[INFO] KML saved to {output_kml} | step={step}")


import json
from datetime import datetime, timedelta
def generate_trajectory_js(txt_path, output_js, step=100):
    with open(txt_path, "r") as f:
        lines = [ln.strip() for ln in f if len(ln.strip().split()) >= 4]

    out = []
    for i, line in enumerate(lines[::step]):
        
        _, lon, lat, alt, *_ = line.split()
        out.append(f"[{i}, {lon}, {lat}, {alt}]")

    with open(output_js, "w") as f:
        f.write("const trajectory = [\n  " + ",\n  ".join(out) + "\n];\n")

def convert_txt_to_czml(txt_path, czml_path, start_time="2025-01-01T00:00:00Z"):
    with open(txt_path, "r") as f:
        lines = [ln.strip() for ln in f if len(ln.strip().split()) >= 4]

    t0 = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%SZ")
    carto = []

    for i, line in enumerate(lines):
        parts = line.split()
        _, lon, lat, alt = parts[:4]
        dt = (t0 + timedelta(seconds=i)).isoformat() + "Z"
        carto.extend([i, float(lon), float(lat), float(alt)])

    czml = [
        {
            "id": "document",
            "name": "Trajectory",
            "version": "1.0",
            "clock": {
                "interval": f"{start_time}/{(t0 + timedelta(seconds=len(lines))).isoformat()}Z",
                "currentTime": start_time,
                "multiplier": 1,
                "loopMode": "LOOP"
            }
        },
        {
            "id": "est_path",
            "name": "EST Trajectory",
            "availability": f"{start_time}/{(t0 + timedelta(seconds=len(lines))).isoformat()}Z",
            "polyline": {
                "positions": {
                    "epoch": start_time,
                    "cartographicDegrees": carto
                },
                "material": {
                    "solidColor": {"color": {"rgba": [255, 255, 0, 255]}}
                },
                "width": 3
            }
        }
    ]

    with open(czml_path, "w") as f:
        json.dump(czml, f, indent=2)

    print(f"[INFO] Saved to {czml_path}")


if __name__=="__main__":
    txt_path = r"Switzerland_seq1.txt"
    output_kml = "batch_fov_pyramids.kml"
    batch_fov_visualization(txt_path, output_kml)
    print("Done. Open 'batch_fov_pyramids.kml' in Google Earth to see multiple FOV pyramids.")
