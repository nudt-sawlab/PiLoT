import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import simplekml

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
    dist_forward=2.0, width=4.0, height=4.0,
    cm_vmin=0.0, cm_vmax=5.0,      # 颜色条范围（m）
    face_alpha=90, line_alpha=200,
    step=10,                         # 隔一帧取一帧
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
        if err > 5:
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
   

if __name__=="__main__":
    txt_path = r"Switzerland_seq1.txt"
    output_kml = "batch_fov_pyramids.kml"
    batch_fov_visualization(txt_path, output_kml)
    print("Done. Open 'batch_fov_pyramids.kml' in Google Earth to see multiple FOV pyramids.")
