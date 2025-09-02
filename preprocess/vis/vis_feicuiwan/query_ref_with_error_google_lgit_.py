import cv2
import os
import numpy as np
from transform import WGS84_to_ECEF
from scipy.spatial.transform import Rotation as R
from get_depth import get_3D_samples_v2, get_points2D_ECEF_projection
from transform import get_matrix, WGS84_to_ECEF ,get_rotation_enu_in_ecef,visualize_matches
from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2
# 配置

def euler_angles_to_matrix_ECEF_w2c(euler_angles, trans):
    lon, lat, _ = trans
    rot_pose_in_enu = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
    rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_enu_in_ecef, rot_pose_in_enu)
    t_c2w = WGS84_to_ECEF(trans)
    
    # R_w2c_in_ecef = R_c2w.transpose() # 和enu的差异是第二行和第三行取负号
    # t_w2c = -R_w2c_in_ecef.dot(t_c2w)

    # T_render_in_ECEF_w2c = np.eye(4)
    # T_render_in_ECEF_w2c[:3, :3] = R_w2c_in_ecef
    # T_render_in_ECEF_w2c[:3, 3] = t_w2c
    return R_c2w
def generate_full_checkerboard(ref_img, query_img, rows, cols):
    h, w = ref_img.shape[:2]
    cell_h, cell_w = h // rows, w // cols
    output = np.zeros_like(ref_img)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            if (r + c) % 2 == 0:
                output[y0:y1, x0:x1] = query_img[y0:y1, x0:x1]
            else:
                output[y0:y1, x0:x1] = ref_img[y0:y1, x0:x1]
    return output
def generate_checkerboard_overlay_transparent_v2(ref_crop, query_crop, rows, cols, alpha=0.7):
    """
    生成带透明度叠加的 checkerboard 图像。
    
    Parameters:
        ref_crop:   底图（numpy array）
        query_crop: 顶图（numpy array）
        rows:       网格行数
        cols:       网格列数
        alpha:      顶图透明度（0~1），越大越偏向 query_crop
    
    Returns:
        output:     合成后的 checkerboard 图像
    """
    h, w = ref_crop.shape[:2]
    cell_h, cell_w = h // rows, w // cols
    output = np.zeros_like(ref_crop, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            ref_block = ref_crop[y0:y1, x0:x1].astype(np.float32)
            query_block = query_crop[y0:y1, x0:x1].astype(np.float32)
            if (r + c) % 2 == 0:
                mix = alpha * query_block + (1 - alpha) * ref_block
            else:
                mix = (1 - alpha) * query_block + alpha * ref_block
            output[y0:y1, x0:x1] = mix.astype(np.uint8)
    return output
def generate_checkerboard_overlay_transparent(ref_crop, query_crop, rows, cols, alpha=0.5):
    """
    生成带透明度叠加的 checkerboard 图像。
    
    Parameters:
        ref_crop:   底图（numpy array）
        query_crop: 顶图（numpy array）
        rows:       网格行数
        cols:       网格列数
        alpha:      顶图透明度（0~1），越大越偏向 query_crop
    
    Returns:
        output:     合成后的 checkerboard 图像
    """
    h, w = ref_crop.shape[:2]
    cell_h, cell_w = h // rows, w // cols
    output = np.zeros_like(ref_crop, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            ref_block = ref_crop[y0:y1, x0:x1].astype(np.float32)
            query_block = query_crop[y0:y1, x0:x1].astype(np.float32)
            if (r + c) % 2 == 0:
                mix = (1 - alpha) * query_block + alpha * ref_block
            else:
                mix = query_block 
            output[y0:y1, x0:x1] = mix.astype(np.uint8)
    return output
# def generate_checkerboard_overlay(ref_crop, query_crop, rows, cols):
#     h, w = ref_crop.shape[:2]
#     cell_h, cell_w = h // rows, w // cols
#     output = np.zeros_like(ref_crop)
#     for r in range(rows):
#         for c in range(cols):
#             y0, y1 = r * cell_h, (r + 1) * cell_h
#             x0, x1 = c * cell_w, (c + 1) * cell_w
#             if (r + c) % 2 == 0:
#                 output[y0:y1, x0:x1] = query_crop[y0:y1, x0:x1]
#             else:
#                 output[y0:y1, x0:x1] = ref_crop[y0:y1, x0:x1]
#     return output
def draw_curved_arrow(img, pt1, pt2, color=(0, 255, 0), thickness=2, arrow_size=8, bend=40):
    """
    绘制从 pt1 到 pt2 的一段平滑弯曲箭头。
    - pt1, pt2: 起点终点 (x, y)
    - bend: 曲线偏移量（越大越弯）
    """
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    direction = pt2 - pt1
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return  # 距离太近不画

    # 法向方向
    direction /= length
    ortho = np.array([-direction[1], direction[0]])

    # 控制点：从中点往外偏移
    midpoint = (pt1 + pt2) / 2
    ctrl = midpoint + ortho * bend

    # 三次贝塞尔插值：pt1 → ctrl → pt2
    curve = []
    for t in np.linspace(0, 1, 60):
        p = (
            (1 - t) ** 2 * pt1 +
            2 * (1 - t) * t * ctrl +
            t ** 2 * pt2
        )
        curve.append(tuple(np.round(p).astype(int)))

    # 画曲线
    for i in range(len(curve) - 1):
        cv2.line(img, curve[i], curve[i + 1], color, thickness, cv2.LINE_AA)

    # 箭头头部（末尾方向）
    p_tip = np.array(curve[-1], dtype=np.float32)
    p_base = np.array(curve[-4], dtype=np.float32)
    dir_vec = p_tip - p_base
    dir_vec /= (np.linalg.norm(dir_vec) + 1e-6)
    ortho = np.array([-dir_vec[1], dir_vec[0]])

    left = p_tip - dir_vec * arrow_size + ortho * arrow_size * 0.5
    right = p_tip - dir_vec * arrow_size - ortho * arrow_size * 0.5
    triangle = np.array([p_tip, left, right], dtype=np.int32)
    cv2.fillPoly(img, [triangle], color)
    pt1i = tuple(np.round(pt1).astype(int))
    pt2i = tuple(np.round(pt2).astype(int))
    cv2.circle(img, pt1i, 5, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, pt1i, 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt2i, 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt2i, 3, (0, 0, 0), -1, cv2.LINE_AA)
def draw_variable_curve_arrow(img, pt1, pt2,
                              color=(0,255,0),
                              thickness=2,
                              arrow_size=8,
                              max_dist_for_max_bend=150,
                              min_bend_frac=0.05,
                              max_bend_frac=0.5,
                              small_thresh=10,
                              small_scale=5,
                              arc_direction='left',
                              n_pts=60):
    """
    小距离 (< small_thresh) 时：绘制一个完整的 360° 圆环（半径 = max(d, small_thresh) * small_scale），
    圆环上恰好包含 pt1 和 pt2，并在 pt2 处画箭头指向圆切线方向。
    大距离 (>= small_thresh) 时：保持原来的可变弯度二次 Bézier 箭头。
    """
    p1 = np.array(pt1, dtype=np.float32)
    p2 = np.array(pt2, dtype=np.float32)
    if np.linalg.norm(p2 - p1) < 1.0:
        direction = np.array([1.0, 0.0])
        p2 += direction * 1.0
    v = p2 - p1
    d = np.linalg.norm(v)
    if d < 1e-3:
        return

    # --- 小距离：绘制完整圆环 + 箭头 ---
    if d < small_thresh:
        # 1) 计算半径和圆心
        R = min(min(d, small_thresh) * small_scale, 10)
        u = v / d
        ortho = np.array([-u[1], u[0]])
        if arc_direction != 'left':
            ortho = -ortho
        midpoint = (p1 + p2) * 0.5
        # 圆心在中点沿法线方向的 h 距离处，使圆经过 p1 和 p2
        h = np.sqrt(max(R*R - (d/2)**2, 0.0))
        center = midpoint + ortho * h

        # 2) 采样整个圆
        circle_pts = [
            (
                int(center[0] + R * np.cos(theta)),
                int(center[1] + R * np.sin(theta))
            )
            for theta in np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        ]
        # 3) 画圆环（闭合）
        for i in range(len(circle_pts)):
            p_curr = circle_pts[i]
            p_next = circle_pts[(i+1) % len(circle_pts)]
            cv2.line(img, p_curr, p_next, color, thickness, cv2.LINE_AA)

        # 4) 在 p2 处画箭头，方向取圆切线方向
        # 计算 p2 在圆上的角度
        ang2 = np.arctan2(p2[1] - center[1], p2[0] - center[0])
        # 切线方向：derivative of (cos, sin) is (-sin, cos)
        tan = np.array([-np.sin(ang2), np.cos(ang2)])
        if arc_direction != 'left':
            tan = -tan
        tan /= (np.linalg.norm(tan) + 1e-6)
        # 构造三角箭头
        tip   = p2
        base  = tip - tan * arrow_size
        ortan = np.array([-tan[1], tan[0]])
        left  = base + ortan * (arrow_size*0.5)
        right = base - ortan * (arrow_size*0.5)
        tri = np.array([tip, left, right], dtype=np.int32)
        # cv2.fillPoly(img, [tri], color)

        # 5) 起、终点高亮
        
        p1i = tuple(p1.astype(int))
        p2i = tuple(p2.astype(int))
        cv2.circle(img, p1i, 3, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(img, p1i, 5, (0,0,0), -1, cv2.LINE_AA)
        cv2.circle(img, p1i, 3, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(img, p2i, 5, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(img, p2i, 3, (0,0,0), -1, cv2.LINE_AA)
        return

    # --- 大距离：原可变弯度 Bézier 箭头 ---
    t = np.clip(d / max_dist_for_max_bend, 0.0, 1.0)
    bend_frac = (1 - t) * (max_bend_frac - min_bend_frac) + min_bend_frac
    u = v / d
    ortho = np.array([-u[1], u[0]])
    mid = (p1 + p2) * 0.5
    ctrl = mid + ortho * (bend_frac * d)

    curve = []
    for tt in np.linspace(0, 1, n_pts):
        pt = (1-tt)**2 * p1 + 2*(1-tt)*tt * ctrl + tt**2 * p2
        curve.append((int(pt[0]+0.5), int(pt[1]+0.5)))
    for i in range(len(curve)-1):
        cv2.line(img, curve[i], curve[i+1], color, thickness, cv2.LINE_AA)

    if len(curve) >= 4:
        p_tip  = np.array(curve[-1], dtype=np.float32)
        p_base = np.array(curve[-4], dtype=np.float32)
        dv = p_tip - p_base
        dv /= (np.linalg.norm(dv) + 1e-6)
        ort = np.array([-dv[1], dv[0]])
        left  = p_tip - dv*arrow_size + ort*arrow_size*0.5
        right = p_tip - dv*arrow_size - ort*arrow_size*0.5
        tri = np.array([p_tip, left, right], dtype=np.int32)
        # cv2.fillPoly(img, [tri], color)

    # 高亮端点
    p1i = tuple(p1.astype(int))
    p2i = tuple(p2.astype(int))
    cv2.circle(img, p1i, 5, (0,0,0), -1, cv2.LINE_AA)
    cv2.circle(img, p1i, 3, (255,255,255), -1, cv2.LINE_AA)
    cv2.circle(img, p2i, 5, (255,255,255), -1, cv2.LINE_AA)
    cv2.circle(img, p2i, 3, (0,0,0), -1, cv2.LINE_AA)



def draw_strong_curved_arrow(img, pt1, pt2, color, thickness=2, arrow_size=6, bend_amplitude=40):
    """
    画从 pt1 到 pt2 的强弯贝塞尔箭头，保持头尾坐标不变。
    """
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    pt2[0] +=1
    # 避免重复点
    if np.linalg.norm(pt2 - pt1) < 1.0:
        direction = np.array([1.0, 0.0])
        pt2 += direction * 1.0

    direction = pt2 - pt1
    length = np.linalg.norm(direction)
    direction = direction / (length + 1e-6)
    ortho = np.array([-direction[1], direction[0]])

    # 控制点为中点基础上弯出
    mid = (pt1 + pt2) / 2
    ctrl1 = mid + ortho * bend_amplitude
    ctrl2 = mid + ortho * bend_amplitude  # 双控制点同位置 → 更平滑拱形

    # 插值路径
    curve = []
    for t in np.linspace(0, 1, 60):
        p = (
            (1 - t) ** 3 * pt1 +
            3 * (1 - t) ** 2 * t * ctrl1 +
            3 * (1 - t) * t ** 2 * ctrl2 +
            t ** 3 * pt2
        )
        curve.append((int(p[0] + 0.5), int(p[1] + 0.5)))

    # 画曲线
    for i in range(len(curve) - 1):
        cv2.line(img, curve[i], curve[i + 1], color, thickness, cv2.LINE_AA)

    # 画箭头头部
    if len(curve) >= 4:
        p_tip = np.array(curve[-1], dtype=np.float32)
        p_base = np.array(curve[-4], dtype=np.float32)
        dir_vec = p_tip - p_base
        dir_vec /= (np.linalg.norm(dir_vec) + 1e-6)
        ortho = np.array([-dir_vec[1], dir_vec[0]])
        p_left = p_tip - dir_vec * arrow_size + ortho * arrow_size * 0.6
        p_right = p_tip - dir_vec * arrow_size - ortho * arrow_size * 0.6
        triangle = np.array([p_tip, p_left, p_right], dtype=np.int32)
        cv2.fillPoly(img, [triangle], color)

    # 起点终点：清晰视觉标记
    pt1i = tuple(np.round(pt1).astype(int))
    pt2i = tuple(np.round(pt2).astype(int))
    cv2.circle(img, pt1i, 5, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, pt1i, 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt2i, 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt2i, 3, (0, 0, 0), -1, cv2.LINE_AA)


def draw_curved_arrow_with_endpoints_v2(img, pt1, pt2, color, thickness=2, arrow_size=6, bend_amplitude=40):
    """
    固定 pt1, pt2 的前提下，绘制夸张弯曲箭头（S型）。
    bend_amplitude: 控制中间“扭出来”的弯曲强度，单位为像素
    """
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
     # === 最小误差可视化处理 ===
    display_pt2 = np.copy(pt2)
    if np.linalg.norm(pt2 - pt1) < 1.0:
        direction = pt2 - pt1
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([1.0, 0.0])
        else:
            direction = direction / np.linalg.norm(direction)
        display_pt2 = pt1 + direction * 1.0  # 至少偏 1 像素
    center = 0.5 * (pt1 + pt2)

    # 构造中间控制点，使曲线强烈弯折
    direction = pt2 - pt1
    length = np.linalg.norm(direction)
    if length < 1e-6:
        direction = np.array([1.0, 0.0])
        length = 1.0
    else:
        direction /= length
    ortho = np.array([-direction[1], direction[0]])

    ctrl = center + ortho * bend_amplitude  # 把曲线“拱出来”！

    # 贝塞尔曲线（可改成三次，但二次就很好）
    curve = []
    for t in np.linspace(0, 1, 40):
        point = (1 - t)**2 * pt1 + 2 * (1 - t) * t * ctrl + t**2 * pt2
        # curve.append(tuple(np.round(point).astype(int)))
        curve.append((int(point[0] + 0.5), int(point[1] + 0.5)))  # 更细腻的手动圆整

    # 曲线主线
    for i in range(len(curve) - 1):
        cv2.line(img, curve[i], curve[i + 1], color, thickness, cv2.LINE_AA)

    # 箭头头部（保持方向）
    if len(curve) >= 4:
        p_tip = np.array(curve[-1], dtype=np.float32)
        p_base = np.array(curve[-4], dtype=np.float32)
        dir_vec = p_tip - p_base
        dir_vec /= (np.linalg.norm(dir_vec) + 1e-6)
        ortho = np.array([-dir_vec[1], dir_vec[0]])

        p_left = p_tip - dir_vec * arrow_size + ortho * arrow_size * 0.6
        p_right = p_tip - dir_vec * arrow_size - ortho * arrow_size * 0.6
        triangle = np.array([p_tip, p_left, p_right], dtype=np.int32)
        cv2.fillPoly(img, [triangle], color)

    # 起点终点圆圈
    pt1i = tuple(np.round(pt1).astype(int))
    pt2i = tuple(np.round(pt2).astype(int))
    cv2.circle(img, pt1i, 5, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, pt1i, 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt2i, 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt2i, 3, (0, 0, 0), -1, cv2.LINE_AA)
def draw_shadow_text(img, text, org, font_scale=1.0):
    x, y = org
    cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1, cv2.LINE_AA)
def get_points_by_mouse(img, save_path=None, max_points=10):
    """
    鼠标点击选取图像中的像素点，并可保存为 .npy / .txt 文件。
    :param img: 图像（BGR）
    :param save_path: 保存路径（.npy 或 .txt）
    :param max_points: 最多点击的点数
    :return: numpy array, shape=(N, 2)
    """
    clicked_points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < max_points:
            clicked_points.append((x, y))
            print(f"[Click] Point {len(clicked_points)}: ({x}, {y})")
            cv2.circle(img, (x, y), 4, (0, 255, 255), -1)
            cv2.imshow("Click Points", img)

    print(f"请在图像中点击 {max_points} 个点...")
    clone = img.copy()
    cv2.imshow("Click Points", clone)
    cv2.setMouseCallback("Click Points", click_event)

    while len(clicked_points) < max_points:
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyWindow("Click Points")
    points = np.array(clicked_points)

    if save_path:
        if save_path.endswith(".npy"):
            np.save(save_path, points)
        elif save_path.endswith(".txt"):
            np.savetxt(save_path, points, fmt='%d')
        print(f"✅ 选点已保存到 {save_path}")

    return points

def draw_curved_arrow_with_endpoints(img, pt1, pt2, color, thickness=2, arrow_size=6):
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    vec = pt2 - pt1
    length = np.linalg.norm(vec)
    if length < 1:
        return

    # 自动拉长短箭头，防止看不见（保留方向）
    MIN_ARROW_LEN = 10.0
    if length < MIN_ARROW_LEN:
        vec = vec / (length + 1e-6)
        pt2 = pt1 + vec * MIN_ARROW_LEN
        length = MIN_ARROW_LEN

    # === 弯曲控制 ===
    unit = vec / length
    ortho = np.array([-unit[1], unit[0]])
    
    # 弯曲幅度：离得越近，弯得越多；最远也不过偏移15%
    bend_strength = min(0.25, 50.0 / length)  # 最多 25% 弯度，太远就不弯
    ctrl = pt1 + 0.5 * vec + ortho * (bend_strength * length)

    # 二次贝塞尔曲线
    curve = []
    for t in np.linspace(0, 1, 30):
        point = (1 - t) ** 2 * pt1 + 2 * (1 - t) * t * ctrl + t ** 2 * pt2
        curve.append(tuple(np.round(point).astype(int)))

    # 画主线
    for i in range(len(curve) - 1):
        cv2.line(img, curve[i], curve[i + 1], color, thickness, cv2.LINE_AA)

    # 画箭头三角头
    if len(curve) >= 4:
        p_tip = np.array(curve[-1], dtype=np.float32)
        p_base = np.array(curve[-4], dtype=np.float32)
        dir_vec = p_tip - p_base
        dir_vec /= (np.linalg.norm(dir_vec) + 1e-6)
        ortho = np.array([-dir_vec[1], dir_vec[0]])

        p_left = p_tip - dir_vec * arrow_size + ortho * arrow_size * 0.6
        p_right = p_tip - dir_vec * arrow_size - ortho * arrow_size * 0.6
        triangle = np.array([p_tip, p_left, p_right], dtype=np.int32)
        cv2.fillPoly(img, [triangle], color)

    # 起点终点圆圈（清晰黑白）
    pt1i = tuple(np.round(pt1).astype(int))
    pt2i = tuple(np.round(pt2).astype(int))
    cv2.circle(img, pt1i, 5, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, pt1i, 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt2i, 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt2i, 3, (0, 0, 0), -1, cv2.LINE_AA)      
def load_pose(file_path):
    xyz = []
    angles = []
    timestamps = []
    ref_T_list = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                name = parts[0]
                if "_" in name:
                    frame_idx = int(name.split("_")[0])
                lon, lat, alt = map(float, parts[1:4])
                roll, pitch, yaw = map(float, parts[4:7])
                xyz.append(WGS84_to_ECEF([lon, lat, alt]))
                e = [pitch, roll, yaw]
                t = [lon, lat, alt]
                R_c2w = euler_angles_to_matrix_ECEF_w2c(e, t)
                angles.append(R_c2w)
                # angles.append([pitch, roll, yaw])
                timestamps.append(frame_idx)
                
                ref_T = get_matrix(t, e, origin=[0,0,0], mode='c2w')
                ref_T_list.append(ref_T)
    return np.array(timestamps), np.array(xyz), np.array(angles), np.array(ref_T_list)
methods = ['GeoPixel', 'Render2Loc', 'Render2ORB', 'Render2RAFT', 'PixLoc']
methods = ['PixLoc']

points_npy = "/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif/vis_video_with_error/clicked_points.npy"
for method in methods:
    ref_dir = "/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif/GT"
    query_dir = os.path.join("/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif/", method)
    output_path = os.path.join("/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif/vis/" )
    query_pose_file = os.path.join(query_dir, "USA_seq5@8@foggy@200.txt")
    ref_pose_file = os.path.join(ref_dir, "USA_seq5@8@cloudy@300-100@200.txt")
    start_idx, end_idx = 775, 779

    # 输出尺寸（与输入图片保持一致）
    W, H = 960, 540
    crop_w, crop_h = 720, 405


    x_off = (W - crop_w) // 2
    y_off = (H - crop_h) // 2
    # 棋盘格参数（行列数）
    grid_rows, grid_cols = 3, 5  # 共 96 块格子

    gt_timestamps, gt_xyz, gt_angles, gt_T_list = load_pose(ref_pose_file)
    es_timestamps, es_xyz, es_angles, es_T_list = load_pose(query_pose_file)
    ang_err = []
    pos_err = []
    for i, fid in enumerate(es_timestamps):
        cos = np.clip((np.trace(np.dot(gt_angles[fid].T, es_angles[i])) - 1) / 2, -1., 1.)
        e_R = np.rad2deg(np.abs(np.arccos(cos)))
        ang_err.append(e_R)
        err = np.linalg.norm(es_xyz[i] - gt_xyz[fid])
        pos_err.append(err)


    # 主循环
    for i in range(start_idx, end_idx + 1):
        ref_path = os.path.join(ref_dir, f"{i}_0.png")
        query_path = os.path.join(query_dir, f"{i}_0.png")
        ref = cv2.imread(ref_path)
        qry = cv2.imread(query_path)
        if qry is None :
            print(f"[跳过] 帧 {i}")
            continue
    #---
        rcamera = [1920, 1080, 2317.6, 2317.6, 960.0, 540.0]
        qcamera = [960, 540, 1158.8, 1158.8, 480.0, 270.0]
        
        ref_depth_image = np.load(ref_path[:-4]+'.npy')
        ref_T = gt_T_list[np.where(gt_timestamps == i)][0]
        query_T = es_T_list[np.where(es_timestamps == i)][0]
        # ey = np.random.randint(0, ref.shape[0], size= 20)
        # ex = np.random.randint(0, ref.shape[1], size= 20)
        # points2d_ref = np.column_stack((ex, ey)) 
        click_save_path = points_npy
        if os.path.exists(click_save_path):
            print(f"📂 已检测到保存的选点，读取中...")
            points2d_ref = np.load(click_save_path)
        else:
            points2d_ref = get_points_by_mouse(ref.copy(), save_path=click_save_path, max_points=8)

        points2d_ref_valid, point3D_from_ref, _= get_3D_samples_v2(points2d_ref, ref_depth_image, ref_T, rcamera)
        points2d_query, _, Points_3D_ECEF_origin, valid = get_points2D_ECEF_projection(np.array(query_T), qcamera, point3D_from_ref, points2d_ref_valid)
        # print(point3D_from_ref[0])
        
        point3D_from_ref_like_query = points2d_query
        points2d_ref_valid, point3D_from_ref_like_query, _= get_3D_samples_v2(point3D_from_ref_like_query, ref_depth_image, ref_T, rcamera)
        point3D_ref_valid = point3D_from_ref[valid]
        points2d_ref = points2d_ref[valid]
        point3D_query_valid = point3D_from_ref_like_query
        points2d_ref = points2d_ref  / 2
        errors_3d = np.linalg.norm(points2d_ref - points2d_query, axis=1)
        
        # errors_3d = np.linalg.norm(point3D_ref_valid - point3D_query_valid, axis=1)
        # vis_save_path = "/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif"
        # visualize_matches(qry, ref, 
        #                 points2d_query, 
        #                 points2d_ref, vis_save_path=vis_save_path)
    #----
        
        
        # 棋盘格交错融合
        ref = cv2.resize(ref, (W, H))
        qry = cv2.resize(qry, (W, H))
        ref_crop = ref[y_off:y_off+crop_h, x_off:x_off+crop_w]
        qry_crop = qry[y_off:y_off+crop_h, x_off:x_off+crop_w]

        # === 构造棋盘格混合区域 ===
        checker_crop = generate_checkerboard_overlay_transparent(ref_crop, qry_crop, grid_rows, grid_cols)
        # checker_crop = generate_checkerboard_overlay(ref_crop, qry_crop, grid_rows, grid_cols)

        # === 放入 frame 中间 ===
        frame = qry.copy()
        frame[y_off:y_off+crop_h, x_off:x_off+crop_w] = checker_crop
        # === 叠加误差信息 ===
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        shadow_offset = 1

        # 获取当前帧误差
        if i in es_timestamps:
            idx = np.where(es_timestamps == i)[0][0]
            err_pos = pos_err[idx]
            err_ang = ang_err[idx]

            pos_str = f"Pos Error: {err_pos:.2f} m"
            ang_str = f"Angle Error: {err_ang:.2f} deg"

            # 位置1
            pos_xy = (30, 40)
            ang_xy = (30, 80)

            # 阴影背景
            cv2.putText(frame, pos_str, (pos_xy[0]+shadow_offset, pos_xy[1]+shadow_offset),
                        font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(frame, ang_str, (ang_xy[0]+shadow_offset, ang_xy[1]+shadow_offset),
                        font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            
            # 主文字
            cv2.putText(frame, pos_str, pos_xy, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            cv2.putText(frame, ang_str, ang_xy, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            if i % 30 ==0 and ('ORB' in query_dir or 'RAFT' in query_dir):
                font_scale_key = 1.1
                thickness_key = 3
                color_key = (0, 255, 255)  # 明黄色
                text = "Keyframe"

                # 获取文字宽高
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale_key, thickness_key)

                # 右下角坐标（偏移 20 像素）
                x = frame.shape[1] - text_w - 20
                y = frame.shape[0] - 20

                # 绘制文字
                cv2.putText(frame, text, (x, y),
                            font, font_scale_key, color_key, thickness_key, cv2.LINE_AA)
                # === 可选：小图显示 Query 全图 ===
        # pip_w, pip_h = 320, 180
        # pip_x = W - pip_w - 20
        # pip_y = H - pip_h - 20
        # qry_pip = cv2.resize(qry, (pip_w, pip_h))
        # cv2.rectangle(frame, (pip_x-2, pip_y-2), (pip_x+pip_w+2, pip_y+pip_h+2), (255,255,255), 2)
        # frame[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w] = qry_pip
        name = ref_dir.split('/')[-2]
        # cv2.imwrite(f'{output_path}/{i}_0.png', frame)
        # frame = cv2.resize(frame, (480, 270))
        
        #----
        # 可选缩放（你有 /2 操作，确认是否一致）
        scale_x = frame.shape[1] / ref.shape[1]
        scale_y = frame.shape[0] / ref.shape[0]
        pts_ref_scaled = points2d_ref * np.array([scale_x, scale_y])
        pts_query_scaled = points2d_query * np.array([scale_x, scale_y])
        # pts_ref_scaled -= np.array([x_off * scale_x, y_off * scale_y])
        # pts_query_scaled -= np.array([x_off * scale_x, y_off * scale_y])

        # 有效像素筛选
        h_img, w_img = frame.shape[:2]
        in_ref = (pts_ref_scaled[:, 0] >= 0) & (pts_ref_scaled[:, 0] < w_img) & (pts_ref_scaled[:, 1] >= 0) & (pts_ref_scaled[:, 1] < h_img)
        in_qry = (pts_query_scaled[:, 0] >= 0) & (pts_query_scaled[:, 0] < w_img) & (pts_query_scaled[:, 1] >= 0) & (pts_query_scaled[:, 1] < h_img)
        valid_mask = in_ref & in_qry

        pts_ref_visible = pts_ref_scaled[valid_mask]
        pts_query_visible = pts_query_scaled[valid_mask]
        errors_3d_visible = errors_3d[valid_mask]

        # 设置映射范围为 0~10 米
        vmin, vmax = 0.0, 3.0
        norm_err = np.clip((errors_3d_visible - vmin) / (vmax - vmin), vmin, vmax)
        cmap = cm.get_cmap('jet')
    
        # for (x1, y1), (x2, y2), err_norm in zip(pts_query_visible, pts_ref_visible, norm_err):
        #     pt1 = tuple(np.round([x1, y1]).astype(int))
        #     pt2 = tuple(np.round([x2, y2]).astype(int))
        #     color = tuple(int(255 * c) for c in cmap(err_norm)[:3][::-1])  # RGB → BGR

        #     draw_variable_curve_arrow(frame, pt1, pt2, color, thickness=2)
        # 保存结果图（无 colorbar）
        method = query_dir.split('/')[-1]
        # cv2.imwrite(f'{output_path}/{i}_{method}.png', frame)
        name = ref_dir.split('/')[-2]
        frame = cv2.resize(frame, (480, 270))
        cv2.imwrite(f'{output_path}/{i}_0.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f'{output_path}/{i}_{method}.png')





print(f"✅ 完成：视频已保存到 {output_path}")
