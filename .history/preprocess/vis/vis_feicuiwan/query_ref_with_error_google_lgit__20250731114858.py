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
ref_dir = "/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif/GT"
query_dir = "/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif/PixLoc"
output_path = "/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif/vis_video_with_error/vis"
query_pose_file = os.path.join(query_dir, "USA_seq5@8@foggy@200.txt")
ref_pose_file = os.path.join(ref_dir, "USA_seq5@8@cloudy@300-100@200.txt")
start_idx, end_idx = 777, 777
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
def generate_checkerboard_overlay_transparent(ref_crop, query_crop, rows, cols, alpha=0.75):
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
def draw_strong_curved_arrow(img, pt1, pt2, color, thickness=2, arrow_size=6, bend_amplitude=40):
    """
    在 pt1 和 pt2 之间，生成夸张弯折但头尾固定的箭头。
    如果像素坐标重合，自动偏移终点方向，以保证可视化可见。
    """
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)

    # 判断像素坐标是否重合，若重合则略微偏移 pt2 方向
    pt1_int = tuple(np.round(pt1).astype(int))
    pt2_int = tuple(np.round(pt2).astype(int))
    if pt1_int == pt2_int:
        direction = pt2 - pt1
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([1.0, 0.0])  # 默认方向
        else:
            direction = direction / np.linalg.norm(direction)
        pt2 = pt2 + direction * 1.0  # 显示上偏移 1 像素

    # 更新方向与垂直方向
    direction = pt2 - pt1
    length = np.linalg.norm(direction)
    if length < 1e-6:
        direction = np.array([1.0, 0.0])
        length = 1.0
    else:
        direction = direction / length
    ortho = np.array([-direction[1], direction[0]])

    # 控制点形成弯曲效果
    ctrl1 = pt1 + 0.3 * (pt2 - pt1) + ortho * bend_amplitude
    ctrl2 = pt1 + 0.7 * (pt2 - pt1) - ortho * bend_amplitude

    
    # 三次贝塞尔曲线插值
    curve = []
    for t in np.linspace(0, 1, 60):
        p = (
            (1 - t) ** 3 * pt1 +
            3 * (1 - t) ** 2 * t * ctrl1 +
            3 * (1 - t) * t ** 2 * ctrl2 +
            t ** 3 * pt2
        )
        curve.append((int(p[0] + 0.5), int(p[1] + 0.5)))  # 保留细节
    # 主曲线
    for i in range(len(curve) - 1):
        cv2.line(img, curve[i], curve[i + 1], color, thickness, cv2.LINE_AA)

    # 箭头三角
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

    # 圆点标记（真实点）
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
# def draw_curved_arrow_with_endpoints(img, pt1, pt2, color, thickness=2, arrow_size=6):
#     """
#     使用贝塞尔曲线 + 圆圈标记起点终点 + 自定义箭头头部。
#     pt1: 起点（query）
#     pt2: 终点（ref）
#     """
#     pt1 = np.array(pt1, dtype=np.float32)
#     pt2 = np.array(pt2, dtype=np.float32)
#     vec = pt2 - pt1
#     length = np.linalg.norm(vec)
#     if length < 1:
#         return

#     # 画圆圈：起点红圈，终点绿圈
#     # 起点（query）：白色圆 + 黑色描边
#     pt1i = tuple(np.round(pt1).astype(int))
#     cv2.circle(img, pt1i, 5, (0, 0, 0), -1, cv2.LINE_AA)   # 黑边
#     cv2.circle(img, pt1i, 3, (255, 255, 255), -1, cv2.LINE_AA)  # 白心

#     # 终点（ref）：黑色圆 + 白色描边（反过来）
#     pt2i = tuple(np.round(pt2).astype(int))
#     cv2.circle(img, pt2i, 5, (255, 255, 255), -1, cv2.LINE_AA)   # 白边
#     cv2.circle(img, pt2i, 3, (0, 0, 0), -1, cv2.LINE_AA)  # 黑心
#     # 控制点（向垂直方向稍微偏移，形成曲线）
#     unit = vec / length
#     ortho = np.array([-unit[1], unit[0]])
#     ctrl = pt1 + 0.5 * vec + ortho * (0.15 * length)  # 控制点偏移程度可调

#     # 生成二次贝塞尔曲线点
#     curve = []
#     for t in np.linspace(0, 1, 30):
#         point = (1 - t) ** 2 * pt1 + 2 * (1 - t) * t * ctrl + t ** 2 * pt2
#         curve.append(tuple(np.round(point).astype(int)))

#     # 画曲线
#     for i in range(len(curve) - 1):
#         cv2.line(img, curve[i], curve[i + 1], color, thickness, cv2.LINE_AA)

#     # 箭头头部（三角形）
#     if len(curve) >= 3:
#         p_tip = np.array(curve[-1], dtype=np.float32)
#         p_base = np.array(curve[-4], dtype=np.float32)
#         dir_vec = p_tip - p_base
#         dir_vec /= (np.linalg.norm(dir_vec) + 1e-6)
#         ortho = np.array([-dir_vec[1], dir_vec[0]])

#         p_left = p_tip - dir_vec * arrow_size + ortho * arrow_size * 0.6
#         p_right = p_tip - dir_vec * arrow_size - ortho * arrow_size * 0.6
#         triangle = np.array([p_tip, p_left, p_right], dtype=np.int32)
#         cv2.fillPoly(img, [triangle], color)
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
    click_save_path = os.path.join(output_path, "clicked_points.npy")
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
    errors_3d = np.linalg.norm(point3D_ref_valid - point3D_query_valid, axis=1)
    # vis_save_path = "/mnt/sda/MapScape/query/estimation/mp4_compare/USA_seq5_foggy_gif"
    # visualize_matches(qry, ref, 
    #                 points2d_query, 
    #                 points2d_ref, vis_save_path=vis_save_path)
#----
    points2d_ref = points2d_ref  / 2
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
    vmin, vmax = 0.0, 300.0
    norm_err = np.clip((errors_3d_visible - vmin) / (vmax - vmin), vmin, vmax)
    cmap = cm.get_cmap('jet')
    arrow_thickness = 3  # 更细

    # OpenCV 绘制箭头（从 query → ref）
    # for (x1, y1), (x2, y2), err_norm in zip(pts_query_visible, pts_ref_visible, norm_err):
    #     pt1 = tuple(np.round([x1, y1]).astype(int))
    #     pt2 = tuple(np.round([x2, y2]).astype(int))
    #     color = tuple(int(255 * c) for c in cmap(err_norm)[:3][::-1])  # RGB → BGR
    #     cv2.arrowedLine(frame, pt1, pt2, color, arrow_thickness, tipLength=0.25)
    for (x1, y1), (x2, y2), err_norm in zip(pts_query_visible, pts_ref_visible, norm_err):
        pt1 = tuple(np.round([x1, y1]).astype(int))
        pt2 = tuple(np.round([x2, y2]).astype(int))
        color = tuple(int(255 * c) for c in cmap(err_norm)[:3][::-1])  # RGB → BGR

        draw_strong_curved_arrow(frame, pt1, pt2, color, thickness=1)
    # 保存结果图（无 colorbar）
    method = query_dir.split('/')[-1]
    cv2.imwrite(f'{output_path}/{i}_{method}.png', frame)
    print(f'{output_path}/{i}_{method}.png')





print(f"✅ 完成：视频已保存到 {output_path}")
